import torch
import roma
import numpy as np
from torch import nn
from aitviewer.configuration import CONFIG as C
from aitviewer.models.smpl import SMPLLayer
from .motion_diffusion import MotionDiffusion
from .gaussian_diffusion import (
    space_timesteps,
    get_named_beta_schedule,
    create_named_schedule_sampler,
    ModelMeanType,
    ModelVarType,
    LossType
)
from .layers import TransformerBlock, FinalLayer
from ponimator.utils.model_utils import zero_module, TimestepEmbedder, PositionalEncoding
from ponimator.utils.utils import process_gender


class ContactMotionGen(nn.Module):
    def __init__(self,cfg):
        super().__init__()
        self.cfg = cfg
        self.latent_dim = cfg.LATENT_DIM
        self.decoder = ContactMotionDiffusion(cfg,sampling_strategy=cfg.STRATEGY, body_model_root=cfg.body_model_root) 

    def compute_loss(self, batch):
        losses = self.decoder.compute_loss(batch)
        return losses["total"], losses

    def decode_motion(self, batch):
        # inference only
        motion = self.decoder(batch)
        batch.update(motion)
        return batch

    def forward(self, batch):
        return self.compute_loss(batch)

    def forward_test(self, batch):
        batch.update(self.decode_motion(batch))
        return batch
    

class ContactMotionDiffusion(nn.Module):
    def __init__(self, cfg, sampling_strategy="ddim50", body_model_root="body_models"):
        super().__init__()
        self.cfg = cfg
        self.nfeats = cfg.INPUT_DIM
        self.latent_dim = cfg.LATENT_DIM
        self.ff_size = cfg.FF_SIZE
        self.num_layers = cfg.NUM_LAYERS
        self.num_heads = cfg.NUM_HEADS
        self.dropout = cfg.DROPOUT
        self.activation = cfg.ACTIVATION

        self.cfg_weight = cfg.CFG_WEIGHT
        self.diffusion_steps = cfg.DIFFUSION_STEPS
        self.beta_scheduler = cfg.BETA_SCHEDULER
        self.sampling_strategy = sampling_strategy

        self.joint_input_dim = 22 # body joints
        
        # interactive pose aug
        self.use_aug = cfg.use_aug if hasattr(cfg, "use_aug") else False
        self.aug_prob = cfg.aug_prob if hasattr(cfg, "aug_prob") else 0.3
        self.rand_angles = cfg.rand_angles if hasattr(cfg, "rand_angles") else 2
        self.rand_trans = cfg.rand_trans if hasattr(cfg, "rand_trans") else 0.02
        self.rand_beta_scale = cfg.rand_beta_scale if hasattr(cfg, "rand_beta_scale") else 0.02
        
        self.net = ContactMotionDenoiser(self.nfeats, self.latent_dim, ff_size=self.ff_size, num_layers=self.num_layers, 
                                     num_heads=self.num_heads, dropout=self.dropout, activation=self.activation, 
                                     cfg_weight=self.cfg_weight, 
                                     joint_input_dim=self.joint_input_dim,
                                     use_condition_embedding=cfg.use_condition_embedding)
            
        
        self.diffusion_steps = self.diffusion_steps
        self.betas = get_named_beta_schedule(self.beta_scheduler, self.diffusion_steps)

        timestep_respacing=[self.diffusion_steps]

    
        C.update_conf({'smplx_models': body_model_root})
        # expose to enable multi-gpu training
        self.smpl_layer_male = SMPLLayer(model_type='smplx',gender='male', num_betas=10, dtype=torch.float32, device=torch.device("cpu")).eval()
        self.smpl_layer_female = SMPLLayer(model_type='smplx', gender='female', num_betas=10, dtype=torch.float32, device=torch.device("cpu")).eval()
        self.smpl_layer_neutral = SMPLLayer(model_type='smplx', gender='neutral', num_betas=10, dtype=torch.float32, device=torch.device("cpu")).eval()
        self.smpl_layer = [self.smpl_layer_male, self.smpl_layer_female, self.smpl_layer_neutral]

        self.diffusion = MotionDiffusion(
            use_timesteps=space_timesteps(self.diffusion_steps, timestep_respacing),
            betas=self.betas,
            model_mean_type=ModelMeanType.START_X,
            model_var_type=ModelVarType.FIXED_SMALL,
            loss_type=LossType.MSE,
            rescale_timesteps = False,
            use_smpl_loss=cfg.use_smpl_loss,
            smpl_layer=self.smpl_layer, smpl_loss_type=cfg.smpl_loss_type, smpl_loss_weight=cfg.smpl_loss_weight, 
            trans_loss_weight=cfg.trans_loss_weight, mse_loss_weight=cfg.mse_loss_weight, 
            use_contact_loss=cfg.use_contact_loss, 
            contact_loss_weight=cfg.contact_loss_weight, distance_loss_weight=cfg.distance_loss_weight if hasattr(cfg, "distance_loss_weight") else 0.0,
            use_foot_contact_loss=cfg.use_foot_contact_loss, foot_contact_loss_weight=cfg.foot_contact_loss_weight, 
            use_relative_rot_loss=cfg.use_relative_rot_loss, relative_rot_loss_weight=cfg.relative_rot_loss_weight,
            use_vel_loss=cfg.use_vel_loss, 
            vel_loss_weight=cfg.vel_loss_weight, 
            foot_vel_loss_scale=cfg.foot_vel_loss_scale)
        
        self.sampler = create_named_schedule_sampler(cfg.SAMPLER, self.diffusion)

    def compute_loss(self, batch):
        # training only

        x_motion = batch["motions"] - batch["inter_pose"] # (B, P, T, D)
        x_trans = batch['trans'] - batch["inter_trans"]
        motion_rep = torch.cat([x_motion, x_trans], dim=-1) # (B, P, T, D+3) diffusion target
        inter_rep = torch.cat([batch["inter_pose"], batch["inter_trans"]], dim=-1) # (B, P, 1, D+3)

        if self.use_aug and np.random.rand() < self.aug_prob: # training augmentation
            B, P = inter_rep.shape[:2]
            inter_pose = batch["inter_pose"].squeeze(2).reshape(B, P, -1, 6)
            root_orient = inter_pose[:, :, 0] # (B, P, 6)
            rest_inter_pose = inter_pose[:, :, 1:] # (B, P, *, 6)
            root_rotmat = roma.special_gramschmidt(root_orient.reshape(-1, 3, 2)).reshape(B, P, 3, 3)
            noise_rot = torch.randn_like(batch["inter_trans"].squeeze(2), device=root_orient.device) * self.rand_angles / 180 * np.pi
            noise_rotmat = roma.euler_to_rotmat('ZYX', noise_rot) # (B, P, 3, 3)
            new_root_orient = torch.matmul(noise_rotmat, root_rotmat)[..., :, :2].reshape(B, P, 6) # (B, P, 6)
            new_inter_pose = torch.cat([new_root_orient.unsqueeze(2), rest_inter_pose], dim=2).reshape(B, P, -1)
            new_inter_pose = new_inter_pose.unsqueeze(2) # (B, P, 1, D)
            noise_trans = torch.randn_like(batch["inter_trans"], device=batch["inter_trans"].device)
            # only add noise to x and z (no height)
            new_inter_trans = batch["inter_trans"].clone()
            new_inter_trans[..., [0, 2]] = batch["inter_trans"][..., [0, 2]] + self.rand_trans * noise_trans[..., [0, 2]]
            new_betas = batch["betas"] + self.rand_beta_scale * torch.randn_like(batch["betas"], device=batch["betas"].device)
            new_inter_rep = torch.cat([new_inter_pose, new_inter_trans], dim=-1) # (B, P, 1, D+3)
        else:
            new_inter_pose = batch["inter_pose"] # (B, P, 1, D)
            new_inter_trans = batch["inter_trans"]
            new_betas = batch["betas"]
            new_inter_rep = None
      
        _, inter_joints = process_gender(self.smpl_layer, new_inter_pose, new_inter_trans, new_betas, batch['gender']) # (B, P, 1, J, 3)
        
        # encode shape information
        inter_joints = inter_joints[:, :, :, :22, :]
        
        B, P, T = batch["motions"].shape[:3]

        mid_index = batch["mid_index"].view(B, 1, 1, 1).expand(B, P, T, 1)
        inter_mask = torch.zeros((B, P, T, 1), device=x_motion.device) # (B, P, T, 1)
        inter_mask.scatter_(2, mid_index, 1)
        
        t, _ = self.sampler.sample(B, x_motion.device)
        
        output = self.diffusion.training_contact_motion_loss(
            # those outside are used for loss computation, x_start, t, model_kwargs are used for model forward
            model=self.net,
            x_start=motion_rep,  # (B, P, T, D)
            t=t,
            # model_kwargs: network input
            model_kwargs={"x_inter": inter_rep, # B, P, 1, D
                        "inter_mask": inter_mask, # B, P, T+1
                        "inter_joints": inter_joints, # B, P, 1, J, 3
                        "valid_mask": batch["valid_mask"], # B, T
                        "betas": batch["betas"],
                        "gender":  batch["gender"],
                        "aug_x_inter": new_inter_rep, 
                        }
            )
        return output

    def forward(self, batch):
        # inference-only
        _, inter_joints = process_gender(self.smpl_layer, batch["inter_pose"], batch["inter_trans"], batch['betas'], batch['gender']) # (B, P, 1, J, 3)
        
        inter_joints = inter_joints[:, :, :, :22, :]    
        inter_rep = torch.cat([batch["inter_pose"], batch["inter_trans"]], dim=-1) # (B, P, 1, D+3)
        
        B, P, T = batch["motions"].shape[:3] # TODO: set T as dataset requires length

        mid_index = batch["mid_index"].view(B, 1, 1, 1).expand(B, P, T, 1)
        inter_mask = torch.zeros((B, P, T, 1), device=inter_rep.device) # (B, P, T, 1)
        inter_mask.scatter_(2, mid_index, 1)

        timestep_respacing= self.sampling_strategy
        self.diffusion_test = MotionDiffusion( # inference diffusion
            use_timesteps=space_timesteps(self.diffusion_steps, timestep_respacing),
            betas=self.betas,
            model_mean_type=ModelMeanType.START_X,
            model_var_type=ModelVarType.FIXED_SMALL,
            loss_type=LossType.MSE,
            rescale_timesteps = False,
            no_normalizer=True
        )

        output = self.diffusion_test.ddim_sample_loop(
            self.net,
            (B, P, T, self.nfeats),
            clip_denoised=False,
            progress=True,
            model_kwargs={
                "x_inter": inter_rep, # B, P, 1, D
                "inter_mask": inter_mask, # B, P, T, 1
                "inter_joints": inter_joints # B, P, 1, J, 3
            },
            x_start=None) # output: B, P, T, D+3
        res_motion = output[:, :, :, :-3] # B, P, T, D
        trans = output[:, :, :, -3:] # B, P, T, 3
        motion = res_motion + batch["inter_pose"] # B, P, T, D
        trans += batch["inter_trans"]
        output = torch.cat([motion, trans], dim=-1) # B, P, T, D+3
        return {"output": output}
    

    def forward_inference(self, inter_pose, inter_trans, inter_joints, seq_len, mid_index=None):
        # in-the-wild inference
        # mid_idx: interactive pose index in the generated sequence (B, ) (default: seq_len // 2)
        inter_joints = inter_joints[:, :, :, :22, :]
        inter_rep = torch.cat([inter_pose, inter_trans], dim=-1) # (B, P, 1, D+3)
        
        B, P, T = 1, 2, seq_len
        if mid_index is None:
            mid_index = seq_len // 2
        if isinstance(mid_index, int):    
            mid_index = torch.tensor(mid_index, device=inter_rep.device)
        mid_index = mid_index.view(B, 1, 1, 1).expand(B, P, T, 1) 
        inter_mask = torch.zeros((B, P, T, 1), device=inter_rep.device) # (B, P, T, 1)
        inter_mask.scatter_(2, mid_index, 1)

        timestep_respacing= self.sampling_strategy
        self.diffusion_test = MotionDiffusion( # inference diffusion
            use_timesteps=space_timesteps(self.diffusion_steps, timestep_respacing),
            betas=self.betas,
            model_mean_type=ModelMeanType.START_X,
            model_var_type=ModelVarType.FIXED_SMALL,
            loss_type=LossType.MSE,
            rescale_timesteps = False)

        output = self.diffusion_test.ddim_sample_loop(
            self.net,
            (B, P, T, self.nfeats),
            clip_denoised=False,
            progress=True,
            model_kwargs={
                "x_inter": inter_rep, # B, P, 1, D
                "inter_mask": inter_mask, # B, P, T, 1
                "inter_joints": inter_joints # B, P, 1, J, 3
            },
            x_start=None) # output: B, P, T, D+3
        res_motion = output[:, :, :, :-3] # B, P, T, D
        trans = output[:, :, :, -3:] # B, P, T, 3
        motion = res_motion + inter_pose # B, P, T, D
        trans += inter_trans
        output = torch.cat([motion, trans], dim=-1) # B, P, T, D+3
        return {"output": output}



class ContactMotionDenoiser(nn.Module):
    def __init__(self,
                 input_feats,
                 latent_dim=512,
                 num_frames=240,
                 ff_size=1024,
                 num_layers=8,
                 num_heads=8,
                 dropout=0.1,
                 activation="gelu",
                 cfg_weight=0.,
                 joint_input_dim=22,
                 use_condition_embedding=False,
                 **kargs):
        super().__init__()

        self.cfg_weight = cfg_weight
        self.num_frames = num_frames
        self.latent_dim = latent_dim
        self.ff_size = ff_size
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.dropout = dropout
        self.activation = activation
        self.input_feats = input_feats
        self.time_embed_dim = latent_dim
        
        self.sequence_pos_encoder = PositionalEncoding(self.latent_dim, dropout=0)
        self.embed_timestep = TimestepEmbedder(self.latent_dim, self.sequence_pos_encoder)

         # Input Embedding
        self.use_condition_embedding = use_condition_embedding
        if not self.use_condition_embedding: # Deprecated: old-architecture
            self.inter_embed = nn.Linear(2 * self.input_feats, self.latent_dim) # 2 person feats
            self.joint_embed_layer = nn.Linear(2 * joint_input_dim * 3, self.latent_dim) # interactive joints
        else:
            self.condition_embed_layer = nn.Sequential(
                nn.Linear(2 * self.input_feats + 2 * joint_input_dim * 3, self.latent_dim), 
                nn.SiLU(),
                nn.Linear(self.latent_dim, self.latent_dim))

            self.conditon_embed = nn.Parameter(torch.randn(self.latent_dim))
  
        self.motion_embed = nn.Linear(self.input_feats+1, self.latent_dim)
        
        self.blocks = nn.ModuleList()
        for i in range(num_layers):
            self.blocks.append(TransformerBlock(num_heads=num_heads,latent_dim=latent_dim, dropout=dropout, ff_size=ff_size))
        
        self.motion_out = zero_module(FinalLayer(self.latent_dim, self.input_feats-3))
        self.trans_out = nn.Linear(self.latent_dim, 3, bias=True)
    
    def forward(self, x, timesteps, x_inter, inter_mask, inter_joints, valid_mask=None, aug_x_inter=None):
        # ensure first two arguments are x_start and time
        """
        x: B, P, T, D+3
        x_inter: B, P, 1, D+3
        inter_mask: B, P, T, 1
        inter_joints: B, P, 1, J, 3
        valid_mask: B, P, T
        """
        if aug_x_inter is not None:
            inter_input = aug_x_inter
        else:
            inter_input = x_inter 

        x = x + inter_input
            
        B, P, T = x.shape[:3]
        x = (1 - inter_mask) * x + inter_mask * inter_input.expand(B, P, T, inter_input.shape[-1])
        
         # (B, P), encode interaction information
        if not self.use_condition_embedding:
            emb = self.embed_timestep(timesteps) + self.inter_embed(inter_input[:, :2].reshape(B, -1)) + self.joint_embed_layer(inter_joints[:, :2].reshape(B, -1))
        else:
            condition_ = torch.cat([inter_input[:, :2].reshape(B, -1), inter_joints[:, :2].reshape(B, -1)], dim=-1)
            condition_ = self.condition_embed_layer(condition_)
            emb = self.embed_timestep(timesteps) + condition_ + self.conditon_embed
        
        x_in = torch.cat([x, inter_mask], dim=-1)
        
        x_emb = self.motion_embed(x_in) # (B, P, T, D)
        D = x_emb.shape[-1]
        
        x_emb = x_emb.reshape(B*P, -1, D)
        h = self.sequence_pos_encoder(x_emb) # PE 
        h = h.reshape(B, P, -1, D)

        if valid_mask is None:
            valid_mask = torch.ones(B, P, T, device=x.device)
        key_padding_mask = ~(valid_mask > 0.5) # for invalid positions, set as 1

        for i, block in enumerate(self.blocks):
            if i%2 == 0: # temporal
                h = h.reshape(B*P, -1, D)
                emb_in = emb.unsqueeze(1).expand(B, P, self.latent_dim).reshape(B*P, -1)
                key_padding_mask = key_padding_mask.reshape(B*P, -1)
                h = block(h, emb_in, key_padding_mask)
                h = h.contiguous().view(B, P, -1, D)
                key_padding_mask = key_padding_mask.reshape(B, P, -1)
            else: # spatial
                h = h.permute(0, 2, 1, 3).contiguous().view(-1, P, D)
                emb_in = emb.unsqueeze(1).expand(B, T, self.latent_dim).reshape(B*T, -1)
                key_padding_mask = key_padding_mask.permute(0, 2, 1).contiguous().view(B*T, P)
                h = block(h, emb_in, key_padding_mask)
                h = h.contiguous().view(B, T, P, -1).permute(0, 2, 1, 3)
                key_padding_mask = key_padding_mask.contiguous().view(B, T, P).permute(0, 2, 1) # (B, P, T)
                
        motion = self.motion_out(h)
        trans = self.trans_out(h)
        output = torch.cat([motion, trans], dim=-1)
        return output
