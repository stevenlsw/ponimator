import torch
from torch import nn
import clip
from aitviewer.configuration import CONFIG as C
from aitviewer.models.smpl import SMPLLayer

from ponimator.utils.model_utils import set_requires_grad, TimestepEmbedder, PositionalEncoding
from ponimator.utils.utils import process_gender
from .cfg_sampler import ClassifierFreePoseModel
from .motion_diffusion import MotionDiffusion
from .gaussian_diffusion import (
    space_timesteps,
    get_named_beta_schedule,
    create_named_schedule_sampler,
    ModelMeanType,
    ModelVarType,
    LossType
)
from .layers import TransformerBlock


class ContactPoseGen(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.latent_dim = cfg.LATENT_DIM
        self.decoder = ContactPoseDiffusion(cfg, sampling_strategy=cfg.STRATEGY, body_model_root=cfg.body_model_root)
        clip_model, _ = clip.load("ViT-L/14@336px", device="cpu", jit=False)

        self.token_embedding = clip_model.token_embedding
        self.clip_transformer = clip_model.transformer
        self.positional_embedding = clip_model.positional_embedding
        self.ln_final = clip_model.ln_final
        self.dtype = clip_model.dtype

        set_requires_grad(self.clip_transformer, False)
        set_requires_grad(self.token_embedding, False)
        set_requires_grad(self.ln_final, False)

        clipTransEncoderLayer = nn.TransformerEncoderLayer(
            d_model=768,
            nhead=8,
            dim_feedforward=2048,
            dropout=0.1,
            activation="gelu",
            batch_first=True)
        self.clipTransEncoder = nn.TransformerEncoder(
            clipTransEncoderLayer,
            num_layers=2)
        self.clip_ln = nn.LayerNorm(768)

    def compute_loss(self, batch):
        batch = self.text_process(batch)
        losses = self.decoder.compute_loss(batch)
        return losses["total"], losses

    def decode_motion(self, batch):
        # inference only
        batch = self.text_process(batch)
        motion = self.decoder(batch)
        batch.update(motion)
        return batch

    def forward(self, batch):
        return self.compute_loss(batch)

    def forward_test(self, batch):
        batch.update(self.decode_motion(batch))
        return batch
    
    def text_process(self, batch):
        device = next(self.clip_transformer.parameters()).device
        raw_text = batch["text"]
        mask = torch.tensor([0 if s == "" else 1 for s in raw_text], dtype=torch.int, device=device)

        with torch.no_grad():

            text = clip.tokenize(raw_text, truncate=True).to(device)
            x = self.token_embedding(text).type(self.dtype)  # [batch_size, n_ctx, d_model]
            pe_tokens = x + self.positional_embedding.type(self.dtype)
            x = pe_tokens.permute(1, 0, 2)  # NLD -> LND
            x = self.clip_transformer(x)
            x = x.permute(1, 0, 2)
            clip_out = self.ln_final(x).type(self.dtype)

        out = self.clipTransEncoder(clip_out)
        out = self.clip_ln(out) # (B, 77, 768)

        cond = out[torch.arange(x.shape[0]), text.argmax(dim=-1)] # text (B, 77)
        
        cond = cond * mask.unsqueeze(-1).float()
        batch["cond"] = cond # (B, 768)

        return batch
    

class ContactPoseDiffusion(nn.Module):
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
        self.joint_input_dim = 22
        
        self.net = ContactPoseDenoiser(self.nfeats, self.latent_dim, ff_size=self.ff_size, num_layers=self.num_layers, 
                                     num_heads=self.num_heads, dropout=self.dropout, activation=self.activation, 
                                     cfg_weight=self.cfg_weight, joint_input_dim=self.joint_input_dim)
        
        self.diffusion_steps = self.diffusion_steps
        self.betas = get_named_beta_schedule(self.beta_scheduler, self.diffusion_steps)

        timestep_respacing=[self.diffusion_steps]

        C.update_conf({'smplx_models': body_model_root})
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
            use_smpl_loss=True,
            smpl_layer=self.smpl_layer,  
            smpl_loss_weight=cfg.smpl_loss_weight, 
            trans_loss_weight=cfg.trans_loss_weight, mse_loss_weight=cfg.mse_loss_weight, 
            use_contact_loss=cfg.use_contact_loss,
            contact_loss_weight=cfg.contact_loss_weight, distance_loss_weight=cfg.distance_loss_weight if hasattr(cfg, "distance_loss_weight") else 0.0,
            use_foot_contact_loss=cfg.use_foot_contact_loss, foot_contact_loss_weight=cfg.foot_contact_loss_weight, 
            use_relative_rot_loss=cfg.use_relative_rot_loss, relative_rot_loss_weight=cfg.relative_rot_loss_weight,
            use_vel_loss=cfg.use_vel_loss, 
            vel_loss_weight=cfg.vel_loss_weight, 
            foot_vel_loss_scale=cfg.foot_vel_loss_scale,
            bone_loss_weight=cfg.bone_loss_weight)
        
        self.sampler = create_named_schedule_sampler(cfg.SAMPLER, self.diffusion)

        self.pose_mask_out_prob = cfg.pose_mask_out_prob
        self.text_mask_out_prob = cfg.text_mask_out_prob

    def mask_cond(self, cond, cond_mask_prob=0.1, force_mask=False):
        bs = cond.shape[0]
        if force_mask:
            return torch.zeros_like(cond)
        elif cond_mask_prob > 0.:
            mask = torch.bernoulli(torch.ones(bs, device=cond.device) * cond_mask_prob).view([bs]+[1]*len(cond.shape[1:]))  # 1-> use null_cond, 0-> use real cond
            return cond * (1. - mask), (1. - mask)
        else:
            return cond, None
    
    def compute_loss(self, batch):
        # training only
        cond = batch["cond"]

        motion_rep = torch.cat([batch['motions'], batch['trans']], dim=-1) # (B, P, D+3)
        
        B, P = batch["motions"].shape[:2]
        D = batch["motions"].shape[-1] // 6
        # in utils/utils.py: we use d6 .reshape(-1, 3, 2) so 1, 0, 0, 1, 0, 0 is 1, 0 // 0, 1 // 0, 0 for the first two column
        template = torch.tensor([1, 0, 0, 1, 0, 0], device=motion_rep.device, dtype=motion_rep.dtype)
        template_motion = template.repeat(B, P, D, 1).reshape (B, P, -1) # (B, P, D*6)
        _, rest_joints = process_gender(self.smpl_layer, template_motion.unsqueeze(2), 0 * batch["trans"].unsqueeze(2), batch['betas'], batch['gender']) # (B, P, 1, J, 3)
        rest_joints = rest_joints.squeeze(2) # (B, P, J, 3)
        rest_joints = rest_joints[:, :, :22]
        rest_joints = rest_joints.reshape(B, P, -1) # (B, P, *)
        
        motion_rep = torch.cat([motion_rep, rest_joints], dim=-1) # (B, P, D*), D*=D+3+J*3
        
        if cond is not None:
            cond, cond_mask = self.mask_cond(cond, self.text_mask_out_prob) # 1 - text_mask_out_prob: keep text

        # generate pose mask condition
        random_values = torch.rand(B, device=motion_rep.device, dtype=motion_rep.dtype)
        pose_mask = torch.zeros((B, 2), device=motion_rep.device, dtype=motion_rep.dtype)
        pose_mask[random_values < 1 - self.pose_mask_out_prob, 0] = 1 # 1 - self.pose_mask_out_prob: keep first-person pose 

        t, _ = self.sampler.sample(B, motion_rep.device)
        output = self.diffusion.training_text2pose_loss(
            # those outside are used for loss computation, x_start, t, model_kwargs are used for model forward
            model=self.net,
            x_start=motion_rep,  # (B, 1, *)
            t=t,
            # model_kwargs: network input
            model_kwargs={"cond": cond,  # (B,  *)
                          "pose_mask": pose_mask,  # (B, 2) # 1 in pose_mask means keep first-person pose
                          "motion_rep": motion_rep,  # (B, P, D*)
                          "betas": batch["betas"],
                          "gender":  batch["gender"]
                          }
        )
        return output

    def forward(self, batch):
        # inference only: for different conditions
        cond = batch["cond"] # if no text input, send text="" and cond are zero tensor
       # pose_mask: (B, P) optional, B=1, P=2
        if "pose_mask" in batch and batch["pose_mask"][0, 0].item() == 1: # given pose as condition
            motion_rep = torch.cat([batch['motions'], batch['trans']], dim=-1) 
            B, P = batch["motions"].shape[:2]
            D = batch["motions"].shape[-1] // 6
            if "rest_joints" in batch:
                rest_joints = batch["rest_joints"] # (B, P, 22, 3)
            else:
                template = torch.tensor([1, 0, 0, 1, 0, 0], device=motion_rep.device, dtype=motion_rep.dtype)
                template_motion = template.repeat(B, P, D, 1).reshape (B, P, -1) # (B, P, D*6)
                trans = torch.zeros((B, P, 3), device=motion_rep.device, dtype=motion_rep.dtype)
                _, rest_joints = process_gender(self.smpl_layer, template_motion.unsqueeze(2), trans.unsqueeze(2), batch['betas'], batch['gender']) # (B, P, 1, J, 3)
                rest_joints = rest_joints.squeeze(2) # (B, P, J, 3)
                rest_joints = rest_joints[:, :, :22]
    
            rest_joints = rest_joints.reshape(B, P, -1) # (B, P, *)
            motion_rep = torch.cat([motion_rep, rest_joints], dim=-1) # (B, P, D*), D*=D+3+J*3
            pose_mask = batch["pose_mask"]
        else:
            B = cond.shape[0]
            motion_rep = torch.zeros((B, 2, self.nfeats), device=cond.device, dtype=cond.dtype)
            pose_mask = torch.zeros((B, 2), device=cond.device, dtype=cond.dtype)

        timestep_respacing= self.sampling_strategy
        self.diffusion_test = MotionDiffusion(
            use_timesteps=space_timesteps(self.diffusion_steps, timestep_respacing),
            betas=self.betas,
            model_mean_type=ModelMeanType.START_X,
            model_var_type=ModelVarType.FIXED_SMALL,
            loss_type=LossType.MSE,
            rescale_timesteps = False,
        )

        self.cfg_model = ClassifierFreePoseModel(self.net, self.cfg_weight)

        output = self.diffusion_test.ddim_sample_loop(
            self.cfg_model,
            (B, 2, self.nfeats),
            clip_denoised=False,
            progress=True,
            model_kwargs={
               "cond": cond,  # (B,  *)
                "pose_mask": pose_mask,  # (B, 2)
                "motion_rep": motion_rep,  # (B, P, D*)
            },
            x_start=None)
        return {"output": output} # output: B, P, *
    

class ContactPoseDenoiser(nn.Module):
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
                 joint_input_dim=55,
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

        self.text_emb_dim = 768
        self.sequence_pos_encoder = PositionalEncoding(self.latent_dim, dropout=0)
        self.embed_timestep = TimestepEmbedder(self.latent_dim, self.sequence_pos_encoder)
        
        self.input_embed_layer = nn.Linear(self.input_feats+1, self.latent_dim)
        self.input_embed = nn.Parameter(torch.randn(self.latent_dim))
        self.text_embed = nn.Linear(self.text_emb_dim, self.latent_dim)
        
        self.blocks = nn.ModuleList()
        for i in range(num_layers):
            self.blocks.append(TransformerBlock(num_heads=num_heads,latent_dim=latent_dim, dropout=dropout, ff_size=ff_size))
      
        self.pose_out = nn.Linear(self.latent_dim, self.input_feats-joint_input_dim*3, bias=True)
        self.joint_out = nn.Linear(self.latent_dim, joint_input_dim*3, bias=True)

    
    def forward(self, x, timesteps, cond, pose_mask, motion_rep, betas=None, gender=None):
        # ensure first two arguments are x_start and time
        """
        x: B, P, *
        cond: B, *
        pose_mask: B, P
        motion_rep: B, P, *
        """
            
        # (B, *), encode interaction information
        B = x.shape[0]

        emb = self.embed_timestep(timesteps) + self.text_embed(cond) # (B, *)
        pose_mask = pose_mask.unsqueeze(-1)
        # pose_mask=1, use motion_rep; pose_mask=0 (not keep first-person pose), use x (noisy)
        x = (1 - pose_mask) * x + pose_mask * motion_rep # pose_mask (B, 2, 1)  motion_rep (B, 1, D*)

        x_in = torch.cat([x, pose_mask], dim=-1) # (B, 2, *+1)
        h = self.input_embed_layer(x_in) # (B, 2, *)
        h += self.input_embed  # (B, 2, *)
        
        for i, block in enumerate(self.blocks):
            h = block(h, emb)

        pose = self.pose_out(h)
        joint = self.joint_out(h)
        output = torch.cat([pose, joint], dim=-1)
        return output
