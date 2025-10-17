import torch
import torch.nn.functional as F
import numpy as np
from .losses import SMPLXLoss, bone_length_loss
from .gaussian_diffusion import GaussianDiffusion, _WrappedModel


class MotionDiffusion(GaussianDiffusion):

    def __init__(self, use_timesteps,
                 smpl_layer=None, use_smpl_loss=True, smpl_loss_type="l2", smpl_loss_weight=1.0,
                 trans_loss_weight=1.0, mse_loss_weight=1.0,
                 use_contact_loss=False, 
                 contact_loss_weight=1.0, distance_loss_weight=1.0,
                 use_foot_contact_loss=False, foot_contact_loss_weight=1.0,
                 use_relative_rot_loss=False, relative_rot_loss_weight=1.0,
                 use_vel_loss=False, vel_loss_weight=1.0, foot_vel_loss_scale=1.0,
                 bone_loss_weight=0.0,
                 **kwargs):
        self.use_timesteps = set(use_timesteps)
        self.timestep_map = []
        self.original_num_steps = len(kwargs["betas"])

        base_diffusion = GaussianDiffusion(
            **kwargs)  # pylint: disable=missing-kwoa
        last_alpha_cumprod = 1.0
        new_betas = []
        for i, alpha_cumprod in enumerate(base_diffusion.alphas_cumprod):
            if i in self.use_timesteps:
                new_betas.append(1 - alpha_cumprod / last_alpha_cumprod)
                last_alpha_cumprod = alpha_cumprod
                self.timestep_map.append(i)
        kwargs["betas"] = np.array(new_betas)

        loss_dict = {
            "loss_type": smpl_loss_type,
            "loss_weight": smpl_loss_weight,
            "use_contact_loss": use_contact_loss,
            "contact_loss_weight": contact_loss_weight,
            "distance_loss_weight": distance_loss_weight,
            "use_foot_contact_loss": use_foot_contact_loss,
            "foot_contact_loss_weight": foot_contact_loss_weight,
            "use_relative_rot_loss": use_relative_rot_loss,
            "relative_rot_loss_weight": relative_rot_loss_weight,
            "use_vel_loss": use_vel_loss,
            "vel_loss_weight": vel_loss_weight,
            "foot_vel_loss_scale": foot_vel_loss_scale,

        }
        if smpl_layer is not None:
            self.smpl_loss = SMPLXLoss(smplx_layers=smpl_layer, **loss_dict)
        else:
            self.smpl_loss = None

        self.trans_loss_weight = trans_loss_weight
        self.mse_loss_weight = mse_loss_weight
        self.bone_loss_weight = bone_loss_weight

        super().__init__(**kwargs)

    def p_mean_variance(self, model, *args, **kwargs):
        return super().p_mean_variance(self._wrap_model(model), *args, **kwargs)

    def training_contact_motion_loss(self, model, *args, **kwargs):
        items = super().training_losses(self._wrap_model(model), *args, **kwargs)

        prediction = items["pred"]  # (B, P, T, D)
        target = items["target"]  # (B, P, T, D)
        valid_mask = kwargs['model_kwargs']['valid_mask']  # (B, P, T)

        pred_motion, pred_trans = prediction[..., :-3], prediction[..., -3:]
        target_motion, target_trans = target[..., :-3], target[..., -3:]

        mse_loss_pose = self.mse_loss_weight * F.mse_loss(pred_motion, target_motion,
                                                          reduction='none')  # (B, P, T, D)
        mse_loss_pose = (mse_loss_pose * valid_mask[:, :, :, None]).sum() / (
            valid_mask.sum() + 1.e-7) / mse_loss_pose.shape[-1]

        losses = {}
        losses["mse_loss_pose"] = mse_loss_pose

        mse_loss_trans = self.trans_loss_weight * F.mse_loss(pred_trans, target_trans, reduction='none')
        mse_loss_trans = (mse_loss_trans * valid_mask[:, :, :, None]).sum() / (
            valid_mask.sum() + 1.e-7) / mse_loss_trans.shape[-1]
        losses["mse_loss_trans"] = mse_loss_trans

        loss = 0
        loss += mse_loss_pose
        loss += mse_loss_trans

        x_inter = kwargs['model_kwargs']['x_inter']
        prediction = prediction + x_inter
        target = target + x_inter

        if self.smpl_loss is not None:
            smpl_losses = self.smpl_loss(prediction[..., :-3], target[..., :-3], mask=valid_mask,
                                         pred_trans=prediction[..., -
                                                               3:], gt_trans=target[..., -3:],
                                         betas=kwargs['model_kwargs']["betas"],
                                         gender=kwargs['model_kwargs']["gender"])  # gender and betas
            for term in smpl_losses.keys():
                loss += smpl_losses[term]
            losses.update(smpl_losses)

        losses["total"] = loss

        return losses

    def training_text2pose_loss(self, model, joint_input_dim=22,*args, **kwargs):
        items = super().training_losses(self._wrap_model(model), *args, **kwargs)

        prediction = items["pred"]
        target = items["target"]

        pred_pose = prediction[..., :-joint_input_dim*3]
        pred_joint = prediction[..., -joint_input_dim*3:]

        target_pose = target[..., :-joint_input_dim*3]
        target_joint = target[..., -joint_input_dim*3:]

        mse_loss = self.mse_loss_weight * \
            F.mse_loss(prediction, target, reduction='mean')

        losses = {}
        losses["mse_loss"] = mse_loss

        loss = 0
        loss += mse_loss

        B, P = pred_joint.shape[:2]
        mask = torch.ones(B, P, 1, device=pred_joint.device)
        if self.smpl_loss is not None:
            smpl_losses = self.smpl_loss(pred_pose[..., :-3].unsqueeze(2), target_pose[..., :-3].unsqueeze(2),
                                         pred_trans=pred_pose[..., -3:].unsqueeze(2), gt_trans=target_pose[..., -3:].unsqueeze(2),
                                         mask=mask,
                                         betas=kwargs['model_kwargs']["betas"],
                                         gender=kwargs['model_kwargs']["gender"])  # gender and betas
            for term in smpl_losses.keys():
                loss += smpl_losses[term]
            losses.update(smpl_losses)

        bone_loss = self.bone_loss_weight * bone_length_loss(pred_joint.reshape(
            B, P, -1, 3).unsqueeze(2), target_joint.reshape(B, P, -1, 3).unsqueeze(2), mask)

        losses["bone_loss"] = bone_loss
        loss += bone_loss

        losses["total"] = loss

        return losses

    def condition_mean(self, cond_fn, *args, **kwargs):
        return super().condition_mean(self._wrap_model(cond_fn), *args, **kwargs)

    def condition_score(self, cond_fn, *args, **kwargs):
        return super().condition_score(self._wrap_model(cond_fn), *args, **kwargs)

    def _wrap_model(self, model):
        if isinstance(model, _WrappedModel):
            return model
        return _WrappedModel(
            model, self.timestep_map, self.rescale_timesteps, self.original_num_steps
        )

    def _scale_timesteps(self, t):
        # Scaling is done by the wrapped model.
        return t
