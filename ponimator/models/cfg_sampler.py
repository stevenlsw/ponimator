import torch
import torch.nn as nn


class ClassifierFreePoseModel(nn.Module):

    def __init__(self, model, cfg_scale):
        super().__init__()
        self.model = model
        self.s = cfg_scale

    def forward(self, x, timesteps, cond, pose_mask, motion_rep, **kwargs):
        B = x.shape[0]

        if pose_mask[0, 0].item() == 1 and cond.abs().sum() != 0: # pose & text condition
            x_combined = torch.cat([x, x, x], dim=0)
            timesteps_combined = torch.cat([timesteps, timesteps, timesteps], dim=0)
            cond = torch.cat([cond, torch.zeros_like(cond), torch.zeros_like(cond)], dim=0)
            pose_mask = torch.cat([0 * pose_mask, pose_mask, 0 * pose_mask], dim=0)
            motion_rep = torch.cat([motion_rep, motion_rep, motion_rep], dim=0)
            out = self.model(x_combined, timesteps_combined, cond=cond, pose_mask=pose_mask, motion_rep=motion_rep, **kwargs)
            out_uncond_pose = out[:B]
            out_uncond_text = out[B:2*B]
            out_uncond = out[2*B:]
            cfg_out = out_uncond + self.s * (out_uncond_text - out_uncond) + self.s * (out_uncond_pose - out_uncond)
         
        elif pose_mask[0, 0].item() == 1 and cond.abs().sum() == 0: # pose only
            x_combined = torch.cat([x, x], dim=0)
            timesteps_combined = torch.cat([timesteps, timesteps], dim=0)
            cond = torch.cat([cond, cond], dim=0)
            pose_mask = torch.cat([0 * pose_mask, pose_mask], dim=0)
            motion_rep = torch.cat([motion_rep, motion_rep], dim=0)

            out = self.model(x_combined, timesteps_combined, cond=cond, pose_mask=pose_mask, motion_rep=motion_rep, **kwargs)
            out_uncond = out[:B]
            out_cond = out[B:]
            cfg_out = out_uncond + self.s * (out_cond - out_uncond)

        elif pose_mask[0, 0].item() == 0 and cond.abs().sum() != 0: # text only
            x_combined = torch.cat([x, x], dim=0)
            timesteps_combined = torch.cat([timesteps, timesteps], dim=0)
            cond = torch.cat([torch.zeros_like(cond), cond], dim=0)
            pose_mask = torch.cat([pose_mask, pose_mask], dim=0)
            motion_rep = torch.cat([motion_rep, motion_rep], dim=0)

            out = self.model(x_combined, timesteps_combined, cond=cond, pose_mask=pose_mask, motion_rep=motion_rep, **kwargs)
            out_uncond = out[:B]
            out_cond = out[B:]
            cfg_out = out_uncond + self.s * (out_cond - out_uncond)

        else:
            out = self.model(x, timesteps, cond=cond, pose_mask=pose_mask, motion_rep=motion_rep, **kwargs)
            cfg_out = out
        
        return cfg_out

