import torch
import torch.nn as nn
import smplx
import roma
import torch.nn.functional as F

from ponimator.utils.utils import rotation_6d_to_aa, process_gender

kinematic_chain = [[0, 2, 5, 8, 11],
                 [0, 1, 4, 7, 10],
                 [0, 3, 6, 9, 12, 15],
                 [9, 14, 17, 19, 21],
                 [9, 13, 16, 18, 20]]


def joint_contact_loss(pred_joints, gt_joints, mask, thresh=1.0, contact_thresh=0.1):
    # pred_joints: (B, P, T, J, 3)
    # gt_joints: (B, P, T, J, 3)
    # mask: (B, P, T)
    B, P, T, J, _ = pred_joints.shape
    pred_joints = pred_joints.permute(0, 2, 1, 3, 4).reshape(B*T, P, J, 3)
    gt_joints = gt_joints.permute(0, 2, 1, 3, 4).reshape(B*T, P, J, 3)

    # TODO: support multiple person
    pred_joints1, pred_joints2 = pred_joints[:, 0], pred_joints[:, 1] # (B*T, J, 3)
    gt_joints1, gt_joints2 = gt_joints[:, 0], gt_joints[:, 1] # (B*T, J, 3)
    mask1, mask2 = mask[:, 0].reshape(-1), mask[:, 1].reshape(-1) # (B * T)

    pred_distance_matrix = torch.cdist(pred_joints1.contiguous(), pred_joints2) # (B*T, J, J)
    tgt_distance_matrix = torch.cdist(gt_joints1.contiguous(), gt_joints2)

    distance_mask = (pred_distance_matrix < thresh).float()
    combine_mask = (mask1 * mask2).unsqueeze(-1).unsqueeze(-1) # (B*T, J, J)
    overall_mask = combine_mask * distance_mask # (B*T, J, J)
    d_loss = (overall_mask * F.mse_loss(pred_distance_matrix, tgt_distance_matrix, reduction='none')).sum() /  (overall_mask.sum() + 1.e-7) # (B*T, J, J)
    
    contact_mask = (tgt_distance_matrix < contact_thresh).float() # (B*T, J, J)
    overall_mask = combine_mask * contact_mask # (B*T, J, J)
    c_loss = (overall_mask * F.mse_loss(pred_distance_matrix, torch.zeros_like(tgt_distance_matrix), reduction='none')).sum() /  (overall_mask.sum() + 1.e-7) # (B*T, J, J)
    
    return d_loss, c_loss


def relative_rot_loss(pred_global_rot, gt_global_rot, mask):
    # pred_global_rot, gt_global_rot: (B, P, T, 3)
    # mask: (B, P, T)
    pred_global_rotmat = roma.rotvec_to_rotmat(pred_global_rot) # (B, P, T, 3, 3)
    gt_global_rotmat = roma.rotvec_to_rotmat(gt_global_rot) # (B, P, T, 3, 3)
    pred_relative_rot = torch.matmul(pred_global_rotmat[:, 1, :], pred_global_rotmat[:, 0, :].permute(0, 1, 3, 2)) # (B, T, 3, 3)
    gt_relative_rot = torch.matmul(gt_global_rotmat[:, 1, :], gt_global_rotmat[:, 0, :].permute(0, 1, 3, 2)) # (B, T, 3, 3)
    mask1, mask2 = mask[:, 0], mask[:, 1] # (B, T)
    overall_mask = (mask1 * mask2) # (B, T)
    # ref: https://github.com/HalfSummer11/CAPTRA/blob/master/network/models/loss.py#L227
    mat_diff = gt_relative_rot - pred_relative_rot
    tmp = torch.matmul(mat_diff, mat_diff.transpose(-1, -2))
    rot_loss = tmp[..., 0, 0] + tmp[..., 1, 1] + tmp[..., 2, 2] # (B, T)
    rot_loss = (overall_mask * rot_loss).sum() /  (overall_mask.sum() + 1.e-7) # Frobenius Norm Loss
    return rot_loss


def vel_loss(pred_j, gt_j, mask, foot_vel_loss_scale=1.0):
    # pred_j, gt_j: (B, P, T, J, 3)
    # mask: (B, P, T)
    fids = [7, 10, 8, 11]
    B, P, T, J = pred_j.shape[:4]
    
    pred_vel = pred_j[:, :, 1:] - pred_j[:, :, :-1] # (B, P, T, J, 3)
    gt_vel = gt_j[:, :, 1:] - gt_j[:, :, :-1]

    all_indices = torch.arange(J)
    pred_feet_vel, gt_feet_vel = pred_vel[:, :, :, fids], gt_vel[:, :, :, fids] # (B, P, T, J*, 3)
    pred_rest_vel, gt_rest_vel = pred_vel[:, :, :, [i for i in all_indices if i not in fids]], gt_vel[:, :, :, [i for i in all_indices if i not in fids]] # (B, P, T, J*, 3)
    
    pred_rest_vel, gt_rest_vel = pred_rest_vel.reshape(B, P, T-1, -1), gt_rest_vel.reshape(B, P, T-1, -1)
    pred_feet_vel, gt_feet_vel = pred_feet_vel.reshape(B, P, T-1, -1), gt_feet_vel.reshape(B, P, T-1, -1)
    
    mask1, mask2 = mask[:, 0:1, :-1], mask[:, 1:2, :-1] # (B, 1, T-1)
    overall_mask = (mask1 * mask2).unsqueeze(-1) # (B, 1, T-1, 1)
    
    rest_vel_loss = (overall_mask * F.mse_loss(pred_rest_vel, gt_rest_vel, reduction='none')).sum() /  (P * pred_rest_vel.shape[-1] * overall_mask.sum() + 1.e-7)
    feet_vel_loss = (overall_mask * F.mse_loss(pred_feet_vel, gt_feet_vel, reduction='none')).sum() /  (P * pred_feet_vel.shape[-1] * overall_mask.sum() + 1.e-7)
    
    return rest_vel_loss + foot_vel_loss_scale* feet_vel_loss
       

def foot_detect(feet_h, front_thre=0.02, back_thre=0.09):
    heightfactor = torch.tensor([back_thre, front_thre, back_thre, front_thre], device=feet_h.device) # TODO
    contact = (feet_h < heightfactor).float()
    return contact


def foot_contact_loss(pred_j, mask, back_thre=0.09, front_thre=0.02):
    # pred_j, gt_j: shape (B, P, T, J, 3)
    # mask: (B, P, T)
    fids = [7, 10, 8, 11]
    feet_vel = pred_j[:, :, 1:, fids, :] - pred_j[:, :, :-1, fids, :] # (B, P, T, J*, 3)
    feet_h = pred_j[:, :, :-1, fids, 1] # (B, P, T, J*)
        
    foot_contact = foot_detect(feet_h, back_thre=back_thre, front_thre=front_thre) # (B, P, T-1, J*)
    mask1, mask2 = mask[:, 0:1, :-1], mask[:, 1:2, :-1] # (B, 1, T-1)
    overall_mask = (mask1 * mask2).unsqueeze(-1) * foot_contact # (B, P, T-1, J*)
    f_loss = (overall_mask[..., None] * F.mse_loss(feet_vel, torch.zeros_like(feet_vel), reduction='none')).sum() /  (3 * overall_mask.sum() + 1.e-7)
    return f_loss
 

def bone_length_loss(pred_j, gt_j, mask=None):
    # pred_j, gt_j: (B, P, T, J, 3)
    # mask: (B, P, T)
    pred_bones = []
    tgt_bones = []
    for chain in kinematic_chain:
        for i, joint in enumerate(chain[:-1]):
            pred_bone = (pred_j[..., chain[i], :] - pred_j[..., chain[i + 1], :]).norm(dim=-1, keepdim=True)  # [B, T, P, 1]
            tgt_bone = (gt_j[..., chain[i], :] - gt_j[..., chain[i + 1], :]).norm(dim=-1, keepdim=True)
            pred_bones.append(pred_bone)
            tgt_bones.append(tgt_bone)

    pred_bones = torch.cat(pred_bones, dim=-1) # [B, P, T, *]
    tgt_bones = torch.cat(tgt_bones, dim=-1) # [B, P, T, *]

    if mask is not None:
        P = pred_j.shape[1]
        mask1, mask2 = mask[:, 0:1, :], mask[:, 1:2, ] # (B, 1, T)
        overall_mask = (mask1 * mask2).unsqueeze(-1) # (B, 1, T, 1)
        bone_loss = (overall_mask * F.mse_loss(pred_bones, tgt_bones, reduction='none')).sum() /  (P * pred_bones.shape[-1] * overall_mask.sum() + 1.e-7)
    else:
        bone_loss = F.mse_loss(pred_bones, tgt_bones, reduction='mean')
    return bone_loss


class SMPLXLoss(nn.Module):
    def __init__(self, smplx_layers, loss_type="l2", loss_weight=1.0, 
                 use_contact_loss=False, 
                 contact_loss_weight=0.0, distance_loss_weight=0.0,
                 contact_thresh=0.1, 
                 use_foot_contact_loss=False, foot_contact_loss_weight=1.0, 
                 use_relative_rot_loss=False, relative_rot_loss_weight=1.0,
                 use_vel_loss=False, vel_loss_weight=1.0, foot_vel_loss_scale=1.0):
        super(SMPLXLoss, self).__init__()
        
        self.smplx_layers = smplx_layers
        for layer in self.smplx_layers:
            layer.eval()
        
        if loss_type == 'l1':
            self.Loss = torch.nn.L1Loss(reduction='none')
        elif loss_type == 'l2':
            self.Loss = torch.nn.MSELoss(reduction='none')
        elif loss_type == 'l1_smooth':
            self.Loss = torch.nn.SmoothL1Loss(reduction='none')
        
        self.loss_weight = loss_weight
        self.use_contact_loss = use_contact_loss
        self.contact_loss_weight = contact_loss_weight
        self.distance_loss_weight = distance_loss_weight
        self.contact_thre = contact_thresh

        self.use_foot_contact_loss = use_foot_contact_loss
        self.foot_contact_loss_weight = foot_contact_loss_weight
        self.use_relative_rot_loss = use_relative_rot_loss
        self.relative_rot_loss_weight = relative_rot_loss_weight
        self.use_vel_loss = use_vel_loss
        self.vel_loss_weight = vel_loss_weight
        self.foot_vel_loss_scale = foot_vel_loss_scale

    def forward(self, pred_6d, gt_6d, mask, betas, gender, pred_trans=None, gt_trans=None, pred_betas=None):
        # 6D representation
        # pred, gt: (B, P, T, D)
        # mask: (B, P, T) could be None
        # betas: (B, 10)
        # gender: (B, P)
        # pred_trans, gt_trans: (B, P, T ,3)
        
        # TODO: current only for male and female
        B, P, T = pred_6d.shape[:3]
    
        pred_v, pred_j = process_gender(self.smplx_layers, pred_6d, pred_trans, 
                                        betas=pred_betas if pred_betas is not None else betas, 
                                        gender=gender)
        gt_v, gt_j = process_gender(self.smplx_layers, gt_6d, gt_trans, betas, gender)
        
        if mask is None:
            mask = torch.ones(B, P, T, device=pred_6d.device)
        mask_expand = mask.unsqueeze(-1).unsqueeze(-1) # (B, P, T, 1, 1)
        
        losses = {}
        J = gt_j.shape[-2]
        j_loss = (self.Loss(pred_j, gt_j) * mask_expand).sum() / (mask_expand.sum() + 1.e-7) / (J * 3)
        j_loss = self.loss_weight *  j_loss
        losses['joint_loss'] = j_loss

        if pred_v is not None and gt_v is not None:
            V = gt_v.shape[-2]
            v_loss = (self.Loss(pred_v, gt_v) * mask_expand).sum() / (mask_expand.sum() + 1.e-7) / (V * 3)
            v_loss = self.loss_weight * v_loss
            losses['vertex_loss'] = v_loss

        if self.use_contact_loss:
            assert pred_trans is not None and gt_trans is not None
            d_loss, c_loss = joint_contact_loss(pred_j, gt_j, mask, thresh=1.0, contact_thresh=self.contact_thre)
            c_loss = self.contact_loss_weight * c_loss
            d_loss = self.distance_loss_weight * d_loss
            losses['distance_loss'] = d_loss
            losses['contact_loss'] = c_loss
        
        if self.use_foot_contact_loss:
            f_loss = foot_contact_loss(pred_j, mask, front_thre=0.02, back_thre=0.09) # TODO tune threshold
            f_loss = self.foot_contact_loss_weight * f_loss
            losses['foot_contact_loss'] = f_loss
        
        if self.use_relative_rot_loss:
            pred_global_rot = rotation_6d_to_aa(pred_6d[..., :6].reshape(-1, 6)).reshape(B, P, T, 3)
            gt_global_rot = rotation_6d_to_aa(gt_6d[..., :6].reshape(-1, 6)).reshape(B, P, T, 3)
            r_loss = relative_rot_loss(pred_global_rot, gt_global_rot, mask)
            r_loss = self.relative_rot_loss_weight * r_loss
            losses['relative_rot_loss'] = r_loss
        
        if self.use_vel_loss:
            v_loss = vel_loss(pred_j, gt_j, mask, foot_vel_loss_scale=self.foot_vel_loss_scale)
            v_loss = self.vel_loss_weight * v_loss
            losses['vel_loss'] = v_loss

        return losses


    