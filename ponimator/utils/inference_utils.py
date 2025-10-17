import re
import glob
import os
import os.path
import sys
import pickle
import numpy as np
import torch
import roma

from aitviewer.renderables.smpl import SMPLSequence
from aitviewer.scene.material import Material

from .utils import rotation_6d_to_aa, rotation_aa_to_6d, process_gender, rotation_matrix_to_align_vectors
from ponimator.models import ContactPoseGen, ContactMotionGen
from ponimator.models.losses import bone_length_loss



def build_models(cfg):
    if cfg.NAME == "ContactPoseGen":
        model = ContactPoseGen(cfg)
    elif cfg.NAME == "ContactMotionGen":
        model = ContactMotionGen(cfg)
    else:
        raise ValueError(f"Invalid model name: {cfg.NAME}")
    return model


def load_model(checkpoint_path, model):
    ckpt = torch.load(checkpoint_path, map_location="cpu")
    for k in list(ckpt["state_dict"].keys()):
        if "model" in k: # replace only first occurence
            ckpt["state_dict"][k.replace("model.", "", 1)] = ckpt["state_dict"].pop(k)
    model.load_state_dict(ckpt["state_dict"], strict=True)
    print("checkpoint state loaded from {}!".format(checkpoint_path))
    

def vis_output(motions, smplx_layers, betas, gender, trans, colors=None):
    # motions: (P, T, D)
    motions = motions.reshape(2, motions.shape[1], -1, 6)
    P, T = motions.shape[:2]
    
    pose = rotation_6d_to_aa(motions.reshape(-1, 6)).reshape(P, T, -1, 3)

    root_orient = pose[:, :, 0] # (P, T, 3)
    body_pose = pose[:, :, 1:22] # (P, T, 21, 3)
    
    if pose.shape[2] > 22:
        left_hand_pose = pose[:, :, 22:22+15] # (P, T, 15, 3)
        right_hand_pose = pose[:, :, 22+15:] # (P, T, 15, 3)
    else:
        # default hand pose
        hand_pose = torch.Tensor([[ 0.1117, -0.0429,  0.4164],
        [ 0.1088,  0.0660,  0.7562],
        [-0.0964,  0.0909,  0.1885],
        [-0.1181, -0.0509,  0.5296],
        [-0.1437, -0.0552,  0.7049],
        [-0.0192,  0.0923,  0.3379],
        [-0.4570,  0.1963,  0.6255],
        [-0.2147,  0.0660,  0.5069],
        [-0.3697,  0.0603,  0.0795],
        [-0.1419,  0.0859,  0.6355],
        [-0.3033,  0.0579,  0.6314],
        [-0.1761,  0.1321,  0.3734],
        [ 0.8510, -0.2769,  0.0915],
        [-0.4998, -0.0266, -0.0529],
        [ 0.5356, -0.0460,  0.2774]])
    
        right_hand_pose = hand_pose.unsqueeze(0).unsqueeze(0).repeat(P, T, 1, 1)
        left_hand_pose = hand_pose.unsqueeze(0).unsqueeze(0).repeat(P, T, 1, 1)
        left_hand_pose[:, :, :, 1] *= -1
        left_hand_pose[:, :, :, 2] *= -1
        
    smpl_seq_p1 = SMPLSequence(poses_body=body_pose[0].reshape(-1, 21*3),
                               betas=betas[0].unsqueeze(dim=0),
                               smpl_layer=smplx_layers[gender[0].item()],
                               poses_root=root_orient[0].reshape(-1, 3),
                               poses_left_hand=left_hand_pose[0].reshape(-1, 15*3),
                               poses_right_hand=right_hand_pose[0].reshape(-1, 15*3),
                               trans=trans[0].reshape(-1, 3),
                               color=(0.11, 0.53, 0.8, 1.0) if colors is None else colors[0],
                               material=Material(ambient=0.3, diffuse=0.3))
    
    smpl_seq_p2 = SMPLSequence(poses_body=body_pose[1].reshape(-1, 21*3),
                                betas=betas[1].unsqueeze(dim=0),
                                smpl_layer=smplx_layers[gender[1].item()],
                                poses_root=root_orient[1].reshape(-1, 3),
                                poses_left_hand=left_hand_pose[1].reshape(-1, 15*3),
                                poses_right_hand=right_hand_pose[1].reshape(-1, 15*3),
                                trans=trans[1].reshape(-1, 3),
                                color=(1.0, 0.27, 0, 1.0) if colors is None else colors[1],
                                material=Material(ambient=0.3, diffuse=0.3))

    return smpl_seq_p1, smpl_seq_p2


def load_buddi_style_motion(meta_data):
    human_data = meta_data['humans']
    motion_data = {
        'root_orient': torch.tensor(human_data['global_orient']).float(),  # (2, 3)
        'pose': torch.tensor(human_data['body_pose']).float(),  # (2, 63)
        'betas': torch.from_numpy(human_data['betas']).float(),  # (2, 10)
        'trans': torch.tensor(human_data['transl']).float()  # (2, 3)
    }
    return motion_data

def load_pose_smpler(file_path):
    with open(file_path, 'rb') as f:
        results = pickle.load(f)
    meta_data = {}
    meta_data['root_orient'] = torch.from_numpy(results['smplx_root_pose'][0]).float() # (3,)
    meta_data['pose'] = torch.from_numpy(results['smplx_body_pose'][0]).reshape(21, 3).float() # (21, 3)
    meta_data['betas'] = torch.from_numpy(results['smplx_shape'][0]).float() # (10, )
    meta_data['trans'] = torch.from_numpy(results['cam_trans'][0]).float() # (3,)
    meta_data['gender'] = "neutral"
    return meta_data


def motion_to_6D_single(motion):
    """
    Convert the motion to 6D representation
    """
    motion_6d = {}
    J = motion['pose'].shape[0]
    root_orient_6d = rotation_aa_to_6d(motion['root_orient'][None])[0] # (3) -> (6)
    pose_6d = rotation_aa_to_6d(motion['pose'].reshape(-1, 3)).reshape(J, 6) # (21, 3) -> (21, 6)
    motion_6d['root_6d'] = root_orient_6d
    motion_6d['pose_6d'] = pose_6d
    motion_6d['betas'] = motion['betas']
    motion_6d['trans'] = motion['trans']
    if "gender" in motion:
        motion_6d['gender'] = motion['gender']
    return motion_6d


def apply_trans(root_orient, trans, betas, smplx_layer, trans_matrix):
    """
    Apply transformation matrix to root orientation and translation.
    Handles both single poses and sequences automatically.
    
    Args:
        root_orient: (3) for single pose or (T, 3) for sequence
        trans: (3) for single pose or (T, 3) for sequence
        betas: (10) for single pose or (T, 10) for sequence
        smplx_layer: SMPL layer
        trans_matrix: (3, 3) transformation matrix
    
    Returns:
        root_orient: transformed root orientation with same shape as input
        trans: transformed translation with same shape as input
    """
    # Check if input is single pose and add batch dimension if needed
    is_single = root_orient.ndim == 1
    if is_single:
        root_orient = root_orient.unsqueeze(0)
        trans = trans.unsqueeze(0)
        betas = betas.unsqueeze(0)
    
    # Process with batch dimension
    T = root_orient.shape[0]
    smpl_seq_ori = SMPLSequence(
                        poses_body=torch.zeros(T, 63),
                        smpl_layer=smplx_layer,
                        betas=betas)
    rot_center = torch.from_numpy(smpl_seq_ori.joints[:, 0, :])  # root joint (T, 3)
    global_rotmat = roma.rotvec_to_rotmat(root_orient)
    global_rotmat = torch.einsum("mn, tnj->tmj", trans_matrix, global_rotmat)
    root_orient = roma.rotmat_to_rotvec(global_rotmat)
    # ref: https://www.dropbox.com/c/scl/fi/zkatuv5shs8d4tlwr8ecc/Change-parameters-to-new-coordinate-system.paper?dl=0&rlkey=lotq1sh6wzkmyttisc05h0in0
    trans = torch.einsum("mn, tn->tm", trans_matrix, (rot_center + trans)) - rot_center
    
    # Remove batch dimension if input was single
    if is_single:
        root_orient = root_orient.squeeze(0)
        trans = trans.squeeze(0)
    
    return root_orient, trans



def align_global_motion(root_orient, body_pose, trans, smplx_layer, betas, device):
    """
    Align global motion by rotating to face forward (+Z direction).
    Handles both single poses and batches automatically.
    
    Args:
        root_orient: (3) for single pose or (P, 3) for batch
        body_pose: (21, 3) or (63) for single, or (P, 21, 3) or (P, 63) for batch
        trans: (3) for single pose or (P, 3) for batch
        smplx_layer: SMPL layer
        betas: (10) for single pose or (P, 10) for batch
        device: torch device
    
    Returns:
        aligned_root_orient: same shape as input root_orient
        aligned_trans: same shape as input trans
    """
    # Check if input is single pose and add batch dimension if needed
    is_single = root_orient.ndim == 1
    if is_single:
        root_orient = root_orient.unsqueeze(0)
        trans = trans.unsqueeze(0)
        betas = betas.unsqueeze(0)
        if body_pose.ndim == 1:  # (63,)
            body_pose = body_pose.unsqueeze(0)
        elif body_pose.ndim == 2:  # (21, 3)
            body_pose = body_pose.unsqueeze(0)
    
    # Ensure body_pose is in shape (P, 63)
    if body_pose.ndim == 3:  # (P, 21, 3)
        body_pose = body_pose.reshape(body_pose.shape[0], -1)
    
    P = root_orient.shape[0]
    
    # Use first person to compute alignment rotation
    smpl_p1 = smplx_layer(
            poses_body=body_pose[0:1].reshape(-1, 21*3).to(device), 
            poses_root=root_orient[0:1].reshape(-1, 3).to(device),
            betas=betas[0:1].to(device),
            trans=trans[0:1].reshape(-1, 3).to(device))
    p1_j = smpl_p1[1][0].cpu()

    face_joint_indx = [2, 1, 17, 16]
    r_hip, l_hip, sdr_r, sdr_l = face_joint_indx
    across1 = p1_j[r_hip] - p1_j[l_hip]
    across2 = p1_j[sdr_r] - p1_j[sdr_l]
    across = across1 + across2
    across = across / across.norm(dim=-1, keepdim=True)

    forward_init = torch.cross(torch.tensor([0, 1, 0]).float(), across, axis=-1)
    forward_init = forward_init / (forward_init.norm(dim=-1, keepdim=True) + 1e-8)
    
    target = torch.tensor([0, 0, 1]).float()
    rotmat = rotation_matrix_to_align_vectors(forward_init, target)

    # Get rotation centers for all people
    rest_pose = smplx_layer(poses_body=torch.zeros(P, 63, device=device), betas=betas.to(device))
    rest_joints = rest_pose[1].cpu()
    rot_center = rest_joints[:, 0, :]  # root joint (P, 3)

    # Apply rotation to all people in batch
    new_trans = torch.einsum("mn, pn->pm", rotmat, (rot_center + trans)) - rot_center
    
    root_rotmat = roma.rotvec_to_rotmat(root_orient)  # (P, 3, 3)
    new_root_rotmat = torch.matmul(rotmat.unsqueeze(0), root_rotmat)  # (P, 3, 3)
    new_root_orient = roma.rotmat_to_rotvec(new_root_rotmat)  # (P, 3)
    
    # Remove batch dimension if input was single
    if is_single:
        new_root_orient = new_root_orient.squeeze(0)
        new_trans = new_trans.squeeze(0)
    
    return new_root_orient, new_trans


def align_global_motion_single_opencv(root_orient, body_pose, trans, smplx_layer, betas, device):
    """
    Improved alignment for poses coming from OpenCV camera coordinates.
    This function:
    1. Ensures the person is upright (Y-axis up)
    2. Rotates the person to face forward (+Z direction)
    3. Handles edge cases where the person might not be perfectly upright
    
    Args:
        root_orient: (3) - root orientation in axis-angle
        body_pose: (63) or (21, 3) - body pose
        trans: (3) - translation
        smplx_layer: SMPL layer
        betas: (10) - shape parameters
        device: torch device
    
    Returns:
        root_orient: (3) - aligned root orientation
        trans: (3) - aligned translation
    """
    if body_pose.ndim == 2:
        body_pose = body_pose.reshape(-1)
    
    # Step 1: Check root orientation to detect if person is upside down
    # Convert root_orient to rotation matrix to extract the body's actual up direction
    root_rotmat = roma.rotvec_to_rotmat(root_orient)
    
    # In SMPL canonical pose, the Y-axis points up along the spine
    # Extract the Y-axis (column 1) from the rotation matrix to see where "up" points in world space
    root_up_direction = root_rotmat[:, 1]  # (3,) - where the body's Y-axis points in world space
    
    # If the root's Y-axis points downward (negative Y in world space), the person is upside down
    # We need to flip them 180 degrees around the X-axis (or Z-axis)
    if root_up_direction[1] < -0.1:  # threshold to handle numerical errors and slight bends
        # Person is upside down - apply 180-degree rotation around X-axis to flip them
        flip_rotmat = torch.tensor([[1.0, 0.0, 0.0],
                                     [0.0, -1.0, 0.0],
                                     [0.0, 0.0, -1.0]]).float()
        root_rotmat = torch.matmul(flip_rotmat, root_rotmat)
        root_orient = roma.rotmat_to_rotvec(root_rotmat)
        
        # Also transform the translation
        trans = torch.matmul(flip_rotmat, trans)
    
    # Get current joint positions with corrected orientation
    smpl_p1 = smplx_layer(
            poses_body=body_pose.reshape(1, 21*3).to(device), 
            poses_root=root_orient.reshape(1, 3).to(device),
            betas=betas.to(device),
            trans=trans.reshape(1, 3).to(device))
    p1_j = smpl_p1[1][0].cpu()

    # Joint indices: [r_hip, l_hip, sdr_r, sdr_l]
    face_joint_indx = [2, 1, 17, 16]
    r_hip, l_hip, sdr_r, sdr_l = face_joint_indx
    
    # Step 2: Compute body orientation vectors for alignment
    # Across vector (right to left in body frame) - use both hips and shoulders for robustness
    across_hips = p1_j[l_hip] - p1_j[r_hip]
    across_shoulders = p1_j[sdr_l] - p1_j[sdr_r]
    across = (across_hips + across_shoulders) / 2.0
    across = across / (across.norm(dim=-1, keepdim=True) + 1e-8)
    
    # Up vector (from hips to shoulders)
    up_vec = (p1_j[sdr_r] + p1_j[sdr_l]) / 2.0 - (p1_j[r_hip] + p1_j[l_hip]) / 2.0
    up_vec = up_vec / (up_vec.norm(dim=-1, keepdim=True) + 1e-8)
    
    # Ensure up vector points upward (should already be corrected by root_orient check)
    if up_vec[1] < 0:
        up_vec = -up_vec
    
    # Forward vector (perpendicular to across and up)
    # cross(across, up_vec) where across is +X and up is +Y gives +Z (forward)
    forward_vec = torch.cross(across, up_vec, dim=-1)
    forward_vec = forward_vec / (forward_vec.norm(dim=-1, keepdim=True) + 1e-8)
    
    # Re-orthogonalize: recompute across to be perpendicular to forward and up
    # This ensures we have a proper right-handed coordinate system
    across = torch.cross(up_vec, forward_vec, dim=-1)
    across = across / (across.norm(dim=-1, keepdim=True) + 1e-8)
    
    # Step 3: Build rotation matrix to align to world coordinates
    # Current body frame: [across, up_vec, forward_vec]
    # Target world frame: [+X, +Y, +Z]
    current_frame = torch.stack([across, up_vec, forward_vec], dim=1)  # (3, 3)
    target_frame = torch.eye(3).float()  # Identity - world frame
    
    # Rotation matrix to align current frame to target frame
    # R * current_frame = target_frame => R = target_frame * current_frame^T
    rotmat = torch.matmul(target_frame, current_frame.T)
    
    # Step 4: Apply rotation to root orientation and translation
    rest_pose = smplx_layer(poses_body=torch.zeros(1, 63, device=device), betas=betas.to(device))
    rest_joints = rest_pose[1].cpu()
    rot_center = rest_joints[0, 0, :]  # root joint (3)

    # Transform translation
    new_trans = torch.einsum("mn, n->m", rotmat, (rot_center + trans)) - rot_center
    
    # Transform root orientation (already corrected for upside-down)
    root_rotmat = roma.rotvec_to_rotmat(root_orient)
    new_root_rotmat = torch.matmul(rotmat, root_rotmat)
    new_root_orient = roma.rotmat_to_rotvec(new_root_rotmat)
    
    return new_root_orient, new_trans


def optimize_beta(smplx_layer, joints, epochs=500, lr=1e-2):
    # optimize beta, gender as neutral
    # gender: (P,)
    # joints: (P, J, 3)
    P = joints.shape[0]
    betas = torch.zeros((P, 10), dtype=torch.float32, device=joints.device, requires_grad=True)
    optimizer = torch.optim.Adam([betas], lr=lr)
    template = torch.tensor([1, 0, 0, 1, 0, 0], device=joints.device, dtype=joints.dtype)
    template_motion = template.repeat(1, P, 22, 1).reshape (1, P, -1) #  hardcode (B, P, D*6)
    trans = torch.zeros((1, P, 3), dtype=joints.dtype, device=joints.device)
    for epoch in range(epochs):
        optimizer.zero_grad()
        _, pred_joints = process_gender([None, None, smplx_layer], template_motion.unsqueeze(2), trans.unsqueeze(2), betas.unsqueeze(0))
        pred_joints = pred_joints.squeeze(0).squeeze(1) # (P, J, 3)
        loss = bone_length_loss(pred_joints, joints)
        loss.backward()
        optimizer.step()
            
        if epoch % 10 == 0:
            print(f'Epoch {epoch}: Loss = {loss.item()}')

    return betas.detach()


def optimize_beta_gender(smplx_layers, gender, joints, epochs=500, lr=1e-2, motion=None, trans=None):
    # optimize beta given gender male/female
    # gender: (P,)
    # motion: (P, T, D*), trans: (P, T, 3)
    # joints: (P, J, 3) / (P, T, J, 3)
    P = joints.shape[0]
    betas = torch.zeros((P, 10), dtype=torch.float32, device=joints.device, requires_grad=True)
    optimizer = torch.optim.Adam([betas], lr=lr)
    if motion is None:
        template = torch.tensor([1, 0, 0, 1, 0, 0], device=joints.device, dtype=joints.dtype)
        motion = template.repeat(1, P, 22, 1).reshape (1, P, -1) #  hardcode (B, P, D*6)
        motion = motion.unsqueeze(2) # (B, P, 1, D)
    else:
        if motion.ndim < 4:
            motion = motion.unsqueeze(0) # (1, P, T, D)
    if trans is None:
        trans = torch.zeros((1, P, 3), dtype=joints.dtype, device=joints.device)
        trans = trans.unsqueeze(2) # (B, P, 1, 3)
    else:
        if trans.ndim < 4:
            trans = trans.unsqueeze(0) # (1, P, T, 3)
    for epoch in range(epochs):
        optimizer.zero_grad()
        _, pred_joints = process_gender(smplx_layers, motion, trans, betas.unsqueeze(0), gender.unsqueeze(0)) # (B, P, T, J, 3)
        loss = bone_length_loss(pred_joints.squeeze(), joints.squeeze())
        loss.backward()
        optimizer.step()
            
        if epoch % 10 == 0:
            print(f'Epoch {epoch}: Loss = {loss.item()}')

    return betas.detach()