import numpy as np
import torch
import roma


face_joint_indx = [2,1,17,16]
fid_l = [7,10]
fid_r = [8,11]


def swap_left_right_position(data):
    assert len(data.shape) == 3 and data.shape[-1] == 3
    data = data.copy()
    data[..., 0] *= -1
    right_chain = [2, 5, 8, 11, 14, 17, 19, 21]
    left_chain = [1, 4, 7, 10, 13, 16, 18, 20]
    left_hand_chain = [22, 23, 24, 34, 35, 36, 25, 26, 27, 31, 32, 33, 28, 29, 30, 52, 53, 54, 55, 56]
    right_hand_chain = [43, 44, 45, 46, 47, 48, 40, 41, 42, 37, 38, 39, 49, 50, 51, 57, 58, 59, 60, 61]

    tmp = data[:, right_chain]
    data[:, right_chain] = data[:, left_chain]
    data[:, left_chain] = tmp
    if data.shape[1] > 24:
        tmp = data[:, right_hand_chain]
        data[:, right_hand_chain] = data[:, left_hand_chain]
        data[:, left_hand_chain] = tmp
    return data


def swap_left_right_rot(data):
    assert len(data.shape) == 3 and data.shape[-1] == 6
    data = data.copy()

    data[..., [1,2,4]] *= -1

    right_chain = np.array([2, 5, 8, 11, 14, 17, 19, 21])-1
    left_chain = np.array([1, 4, 7, 10, 13, 16, 18, 20])-1
    left_hand_chain = np.array([22, 23, 24, 34, 35, 36, 25, 26, 27, 31, 32, 33, 28, 29, 30,])-1
    right_hand_chain = np.array([43, 44, 45, 46, 47, 48, 40, 41, 42, 37, 38, 39, 49, 50, 51,])-1

    tmp = data[:, right_chain]
    data[:, right_chain] = data[:, left_chain]
    data[:, left_chain] = tmp
    if data.shape[1] > 24:
        tmp = data[:, right_hand_chain]
        data[:, right_hand_chain] = data[:, left_hand_chain]
        data[:, left_hand_chain] = tmp
    return data


def swap_left_right(data, n_joints):
    T = data.shape[0]
    new_data = data.copy()
    positions = new_data[..., :3*n_joints].reshape(T, n_joints, 3)
    rotations = new_data[..., 3*n_joints:].reshape(T, -1, 6)

    positions = swap_left_right_position(positions)
    rotations = swap_left_right_rot(rotations)

    new_data = np.concatenate([positions.reshape(T, -1), rotations.reshape(T, -1)], axis=-1)
    return new_data


def rotation_6d_to_aa(d6):
    rot_matrix = roma.special_gramschmidt(d6.reshape(-1, 3, 2))
    rot_vec = roma.rotmat_to_rotvec(rot_matrix)
    return rot_vec


def rotation_aa_to_6d(rot_vec):
    rot_matrix = roma.rotvec_to_rotmat(rot_vec)
    rot_6d = rot_matrix[..., :, :2].reshape(-1, 6)
    return rot_6d


def save_logfile(log_loss, save_path):
    with open(save_path, 'wt') as f:
        for k, v in log_loss.items():
            w_line = k
            for digit in v:
                w_line += ' %.3f' % digit
            f.write(w_line + '\n')


def rotation_matrix_to_align_vectors(v1, v2):
    """
    Calculate the 3x3 rotation matrix that aligns vector v1 to v2 using PyTorch.
    
    Args:
    - v1: PyTorch tensor of shape (3,) representing the first vector.
    - v2: PyTorch tensor of shape (3,) representing the second vector.
    
    Returns:
    - 4x4 rotation matrix that aligns v1 with v2.
    """
    # Normalize the input vectors
    v1 = v1 / torch.norm(v1)
    v2 = v2 / torch.norm(v2)

    # Compute the rotation axis using the cross product
    axis = torch.cross(v1, v2)

    # Handle the case where v1 and v2 are already aligned (i.e., cross product is zero)
    if torch.norm(axis) == 0:
        return torch.eye(3, dtype=v1.dtype, device=v1.device)

    # Normalize the rotation axis
    axis = axis / torch.norm(axis)

    # Compute the angle between the two vectors using dot product
    angle = torch.acos(torch.dot(v1, v2))

    # Create the skew-symmetric cross-product matrix for the axis
    rotation_matrix = roma.rotvec_to_rotmat(angle * axis)

    return rotation_matrix



def process_gender(smplx_layers, tensor_6d, trans, betas, gender=None):
    """
    Process SMPLX parameters with gender-specific or neutral SMPLX layers.
    Merged from process_gender and process_gender_uniform.
    
    Args:
        smplx_layers: List of SMPLLayers [male, female, neutral]
        tensor_6d: (B, P, T, D) where P=1/2, rotation in 6D representation
        trans: (B, P, T, 3) translation
        betas: (B, P, 10) shape parameters
        gender: (B, P) gender tensor where 0=male, 1=female, 2=neutral. 
                If None, treats all as neutral (default)
    
    Returns:
        v: (B, P, T, 10475, 3) vertices
        j: (B, P, T, 55/127, 3) joints
    """
    B, P, T, D = tensor_6d.shape
    
    # If gender is None, treat all as neutral
    if gender is None:
        gender = torch.full((B, P), 2, dtype=torch.long, device=tensor_6d.device)
    
    male_mask = gender == 0
    female_mask = gender == 1
    neutral_mask = gender == 2
    
    tensor_6d = tensor_6d.reshape(B, P, T, -1, 6).reshape(-1, 6)
    tensor_aa = rotation_6d_to_aa(tensor_6d).reshape(B, P, T, -1, 3)
    if tensor_aa.shape[-2] == 22: # global + 21 body
        poses_root = tensor_aa[..., 0, :].reshape(B, P, T, 3)
        poses_body = tensor_aa[..., 1:, :].reshape(B, P, T, 21*3)
        poses_left_hand = None
        poses_right_hand = None
    else: # global + 21 body + 15 left hand + 15 right hand
        poses_root = tensor_aa[..., 0, :].reshape(B, P, T, 3)
        poses_body = tensor_aa[..., 1:22, :].reshape(B, P, T, 21*3)
        poses_left_hand = tensor_aa[..., 22:22+15, :].reshape(B, P, T, 15*3)
        poses_right_hand = tensor_aa[..., 22+15:, :].reshape(B, P, T, 15*3)

    J = 127
    j = torch.zeros((B, P, T, J, 3), device=tensor_6d.device)
    v = torch.zeros((B, P, T, 10475, 3), device=tensor_6d.device)
    
    if male_mask.any():
        male_poses_body = poses_body[male_mask].view(-1, T, 21*3).view(-1, 21*3)
        male_betas = betas[male_mask].unsqueeze(1).expand(-1, T, betas.shape[-1]).reshape(-1, betas.shape[-1])
        male_poses_root = poses_root[male_mask].view(-1, T, 3).view(-1, 3)
        if poses_left_hand is not None:
            male_poses_left_hand = poses_left_hand[male_mask].view(-1, T, 15*3).view(-1, 15*3)
        if poses_right_hand is not None:
            male_poses_right_hand = poses_right_hand[male_mask].view(-1, T, 15*3).view(-1, 15*3)
        if trans is not None:
            male_trans = trans[male_mask].reshape(-1, T, 3).reshape(-1, 3)

        v_male, j_male = smplx_layers[0](
            poses_body=male_poses_body, 
            betas=male_betas, 
            poses_root=male_poses_root,  
            poses_left_hand=male_poses_left_hand if poses_left_hand is not None else None,
            poses_right_hand=male_poses_right_hand if poses_right_hand is not None else None,
            trans=male_trans if trans is not None else None)

        j[male_mask] = j_male.reshape(-1, T, J, 3)
        v[male_mask] = v_male.reshape(-1, T, 10475, 3)

    if female_mask.any():
        female_poses_body = poses_body[female_mask].view(-1, T, 21*3).view(-1, 21*3)
        female_betas = betas[female_mask].unsqueeze(1).expand(-1, T, betas.shape[-1]).reshape(-1, betas.shape[-1])
        female_poses_root = poses_root[female_mask].view(-1, T, 3).view(-1, 3)
        if poses_left_hand is not None:
            female_poses_left_hand = poses_left_hand[female_mask].view(-1, T, 15*3).view(-1, 15*3)
        if poses_right_hand is not None:
            female_poses_right_hand = poses_right_hand[female_mask].view(-1, T, 15*3).view(-1, 15*3)
        if trans is not None:
            female_trans = trans[female_mask].reshape(-1, T, 3).reshape(-1, 3)

        v_female, j_female = smplx_layers[1](
            poses_body=female_poses_body, 
            betas=female_betas, 
            poses_root=female_poses_root,  
            poses_left_hand=female_poses_left_hand if poses_left_hand is not None else None,
            poses_right_hand=female_poses_right_hand if poses_right_hand is not None else None,
            trans=female_trans if trans is not None else None)

        j[female_mask] = j_female.reshape(-1, T, J, 3)
        v[female_mask] = v_female.reshape(-1, T, 10475, 3)
    
    if neutral_mask.any():
        neutral_poses_body = poses_body[neutral_mask].view(-1, T, 21*3).view(-1, 21*3)
        neutral_betas = betas[neutral_mask].unsqueeze(1).expand(-1, T, betas.shape[-1]).reshape(-1, betas.shape[-1])
        neutral_poses_root = poses_root[neutral_mask].view(-1, T, 3).view(-1, 3)
        if poses_left_hand is not None:
            neutral_poses_left_hand = poses_left_hand[neutral_mask].view(-1, T, 15*3).view(-1, 15*3)
        if poses_right_hand is not None:
            neutral_poses_right_hand = poses_right_hand[neutral_mask].view(-1, T, 15*3).view(-1, 15*3)
        if trans is not None:
            neutral_trans = trans[neutral_mask].reshape(-1, T, 3).reshape(-1, 3)

        v_neutral, j_neutral = smplx_layers[2](
            poses_body=neutral_poses_body, 
            betas=neutral_betas, 
            poses_root=neutral_poses_root,  
            poses_left_hand=neutral_poses_left_hand if poses_left_hand is not None else None,
            poses_right_hand=neutral_poses_right_hand if poses_right_hand is not None else None,
            trans=neutral_trans if trans is not None else None)

        j[neutral_mask] = j_neutral.reshape(-1, T, J, 3)
        v[neutral_mask] = v_neutral.reshape(-1, T, 10475, 3)
    
    return v, j