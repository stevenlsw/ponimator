import numpy as np
import torch 
from aitviewer.renderables.smpl import SMPLSequence
from ponimator.utils.inference_utils import apply_trans, align_global_motion


def load_buddi_style_motion(meta_data):
    human_data = meta_data['humans']
    meta_data['root_orient'] =  torch.tensor(human_data['global_orient']).float() # (2, 3)
    meta_data['pose'] = torch.tensor(human_data['body_pose']).float() # (2, 63)
    meta_data['betas'] = torch.from_numpy(human_data['betas']).float() # (2, 10)
    meta_data['trans'] = torch.tensor(human_data['transl']).float() # (2, 3)
    return meta_data


def buddi_style_motion_process(meta_data, smplx_layer, device):
    motion_data = load_buddi_style_motion(meta_data)
    trans_mat = torch.Tensor([[-1.0, 0.0, 0.0],
                            [0.0, -1.0, 0.0],
                            [0.0, 0.0, 1.0]])
    global_orient, transl = apply_trans(motion_data['root_orient'], motion_data['trans'], motion_data['betas'], smplx_layer, trans_matrix=trans_mat)
    body_pose = motion_data['pose'].reshape(2, -1, 3)
    global_orient, transl = align_global_motion(global_orient, body_pose, transl, smplx_layer, motion_data['betas'], device=device)

    root_init_xz = transl[0:1, [0, 2]] # (1, 2)
    transl[:, [0, 2]] = transl[:, [0, 2]] - root_init_xz
    
    motions = torch.cat([global_orient.reshape(2, -1, 3), body_pose], dim=1) # (P, *, 3)
    smpl_seq_p1 = SMPLSequence(poses_body=body_pose[0].reshape(-1, 21*3),
                        betas=motion_data['betas'][0].unsqueeze(dim=0),
                        smpl_layer=smplx_layer,
                        poses_root=global_orient[0].reshape(-1, 3),
                        trans=transl[0].reshape(-1, 3))

    smpl_seq_p2 = SMPLSequence(poses_body=body_pose[1].reshape(-1, 21*3),
                                betas=motion_data['betas'][1].unsqueeze(dim=0),
                                smpl_layer=smplx_layer,
                                poses_root=global_orient[1].reshape(-1, 3),
                                trans=transl[1].reshape(-1, 3))

    joints = np.concatenate([smpl_seq_p1.joints, smpl_seq_p2.joints], axis=0) # (P, J, 3)
    floor_height = joints.min(axis=0).min(axis=0)[1]
    transl[:, 1] -= floor_height
    
    motion_data['trans'] = transl
    motion_data['root_orient'] = global_orient
    return motion_data