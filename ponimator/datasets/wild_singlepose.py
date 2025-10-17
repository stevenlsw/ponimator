import numpy as np
import os
import pickle
import glob
import torch
from torch.utils import data
from tqdm import tqdm

from ponimator.utils.inference_utils import apply_trans, align_global_motion_single_opencv, motion_to_6D_single, load_pose_smpler


class SMPLerDataset(data.Dataset):
    def __init__(self, data_dir, smplx_layer, device): 
        self.data_dir = data_dir
        self.smplx_layer = smplx_layer
        self.device = device
       
    def __len__(self):
        return 1
    
    def __getitem__(self, idx):
        if idx >= len(self):
            raise IndexError(f"Index {idx} out of range for dataset of size {len(self)}")
        
        data_file = glob.glob(os.path.join(self.data_dir, "*.pkl"))[0]
        name = data_file.split("/")[-1].split(".")[0]
            
        pose_data = load_pose_smpler(data_file)
        global_orient = pose_data['root_orient'] # (3, )
        trans = pose_data['trans'] # (3, )
        betas = pose_data['betas'] # (D, )
        pose = pose_data['pose'] # (J, 3)
        
        # Convert from OpenCV camera coordinates to Y-up world coordinates
        # OpenCV: +X=right, +Y=down, +Z=forward -> World: +X=right, +Y=up, +Z=forward
        trans_matrix = torch.tensor([[1.0, 0.0, 0.0],
                                     [0.0, -1.0, 0.0],
                                     [0.0, 0.0, 1.0]]).float()
        global_orient, trans = apply_trans(global_orient, trans, betas, self.smplx_layer, trans_matrix=trans_matrix)
        
        # Align the person to face forward and be upright using improved OpenCV alignment
        global_orient, trans = align_global_motion_single_opencv(global_orient, pose, trans, self.smplx_layer, betas, device=self.device)
        
        smpl_p1 = self.smplx_layer(
            poses_body=pose.reshape(1, 21*3).to(self.device), 
            poses_root=global_orient.reshape(1, 3).to(self.device),
            betas=betas.to(self.device),
            trans=trans.reshape(1, 3).to(self.device))
        p1_j = smpl_p1[1][0].cpu().numpy() # (21, 3)
        floor_height = p1_j.min(axis=0)[1]
        trans[1] -= floor_height
        trans[[0, 2]] = 0

        pose_data['root_orient'] = global_orient
        pose_data['trans'] = trans
        pose_data = motion_to_6D_single(pose_data)
        
        condition_motion = torch.cat([pose_data['root_6d'].reshape(1, 6), pose_data['pose_6d'].reshape(-1, 6)], dim=0)
        condition_motion = condition_motion.reshape(-1)
        condition_rep = torch.cat([condition_motion, pose_data['trans'].reshape(-1)], dim=0) # (D*, )
        
        sample = {"condition_rep": condition_rep[None], # (1, D), 
                "motions": condition_motion[None], # (P=1, D)
                "trans": pose_data['trans'].reshape(-1)[None], # (P=1, 3)
                "betas": betas[None], 
                "data_file": data_file, 
                "name": name}
        return sample



class MotionXSinglePoseDataset(data.Dataset):
    def __init__(self, data_dir, smplx_layer, device): 
        self.data_dir = data_dir
        self.smplx_layer = smplx_layer
        self.device = device
    
    def __len__(self):
        return 1
    
    def __getitem__(self, idx):
        if idx >= len(self):
            raise IndexError(f"Index {idx} out of range for dataset of size {len(self)}")
        
        data_file = os.path.join(self.data_dir, "result_dict.pkl")
        name = self.data_dir.split("/")[-1]
            
        with open(data_file, 'rb') as f:
            pose_data = pickle.load(f)
        global_orient = torch.from_numpy(pose_data['root_pose']).float() # (3, )
        trans = torch.from_numpy(pose_data['trans']).float() # (3, )
        betas = torch.from_numpy(pose_data['betas']).float() # (D, )
        pose = torch.from_numpy(pose_data['body_pose']).float().reshape(21, 3) # (J, 3)
        global_orient, trans = align_global_motion_single_opencv(global_orient, pose, trans, self.smplx_layer, betas, device=self.device)
        
        smpl_p1 = self.smplx_layer(
            poses_body=pose.reshape(1, 21*3).to(self.device), 
            poses_root=global_orient.reshape(1, 3).to(self.device),
            betas=betas.to(self.device),
            trans=trans.reshape(1, 3).to(self.device))
        p1_j = smpl_p1[1][0].cpu().numpy() # (21, 3)
        floor_height = p1_j.min(axis=0)[1]
        trans[1] -= floor_height
        trans[[0, 2]] = 0

        pose_data['root_orient'] = global_orient
        pose_data['trans'] = trans
        pose_data['pose'] = pose
        pose_data['betas'] = betas
        pose_data = motion_to_6D_single(pose_data)
        
        condition_motion = torch.cat([pose_data['root_6d'].reshape(1, 6), pose_data['pose_6d'].reshape(-1, 6)], dim=0)
        condition_motion = condition_motion.reshape(-1)
        condition_rep = torch.cat([condition_motion, pose_data['trans'].reshape(-1)], dim=0) # (D*, )
        
        sample = {"condition_rep": condition_rep[None], # (1, D), 
                "motions": condition_motion[None], # (P=1, D)
                "trans": pose_data['trans'].reshape(-1)[None], # (P=1, 3)
                "betas": betas[None], 
                "data_file": data_file, 
                "name": name}
        return sample


