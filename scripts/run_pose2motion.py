import os
import sys
sys.path.append(sys.path[0] + r"/../")
import shutil
import torch
import numpy as np
from aitviewer.renderables.meshes import Meshes
from aitviewer.models.smpl import SMPLLayer
from aitviewer.headless import HeadlessRenderer
from aitviewer.configuration import CONFIG as C

from ponimator.utils.inference_utils import load_model, vis_output
from ponimator.utils.utils import rotation_aa_to_6d, process_gender
from ponimator.utils.config_utils import get_config
from ponimator.datasets.wild_interpose import Buddi_Dataset
from ponimator.models import ContactMotionGen


def map_gender(gender_list):
    """Map gender strings to integers: male=0, female=1, neutral=2"""
    gender_map = {'male': 0, 'female': 1, 'neutral': 2}
    if gender_list is None:
        return None
    if len(gender_list) != 2:
        raise ValueError("Gender must be a list of exactly 2 strings")
    return torch.tensor([gender_map.get(g.lower(), 2) for g in gender_list])


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--save_dir", type=str, default="outputs")
    parser.add_argument("--model_path", type=str, default="checkpoints/contactmotion.ckpt")
    parser.add_argument("--model_config_path", type=str, default="configs/motion.yaml")
    parser.add_argument("--body_model_root", type=str, default="body_models")
    parser.add_argument("--seq_len", type=int, default=30, help="defaultsequence length (trained on 30)")
    # inference settings
    parser.add_argument("--data_source", choices=["buddi"], default="buddi", help="interactive pose data source")
    parser.add_argument("--data_dir", type=str, default="data/buddi/Couple_6806", help="data dir")
    parser.add_argument("--inter_time_idx", type=int, default=None, help="Index of the interactive pose range (0-29)")
    parser.add_argument("--gender", nargs="+", default=None, help="gender for two persons as list of strings (e.g., male female), where male=0, female=1, neutral=2")
    
    # utilization
    parser.add_argument("--save", action="store_true", help="save the results")
    parser.add_argument("--disable_vis", action="store_true", help="disable visualization")
    parser.add_argument("--vis_interactive_pose", action="store_true", help="visualize interactive pose")
    args = parser.parse_args()
    
    gender_arg = map_gender(args.gender)
    
    C.update_conf({'smplx_models': args.body_model_root})
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    smplx_layers = [
        SMPLLayer(model_type='smplx',gender='male', num_betas=10, device=device),
        SMPLLayer(model_type='smplx',gender='female', num_betas=10, device=device),
        SMPLLayer(model_type='smplx',gender='neutral', num_betas=10, device=device)
    ]

    model_cfg = get_config(args.model_config_path) 
    model = ContactMotionGen(model_cfg)
    load_model(args.model_path, model)
    model.eval()
    model.to(device)

    # Create a viewer.
    if not args.disable_vis:
        v = HeadlessRenderer()
    
    if args.data_source == "buddi":
        dataset = Buddi_Dataset(args.data_dir, smplx_layer=smplx_layers[2], device=device)
    else:
        raise NotImplementedError

    for sample in dataset:
        
        motions = torch.cat([sample['root_orient'].reshape(2, -1, 3), sample['pose'].reshape(2, -1, 3)], dim=1)
        
        inter_pose = rotation_aa_to_6d(motions.reshape(-1, 3)).reshape(2, -1, 6) # (P, *, 6)
        inter_pose = inter_pose.reshape(2, -1)
        inter_pose = inter_pose[None, :, None, :].to(device) # (B, P, 1, D)
        inter_trans = sample['trans'][None, :, None, :].to(device) # (B, P, 1, 3)
            
        if "gender" in sample:
            gender = sample['gender']
        elif gender_arg is not None:
            gender = gender_arg
        else:
            gender = torch.tensor([2, 2]) # (B, P) - default to neutral for both

        if args.inter_time_idx is None:
            args.inter_time_idx = args.seq_len // 2
            
        _, inter_joints = process_gender(smplx_layers, inter_pose, inter_trans, sample['betas'].unsqueeze(0).to(device),  gender.unsqueeze(0).to(device))
        inter_joints = inter_joints[:, :, :, :22]

        with torch.no_grad():
            sample_output = model.decoder.forward_inference(inter_pose, inter_trans, inter_joints, args.seq_len, mid_index=args.inter_time_idx)
        output = sample_output["output"].squeeze(0).cpu() # (P, T, D)
        
        trans_pred = output[:, :, -3:]
        motion_pred = output[:, :, :-3]
        name = sample['name']
        
        save_dir = os.path.join(args.save_dir, name)
        os.makedirs(save_dir, exist_ok=True)
        
        if args.save:
            output_results = {
                "betas": sample['betas'].cpu(),
                "motion_pred": motion_pred.cpu(),
                "trans_pred": trans_pred.cpu(),
                "gender": gender.cpu(), 
                "inter_time_idx": args.inter_time_idx,
            }

            save_path = os.path.join(save_dir, f"motion_pred.pkl")
            with open(save_path, "wb") as f:
                torch.save(output_results, f)
            print(f"Saved results to {save_path}")
        
        if not args.disable_vis:
            smpl_seq_p1_pred, smpl_seq_p2_pred = vis_output(motion_pred, smplx_layers, sample['betas'], gender, trans_pred)

            v.scene.add(smpl_seq_p1_pred)
            v.scene.add(smpl_seq_p2_pred)
            
            v.scene.camera.position = np.array([2.5, 2.5, 5]) # adjust camera location
            v.scene.camera.target = (smpl_seq_p1_pred.vertices.mean(axis=0).mean(axis=0) + smpl_seq_p2_pred.vertices.mean(axis=0).mean(axis=0))/2
            v.playback_fps = 10
            v.scene.fps = 10
    
            if args.vis_interactive_pose:
                vertices_p1_mid = smpl_seq_p1_pred.vertices[args.mid_index:args.mid_index+1]  # (1, V, 3)
                vertices_p2_mid = smpl_seq_p2_pred.vertices[args.mid_index:args.mid_index+1]  # (1, V, 3)
                faces = smpl_seq_p1_pred.faces  # (F, 3)
                
                # Create purple meshes for the mid frame
                purple = (0.6, 0.2, 0.8, 1.0)
                mesh_p1_mid = Meshes(vertices=vertices_p1_mid, faces=faces, color=purple)
                mesh_p2_mid = Meshes(vertices=vertices_p2_mid, faces=faces, color=purple)
                
                v.scene.add(mesh_p1_mid)
                v.scene.add(mesh_p2_mid)
            
            v.save_video(video_dir=os.path.join(save_dir, f"vis_motion_pred.mp4"))
            v.scene.remove(smpl_seq_p1_pred)
            v.scene.remove(smpl_seq_p2_pred)
            
            if args.vis_interactive_pose:
                v.scene.remove(mesh_p1_mid)
                v.scene.remove(mesh_p2_mid)

    print("Done!")