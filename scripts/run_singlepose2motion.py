import os
import os.path
import sys
sys.path.append(sys.path[0] + r"/../")
import numpy as np
import torch
import random

from aitviewer.models.smpl import SMPLLayer
from aitviewer.headless import HeadlessRenderer
from aitviewer.configuration import CONFIG as C
from aitviewer.renderables.meshes import Meshes

from ponimator.models import ContactPoseGen, ContactMotionGen
from ponimator.utils.config_utils import get_config
from ponimator.datasets.wild_singlepose import SMPLerDataset, MotionXSinglePoseDataset
from ponimator.utils.inference_utils import load_model, vis_output, optimize_beta_gender
from ponimator.utils.utils import process_gender


def map_gender(gender_str):
    """Map gender string to integer: male=0, female=1, neutral=2"""
    if gender_str is None:
        return 2  # default to neutral
    gender_map = {'male': 0, 'female': 1, 'neutral': 2}
    return gender_map.get(gender_str.lower(), 2)
    

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--save_dir", type=str, default="outputs")
    parser.add_argument("--pose_checkpoint_path", type=str, default="checkpoints/contactpose.ckpt")
    parser.add_argument("--motion_checkpoint_path", type=str, default="checkpoints/contactmotion.ckpt")
    parser.add_argument("--pose_config_path", type=str, default="configs/pose.yaml")
    parser.add_argument("--motion_config_path", type=str, default="configs/motion.yaml")
    parser.add_argument("--body_model_root", type=str, default="body_models")
    parser.add_argument("--seq_len", type=int, default=30, help="sequence length (trained on 30)")
    # inference settings
    parser.add_argument("--data_source", choices=["smpler", "motion-x-singlepose"], default="motion-x-singlepose", help="single-pose data source")
    parser.add_argument("--data_dir", type=str, default="data/motionx/Back_Flip_Kungfu_wushu_Trim9_clip1", help="data dir")
    parser.add_argument("--inter_time_idx", type=int, default=None, help="Index of the interactive pose range (0-29)")
    parser.add_argument("--gender", type=str, default=None, help="gender for second person (male, female, or neutral)")
    parser.add_argument("--seed", type=int, default=4, help="random seed")
    parser.add_argument("--cfg_weight", type=float, default=2.5, help="CFG weight")

    # additional settings
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--lr", type=float, default=1e-2)
    parser.add_argument("--text", type=str, default=None, help="text input")
    parser.add_argument("--save", action="store_true", help="save the results")
    parser.add_argument("--disable_vis", action="store_true", help="disable visualization")
    parser.add_argument("--vis_interactive_pose", action="store_true", help="visualize generated interactive pose")
    
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    
    gender_arg = map_gender(args.gender)
    
    C.update_conf({'smplx_models': args.body_model_root})
    smplx_layers = [SMPLLayer(model_type='smplx', gender='male', num_betas=10), 
                    SMPLLayer(model_type='smplx', gender='female', num_betas=10),
                    SMPLLayer(model_type='smplx', gender='neutral', num_betas=10)]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Create a viewer.
    if not args.disable_vis:
        v = HeadlessRenderer()
    
    # Load pose model
    pose_model_cfg = get_config(args.pose_config_path)
    pose_model_cfg.defrost()
    pose_model_cfg.CFG_WEIGHT = args.cfg_weight
    pose_model_cfg.freeze()
    pose_model = ContactPoseGen(pose_model_cfg)
    
    load_model(args.pose_checkpoint_path, pose_model)
    pose_model.eval()
    pose_model.to(device)
    
    # Load motion model
    motion_model_cfg = get_config(args.motion_config_path)
    motion_model = ContactMotionGen(motion_model_cfg)
    load_model(args.motion_checkpoint_path, motion_model)
    motion_model.eval()
    motion_model.to(device)

    if args.data_source == "smpler":
        dataset = SMPLerDataset(args.data_dir, smplx_layer=smplx_layers[2], device=device) # neutral gender
    
    elif args.data_source == "motion-x-singlepose":
        dataset = MotionXSinglePoseDataset(args.data_dir, smplx_layer=smplx_layers[2], device=device) # neutral gender
    else:
        raise ValueError(f"Invalid data source: {args.data_source}")

    for idx, sample in enumerate(dataset):
        
        sample['pose_mask'] = torch.tensor([1, 0]).long() # (2, )

        if args.text is not None:
            sample['text'] = [args.text]
        else:
            sample['text'] = [""]

        for key in sample:
            if isinstance(sample[key], torch.Tensor):
                sample[key] = sample[key][None].to(device)
         
        template = torch.tensor([1, 0, 0, 1, 0, 0], device=sample['motions'].device, dtype=sample['motions'].dtype)
        B, P = 1, 1
        template_motion = template.repeat(B, P, 22, 1).reshape (B, P, -1)
        
        if "gender" in sample:
            _, rest_joints = process_gender(smplx_layers, template_motion.unsqueeze(2), 0 * sample["trans"].unsqueeze(2), sample['betas'], sample['gender']) # (B, P, 1, J, 3)
        else:
            _, rest_joints = process_gender(smplx_layers, template_motion.unsqueeze(2), 0 * sample["trans"].unsqueeze(2), sample['betas']) # (B, P, 1, J, 3)
        rest_joints = rest_joints.squeeze(2) # (B, P, J, 3)
        rest_joints = rest_joints[:, :, :22]
        
        sample["rest_joints"] = rest_joints
        
        with torch.no_grad():
            sample = pose_model.forward_test(sample)
        output = sample["output"].squeeze(0) # (1, D)
        
        joint_input_dim = 22
        pred_pose = output[:, :-joint_input_dim*3]
        
        if pred_pose.shape[0] == 2:
            motion_pred = pred_pose[:, :-3].unsqueeze(1) # (P, T, D)
            trans_pred = pred_pose[:, -3:].unsqueeze(1) # (P, T, 3)
            pred_joints = output[1:2, -joint_input_dim*3:].reshape(1, joint_input_dim, 3)
        else:
            condition_pose = sample["motions"][0, 0:1] # (1, D)
            condition_trans = sample["trans"][0, 0:1]
            pred_joints = output[:, -joint_input_dim*3:].reshape(1, joint_input_dim, 3)

            motion_pred = torch.cat([condition_pose, pred_pose[:, :-3]], dim=0).unsqueeze(1) # (P, T, D)
            trans_pred = torch.cat([condition_trans, pred_pose[:, -3:]], dim=0).unsqueeze(1) # (P, T, 3) 
       
        # optimize beta using specified gender
        second_person_gender_tensor = torch.tensor([gender_arg], device=device)
        pred_betas = optimize_beta_gender(smplx_layers, second_person_gender_tensor, pred_joints, args.epochs, args.lr) # (1, 10)
        pred_betas = torch.cat([sample["betas"][0, 0:1], pred_betas], dim=0) # (2, 10)
        
        inter_pose = motion_pred.unsqueeze(0) # (B, P, 1, D)
        inter_trans = trans_pred.unsqueeze(0) # (B, P, 1, 3)
        pred_betas = pred_betas.unsqueeze(0) # (B, P, 10)

        # Create gender tensor for both persons
        first_person_gender = 2 if "gender" not in sample else sample['gender'][0].item()
        gender = torch.tensor([first_person_gender, gender_arg], device=device)
        
        # Use process_gender to handle both persons
        _, inter_joints = process_gender(smplx_layers, inter_pose, inter_trans, pred_betas, gender.unsqueeze(0))
        inter_joints = inter_joints[:, :, :, :22]
        
        with torch.no_grad():
            sample_output = motion_model.decoder.forward_inference(inter_pose, inter_trans, inter_joints, args.seq_len, mid_index=args.inter_time_idx)
        
        output = sample_output["output"].squeeze(0).cpu() # (P, T, D)
        
        trans_pred = output[:, :, -3:]
        motion_pred = output[:, :, :-3]
        
        pred_betas = pred_betas.squeeze(0) # (P, 10)
        name = sample["name"]
        
        save_dir = os.path.join(args.save_dir, name)
        os.makedirs(save_dir, exist_ok=True)
        
        if args.save:
            output_results = {
                "betas_pred": pred_betas.cpu(),
                "motion_pred": motion_pred.cpu(),
                "trans_pred": trans_pred.cpu(),
                "gender": gender.cpu(),
            }
            save_path = os.path.join(save_dir, f"motion_pred.pkl")
            with open(save_path, "wb") as f:
                torch.save(output_results, f)
            print(f"Saved results to {save_path}")
        
        if not args.disable_vis:
            smpl_seq_p1_pred, smpl_seq_p2_pred = vis_output(motion_pred, smplx_layers, pred_betas, gender, trans_pred)

            v.scene.add(smpl_seq_p1_pred)
            v.scene.add(smpl_seq_p2_pred)

            if args.vis_interactive_pose:
                purple = (0.6, 0.2, 0.8, 1.0)
                vertices_p1_input = smpl_seq_p1_pred.vertices[0:1]  # (1, V, 3)
                faces = smpl_seq_p1_pred.faces  # (F, 3)
                mesh_p1_input = Meshes(vertices=vertices_p1_input, faces=faces, color=purple)
                v.scene.add(mesh_p1_input)

                # Overlay the initial pose of the generated second person in a complementary teal
                teal = (0.2, 0.8, 0.7, 1.0)
                vertices_p2_input = smpl_seq_p2_pred.vertices[0:1]  # (1, V, 3)
                mesh_p2_input = Meshes(vertices=vertices_p2_input, faces=faces, color=teal)
                v.scene.add(mesh_p2_input)

            v.scene.camera.position = np.array([2.5, 2.5, 5])
            v.scene.camera.target = (smpl_seq_p1_pred.vertices.mean(axis=0).mean(axis=0) + smpl_seq_p2_pred.vertices.mean(axis=0).mean(axis=0))/2
            v.playback_fps = 10
            v.scene.fps = 10

            v.save_video(video_dir=os.path.join(save_dir, f"vis_motion_pred.mp4"))
            
            v.scene.remove(smpl_seq_p1_pred)
            v.scene.remove(smpl_seq_p2_pred)
            if args.vis_interactive_pose:
                v.scene.remove(mesh_p1_input)
                v.scene.remove(mesh_p2_input)
        
    print("Done!")
