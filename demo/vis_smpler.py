import glob
import imageio
import numpy as np
import os
import os.path
import sys
import pickle
import shutil
sys.path.append(sys.path[0] + r"/../")
import torch
import argparse
from PIL import Image
from tqdm import tqdm


from aitviewer.models.smpl import SMPLLayer
from aitviewer.configuration import CONFIG as C

from blendify import scene
from blendify.colors import UniformColors
from blendify.materials import PrincipledBSDFMaterial

from ponimator.utils.vis_utils import load_result, align_to_reference, get_intrinsics_from_K_matrix


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="data/motionx/Ways_to_Catch_Twisted_Ankle_clip1")
    parser.add_argument("--result_dir", type=str, default="outputs/Ways_to_Catch_Twisted_Ankle_clip1")
    parser.add_argument("--save_dir", type=str, default=None)
    parser.add_argument("--body_model_root", type=str, default="body_models")
    parser.add_argument("-n", "--n-samples", default=256, type=int,
                        help="Number of paths to trace for each pixel in the render (default: 256)")
    args = parser.parse_args()

    C.update_conf({'smplx_models': args.body_model_root})
    smplx_layer = SMPLLayer(model_type='smplx', gender='neutral', num_betas=10)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    if args.save_dir is None:
        args.save_dir = args.result_dir
    os.makedirs(args.save_dir, exist_ok=True)
    
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
    right_hand_pose = hand_pose.unsqueeze(0).unsqueeze(0).repeat(2, 30, 1, 1)
    left_hand_pose = hand_pose.unsqueeze(0).unsqueeze(0).repeat(2, 30, 1, 1)
    left_hand_pose[:, :, :, 1] *= -1
    left_hand_pose[:, :, :, 2] *= -1
    
    input_file = glob.glob(os.path.join(args.data_dir, "*.pkl"))[0]
    result_file = os.path.join(args.result_dir, "motion_pred.pkl")
    
    with open(input_file, 'rb') as f:
        input_data = pickle.load(f)
    focal = input_data['focal']
    princpt = input_data['princpt']
    img_shape = input_data['img_size_wh']
    
    img_path = os.path.join(args.data_dir, "image_ori.png")
    img_inpaint_path = os.path.join(args.data_dir, "image_inpaint.png")
    
    img_inpaint = Image.open(img_inpaint_path)
    img_ori = Image.open(img_path)
    img_ori = np.array(img_ori)
    img_inpaint = img_inpaint.resize((img_ori.shape[1], img_ori.shape[0]), Image.Resampling.LANCZOS)
    img_inpaint = np.array(img_inpaint)
    result_data = load_result(result_file)
    
    root_pose = torch.FloatTensor(np.array(input_data['smplx_root_pose'])).reshape(1, 3, )
    body_pose = torch.FloatTensor(np.array(input_data['smplx_body_pose'])).reshape(1, 21, 3)
    shape = torch.FloatTensor(np.array(input_data['smplx_shape'])).reshape(1, 10, )
    trans = torch.FloatTensor(np.array(input_data['cam_trans'])).reshape(1, 3,)
    
    lhand_pose_person1 = torch.FloatTensor(np.array(input_data['smplx_lhand_pose'])).reshape(1, 15, 3).repeat(30, 1, 1)
    rhand_pose_person1 = torch.FloatTensor(np.array(input_data['smplx_rhand_pose'])).reshape(1, 15, 3).repeat(30, 1, 1)
        
    motion_data = {
        "root_orient": root_pose,
        "pose": body_pose,
        "betas": shape,
        "trans": trans,
        "gender": "neutral"
    }
    motion_idx = 0 # single pose in motion_data
    inter_time_idx = 15 # single pose in result_data
        
    # Use unified alignment function with database annotation
    # reference_index=motion_idx (for motion_data), result_index=inter_time_idx (for result_data)
    motionx_aligned, result_aligned, transform_info = align_to_reference(
        motion_data, result_data, smplx_layer, device,
        reference_index=motion_idx,
        result_index=inter_time_idx,
        trans_matrix=None)
        
    smpl_p1_ours = smplx_layer(
        poses_body=result_aligned['pose'][0].reshape(-1, 63).to(device),
        poses_root=result_aligned['root_orient'][0].reshape(-1, 3).to(device),
        # poses_left_hand=lhand_pose_person1.reshape(-1, 45).to(device),
        # poses_right_hand=rhand_pose_person1.reshape(-1, 45).to(device),
        poses_left_hand=left_hand_pose[0].reshape(-1, 45).to(device),
        poses_right_hand=right_hand_pose[0].reshape(-1, 45).to(device),
        betas=result_aligned['betas'][0].reshape(1, 10).to(device),
        trans=result_aligned['trans'][0].reshape(-1, 3).to(device))
    p1_v_ours = smpl_p1_ours[0].cpu().numpy()
    
    smpl_p2_ours = smplx_layer(
        poses_body=result_aligned['pose'][1].reshape(-1, 63).to(device),
        poses_root=result_aligned['root_orient'][1].reshape(-1, 3).to(device),
        poses_left_hand=left_hand_pose[1].reshape(-1, 45).to(device),
        poses_right_hand=right_hand_pose[1].reshape(-1, 45).to(device),
        betas=result_aligned['betas'][1].reshape(1, 10).to(device),
        trans=result_aligned['trans'][1].reshape(-1, 3).to(device))
    p2_v_ours = smpl_p2_ours[0].cpu().numpy()

    intrinsic = np.array(
        [[focal[0], 0, princpt[0]],
        [0, focal[1], princpt[1]],
        [0, 0, 1]]
    )

    T_opencv_to_blender = np.array([
    [1, 0, 0, 0],
    [0, -1, 0, 0],
    [0, 0, -1, 0],
    [0, 0, 0, 1]
    ])

    R_opencv_to_blender = np.array([
    [1, 0, 0],
    [0, -1, 0],
    [0, 0, -1]
    ])

    p1_v = np.einsum('ij,tkj->tki', R_opencv_to_blender, p1_v_ours)
    p2_v = np.einsum('ij,tkj->tki', R_opencv_to_blender, p2_v_ours)
    
    iw, ih = img_ori.shape[1], img_ori.shape[0]
    camera = scene.set_perspective_camera((iw, ih), focal_dist=5000)
    f, shift_x, shift_y, pixel_aspect_x, pixel_aspect_y = get_intrinsics_from_K_matrix(intrinsic, camera.blender_camera, iw, ih)
    camera.blender_camera.data.lens = f
    camera.blender_camera.data.shift_x = shift_x
    camera.blender_camera.data.shift_y = shift_y
    
    # Define the materials
    # Material and Colors for SMPL mesh
    smpl_material = PrincipledBSDFMaterial()
    smpl_color = UniformColors((0.11, 0.53, 0.8))
    smpl_color2 = UniformColors((1.0, 0.27, 0))

    # Set the lights; one main sunlight and a secondary light without visible shadows to make the scene overall brighter
    sunlight = scene.lights.add_sun(
        strength=2.3, rotation_mode="euleryz", rotation=(-45, -90)
    )
    sunlight2 = scene.lights.add_sun(
        strength=3, rotation_mode="euleryz", rotation=(-45, 165)
    )
    sunlight2.cast_shadows = True
    camera_pose = np.eye(4)

    smpl_faces = smplx_layer.faces.cpu().numpy()
    smpl_mesh = scene.renderables.add_mesh(p1_v[inter_time_idx], smpl_faces, smpl_material, smpl_color)
    smpl_mesh2 = scene.renderables.add_mesh(p2_v[inter_time_idx], smpl_faces, smpl_material, smpl_color2)
    smpl_mesh.set_smooth()
    smpl_mesh2.set_smooth()

    video_tmp_save_dir = os.path.join(args.save_dir, "video")
    os.makedirs(video_tmp_save_dir, exist_ok=True)
    images = []
    for time_idx in tqdm(range(len(p1_v))):
        v1 = p1_v[time_idx]
        v2 = p2_v[time_idx]
        smpl_mesh.update_vertices(v1)
        smpl_mesh2.update_vertices(v2)
        
        img = scene.render(use_gpu=True, samples=args.n_samples) 
        alpha = img[:, :, 3:4] / 255.0
        
        img_with_bkg = (img[:, :, :3] * alpha + img_inpaint * (1. - alpha))
        img = img_with_bkg.astype(np.uint8)
        
        save_path = os.path.join(video_tmp_save_dir, f"{time_idx:02d}.png")
        imageio.imwrite(save_path, img)
        images.append(img)

    video_path = os.path.join(args.save_dir, "rendered_video.mp4")
    with imageio.get_writer(video_path, format='FFMPEG', fps=10, codec='libx264', pixelformat='yuv420p', quality=8) as writer:
        for img in images:
            # Ensure image is uint8
            if img.dtype != np.uint8:
                img = img.astype(np.uint8)
            writer.append_data(img)
    
    shutil.rmtree(video_tmp_save_dir)
    print("video saved to {}".format(args.save_dir))
    # clear the scene
    scene.clear()
