
import numpy as np
import torch
import roma
from ponimator.utils.utils import rotation_6d_to_aa



def load_result(result_file):
    """Load result motion data from pkl file."""
    name = result_file.split("/")[-1].split(".")[0]
    motion_data = {"name": name}
    with open(result_file, 'rb') as f:
            meta_data = torch.load(f)

    betas = meta_data["betas_pred"] if "betas_pred" in meta_data else meta_data["betas"] # (P, 10)
    if isinstance(betas, np.ndarray):
        betas = torch.from_numpy(betas)
    motion_data["betas"] = betas
    
    motions = meta_data["motion_pred"] # (P, T, D)
    if isinstance(motions, np.ndarray):
        motions = torch.from_numpy(motions)
    
    motions = motions.reshape(2, motions.shape[1], -1, 6)
    P, T = motions.shape[:2]
    
    pose = rotation_6d_to_aa(motions.reshape(-1, 6)).reshape(P, T, -1, 3)
    motion_data["pose"] = pose
    
    trans = meta_data["trans_pred"]
    if isinstance(trans, np.ndarray):
        trans = torch.from_numpy(trans)
    motion_data["trans"] = trans # (P, T, 3)
    
    gender = meta_data["gender"]
    if isinstance(gender, np.ndarray):
        gender = torch.from_numpy(gender)
    motion_data["gender"] = gender # (P, )
    
    if "text" in meta_data and meta_data["text"] is not None:
        motion_data["text"] = meta_data["text"]
    
    if "inter_time_idx" in meta_data:
        motion_data["inter_time_idx"] = meta_data["inter_time_idx"]
    return motion_data
   

def align_to_reference(reference_motion, result_motion, smplx_layer, device, 
                       reference_index=0, result_index=0, trans_matrix=None):
    """
    Align both reference and generated motions to a common reference frame.
    Works for both BUDDI (2-person static) and MotionX (single-person temporal) reference.
    
    Args:
        reference_motion: dict with 'root_orient', 'pose', 'betas', 'trans'
                         - BUDDI: (2,3), (2,63), (2,10), (2,3)
                         - MotionX: (T,3), (T,63), (10,), (T,3) 
        result_motion: dict with 'pose' (2,T,22,3), 'betas' (2,10), 'trans' (2,T,3)
        smplx_layer: SMPL-X layer
        device: torch device
        reference_index: index for reference motion frame (motion_idx for MotionX)
        result_index: index for result motion frame alignment (inter_time_idx)
        trans_matrix: coordinate transformation matrix (3,3), default is identity
        
    Returns:
        reference_aligned: aligned reference motion
        result_aligned: aligned result motion  
        transform_info: transformation information for rendering
    """
    if trans_matrix is None:
        trans_matrix = torch.eye(3, dtype=torch.float32)
    
    # Extract reference parameters - handle both BUDDI (2,3) and MotionX (T,3) formats
    # For BUDDI: root_orient is (2,3), for MotionX: root_orient is (T,3), extract at reference_index
    if len(reference_motion['root_orient'].shape) == 2 and reference_motion['root_orient'].shape[0] != 1:
        # Multi-person static (BUDDI) or temporal sequence
        if reference_motion['root_orient'].shape[0] == 2:
            # BUDDI: static 2-person, use as-is
            ref_root_orient = reference_motion['root_orient']  # (2, 3)
            ref_trans = reference_motion['trans']  # (2, 3)
            ref_betas = reference_motion['betas']  # (2, 10)
            ref_pose = reference_motion['pose']  # (2, 63)
        else:
            # MotionX: temporal sequence, extract frame at reference_index
            ref_root_orient = reference_motion['root_orient'][reference_index:reference_index+1]  # (1, 3)
            ref_trans = reference_motion['trans'][reference_index:reference_index+1]  # (1, 3)
            # MotionX betas is (10,) for single person, reshape to (1, 10)
            if len(reference_motion['betas'].shape) == 1:
                ref_betas = reference_motion['betas'].unsqueeze(0)  # (1, 10)
            else:
                ref_betas = reference_motion['betas']
            # MotionX pose is (T, 21, 3), extract one frame and reshape
            if len(reference_motion['pose'].shape) == 3:
                ref_pose = reference_motion['pose'][reference_index:reference_index+1].reshape(1, -1)  # (1, 63)
            else:
                ref_pose = reference_motion['pose']
    else:
        # Already extracted single frame
        ref_root_orient = reference_motion['root_orient']
        ref_trans = reference_motion['trans']
        ref_betas = reference_motion['betas'] if len(reference_motion['betas'].shape) == 2 else reference_motion['betas'].unsqueeze(0)
        ref_pose = reference_motion['pose']
    
    # Extract result parameters (temporal sequence)
    result_root_orient = result_motion['pose'][:, :, 0]  # (2, T, 3)
    result_body_pose = result_motion['pose'][:, :, 1:22]  # (2, T, 21, 3)
    result_trans = result_motion['trans']  # (2, T, 3)
    result_betas = result_motion['betas']  # (2, 10)
    
    # Step 1: Get rotation center from rest pose
    num_ref = ref_betas.shape[0]
    rest_pose = smplx_layer(poses_body=torch.zeros(num_ref, 63, device=device), betas=ref_betas.to(device))
    rot_center = rest_pose[1][:, 0, :].cpu()  # (num_ref, 3) - root joint positions
    
    # Step 2: Apply coordinate transformation to reference if provided
    ref_root_mat = roma.rotvec_to_rotmat(ref_root_orient)
    ref_root_mat_transformed = torch.einsum("mn,pnj->pmj", trans_matrix, ref_root_mat)
    ref_root_orient_transformed = roma.rotmat_to_rotvec(ref_root_mat_transformed)
    ref_trans_transformed = torch.einsum("mn,pn->pm", trans_matrix, rot_center + ref_trans) - rot_center
    
    # Step 3: Align result motion at result_index to reference pose
    # Get result pose at result frame
    result_root_ref = result_root_orient[:, result_index]  # (2, 3)
    result_trans_ref = result_trans[:, result_index]  # (2, 3)
    
    # Calculate rigid transformation from result to reference at interaction time
    # Use person 0 (or only person if MotionX)
    result_root_mat_ref = roma.rotvec_to_rotmat(result_root_ref[0:1])  # (1, 3, 3)
    ref_root_mat_ref = ref_root_mat_transformed[0:1]  # (1, 3, 3)

    # Rotation from result to reference
    rot_result_to_ref = ref_root_mat_ref @ torch.inverse(result_root_mat_ref)  # (1, 3, 3)
    
    # Translation alignment
    trans_offset = ref_trans_transformed[0] + rot_center[0] - rot_result_to_ref[0] @ (rot_center[0] + result_trans_ref[0])
    
    # Step 4: Apply transformation to all result frames
    result_root_mat = roma.rotvec_to_rotmat(result_root_orient)  # (2, T, 3, 3)
    # Apply rotation: rot_result_to_ref[0] @ result_root_mat for all frames
    rot_mat = rot_result_to_ref[0]  # (3, 3)
    result_root_mat_aligned = torch.einsum("ij,ptjk->ptik", rot_mat, result_root_mat)  # (2, T, 3, 3)
    result_root_orient_aligned = roma.rotmat_to_rotvec(result_root_mat_aligned)  # (2, T, 3)
    
    # Apply translation transformation
    trans_with_center = rot_center[0].unsqueeze(0).unsqueeze(1) + result_trans  # (2, T, 3)
    result_trans_aligned = torch.einsum("pti,ji->ptj", trans_with_center, rot_mat) - rot_center[0].unsqueeze(0).unsqueeze(1) + trans_offset.unsqueeze(0).unsqueeze(1)
    
    # Package aligned results
    # Keep dimensions as transformed (squeeze handled by caller if needed)
    reference_aligned = {
        'root_orient': ref_root_orient_transformed,
        'pose': ref_pose,
        'betas': ref_betas,
        'trans': ref_trans_transformed
    }
    
    result_aligned = {
        'root_orient': result_root_orient_aligned,
        'pose': result_body_pose,
        'betas': result_betas,
        'trans': result_trans_aligned,
        'gender': result_motion.get('gender', None)
    }
    
    transform_info = {
        'rot_center': rot_center,
        'rot_result_to_ref': rot_result_to_ref,
        'trans_matrix': trans_matrix,
        'trans_offset': trans_offset,
        'ref_root_mat_transformed': ref_root_mat_transformed,
        'result_root_mat_aligned': result_root_mat_aligned
    }
    
    return reference_aligned, result_aligned, transform_info



def get_view_fac_in_px(cam, pixel_aspect_x: float, pixel_aspect_y: float,
                       resolution_x_in_px: int, resolution_y_in_px: int) -> int:
    """ Returns the camera view in pixels.

    :param cam: The camera object.
    :param pixel_aspect_x: The pixel aspect ratio along x.
    :param pixel_aspect_y: The pixel aspect ratio along y.
    :param resolution_x_in_px: The image width in pixels.
    :param resolution_y_in_px: The image height in pixels.
    :return: The camera view in pixels.
    """
    # Determine the sensor fit mode to use
    if cam.sensor_fit == 'AUTO':
        if pixel_aspect_x * resolution_x_in_px >= pixel_aspect_y * resolution_y_in_px:
            sensor_fit = 'HORIZONTAL'
        else:
            sensor_fit = 'VERTICAL'
    else:
        sensor_fit = cam.sensor_fit

    # Based on the sensor fit mode, determine the view in pixels
    pixel_aspect_ratio = pixel_aspect_y / pixel_aspect_x
    if sensor_fit == 'HORIZONTAL':
        view_fac_in_px = resolution_x_in_px
    else:
        view_fac_in_px = pixel_aspect_ratio * resolution_y_in_px

    return view_fac_in_px



def get_sensor_size(cam) -> float:
    """ Returns the sensor size in millimeters based on the configured sensor_fit.

    :param cam: The camera object.
    :return: The sensor size in millimeters.
    """
    if cam.sensor_fit == 'VERTICAL':
        sensor_size_in_mm = cam.sensor_height
    else:
        sensor_size_in_mm = cam.sensor_width
    return sensor_size_in_mm


# ref: blenderproc
def get_intrinsics_from_K_matrix(K, camera, image_width: int, image_height: int):
    """ Set the camera intrinsics via a K matrix.

    The K matrix should have the format:
        [[fx, 0, cx],
         [0, fy, cy],
         [0, 0,  1]]

    This method is based on https://blender.stackexchange.com/a/120063.

    :param K: The 3x3 K matrix.
    :param image_width: The image width in pixels.
    :param image_height: The image height in pixels.
    :param clip_start: Clipping start.
    :param clip_end: Clipping end.
    """

    cam = camera.data

    if abs(K[0][1]) > 1e-7:
        raise ValueError(f"Skew is not supported by blender and therefore "
                         f"not by BlenderProc, set this to zero: {K[0][1]} and recalibrate")

    fx, fy = K[0][0], K[1][1]
    cx, cy = K[0][2], K[1][2]

    # If fx!=fy change pixel aspect ratio
    pixel_aspect_x = pixel_aspect_y = 1
    if fx > fy:
        pixel_aspect_y = fx / fy
    elif fx < fy:
        pixel_aspect_x = fy / fx

    # Compute sensor size in mm and view in px
    pixel_aspect_ratio = pixel_aspect_y / pixel_aspect_x
    view_fac_in_px = get_view_fac_in_px(cam, pixel_aspect_x, pixel_aspect_y, image_width, image_height)
    sensor_size_in_mm = get_sensor_size(cam)

    # Convert focal length in px to focal length in mm
    f_in_mm = fx * sensor_size_in_mm / view_fac_in_px

    # Convert principal point in px to blenders internal format
    shift_x = (cx - (image_width - 1) / 2) / -view_fac_in_px
    shift_y = (cy - (image_height - 1) / 2) / view_fac_in_px * pixel_aspect_ratio

    # Finally set all intrinsics
    return f_in_mm, shift_x, shift_y, pixel_aspect_x, pixel_aspect_y
