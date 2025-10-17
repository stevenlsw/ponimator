import os
import sys
import os.path as osp

# Add parent directory to PYTHONPATH for human_models import
parent_dir = osp.dirname(osp.dirname(osp.abspath(__file__)))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

import argparse
import numpy as np
import torchvision.transforms as transforms
import torch.backends.cudnn as cudnn
import torch
import cv2
import datetime
import pickle
from tqdm import tqdm
from pathlib import Path
from human_models.human_models import SMPLX
from ultralytics import YOLO
from main.base import Tester
from main.config import Config
from utils.data_utils import load_img, process_bbox, generate_patch_image
from utils.visualization_utils import render_mesh
from utils.inference_utils import non_max_suppression


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_gpus', type=int, dest='num_gpus')
    parser.add_argument('--file_name', type=str, required=True, help='Full path to input image file')
    parser.add_argument('--ckpt_name', type=str, default='model_dump')
    parser.add_argument('--output_folder', type=str, default='./demo/output', help='Output folder path')
    parser.add_argument('--save_pkl', action='store_true', help='Save model outputs to pickle file')
    args = parser.parse_args()
    return args

def main():
    args = parse_args()
    cudnn.benchmark = True

    # init config
    time_str = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    root_dir = Path(__file__).resolve().parent.parent
    config_path = osp.join('./pretrained_models', args.ckpt_name, 'config_base.py')
    cfg = Config.load_config(config_path)
    checkpoint_path = osp.join('./pretrained_models', args.ckpt_name, f'{args.ckpt_name}.pth.tar')
    
    # Use the full file path as input
    img_path = args.file_name
    
    # Create output path based on input filename and specified output folder
    input_filename = osp.basename(img_path)
    output_filename = f'result_{input_filename}'
    output_folder = args.output_folder
    os.makedirs(output_folder, exist_ok=True)
    output_path = osp.join(output_folder, output_filename)
    
    exp_name = f'inference_{osp.splitext(input_filename)[0]}_{args.ckpt_name}_{time_str}'

    new_config = {
        "model": {
            "pretrained_model_path": checkpoint_path,
        },
        "log":{
            'exp_name':  exp_name,
            'log_dir': osp.join(root_dir, 'outputs', exp_name, 'log'),  
            }
    }
    cfg.update_config(new_config)
    cfg.prepare_log()
    
    # init human models
    smpl_x = SMPLX(cfg.model.human_model_path)

    # init tester
    demoer = Tester(cfg)
    demoer.logger.info(f"Using 1 GPU.")
    demoer.logger.info(f'Inference [{img_path}] with [{cfg.model.pretrained_model_path}].')
    demoer._make_model()

    # init detector
    bbox_model = getattr(cfg.inference.detection, "model_path", 
                        './pretrained_models/yolov8x.pt')
    detector = YOLO(bbox_model)

    # prepare input image
    transform = transforms.ToTensor()
    original_img = load_img(img_path)
    vis_img = original_img.copy()
    original_img_height, original_img_width = original_img.shape[:2]
    
    # detection, xyxy
    yolo_bbox = detector.predict(original_img, 
                            device='cuda', 
                            classes=00, 
                            conf=cfg.inference.detection.conf, 
                            save=cfg.inference.detection.save, 
                            verbose=cfg.inference.detection.verbose
                                )[0].boxes.xyxy.detach().cpu().numpy()

    if len(yolo_bbox)<1:
        # save original image if no bbox
        num_bbox = 0
    else:
        # only select the largest bbox (single person)
        num_bbox = 1

    # loop all detected bboxes
    for bbox_id in range(num_bbox):
        yolo_bbox_xywh = np.zeros((4))
        yolo_bbox_xywh[0] = yolo_bbox[bbox_id][0]
        yolo_bbox_xywh[1] = yolo_bbox[bbox_id][1]
        yolo_bbox_xywh[2] = abs(yolo_bbox[bbox_id][2] - yolo_bbox[bbox_id][0])
        yolo_bbox_xywh[3] = abs(yolo_bbox[bbox_id][3] - yolo_bbox[bbox_id][1])
        
        # xywh
        bbox = process_bbox(bbox=yolo_bbox_xywh, 
                            img_width=original_img_width, 
                            img_height=original_img_height, 
                            input_img_shape=cfg.model.input_img_shape, 
                            ratio=getattr(cfg.data, "bbox_ratio", 1.25))                
        img, _, _ = generate_patch_image(cvimg=original_img, 
                                            bbox=bbox, 
                                            scale=1.0, 
                                            rot=0.0, 
                                            do_flip=False, 
                                            out_shape=cfg.model.input_img_shape)
            
        img = transform(img.astype(np.float32))/255
        img = img.cuda()[None,:,:,:]
        inputs = {'img': img}
        targets = {}
        meta_info = {}

        # mesh recovery
        with torch.no_grad():
            out = demoer.model(inputs, targets, meta_info, 'test')

        mesh = out['smplx_mesh_cam'].detach().cpu().numpy()[0]
        
        # render mesh
        focal = [cfg.model.focal[0] / cfg.model.input_body_shape[1] * bbox[2], 
                 cfg.model.focal[1] / cfg.model.input_body_shape[0] * bbox[3]]
        princpt = [cfg.model.princpt[0] / cfg.model.input_body_shape[1] * bbox[2] + bbox[0], 
                   cfg.model.princpt[1] / cfg.model.input_body_shape[0] * bbox[3] + bbox[1]]
        
        
        # Save model outputs to pickle if requested
        if args.save_pkl:
            pkl_filename = f'result_{osp.splitext(input_filename)[0]}_params.pkl'
            pkl_path = osp.join(output_folder, pkl_filename)
            
            # Extract the specified parameters from model output
            model_outputs = {}
            if 'smplx_root_pose' in out:
                model_outputs['smplx_root_pose'] = out['smplx_root_pose'].detach().cpu().numpy()
            if 'smplx_body_pose' in out:
                model_outputs['smplx_body_pose'] = out['smplx_body_pose'].detach().cpu().numpy()
            if 'smplx_lhand_pose' in out:
                model_outputs['smplx_lhand_pose'] = out['smplx_lhand_pose'].detach().cpu().numpy()
            if 'smplx_rhand_pose' in out:
                model_outputs['smplx_rhand_pose'] = out['smplx_rhand_pose'].detach().cpu().numpy()
            if 'smplx_shape' in out:
                model_outputs['smplx_shape'] = out['smplx_shape'].detach().cpu().numpy()
            if 'cam_trans' in out:
                model_outputs['cam_trans'] = out['cam_trans'].detach().cpu().numpy()
            model_outputs['focal'] = focal
            model_outputs['princpt'] = princpt
            model_outputs['img_size_wh'] = [original_img_width, original_img_height]
            
            # Save to pickle file
            with open(pkl_path, 'wb') as f:
                pickle.dump(model_outputs, f)
            print(f"Model outputs saved to: {pkl_path}")

        # draw the bbox on img
        vis_img = cv2.rectangle(vis_img, (int(yolo_bbox[bbox_id][0]), int(yolo_bbox[bbox_id][1])), 
                                (int(yolo_bbox[bbox_id][2]), int(yolo_bbox[bbox_id][3])), (0, 255, 0), 1)
        # draw mesh
        vis_img = render_mesh(vis_img, mesh, smpl_x.face, {'focal': focal, 'princpt': princpt}, mesh_as_vertices=False)

    # save rendered image
    cv2.imwrite(output_path, vis_img[:, :, ::-1])
    print(f"Result saved to: {output_path}")


if __name__ == "__main__":
    main()