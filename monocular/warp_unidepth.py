
from UniDepth.unidepth.models import UniDepthV2
import numpy as np
from PIL import Image
import torch
import cv2, os
from tqdm import tqdm
from diffusers.utils import export_to_video
import argparse

from utils import get_look_up_camera_seq, get_look_right_camera_seq, get_circle_camera_seq, project_points_to_image_pytorch, resize_pil_image


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('image_path', type=str, help='image_path')
    parser.add_argument('--direction', type=str, choices=['up','down','left','right'], default='left', help='direction')
    parser.add_argument('--degree', type=float, default=15, help='degree')
    parser.add_argument('--use_confidence_mask', action='store_true', help='use_confidence_mask')
    args = parser.parse_args()

    os.makedirs('logs/imgs', exist_ok=True)
    image_path = args.image_path

    model = UniDepthV2.from_pretrained("lpiccinelli/unidepth-v2-vitl14")

    # Move to CUDA, if any
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    # Load the RGB image and the normalization will be taken care of by the model
    image = resize_pil_image(Image.open(image_path).convert('RGB'), 1024)
    rgb = torch.from_numpy(np.array(image)).to(torch.float32).div(255).permute(2, 0, 1).to(device) # C, H, W

    predictions = model.infer(rgb)
    print('Got the predictions from UniDepth.')

    # Confidence Prediction
    confidence = predictions["confidence"].cpu().numpy().reshape(-1) # (H*W,)
    confidence_mask = confidence > np.quantile(confidence, 0.2) if args.use_confidence_mask else np.ones_like(confidence).astype(bool)

    rgb = rgb.permute(1, 2, 0).reshape(-1, 3) # (H*W, 3)
    rgb = rgb[confidence_mask]

    # Metric Depth Estimation
    depth = predictions["depth"].reshape(-1) # (H*W,)
    depth = depth[confidence_mask]

    # Point Cloud in Camera Coordinate
    points = predictions["points"] # B=1, 3, H, W
    points = points.permute(0, 2, 3, 1).reshape(-1, 3) # (H*W, 3)
    points = points[confidence_mask]

    # Intrinsics Prediction
    intrinsics = predictions["intrinsics"][0]

    # Camera Extrinsics
    extrinsics = np.eye(4)

    # calculate the camera parameters
    look_at_depth = np.median(depth.cpu().numpy())
    frame_num = 25

    max_degree = args.degree
    # circle camera
    # radius = np.median(depth.cpu().numpy())/30
    # cams = get_circle_camera_seq(extrinsics, radius, frame_num, look_at_depth, direction='right')

    # direction camera
    if args.direction == 'up':
        cams = get_look_up_camera_seq(extrinsics, max_degree, frame_num, look_at_depth)
    elif args.direction == 'down':
        cams = get_look_up_camera_seq(extrinsics, -max_degree, frame_num, look_at_depth)
    elif args.direction == 'right':
        cams = get_look_right_camera_seq(extrinsics, max_degree, frame_num, look_at_depth)
    elif args.direction == 'left':
        cams = get_look_right_camera_seq(extrinsics, -max_degree, frame_num, look_at_depth)
    else:
        raise ValueError('Invalid direction')

    frames = []
    masks = []
    for i,cam in tqdm(enumerate(cams)):
        scale_factor = 1
        rendered_image, mask = project_points_to_image_pytorch(points, rgb, torch.from_numpy(cam).to(device), 
                                                            intrinsics/scale_factor, 
                                                            torch.tensor([size//scale_factor for size in image.size[::-1]], device=device),
                                                            morph=True)
        frames.append(rendered_image)
        masks.append(mask)

        cv2.imwrite(f'logs/imgs/rendered_image_{i}.jpg', cv2.cvtColor(rendered_image*255, cv2.COLOR_BGR2RGB))
        cv2.imwrite(f'logs/imgs/mask_{i}.jpg', mask*255)
    export_to_video(frames, 'logs/video.mp4', fps=6)
    export_to_video(masks, 'logs/mask.mp4', fps=6)