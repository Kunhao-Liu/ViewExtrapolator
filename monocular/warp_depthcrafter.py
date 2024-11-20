
import numpy as np
from PIL import Image
import torch
import cv2, os
from tqdm import tqdm
from diffusers.utils import export_to_video
import argparse
import glob
from torchvision.transforms.functional import resize

from utils import get_look_up_camera_seq, get_look_right_camera_seq, get_circle_camera_seq, project_points_to_image_pytorch, resize_pil_image


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--output_folder', type=str, default='camel')
    parser.add_argument('--input_path', type=str, default='examples/camel.mp4')
    
    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    os.makedirs('logs/imgs', exist_ok=True)

    input_path = args.input_path
    output_folder = args.output_folder

    # read depth
    depth_path = glob.glob(f'{output_folder}/*.npz')[0]
    depth = np.load(depth_path)['depth'] # (F=25, H, W)
    depth = torch.from_numpy(depth).to(device) # in 0-1


    # read video
    cap = cv2.VideoCapture(input_path)
    frames = []
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(frame)    
    cap.release()
    frames = np.array(frames)
    frames = torch.from_numpy(frames).to(device)/255.0
    frames = frames.permute(0, 3, 1, 2) # (F, C, H, W)
    frames = resize(frames, (depth.shape[-2], depth.shape[-1])) # (F, C, H, W)

    # manually set camera intrinsics
    H = depth.shape[-2]
    W = depth.shape[-1]
    c_x = 0.5 * W
    c_y = 0.5 * H
    f_x = 525.0
    f_y = 525.0

    K = torch.tensor([[f_x, 0, c_x],
                      [0, f_y, c_y],
                      [0, 0, 1]]).to(device)
    
    # Camera Extrinsics
    extrinsics = np.eye(4)

    # calculate the camera parameters
    look_at_depth = np.median((1.0 / (depth[0] + 0.1)).cpu().numpy())
    frame_num = 25

    # direction camera
    max_degree = 10
    cams = get_look_right_camera_seq(extrinsics, max_degree, frame_num, look_at_depth)

    rendered_frames = []
    masks = []
    for idx,cam in tqdm(enumerate(cams)):
        rgb = frames[idx].permute(1, 2, 0).reshape(-1, 3) # (H*W, 3)

        depth_frame = depth[idx]
        depth_frame = 1.0 / (depth_frame + 0.1)

        ii, jj = torch.from_numpy(np.indices((H, W))).to(device)
        X_cam = (jj - c_x) * depth_frame / f_x
        Y_cam = (ii - c_y) * depth_frame / f_y
        Z_cam = depth_frame

        # Stack X_cam, Y_cam, Z_cam to get the point cloud (shape: (H * W, 3))
        point_cloud = torch.stack((X_cam, Y_cam, Z_cam), axis=-1).reshape(-1, 3)
        
        
        scale_factor = 1
        rendered_image, mask = project_points_to_image_pytorch(point_cloud, rgb, torch.from_numpy(cam).to(device), 
                                                            K/scale_factor, 
                                                            torch.tensor([H//scale_factor,W//scale_factor], device=device),
                                                            morph=True)
        rendered_frames.append(rendered_image)
        masks.append(mask)

        cv2.imwrite(f'logs/imgs/rendered_image_{idx}.jpg', cv2.cvtColor(rendered_image*255, cv2.COLOR_BGR2RGB))
        cv2.imwrite(f'logs/imgs/mask_{idx}.jpg', mask*255)

    export_to_video(rendered_frames, 'logs/rendered_video.mp4', fps=6)
    export_to_video(masks, 'logs/mask_video.mp4', fps=6)