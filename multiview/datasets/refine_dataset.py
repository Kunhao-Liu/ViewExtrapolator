import os
import imageio
import glob
import numpy as np
import cv2


import torch
from torch.utils.data import Dataset


class Refine_Dataset(Dataset):
    def __init__(self, data_dir):
        # check if the directory exists
        if not os.path.exists(data_dir):
            raise ValueError(f"Directory {data_dir} does not exist")

        image_paths = glob.glob(os.path.join(data_dir, 'refined', 'images', "*.png"))
        self.all_image_paths = sorted(image_paths)

        # load the camera poses
        self.closest_K = np.load(os.path.join(data_dir, 'closest_K.npy'))
        self.all_interp_c2ws = np.load(os.path.join(data_dir, 'interp_c2ws.npy'))
        self.img_wh = np.load(os.path.join(data_dir, 'closest_img_wh.npy'))

        # subset
        select_idx = [0,4,9,14,19,24]
        self.image_paths = [self.all_image_paths[i] for i in select_idx]
        self.interp_c2ws = self.all_interp_c2ws[select_idx]
        print(f"Loading refine dataset from {data_dir} with {len(self.image_paths)} images.")

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        
        image = imageio.imread(self.image_paths[idx])[..., :3] # [H, W, 3]
        image = cv2.resize(image, (self.img_wh[0], self.img_wh[1]))
        
        data = {
            "K": torch.from_numpy(self.closest_K).float(),
            "camtoworld": torch.from_numpy(self.interp_c2ws[idx]).float(),
            "image": torch.from_numpy(image).float(),
            "image_id": idx,  # the index of the image in the dataset
        }

        return data