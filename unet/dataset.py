import os
import pandas as pd
from torchvision.io import decode_image
from torch.utils.data import Dataset
import numpy as np
from run_and_visualized_diff import read_gt_depth, read_estimated_depth,depth_project_3d_to_2d
from apply_RT import map_estimated_depth_to_gt, read_color_intrinsics
from torchvision.transforms import ToTensor
import yaml
import torch

with open("params.yaml", 'r') as stream:
    try:
        params = yaml.safe_load(stream)
        parameters = params.get("params", {})
        gt_depth_folder = parameters.get("depth_folder", "")
        estimated_depth_npy_folder = parameters.get("estimated_depth_npy_folder", "")
        depth_internal = parameters.get("depth_internal", 1440)
        color_intrinsics_file = parameters.get("color_intrinsics_file", "camera_matrix.csv")
        depth_intrinsics_file = parameters.get("depth_intrinsics_file", "data/depth_intrinsics.csv")
        depth2cam_extrinsics_file = parameters.get("depth2color_extrinsics_file", "depth2cam_ext.npz")
        fixed_params_file = parameters.get("fixed_params_file", "data/scaling_factors.npz")



    except yaml.YAMLError as exc:
        print(exc)

class UNetDataset(Dataset):
    def __init__(self,):
        self.estimated_depth_npy_list = sorted(os.listdir(estimated_depth_npy_folder), key=lambda x: int(x.split('_')[1].split('.')[0]))
        self.estimated_depth_folder = estimated_depth_npy_folder
        self.gt_depth_folder = gt_depth_folder
        self.color_intrinsics = read_color_intrinsics(color_intrinsics_file)
        self.depth_intrinsics = read_color_intrinsics(depth_intrinsics_file)
        self.depth2cam_extrinsics = np.load(depth2cam_extrinsics_file, allow_pickle=True)
        self.scaling_factors = np.load(fixed_params_file, allow_pickle=True)
        print(f"scaling factors: {self.scaling_factors['m']}, {self.scaling_factors['c']}")

    def __len__(self):
        return len(self.estimated_depth_npy_list)

    def __getitem__(self, idx):
        estimated_depth = read_estimated_depth(idx, self.estimated_depth_folder)
        gt_depth = read_gt_depth(self.gt_depth_folder, idx, depth_internal)

        try:
            cam_space, _ = map_estimated_depth_to_gt(
                estimated_depth, 
                self.color_intrinsics, 
                self.depth2cam_extrinsics, 
                self.scaling_factors)
        except Exception as e:
            print(f"Error processing index {idx}: {e}")
            return None, None

        estimated_depth = depth_project_3d_to_2d(
            cam_space, 
            self.depth_intrinsics, 
            depth_internal, 
            {"height": gt_depth.shape[0], "width": gt_depth.shape[1]})

        # estimated_depth = ToTensor()(estimated_depth)
        # gt_depth = ToTensor()(gt_depth)
        estimated_depth = torch.tensor(estimated_depth, dtype=torch.float32).unsqueeze(0)  # Add channel dimension
        gt_depth = torch.tensor(gt_depth, dtype=torch.float32).unsqueeze(0)  # Add channel dimension
        valid_mask = (gt_depth > 0) & (estimated_depth > 0)
        return estimated_depth.float(), gt_depth.float(), valid_mask


if __name__ == "__main__":
    dataset = UNetDataset()
    for i in range(len(dataset)):
        estimated_depth, gt_depth = dataset[i]
        print(f"Estimated Depth Shape: {estimated_depth.shape}, GT Depth Shape: {gt_depth.shape}")