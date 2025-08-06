import numpy as np
import os
import cv2

def depth_data_from_npy_to_bin(depth_data, depth_bin_folder, idx):
    depth_data.astype(np.uint16).tofile(f'{depth_bin_folder}/{idx:04d}.bin')

def read_gt_depth(gt_depth_folder, idx, depth_internal):
    gt_idx = idx // depth_internal
    gt_i = idx % depth_internal
    gt_depth_file = os.path.join(gt_depth_folder, f"depth_{gt_idx+1}.npy")
    gt_depth_data = np.load(gt_depth_file)
    return gt_depth_data[gt_i]


def read_estimated_depth(idx:int, estimated_depth_npy_folder:str) -> np.ndarray:
    estimated_depth_file_name_no_zero_padding = os.path.join(estimated_depth_npy_folder, f"frame_{idx}.npy")
    estimated_depth_file_name_with_zero_padding = os.path.join(estimated_depth_npy_folder, f"frame_{idx:04d}.npy")

    if not os.path.exists(estimated_depth_file_name_no_zero_padding):
        if not os.path.exists(estimated_depth_file_name_with_zero_padding):
            raise FileNotFoundError(f"Estimated depth file {estimated_depth_file_name_no_zero_padding} does not exist.")
        else:
            estimated_depth_file = estimated_depth_file_name_with_zero_padding
    else:
        estimated_depth_file = estimated_depth_file_name_no_zero_padding

    return np.load(estimated_depth_file)

def visualize_depth(depth_data, visualize_folder, idx, factor = 0.03):
    os.makedirs(visualize_folder, exist_ok=True)
    depth_data = cv2.applyColorMap(cv2.convertScaleAbs(depth_data, alpha=factor), cv2.COLORMAP_JET).astype(np.float32)
    cv2.imwrite(os.path.join(visualize_folder, f'frame_{idx}.png'), depth_data)