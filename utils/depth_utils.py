import numpy as np
import os
import cv2

def depth_data_from_npy_to_bin(depth_data, depth_bin_folder, idx):
    depth_data.astype(np.uint16).tofile(f'{depth_bin_folder}/{idx:04d}.bin')

def depth_to_np_arr(depth):
    cv2.applyColorMap(cv2.convertScaleAbs(depth, alpha=0.03), cv2.COLORMAP_JET)
    depth = cv2.cvtColor(depth, cv2.COLOR_BGR2RGB)
    depth = depth.astype(np.float32)
    return depth

def read_gt_depth_given_frame_idx(idx, gt_depth_folder, depth_interval):
    gt_idx = idx // depth_interval
    gt_i = idx % depth_interval
    gt_depth_file = os.path.join(gt_depth_folder, f"depth_{gt_idx+1}.npy")
    gt_depth_data = np.load(gt_depth_file)
    return gt_depth_data[gt_i]


def read_estimated_depth(idx:int, estimated_depth_npy_folder:str) -> np.ndarray:
    estimated_depth_file_name_no_zero_padding = os.path.join(estimated_depth_npy_folder, f"frame_{idx}.npy")
    estimated_depth_file_name_with_zero_padding = os.path.join(estimated_depth_npy_folder, f"frame_{idx:05d}.npy")

    if not os.path.exists(estimated_depth_file_name_no_zero_padding):
        if not os.path.exists(estimated_depth_file_name_with_zero_padding):
            raise FileNotFoundError(f"Estimated depth file {estimated_depth_file_name_no_zero_padding} does not exist.")
        else:
            estimated_depth_file = estimated_depth_file_name_with_zero_padding
    else:
        estimated_depth_file = estimated_depth_file_name_no_zero_padding

    return np.load(estimated_depth_file)

def from_depth_file_name_to_frame_idx(depth_file_name: str) -> int:
    base_name = os.path.basename(depth_file_name)
    frame_idx_str = base_name.split('.')[0].split('_')[-1]
    return int(frame_idx_str)

def from_depth_folder_to_frame_indices(depth_folder: str) -> list[int]:
    depth_files = [f for f in sorted(os.listdir(depth_folder)) if f.endswith('.npy')]
    frame_indices = [from_depth_file_name_to_frame_idx(f) for f in depth_files]
    return frame_indices

def visualize_depth(depth_data, visualize_folder, idx, factor = 0.03, to_meter_sf=1):
    if to_meter_sf != 1:
        depth_data = depth_data * to_meter_sf
    os.makedirs(visualize_folder, exist_ok=True)
    depth_data = depth_data.astype(np.float32) # convertScaleAbs not support float16 
    exception_mask = depth_data == 0
    depth_data = cv2.applyColorMap(cv2.convertScaleAbs(depth_data, alpha=factor), cv2.COLORMAP_JET)
    depth_data[exception_mask] = 255

    cv2.imwrite(os.path.join(visualize_folder, f'frame_{idx:05d}.png'), depth_data)
    # print(f"Saved depth visualization to {os.path.join(visualize_folder, f'frame_{idx:05d}.png')}")