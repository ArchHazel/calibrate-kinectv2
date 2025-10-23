import numpy as np
import os
import cv2

def applied_sf_to_estimated_depth(estimated_depth:np.ndarray, scaling_factors:dict, resolution = (1920,10080)) -> np.ndarray:
    '''
    This function is used to map the estimated depth to the ground truth depth in the terms of resolution
    Args:
        estimated_depth: estimated depth, shape: (H, W) not (1080, 1920)
        scaling_factors: scaling factors, shape: {'m': float, 'c': float}
    Returns:
        rescale_depth: rescale depth, shape: (1080, 1920)
    '''

    rescale_depth = estimated_depth * scaling_factors['m'] + scaling_factors['c']



    rescale_depth = rescale_depth.astype(np.int16)
    return rescale_depth

def read_rgb_img(idx, rgb_folder):
    file_name = os.path.join(rgb_folder, f"frame_{idx:05d}.png")
    if not os.path.exists(file_name):
        raise FileNotFoundError(f"RGB file {file_name} does not exist.")
    return cv2.imread(file_name, cv2.IMREAD_UNCHANGED)

def read_color_intrinsics(file_path):
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Color intrinsics file {file_path} does not exist.")
    return np.loadtxt(file_path, delimiter=',').reshape((3, 3))

