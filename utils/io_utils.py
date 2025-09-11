import numpy as np
import os

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


def read_color_intrinsics(file_path):
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Color intrinsics file {file_path} does not exist.")
    return np.loadtxt(file_path, delimiter=',').reshape((3, 3))

