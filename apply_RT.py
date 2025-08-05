import os
import numpy as np
import cv2
import yaml
from matplotlib import pyplot as plt
from tqdm import trange


with open("params.yaml", 'r') as stream:
    try:
        params = yaml.safe_load(stream)
        parameters = params.get("params", {})
        rescale_depth = params.get("rescale_depth", {})
        color_intrinsics_file = parameters.get("color_intrinsics_file", "camera_matrix.csv")
        gt_depth_folder = parameters.get("depth_folder", "")
        depth2cam_extrinsics_file = parameters.get("depth2color_extrinsics_file", "depth2cam_ext.npz")
        cam2depth_folder = parameters.get("cam2depth_folder", "")
        predicted_camspace_folder = parameters.get("predicted_camspace_folder", "")
        estimated_depth_npy_folder = parameters.get("estimated_depth_npy_folder", "")
        models = params.get("models", {})
        depthpro_estimated_depth_npy_folder = parameters.get("depthpro_estimated_depth_npy_folder", "")
        flashdepth_estimated_depth_npy_folder = parameters.get("flashdepth_estimated_depth_npy_folder", "")
        extracted_frame_idx = parameters.get("pick_up_idx", [])
        fixed_params_file = parameters.get("fixed_params_file", "data/scaling_factors.npz")
        # pred_depth_rgb_view_folder = parameters.get("pred_depth_rgb_view_folder", "")
        visualize_folder = parameters.get("visualized_estimated_depth_frames_folder", "")
        depth_estimator = models.get("depth_estimator", "")
        depth_internal = parameters.get("depth_internal", 1440)
        if depth_estimator == "depthpro":
            estimated_depth_npy_folder = depthpro_estimated_depth_npy_folder
        elif depth_estimator == "flashdepth":
            estimated_depth_npy_folder = flashdepth_estimated_depth_npy_folder
        else:
            raise ValueError(f"Depth estimator {depth_estimator} not supported.")
    except yaml.YAMLError as exc:
        print(exc)


def depth_to_np_arr(depth):
    cv2.applyColorMap(cv2.convertScaleAbs(depth, alpha=0.03), cv2.COLORMAP_JET)
    depth = cv2.cvtColor(depth, cv2.COLOR_BGR2RGB)
    depth = depth.astype(np.float32)
    return depth


def map_estimated_depth_to_gt(estimated_depth:np.ndarray, scaling_factors:dict, depth_estimator:str, resolution = (1920,10080)) -> np.ndarray:
    '''
    This function is used to map the estimated depth to the ground truth depth in the terms of resolution
    Args:
        estimated_depth: estimated depth, shape: (H, W) not (1080, 1920)
        scaling_factors: scaling factors, shape: {'m': float, 'c': float}
    Returns:
        rescale_depth: rescale depth, shape: (1080, 1920)
    '''
    if depth_estimator == "flashdepth":
        depth_data = cv2.resize(estimated_depth, resolution, interpolation=cv2.INTER_CUBIC)
        rescale_depth = 1/depth_data * scaling_factors['m'] + scaling_factors['c']
    elif depth_estimator == "depthpro":
        rescale_depth = estimated_depth * scaling_factors['m'] + scaling_factors['c']

    rescale_depth = rescale_depth.astype(np.int16)
    return rescale_depth

def create_point_cloud_from_rgbd_pair(rescale_depth:np.ndarray, color_intrinsics:np.ndarray, depth2rgb_extrinsics:np.ndarray, )-> np.ndarray:
    # create point cloud from estimated depth by using the inverse of the intrinsic matrix of "color" camera.
    cam_space = np.zeros((rescale_depth.shape[0], rescale_depth.shape[1], 3), dtype=np.float32) # (1080, 1920, 3)
    cam_space[:, :, 0] = (np.arange(rescale_depth.shape[1]))
    cam_space[:, :, 1] = (np.arange(rescale_depth.shape[0]-1, -1, -1))[:, np.newaxis]
    cam_space[:, :, 2] = np.ones(rescale_depth.shape, dtype=np.float32)
    cam_space = cam_space.reshape((-1, 3))
    cam_space = np.dot(np.linalg.inv(color_intrinsics), cam_space.T).T
    cam_space *= (rescale_depth.flatten())[:,np.newaxis] * 0.001  # Convert mm to meters
    cam_space = cam_space.reshape((-1, 3))

    # change view from color camera to depth camera by using the depth2rgb extrinsics.
    r = np.linalg.inv(depth2rgb_extrinsics['rotation'])
    t = -np.dot(r, depth2rgb_extrinsics['translation'])
    cam_space = np.dot(r, cam_space.T).T + t.reshape((1, 3))
    return cam_space

def read_color_intrinsics(file_path):
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Color intrinsics file {file_path} does not exist.")
    return np.loadtxt(file_path, delimiter=',').reshape((3, 3))



def visualize_depth(depth_data, visualize_folder, idx,factor = 0.03):
    os.makedirs(visualize_folder, exist_ok=True)
    depth_data = cv2.applyColorMap(cv2.convertScaleAbs(depth_data, alpha=factor), cv2.COLORMAP_JET).astype(np.float32)
    cv2.imwrite(os.path.join(visualize_folder, f'frame_{idx}.png'), depth_data)

if __name__ == "__main__":
    from run_and_visualized_diff import read_estimated_depth, read_gt_depth

    os.makedirs(visualize_folder, exist_ok=True)


    min_error = 100000  # Initialize error to a large value
    m_ = 0
    c_ = 0


    for i,idx in enumerate(extracted_frame_idx):


        
        
        depth_data = read_estimated_depth(idx, estimated_depth_npy_folder)
        # Do interpolation to (1920, 1080) using cubic interpolation 
        # inherit problem from DAv2
        depth_data = cv2.resize(depth_data, (1920, 1080), interpolation=cv2.INTER_CUBIC)


        mapping_file = os.path.join(cam2depth_folder, f"depth_frame_{idx:04d}_color_to_depth_mapping.csv")
        # read mapping file
        if not os.path.exists(mapping_file):
            print(f"Mapping file {mapping_file} does not exist.")
            continue
        mapping = np.loadtxt(mapping_file, delimiter=',', skiprows=1)
        mapped_depth = np.zeros((424, 512), dtype=np.float32)
        for i in range(mapping.shape[0]):
            if not np.isfinite(mapping[i, 2]) or not np.isfinite(mapping[i, 3]):
                continue
            if mapped_depth[int(mapping[i, 3]), int(mapping[i, 2])] == 0 \
            or mapped_depth[int(mapping[i, 3]), int(mapping[i, 2])] > depth_data[int(mapping[i, 1]), int(mapping[i, 0])]:  
                mapped_depth[int(mapping[i, 3]), int(mapping[i, 2])] = depth_data[int(mapping[i, 1]), int(mapping[i, 0])]

       
        
        gt_depth_data = read_gt_depth(gt_depth_folder, idx, depth_internal)


        valid_mask  = (mapped_depth > 0 ) & (gt_depth_data > 0) 
        x = mapped_depth[valid_mask]
        if depth_estimator == "flashdepth":
            x = 1/ x
        y = gt_depth_data[valid_mask]
        # minimum square error 
        A = np.vstack([x, np.ones(len(x))]).T
        m, c = np.linalg.lstsq(A, y, rcond=None)[0]
        print(f"Fitted line: y = {m} * x + {c}")
        error = np.abs(m * x + c - y)
        error = np.mean(error)
        # plot scatter plot
        plt.scatter(x, y, label=f'Frame {idx}', alpha=0.5, s=1)
        plt.plot(x, m * x + c, label=f'Fitted line {idx}')
        plt.xlabel('Estimated Depth ')
        plt.ylabel('Ground Truth Depth (mm)')
        plt.title('Depth Calibration')
        plt.legend()
        plt.savefig(f'calibration_frame_{idx}.png')
        plt.close()
        if error < min_error:
            min_error = error
            m_ = m
            c_ = c

    # save the scaling factors
    np.savez(fixed_params_file, m=m_, c=c_)
    print(f"Minimum error: {min_error}, m: {m_}, c: {c_}")