import numpy as np
import cv2
import os
import yaml
import matplotlib.pyplot as plt
from apply_RT import map_estimated_depth_to_gt, read_color_intrinsics,visualize_depth, create_point_cloud_from_rgbd_pair
from matplotlib.colors import TwoSlopeNorm
from tqdm import tqdm
with open("params.yaml", 'r') as stream:
    try:
        params = yaml.safe_load(stream)
        parameters = params.get("params", {})
        models = params.get("models", {})
        all_depth_frames_folder = parameters.get("all_depth_frames_folder", "data/depth_frames/")
        all_pred_depth_frames_folder = parameters.get("all_pred_depth_frames_folder", "data/predicted_depth_frames/")
        all_diff_depth_frames_folder = parameters.get("all_diff_depth_frames_folder", "data/diff_depth_frames/")
        gt_depth_folder = parameters.get("depth_folder", "")
        color_intrinsics_file = parameters.get("color_intrinsics_file", "camera_matrix.csv")
        # estimated_depth_npy_folder = parameters.get("estimated_depth_npy_folder", "")
        fixed_params_file = parameters.get("fixed_params_file", "data/scaling_factors.npz")
        depth2cam_extrinsics_file = parameters.get("depth2color_extrinsics_file", "depth2cam_ext.npz")
        depth_intrinsics_file = parameters.get("depth_intrinsics_file", "data/depth_intrinsics.csv")
        depth_internal = parameters.get("depth_internal", 1440)
        flashdepth_estimated_depth_npy_folder = parameters.get("flashdepth_estimated_depth_npy_folder", "")
        depthpro_estimated_depth_npy_folder = parameters.get("depthpro_estimated_depth_npy_folder", "")
        depth_estimator = models.get("depth_estimator", "")

    except yaml.YAMLError as exc:
        print(exc)


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

def depth_project_3d_to_2d(cam_space, depth_intrinsics, depth_internal, dimension):
    
    estimated_depth = np.zeros((dimension["height"], dimension["width"]))

    # divide by depth into unit coordinates
    cam_space[:, 0] /= cam_space[:, 2]
    cam_space[:, 1] /= cam_space[:, 2]

    cam_space[:, 0] = (cam_space[:, 0] * depth_intrinsics[0, 0]) + depth_intrinsics[0, 2]
    cam_space[:, 1] = (cam_space[:, 1] * depth_intrinsics[1, 1]) + depth_intrinsics[1, 2]

    # assign depth values to the estimated depth array
    for i in range(cam_space.shape[0]):
        x, y = int(cam_space[i, 0]), int(cam_space[i, 1])
        if 0 <= x < estimated_depth.shape[1] and 0 <= y < estimated_depth.shape[0]:
            if estimated_depth[dimension["height"]-1-y, x] == 0:
                estimated_depth[dimension["height"]-1-y, x] = cam_space[i, 2] * 1000
            elif cam_space[i, 2] * 1000 < estimated_depth[dimension["height"]-1-y, x]:
                estimated_depth[dimension["height"]-1-y, x] = cam_space[i, 2] * 1000
    return estimated_depth



        


if __name__ == "__main__":
    os.makedirs(all_depth_frames_folder, exist_ok=True)
    os.makedirs(all_pred_depth_frames_folder, exist_ok=True)
    os.makedirs(all_diff_depth_frames_folder, exist_ok=True)    


    # Load necessary parameters
    scaling_factors = np.load(fixed_params_file)
    depth2cam_extrinsics = np.load(depth2cam_extrinsics_file)
    color_intrinsics = read_color_intrinsics(color_intrinsics_file)
    depth_intrinsics = np.loadtxt(depth_intrinsics_file, delimiter=',').reshape((3, 3))


    # set up the estimated depth folder
    if depth_estimator == "depthpro":
        estimated_depth_npy_folder = depthpro_estimated_depth_npy_folder
    elif depth_estimator == "flashdepth":  
        estimated_depth_npy_folder = flashdepth_estimated_depth_npy_folder

    # process each frame index
    for idx in tqdm(range(len(os.listdir(estimated_depth_npy_folder)))):
        gt_depth_data = read_gt_depth(gt_depth_folder, idx, depth_internal).astype(np.float32)
        visualize_depth(gt_depth_data, all_depth_frames_folder, idx)
        estimated_depth = read_estimated_depth(idx, estimated_depth_npy_folder)

        

        rescale_depth = map_estimated_depth_to_gt(estimated_depth, scaling_factors, depth_estimator)




        cam_space = create_point_cloud_from_rgbd_pair(rescale_depth, color_intrinsics, depth2cam_extrinsics)

        estimated_depth = depth_project_3d_to_2d(
            cam_space, 
            depth_intrinsics, 
            depth_internal, 
            {"height": gt_depth_data.shape[0], "width": gt_depth_data.shape[1]})


        visualize_depth(estimated_depth, all_pred_depth_frames_folder, idx)

        # Calculate the difference
        valid_mask = (gt_depth_data > 0) & (estimated_depth > 0)
        diff_depth = gt_depth_data - estimated_depth
        diff_depth[~valid_mask] = np.nan  # Set invalid areas to NaN for visualization

        
        

        vmin, vmax = -500, 500
        norm = TwoSlopeNorm(vmin=vmin, vcenter=0, vmax=vmax)


        target_height_px = 424
        dpi = 100  
        height_inch = target_height_px / dpi


        h, w = diff_depth.shape
        aspect_ratio = w / h
        width_inch = height_inch * aspect_ratio
        fig = plt.figure(figsize=(width_inch, height_inch), dpi=dpi)
        ax = fig.add_axes([0, 0, 1, 1])

        cmap = plt.colormaps.get_cmap('seismic').with_extremes(bad='gray')

        im = ax.imshow(diff_depth, cmap=cmap, norm=norm)

        # fig.colorbar(im, ax=ax)
        ax.set_title("")            # 清除标题（如果有的话）
        ax.axis('off')              # 隐藏坐标轴（坐标刻度和边框）

        plt.savefig(os.path.join(all_diff_depth_frames_folder, f'frame_{idx}.png'), pad_inches=0)
        plt.close(fig)
