import numpy as np
import cv2
import os
from tqdm import tqdm
import hydra
from omegaconf import DictConfig
from calibrateKinectv2.utils.depth_utils import *
from calibrateKinectv2.utils.proj_utils import *
from calibrateKinectv2.utils.io_utils import *
from depth2loc.utils.basic_draw import stitch_three_images_side_by_side_and_save



def apply_colormap_to_depth(depth_data, colormap = 'seismic', vmin=-0.5, vmax=0.5, sf_to_m = 1):
    import matplotlib.pyplot as plt
    from matplotlib.colors import TwoSlopeNorm

    # colormap and normalization
    if sf_to_m != 1:
        depth_data = depth_data * sf_to_m
    norm = TwoSlopeNorm(vmin=vmin, vcenter=0, vmax=vmax)
    cmap = plt.colormaps.get_cmap(colormap).with_extremes(bad='gray')

    depth_colored = cmap(norm(depth_data))
    depth_colored = (depth_colored[:, :, :3] * 255).astype(np.uint8)  # Convert to uint8 RGB format
    return depth_colored

def draw_depth_difference_using_plt_colormap(diff_depth, output_folder, idx, sf_to_m = 1, debug=False):
    if debug:
        print("diff_depth min and max:", diff_depth.min(), diff_depth.max())
    depth_colored = apply_colormap_to_depth(diff_depth, colormap='seismic', vmin=-0.5, vmax=0.5, sf_to_m=sf_to_m)
    depth_colored = cv2.cvtColor(depth_colored, cv2.COLOR_RGB2BGR)  # Convert RGB to BGR for OpenCV
    cv2.imwrite(os.path.join(output_folder, f'frame_{idx:05d}.png'), depth_colored)

def draw_diff_one_from_kinect_one_from_estimation(gt_depth, estimated_depth, output_folder, idx):
    valid_mask = (gt_depth > 0) & (estimated_depth > 0)
    diff_depth = gt_depth - estimated_depth
    diff_depth[~valid_mask] = np.nan  # Set invalid areas to NaN for visualization
    draw_depth_difference_using_plt_colormap(diff_depth, output_folder, idx)

def draw_diff_both_from_estimation(estimated_depth1, estimated_depth2, output_folder, idx,debug=False):
    if debug:
       print("estimated depth1 min and max:",estimated_depth1.min(), estimated_depth1.max())
       print("estimated depth2 min and max:",estimated_depth2.min(), estimated_depth2.max())
    valid_mask = (estimated_depth1 > 0) & (estimated_depth2 > 0)
    diff_depth = estimated_depth1 - estimated_depth2
    diff_depth[~valid_mask] = np.nan  # Set invalid areas to NaN for visualization
    draw_depth_difference_using_plt_colormap(diff_depth, output_folder, idx)

def get_depth_diff_npy_folder_wise(with_folder, without_folder, output_folder):
    print(f"Comparing {with_folder} and {without_folder}, saving to {output_folder}")
    os.makedirs(output_folder, exist_ok=True)
    for idx in tqdm(from_depth_folder_to_frame_indices(with_folder)):
        with_img = read_estimated_depth(idx, with_folder).astype(np.float32)
        without_img = read_estimated_depth(idx, without_folder).astype(np.float32)
        draw_diff_both_from_estimation(with_img, without_img, output_folder, idx)

def get_depth_diff_png_folder_wise(gt_folder, estimated_folder, output_folder):
    print(f"Comparing {gt_folder} and {estimated_folder}, saving to {output_folder}")
    os.makedirs(output_folder, exist_ok=True)
    for image_file in tqdm(os.listdir(gt_folder)):
        gt_img = cv2.imread(os.path.join(gt_folder, image_file), cv2.IMREAD_UNCHANGED).astype(np.float32)
        estimated_img = cv2.imread(os.path.join(estimated_folder, image_file), cv2.IMREAD_UNCHANGED).astype(np.float32)
        draw_diff_one_from_kinect_one_from_estimation(gt_img, estimated_img, output_folder, int(image_file.split('_')[1].split('.')[0]))



@hydra.main(config_path="/home/hhan2/Scripts/hof/", config_name="config",version_base=None)
def main(cfg: DictConfig):
    scaling_factors = np.load(cfg.dataset.paths.fixed_sf_to_metric_depth_file)
    depth2cam_extrinsics = np.load(cfg.dataset.paths.depth2color_extrinsics_file)
    color_intrinsics = read_color_intrinsics(cfg.dataset.paths.color_intrinsics_file)
    depth_intrinsics = np.loadtxt(cfg.dataset.paths.depth_intrinsics_file, delimiter=',').reshape((3, 3))
    estimated_depth_npy_folder = cfg.dataset.paths.depthpro_estimated_depth_npy_folder

    gt_depth_data = read_gt_depth_given_frame_idx( 0,cfg.dataset.paths.depth_r_F_each_npy_containing_1440_frames, cfg.dataset.depth_interval)
    
    # process each frame index
    for idx in tqdm(from_depth_folder_to_frame_indices(estimated_depth_npy_folder)):

        estimated_depth = read_estimated_depth(idx, estimated_depth_npy_folder)
        rescale_depth = applied_sf_to_estimated_depth(estimated_depth, scaling_factors)
        cam_space = create_point_cloud_from_rgbd_pair(rescale_depth, color_intrinsics, depth2cam_extrinsics)

        estimated_depth = depth_project_3d_to_2d(
            cam_space, 
            depth_intrinsics, 
            {"height": gt_depth_data.shape[0], "width": gt_depth_data.shape[1]})


        visualize_depth(estimated_depth, cfg.dataset.paths.derived_depth_png_folder, idx)
    

@hydra.main(config_path="/home/hhan2/Scripts/hof/", config_name="config",version_base=None)
def ad_hoc_test_depth_diff(cfg: DictConfig):
    gt_folder = cfg.model.paths.depth_gt_with_human_F
    derived_folder = cfg.dataset.paths.derived_depth_png_folder
    diff_folder = cfg.dataset.paths.depthpro_diff_depth_png_folder
    get_depth_diff_png_folder_wise(gt_folder, derived_folder, diff_folder)
    stitch_three_images_side_by_side_and_save(gt_folder, derived_folder, diff_folder)


if __name__ == "__main__":
    main()







