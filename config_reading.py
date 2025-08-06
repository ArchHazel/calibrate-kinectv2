import yaml
import os
import cv2

params_file = "params_HAR6.yaml"

with open(params_file, 'r') as stream:
    try:
        params = yaml.safe_load(stream)
        camera_number = params.get("camera_number", 0)
        parameters = params.get("params", {})
        rescale_depth = params.get("rescale_depth", {})
        depth_frames_folder_as_bin = parameters.get("depth_frames_folder_as_bin", "")
        color_intrinsics_file = parameters.get("color_intrinsics_file", "camera_matrix.csv")
        # depth related files and folders
        depth_r_F_each_npy_containing_1440_frames = parameters.get("depth_r_F_each_npy_containing_1440_frames", "")
        depth_r_F_bin = parameters.get("depth_r_F_bin", "")
        depth_rv_F = parameters.get("depth_rv_F", "")

        # rgb related files and folders
        rgb_r_f_avi = parameters.get("rgb_r_f_avi", "")
        rgb_r_F_png = parameters.get("rgb_r_F_png", "")

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
        depth_interval = parameters.get("depth_interval", 1440)
        if depth_estimator == "depthpro":
            estimated_depth_npy_folder = depthpro_estimated_depth_npy_folder
        elif depth_estimator == "flashdepth":
            estimated_depth_npy_folder = flashdepth_estimated_depth_npy_folder
        else:
            raise ValueError(f"Depth estimator {depth_estimator} not supported.")
    except yaml.YAMLError as exc:
        print(exc)

if not os.path.exists(depth_r_F_bin):
    os.makedirs(depth_r_F_bin)
if not os.path.exists(depth_rv_F):
    os.makedirs(depth_rv_F)

# read rgb avi file
cap = cv2.VideoCapture(rgb_r_f_avi)
# frame count
frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
# close the video file
cap.release()

print(f"camera #{camera_number}")