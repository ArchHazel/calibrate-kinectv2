import numpy as np
import cv2
import os
import yaml


from tqdm import tqdm

with open("params.yaml", 'r') as stream:
    try:
        params = yaml.safe_load(stream)
        parameters = params.get("params", {})
        models = params.get("models", {})
        all_depth_frames_folder = parameters.get("all_depth_frames_folder", "data/depth_frames/")
        all_pred_depth_frames_folder = parameters.get("all_pred_depth_frames_folder", "data/predicted_depth_frames/")
        all_diff_depth_frames_folder = parameters.get("all_diff_depth_frames_folder", "data/diff_depth_frames/")
        
        gt_pred_diff_video = parameters.get("gt_pred_diff_video", "")
        frames_folder = parameters.get("frames_folder", "")
    except yaml.YAMLError as exc:
        print(exc)


if __name__ == "__main__":

    gt_frames = os.listdir(all_depth_frames_folder)
    # sort frames to ensure they are in order
    gt_frames.sort(key=lambda x: int(x.split('_')[1].split('.')[0]))
    pred_frames = os.listdir(all_pred_depth_frames_folder)
    pred_frames.sort(key=lambda x: int(x.split('_')[1].split('.')[0]))
    diff_frames = os.listdir(all_diff_depth_frames_folder)
    diff_frames.sort(key=lambda x: int(x.split('_')[1].split('.')[0]))
    color_frames = os.listdir(frames_folder)
    color_frames.sort(key=lambda x: int(x.split('_')[1].split('.')[0]))

    # create video writer
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    video_writer = cv2.VideoWriter(gt_pred_diff_video, fourcc, 20.0, (512*3+754, 424))


    for gt_frame, pred_frame, diff_frame, color_frame in tqdm(zip(gt_frames, pred_frames, diff_frames,color_frames), total=len(gt_frames)):
        gt_frame_path = os.path.join(all_depth_frames_folder, gt_frame)
        pred_frame_path = os.path.join(all_pred_depth_frames_folder, pred_frame)
        diff_frame_path = os.path.join(all_diff_depth_frames_folder, diff_frame)
        color_frame_path = os.path.join(frames_folder, color_frame)

        gt_depth = cv2.imread(gt_frame_path, cv2.IMREAD_UNCHANGED)
        pred_depth = cv2.imread(pred_frame_path, cv2.IMREAD_UNCHANGED)
        diff_depth = cv2.imread(diff_frame_path, cv2.IMREAD_UNCHANGED)
        diff_depth = diff_depth[:,:,:3]  # Ensure it's 3-channel for visualization
        diff_depth = cv2.resize(diff_depth, (512, 424))

        color_frame = cv2.imread(color_frame_path)
        color_frame = cv2.resize(color_frame, (754, 424))




        # side by side
        combined = np.hstack((gt_depth, pred_depth, diff_depth,color_frame))
        video_writer.write(combined)
    video_writer.release()
    print("Video created successfully: gt_pred_diff_video.avi")

