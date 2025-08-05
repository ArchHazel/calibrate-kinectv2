import numpy as np
import cv2
import os
from calibrate_kinect import pick_up_idx
import yaml




with open("params.yaml", 'r') as stream:
    try:
        params = yaml.safe_load(stream)
        parameters = params.get("params", {})
        depth_folder = parameters.get("depth_folder", "")
        output_folder = parameters.get("depth_frames_folder", "")
        visualize_folder = parameters.get("visualized_estimated_depth_frames_folder", "")
        depth_internal = parameters.get("depth_internal", 1440)
    except yaml.YAMLError as exc:
        print(exc)

if not os.path.exists(output_folder):
    os.makedirs(output_folder)
if not os.path.exists(visualize_folder):
    os.makedirs(visualize_folder)

# list all npy files in the folder
depth_files = [f for f in os.listdir(depth_folder) if f.endswith('.npy')]
depth_files.sort()  # Sort files to maintain order

if __name__ == "__main__":
    files_id = []
    idxs = []
    for i in pick_up_idx:
        idxs.append(i % depth_internal)
        file_id = i // depth_internal
        files_id.append(file_id)
    files_id = np.array(files_id)
    idxs = np.array(idxs)
        



    for depth_file in depth_files:
        depth_idx = int(depth_file.split('_')[1].split('.')[0])  # Extract index from filename
        depth_idx = depth_idx -1 
        depth_path = os.path.join(depth_folder, depth_file)
        print(f"Processing {depth_path}")
        
        # Load the depth data
        depth_data = np.load(depth_path)
        print(f"Depth data shape: {depth_data.shape}")

        selected_idxs = idxs[files_id == depth_idx]

        print(f"Selected indices for {depth_file}: {selected_idxs}")
        for selected_idx in selected_idxs:
            if selected_idx < depth_data.shape[0]:
                depth_frame = depth_data[selected_idx]

                print("max depth value:", np.max(depth_frame))
                print("min depth value:", np.min(depth_frame))
                # Save the depth frame as an image
                output_filename = os.path.join(output_folder, f'depth_frame_{selected_idx:04d}.npy')
                depth_frame.astype(np.uint16).tofile(f'{output_filename[:-4]}.bin')

                np.save(output_filename, depth_frame)
                # Visualize the depth frame
                depth_frame_visualized = cv2.applyColorMap(cv2.convertScaleAbs(depth_frame, alpha=0.03), cv2.COLORMAP_JET)
                visualize_filename = os.path.join(visualize_folder, f'depth_frame_{selected_idx:04d}.png')
                cv2.imwrite(visualize_filename, depth_frame_visualized)
                print(f"Saved {output_filename}")
            else:
                print(f"Index {selected_idx} is out of bounds for depth data with shape {depth_data.shape}")