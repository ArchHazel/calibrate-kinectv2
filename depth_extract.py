import numpy as np
import cv2
import os
from calibrate_kinect import pick_up_idx
depth_folder = '/home/hazel/Downloads/2025_07_11_13_22_47_IntrinsicsMeasure/depth/'  
output_folder = 'selected_depth_frames/'  # 输出深度帧的文件夹
visualize_folder = 'visualize_depth_frames/'  # 可视化深度帧的文件夹
if not os.path.exists(output_folder):
    os.makedirs(output_folder)
if not os.path.exists(visualize_folder):
    os.makedirs(visualize_folder)

# list all npy files in the folder
depth_files = [f for f in os.listdir(depth_folder) if f.endswith('.npy')]
depth_files.sort()  # Sort files to maintain order
interval = 1440
if __name__ == "__main__":
    files_id = []
    idxs = []
    for i in pick_up_idx:
        idxs.append(i % interval)
        file_id = i // interval
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
                print(depth_frame[0:10, 0:10])  # Print a small part of the depth frame for debugging
                print("shape of depth frame:", depth_frame.shape)
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