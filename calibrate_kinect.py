import cv2
import numpy as np
import glob
import json
import os
import yaml
import tqdm

with open("params.yaml", 'r') as stream:
    try:
        params = yaml.safe_load(stream)
        parameters = params.get("params", {})
        pipeline = params.get("pipeline", {})
        avi_path = parameters.get("avi_path", "")
        pick_up_idx = parameters.get("pick_up_idx", [])
        image_folder = parameters.get("frames_folder", "")
        output_folder = parameters.get("corners_vis_folder", "")
        corner_npy_path = parameters.get("corners_npy_folder", "")
        c2d_folder = parameters.get("cam2depth_folder", "")
        camspace_folder = parameters.get("camspace_folder", "")
        color_intrinsics_file = parameters.get("color_intrinsics_file", "camera_matrix.csv")
        visualized_depth_folder = parameters.get("visualized_depth_folder", "vis_data/depth/")
        visualized_depth_w_corners_folder = parameters.get("visualized_depth_w_corners_folder", "vis_data/depth_with_corners/")
        depth2cam_extrinsics_file = parameters.get("depth2color_extrinsics_file", "depth2cam_ext.npz")
        frame_extracting = pipeline.get("frame_extracting", False)
        corners_finding = pipeline.get("corners_finding", False)
        cam2depth_calibrating = pipeline.get("cam2depth_calibration", False)

    except yaml.YAMLError as exc:
        print(exc)



clicked_points = []

def click_event(event, x, y, flags, param):
    global clicked_points
    if event == cv2.EVENT_LBUTTONDOWN:
        clicked_points.append((x, y))
        print(f"Clicked point: ({x}, {y})")

def get_manual_corners(image, num_points=10):
    global clicked_points
    clicked_points = []

    cv2.imshow("Click 10 corners", image)
    cv2.setMouseCallback("Click 10 corners", click_event)

    print("Please click 10 corner points on the image...")
    while len(clicked_points) < num_points:
        cv2.waitKey(1)

    cv2.destroyWindow("Click 10 corners")
    # 转成你的格式 (N, 1, 2)
    return np.array(clicked_points, dtype=np.float32).reshape(-1, 1, 2)

def extract_frames(avi_path:str, output_folder:str, frame_extracting:bool):
    if not frame_extracting:
        print(f"{"Frame extracting is disabled.".ljust(40)} Skipping frame extraction.")
        return

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    cap = cv2.VideoCapture(avi_path)
    frame_count = 0
    # use tqdm to show progress
    pbar = tqdm.tqdm(total=int(cap.get(cv2.CAP_PROP_FRAME_COUNT)), desc="Extracting frames")
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        pbar.update(1)
        frame_filename = os.path.join(output_folder, f'frame_{frame_count:04d}.png')
        cv2.imwrite(frame_filename, frame)
        frame_count += 1
    pbar.close()
    cap.release()

def color_intrisic_calibration(image_folder:str, output_folder:str, corner_npy_path:str, pick_up_idx:list,  corners_finding:bool):
    if not corners_finding:
        print(f"{'Corners finding is disabled.'.ljust(40)} Skipping corner detection.")
        return
    if not os.path.exists(corner_npy_path):
        os.makedirs(corner_npy_path)
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    chessboard_size = (5, 8)
    square_size = 25  # unit: mm

    objp = np.zeros((np.prod(chessboard_size), 3), np.float32)
    objp[:, :2] = np.indices(chessboard_size).T.reshape(-1, 2)
    objp *= square_size

    objpoints = []
    imgpoints = []

    
    image_paths = [f'{image_folder}frame_{i:04d}.png' for i in pick_up_idx]

    os.makedirs(output_folder, exist_ok=True)
    frame_corners = {}

    for fname in tqdm.tqdm(image_paths, desc="finding corners"):
        img = cv2.imread(fname)
        fidx = int(os.path.basename(fname).split('_')[1].split('.')[0])


        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        ret, corners = cv2.findChessboardCorners(gray, chessboard_size)
        # visualize corners
        
        if ret:
            frame_corners[fidx] = corners
            cv2.drawChessboardCorners(img, chessboard_size, corners, ret)
            cv2.imwrite(os.path.join(output_folder, os.path.basename(fname)), img)
            cv2.waitKey(0)
            objpoints.append(objp)
            imgpoints.append(corners)
            np.save(os.path.join(corner_npy_path, f'corners_{fidx}.npy'), corners)
        else:
            print(f"Chessboard corners not found in {fname}")



    ret, mtx, dist, _, _ = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None) 
    np.savetxt(color_intrinsics_file, mtx, delimiter=',') # for now, I ignore the distortion coefficients




def cam2depth_calibration(
    c2d_folder: str, 
    camspace_folder: str, 
    corner_npy_path:str, 
    visualized_depth_folder:str, 
    visualized_depth_w_corners_folder:str, 
    color_intrinsics_path: str,
    d2c_file:str,
    cam2depth_calibrating: bool):
    '''
    Calibrate the camera to depth mapping using the provided CSV files.
    c2d_folder: folder containing the CSV files for color to depth mapping
    camspace_folder: folder containing the camera space data
    cam2depth_calibrating: boolean flag to enable/disable the calibration
    '''

    if not cam2depth_calibrating:
        print(f"{'Cam2Depth calibration is disabled.'.ljust(40)} Skipping calibration.")
        return
    if not os.path.exists(visualized_depth_w_corners_folder):
        os.makedirs(visualized_depth_w_corners_folder)

    c2d_csv_files = [f for f in os.listdir(c2d_folder) if f.endswith('.csv')]
    c2d_csv_files.sort(key=lambda x: int(x.split('_')[2]))  # Sort by frame index





    # read color camera intrinsics, csv
    color_intrinsics_matrix = np.loadtxt(color_intrinsics_path, delimiter=',')
    intrinsics_matrix = color_intrinsics_matrix.reshape((3, 3)) 
    print(f"Color camera intrinsics matrix:\n{intrinsics_matrix}")

    rot = []
    trans = []
    error = 1000

    for csv_file in tqdm.tqdm(c2d_csv_files, desc="color to depth mapping"):
        pixel_coords = []
        cam_coords = []

        
        csv_path = os.path.join(c2d_folder, csv_file)
        frame_idx = int(csv_file.split('_')[2])


        # Load the CSV file, skip the first row if it contains headers
        data_c2d = np.loadtxt(csv_path, delimiter=',', skiprows=1)
        color_X = data_c2d[:, 0]
        color_Y = data_c2d[:, 1]
        depth_X = data_c2d[:, 2]
        depth_Y = data_c2d[:, 3]
        # build a dictionary to map color coordinates to depth coordinates, rounding to nearest integer
        color_to_depth_map = {}

        valid_mash = np.isfinite(color_X) & np.isfinite(color_Y) & np.isfinite(depth_X) & np.isfinite(depth_Y)
        color_X = color_X[valid_mash]
        color_Y = color_Y[valid_mash]
        depth_X = depth_X[valid_mash]
        depth_Y = depth_Y[valid_mash]   

        color_coords = np.column_stack((color_X, color_Y)).astype(int)
        depth_coords = np.column_stack((depth_X + 0.5, depth_Y+ 0.5)).astype(int)
        color_to_depth_map = { tuple(color_coord): tuple(depth_coord) for color_coord, depth_coord in zip(color_coords, depth_coords) }


        camspace_path = os.path.join(camspace_folder, os.path.basename(csv_path).replace("color_to_depth_mapping", "camera_space"))
        data_camSpace = np.loadtxt(camspace_path, delimiter=',', skiprows=1)
        depth_X_index = data_camSpace[:, 0]
        depth_Y_index = data_camSpace[:, 1]
        cam_space_X = data_camSpace[:, 2]
        cam_space_Y = data_camSpace[:, 3]   
        cam_space_Z = data_camSpace[:, 4]

        valid_mash = np.isfinite(depth_X_index) & np.isfinite(depth_Y_index) & np.isfinite(cam_space_X) & np.isfinite(cam_space_Y) & np.isfinite(cam_space_Z)
        depth_X_index = depth_X_index[valid_mash]
        depth_Y_index = depth_Y_index[valid_mash]
        cam_space_X = cam_space_X[valid_mash]
        cam_space_Y = cam_space_Y[valid_mash]
        cam_space_Z = cam_space_Z[valid_mash]
        depth_coords = np.column_stack((depth_X_index , depth_Y_index)).astype(int)
        cam_space_coords = np.column_stack((cam_space_X, cam_space_Y, cam_space_Z)).astype(float)
        depth2cam_space_map = { tuple(depth_coord): cam_space_coord for depth_coord, cam_space_coord in zip(depth_coords, cam_space_coords) }


        #visualize cam space as ply
        cam_space_ply_path = os.path.join(camspace_folder, f'cam_space_{frame_idx}.ply')
        points = np.column_stack((cam_space_X, cam_space_Y, cam_space_Z))
        with open(cam_space_ply_path, 'w') as f:
            f.write("ply\n")
            f.write("format ascii 1.0\n")
            f.write(f"element vertex {len(points)}\n")
            f.write("property float x\n")
            f.write("property float y\n")
            f.write("property float z\n")
            f.write("end_header\n")
            for point in points:
                f.write(f"{point[0]} {point[1]} {point[2]}\n")
        

        random_x = [619,850,1075,330,1390,1539] # pick up from color image
        random_y = [872,884,813,765,775,764] # pick up from color image
        random_points = np.column_stack((random_x, random_y)).astype(int)



        depth_img =  cv2.imread(os.path.join(visualized_depth_folder, f'depth_frame_{frame_idx:04d}.png'))

        # rounded_corners = get_manual_corners(depth_img, num_points=10)

        # find depth coordinates for corners
        corner_depth_coords = []
        for corner in random_points:
            # corner is a 1x2 array, we need to convert it to a tuple of integers
            key = (corner[0], corner[1])
            if key in color_to_depth_map:
                depth_coord = color_to_depth_map[key]
                corner_depth_coords.append(depth_coord)

        
        
        for color_coord, depth_coord in zip(random_points, corner_depth_coords):
            # looking for cam space coordinates for this depth coordinate
            depth_coord_tuple = (depth_coord[0], depth_coord[1])
            if depth_coord_tuple in depth2cam_space_map :
                cam_space_coord = depth2cam_space_map[depth_coord_tuple]

                pixel_coords.append(color_coord)
                cam_coords.append(cam_space_coord)
                

        # draw this corners on depth image
        for depth_coord in corner_depth_coords:
            cv2.circle(depth_img, (depth_coord[0], depth_coord[1]), 1, (0, 0, 255), -1)
        cv2.imwrite(os.path.join(visualized_depth_w_corners_folder, f'depth_with_corners_{frame_idx:04d}.png'), depth_img)

        # do pnp calibration
        success, rvec, tvec = cv2.solvePnP(np.array(cam_coords).astype(float), np.array(pixel_coords).astype(float), intrinsics_matrix, None, flags=cv2.SOLVEPNP_ITERATIVE)
        R = cv2.Rodrigues(rvec)[0]  # Convert rotation vector to rotation matrix
        cam_in_object_coords =  -R.T @ tvec
        
        if success:
            print(f"Frame {frame_idx}: PnP calibration successful.")


        else:
            print(f"Frame {frame_idx}: PnP calibration failed.")


        projected, _ = cv2.projectPoints(np.array(cam_coords).astype(float), rvec, tvec, intrinsics_matrix, None)
        error_curr = np.linalg.norm(pixel_coords - projected.squeeze(), axis=1).mean()
        if error_curr < error:
            error = error_curr
            rot = R
            trans = tvec

    print("Reprojection error:", error)
    print("Rotation matrix:\n", rot)
    print("Translation vector:\n", trans)
    np.savez(d2c_file, rotation=rot, translation=trans)


if __name__ == "__main__":

    extract_frames(avi_path, image_folder, frame_extracting)
    color_intrisic_calibration(image_folder, output_folder, corner_npy_path, pick_up_idx, corners_finding)
    cam2depth_calibration(c2d_folder, 
    camspace_folder, 
    corner_npy_path, 
    visualized_depth_folder, 
    visualized_depth_w_corners_folder,
    color_intrinsics_file, 
    depth2cam_extrinsics_file,
    cam2depth_calibrating)
