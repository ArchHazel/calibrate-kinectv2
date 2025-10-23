import hydra
from omegaconf import DictConfig
import numpy as np
from pathlib import Path
import json

from calibrateKinectv2.utils.depth_utils import read_gt_depth_given_frame_idx
from calibrateKinectv2.utils.io_utils import read_color_intrinsics
from calibrateKinectv2.utils.proj_utils import *
from calibrateKinectv2.utils.io_utils import read_rgb_img
from calibrateKinectv2.utils.depth_utils import visualize_depth
from tqdm import tqdm
def from_keyword_search_session_folder(base_folder, session_name_keyword):
    import os
    base_folder = Path(base_folder)
    session_folders = [f for f in os.listdir(base_folder) if session_name_keyword in f]
    if len(session_folders) == 0:
        raise ValueError(f"No session folder found with keyword {session_name_keyword}")
    elif len(session_folders) > 1:
        print(f"Multiple session folders found with keyword {session_name_keyword}, using the first one: {session_folders[0]}")
    return session_folders[0]

def parse_idx_from_pose_keypoints_json_key(key):
    return int(Path(key).stem.split('_')[-1])

def save_json(data, save_path):
    with open(save_path, 'w') as f:
        json.dump(data, f, indent=4)

def identity_session_folder(base_folder, session_name_keyword):
    import os
    base_folder = Path(base_folder)
    session_folders = [f for f in os.listdir(base_folder) if session_name_keyword in f]
    if len(session_folders) == 0:
        raise ValueError(f"No session folder found with keyword {session_name_keyword}")
    elif len(session_folders) > 1:
        print(f"Multiple session folders found with keyword {session_name_keyword}, using the first one: {session_folders[0]}")
    return base_folder / session_folders[0]


@hydra.main(config_path="/home/hhan2/Scripts/hof/", config_name="config",version_base=None)

def main(cfg: DictConfig):
    
    depth2cam_extrinsics = np.load(cfg.dataset.paths.depth2color_extrinsics_file)
    color_intrinsics = read_color_intrinsics(cfg.dataset.paths.color_intrinsics_file)
    depth_intrinsics = np.loadtxt(cfg.dataset.paths.depth_intrinsics_file, delimiter=',').reshape((3, 3))
    
    for node in cfg.dataset.yolo.nodes_folder_list:

        
        session_path = identity_session_folder(Path(cfg.dataset.yolo.nodes_parent_path) / node, cfg.dataset.yolo.scene_name)
        res_json_path = session_path / cfg.dataset.yolo.result_path
        gt_depth_folder = session_path / 'depth'
        color_img = read_rgb_img(0, cfg.model.paths.rgb_F)

    

        with open(res_json_path, 'r') as f:
            data = json.load(f)

        threed_joint_data = {}
        centers = {}


        for frame_img_path, joint_data in tqdm(data.items()):
            frame_idx = parse_idx_from_pose_keypoints_json_key(frame_img_path)
            gt_depth = read_gt_depth_given_frame_idx(frame_idx, gt_depth_folder, cfg.dataset.depth_interval)
            # get depth cooridinates of all joints
            cam_space = project_depth_img_to_color_coordinate(gt_depth, depth_intrinsics, depth2cam_extrinsics)
            estimated_depth = depth_project_3d_to_2d(
                cam_space, 
                color_intrinsics, 
                {"height": color_img.shape[0], "width": color_img.shape[1]})
            # save estimated depth image
        
            visualize_depth(estimated_depth, gt_depth_folder / "derived_estimated_depth_png", frame_idx)
            
            use_center = True
            if use_center :
                center = list()
                for joint in joint_data:
                    x, y = joint
                    x, y = int(x), int(y)
                    if 0 <= x < estimated_depth.shape[1] and 0 <= y < estimated_depth.shape[0]:
                        center.append((x,y))
                
                
                center = np.mean(center, axis=0).astype(int)
                centers[frame_img_path] = center.tolist()
            
            
                x, y = center
                depth_neighbor = estimated_depth[y-4:y+5, x-4:x+5]
                # if np.count_nonzero(depth_neighbor) > 0:
                if depth_neighbor[depth_neighbor > 0].size == 0:
                    continue
                estimated_depth_value = np.median(depth_neighbor[depth_neighbor > 0])
                xyz_homo = np.array([[x, y, 1.0]]).T
                xyz_reproject = np.linalg.inv(color_intrinsics) @ xyz_homo
                xyz_reproject *= estimated_depth_value
                threed_joint_data[frame_img_path] = xyz_reproject.flatten().tolist()
            else:
                center = []
                for joint in joint_data:
                    # find_given_joint(joint, estimated_depth, padding)
                    x, y = joint
                    x, y = int(x), int(y)
                    padding = 4
                    if padding <= x < estimated_depth.shape[1]-padding and padding <= y < estimated_depth.shape[0]-padding:
                        depth_neighbor = estimated_depth[y-4:y+5, x-4:x+5]
                        if depth_neighbor[depth_neighbor > 0].size == 0:
                            continue
                        estimated_depth_value = np.median(depth_neighbor[depth_neighbor > 0])
                        xyz_homo = np.array([[x, y, 1.0]]).T
                        xyz_reproject = np.linalg.inv(color_intrinsics) @ xyz_homo
                        xyz_reproject *= estimated_depth_value
                        center.append(xyz_reproject.flatten().tolist())
                threed_joint_data[frame_img_path] = np.mean(center, axis=0).tolist()



        # Save the 3D joint data to a file
        save_json(threed_joint_data, gt_depth_folder / "3d_joint_data.json")
        save_json(centers, gt_depth_folder / "centers.json")




if __name__ == "__main__":
    main()
    


