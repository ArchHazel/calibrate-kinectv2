from config_reading import *    
from depth_utils import *
from tqdm import tqdm



if __name__ == "__main__":
    for i in tqdm(range(frame_count)):
        depth_data = read_gt_depth(depth_r_F_each_npy_containing_1440_frames, i, depth_interval)
        depth_data_from_npy_to_bin(depth_data, depth_r_F_bin, i)
        visualize_depth(depth_data, depth_rv_F, i)