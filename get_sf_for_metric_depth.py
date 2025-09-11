import numpy as np
from matplotlib import pyplot as plt
from calibrateKinectv2.utils.depth_utils import read_estimated_depth, read_gt_depth_given_frame_idx
import hydra
from omegaconf import DictConfig

@hydra.main(config_path="/home/hhan2/Scripts/hof/", config_name="config",version_base=None)
def main(cfg: DictConfig):
    
    min_error = 100000  # Initialize error to a large value
    m_ = 0
    c_ = 0


    for i,idx in enumerate([6327]):

        depth_data = read_estimated_depth(idx, cfg.dataset.paths.depth_r_F_each_npy_containing_1440_frames)
        gt_depth_data = read_gt_depth_given_frame_idx(idx, cfg.dataset.paths.gt_depth_folder,  cfg.dataset.depth_interval)




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
    np.savez(fixed_sf_to_metric_depth_file, m=m_, c=c_)
    print(f"Minimum error: {min_error}, m: {m_}, c: {c_}")




if __name__ == "__main__":
    main()
