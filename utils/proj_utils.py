import numpy as np

def depth_project_3d_to_2d(cam_space, depth_intrinsics, dimension):
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


def create_point_cloud_from_rgbd_pair(rescale_depth:np.ndarray, color_intrinsics:np.ndarray, depth2rgb_extrinsics:np.ndarray, )-> np.ndarray:
    # create point cloud from estimated depth by using the inverse of the intrinsic matrix of "color" camera.
    cam_space = np.zeros((rescale_depth.shape[0], rescale_depth.shape[1], 3), dtype=np.float32) # (1080, 1920, 3)
    cam_space[:, :, 0] = (np.arange(rescale_depth.shape[1]))
    cam_space[:, :, 1] = (np.arange(rescale_depth.shape[0]-1, -1, -1))[:, np.newaxis]
    cam_space[:, :, 2] = np.ones(rescale_depth.shape, dtype=np.float32)
    cam_space = cam_space.reshape((-1, 3))
    cam_space = np.dot(np.linalg.inv(color_intrinsics), cam_space.T).T
    cam_space *= (rescale_depth.flatten())[:,np.newaxis] * 0.001  # Convert mm to meters
    cam_space = cam_space.reshape((-1, 3))

    # change view from color camera to depth camera by using the depth2rgb extrinsics.
    r = np.linalg.inv(depth2rgb_extrinsics['rotation'])
    t = -np.dot(r, depth2rgb_extrinsics['translation'])
    cam_space = np.dot(r, cam_space.T).T + t.reshape((1, 3))
    return cam_space