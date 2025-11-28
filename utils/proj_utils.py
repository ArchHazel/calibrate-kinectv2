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

def create_camspace_from_depth(depth_data:np.ndarray, depth_intrinsics:np.ndarray, sf_to_m:float = 0.001)-> np.ndarray:
    # create point cloud from estimated depth by using the inverse of the intrinsic matrix of "depth" camera.
    cam_space = np.zeros((depth_data.shape[0], depth_data.shape[1], 3), dtype=np.float32) # (1080, 1920, 3)
    cam_space[:, :, 0] = (np.arange(depth_data.shape[1]))
    cam_space[:, :, 1] = (np.arange(depth_data.shape[0]-1, -1, -1))[:, np.newaxis]
    cam_space[:, :, 2] = np.ones(depth_data.shape, dtype=np.float32)
    cam_space = cam_space.reshape((-1, 3))
    cam_space = np.dot(np.linalg.inv(depth_intrinsics), cam_space.T).T
    cam_space *= (depth_data.flatten())[:,np.newaxis] * sf_to_m  # Convert mm to meters
    cam_space = cam_space.reshape((-1, 3))

    return cam_space

def transform_cam_space_inversely(cam_space:np.ndarray, extrinsics:np.ndarray) -> np.ndarray:
    r = np.linalg.inv(extrinsics['rotation'])
    t = -np.dot(r, extrinsics['translation'])
    cam_space = np.dot(r, cam_space.T).T + t.reshape((1, 3))
    return cam_space

def trasnform_cam_space(cam_space:np.ndarray, extrinsics:np.ndarray) -> np.ndarray:
    r = extrinsics['rotation']
    t = extrinsics['translation']
    cam_space = np.dot(r, cam_space.T).T + t.reshape((1, 3))
    return cam_space


def create_point_cloud_from_rgbd_pair(rescale_depth:np.ndarray, color_intrinsics:np.ndarray, depth2rgb_extrinsics:np.ndarray, )-> np.ndarray:
    cam_space = create_camspace_from_depth(rescale_depth, color_intrinsics, sf_to_m=0.001)
    cam_space = transform_cam_space_inversely(cam_space, depth2rgb_extrinsics)
    return cam_space



def project_depth_img_to_color_coordinate(depth_data, depth_intrinsics, depth2rgb_extrinsics):
    # print(" max and min of depth data:", depth_data.max(), depth_data.min())
    cam_space = create_camspace_from_depth(depth_data, depth_intrinsics, sf_to_m=0.001)
    cam_space = trasnform_cam_space(cam_space, depth2rgb_extrinsics)
    return cam_space
