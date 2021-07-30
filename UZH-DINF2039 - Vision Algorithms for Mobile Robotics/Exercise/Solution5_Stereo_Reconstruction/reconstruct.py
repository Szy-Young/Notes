import os, sys
currentdir = os.path.dirname(os.path.realpath(__file__))
parentdir = os.path.dirname(currentdir)
sys.path.append(parentdir)

import cv2
import numpy as np
from plyfile import PlyData, PlyElement
from global_utils.camera import load_K, load_cam_pose, coord_to_hom, project_points
from get_disparity import get_disparity


def triangulate(group_K_mat, group_pose_mat, group_points_uv):
    """
    Reconstruct 3D points given their projections on a group of cameras.
    Both intrinsic and extrinsic parameters of cameras are known.
    Parameters:
        group_K_mat: (M, 3, 3) array.
        group_pose_mat: (M, 3, 4) array.
        group_points_uv: (M, N, 2) array.
    """
    n_cams, n_points = group_points_uv.shape[:2]
    group_cam_mat = np.einsum('mij,mjk->mik', group_K_mat, group_pose_mat)

    # Construct and solve the least squares approximation problem
    points_xyz = np.zeros((n_points, 3))
    for n in range(n_points):
        group_point_uv = coord_to_hom(group_points_uv[:, n])
        # Vector to skew-symmetric matrix
        skew_uv = np.zeros((n_cams, 9))
        skew_uv[:, 1] = - group_point_uv[:, 2]
        skew_uv[:, 2] = group_point_uv[:, 1]
        skew_uv[:, 5] = - group_point_uv[:, 0]
        skew_uv[:, 3] = group_point_uv[:, 2]
        skew_uv[:, 6] = - group_point_uv[:, 1]
        skew_uv[:, 7] = group_point_uv[:, 0]
        skew_uv = skew_uv.reshape((n_cams, 3, 3))
        A_mat = np.einsum('mij,mjk->mik', skew_uv, group_cam_mat)
        A_mat = A_mat.reshape((n_cams*3, 4))
        u, sigma, vh = np.linalg.svd(A_mat)
        point_xyz = vh[-1]
        points_xyz[n] = point_xyz[:3] / (point_xyz[3] + 1e-6)
    return points_xyz


if __name__ == '__main__':
    patch_radius = 5
    disp_min = 5
    disp_max = 50
    rescale_factor = 0.5    # Rescale the original image for efficiency
    baseline = 0.54     # Given by KITTI Dataset
    xlim = (-10, 6)
    ylim = (-5, 5)
    zlim = (7, 20)

    img_left_dir = 'data/left'
    img_right_dir = 'data/right'
    K_file = 'data/K.txt'
    pose_file = 'data/poses.txt'
    save_dir = 'results'
    os.makedirs(save_dir, exist_ok=True)
    save_pcd_file = 'results/scene.ply'

    K_mat = load_K(K_file)
    K_mat[:2] = K_mat[:2] * rescale_factor
    group_K_mat = np.stack((K_mat, K_mat), axis=0)
    pose_mats = load_cam_pose(pose_file)

    points_xyz_all = []
    points_color_all = []
    img_files = sorted(os.listdir(img_left_dir))
    # Test on the sequence
    for t, img_file in enumerate(img_files):
        img_left = cv2.imread(os.path.join(img_left_dir, img_file))
        img_right = cv2.imread(os.path.join(img_right_dir, img_file))
        img_h, img_w = img_left.shape[:2]
        img_left = cv2.resize(img_left, (int(rescale_factor*img_w), int(rescale_factor*img_h)))
        img_right = cv2.resize(img_right, (int(rescale_factor*img_w), int(rescale_factor*img_h)))
        img_left_gray = cv2.cvtColor(img_left, cv2.COLOR_BGR2GRAY).astype(np.float64)
        img_right_gray = cv2.cvtColor(img_right, cv2.COLOR_BGR2GRAY).astype(np.float64)
        # Calculate or load the disparity
        # img_disp = get_disparity(img_left_gray, img_right_gray, patch_radius, disp_min, disp_max)
        img_disp = np.load(os.path.join(save_dir, 'disparity', img_file[:-4]+'.npy'))
        # Collect 2D point correspondances
        points_uv_left = np.where(img_disp > 0)
        points_color = cv2.cvtColor(img_left, cv2.COLOR_BGR2RGB)[points_uv_left]
        points_uv_disp = img_disp[points_uv_left]
        points_uv_left = np.stack((points_uv_left[1], points_uv_left[0]), axis=1).astype(np.float64)
        points_uv_right = points_uv_left.copy()
        points_uv_right[:, 0] -= points_uv_disp
        group_points_uv = np.stack((points_uv_left, points_uv_right), axis=0)
        # Triangulate the 3D points
        pose_mat_left = np.concatenate((np.eye(3), np.zeros((3, 1))), axis=1)
        pose_mat_right = pose_mat_left.copy()
        pose_mat_right[:, 3] -= np.array([baseline, 0, 0], dtype=np.float64)
        group_pose_mat = np.stack((pose_mat_left, pose_mat_right), axis=0)
        points_xyz = triangulate(group_K_mat, group_pose_mat, group_points_uv)
        # Filter out invalid points
        for axis, lim in enumerate([xlim, ylim, zlim]):
            valid = np.where(points_xyz[:, axis] > lim[0])
            points_xyz = points_xyz[valid]
            points_color = points_color[valid]
            valid = np.where(points_xyz[:, axis] < lim[1])
            points_xyz = points_xyz[valid]
            points_color = points_color[valid]
        # Transform from camera frame to world frame
        c2w_mat = np.concatenate((pose_mats[t], np.array([[0, 0, 0, 1]])), axis=0)
        points_xyz = project_points(c2w_mat, coord_to_hom(points_xyz))
        points_xyz_all.append(points_xyz)
        points_color_all.append(points_color)

    # Save as point cloud
    points_xyz = np.concatenate(points_xyz_all, axis=0)
    points_color = np.concatenate(points_color_all, axis=0)
    n_points = points_xyz.shape[0]
    vertices = np.empty(n_points, dtype=[('x', 'f4'), ('y', 'f4'), ('z', 'f4'), ('red', 'u1'), ('green', 'u1'), ('blue', 'u1')])
    vertices['x'] = points_xyz[:, 0].astype('f4')
    vertices['y'] = points_xyz[:, 1].astype('f4')
    vertices['z'] = points_xyz[:, 2].astype('f4')
    vertices['red'] = points_color[:, 0].astype('u1')
    vertices['green'] = points_color[:, 1].astype('u1')
    vertices['blue'] = points_color[:, 2].astype('u1')
    ply = PlyData([PlyElement.describe(vertices, 'vertex')], text=False)
    ply.write(save_pcd_file)