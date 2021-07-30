import os, sys
currentdir = os.path.dirname(os.path.realpath(__file__))
parentdir = os.path.dirname(currentdir)
sys.path.append(parentdir)

from itertools import product
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d

from eight_point import normed_eight_point
from global_utils.mvg import triangulate
from global_utils.camera import coord_to_hom, project_points


def decompose_E_mat(E_mat):
    """
    Extract rot and transl from E matrix.
    """
    u, sigma, vh = np.linalg.svd(E_mat)
    w = np.array([[0, -1, 0],
                  [1, 0, 0],
                  [0, 0, 1]], dtype=np.float64)
    rot1 = np.matmul(np.matmul(u, w), vh)
    rot2 = np.matmul(np.matmul(u, w.T), vh)
    if np.linalg.det(rot1) < 0:
        rot1 = - rot1
    if np.linalg.det(rot2) < 0:
        rot2 = - rot2
    transl1 = u[:, 2:]
    transl2 = - u[:, 2:]

    pose_mats = []
    for pair in list(product([rot1, rot2], [transl1, transl2])):
        pose_mat = np.concatenate(pair, axis=1)
        if np.linalg.det(pose_mat[:, :3]) < 0:
            pose_mat = - pose_mat
        pose_mats.append(pose_mat)
    return pose_mats


def disambiguate_pose(pose_mats, group_K_mat, group_points_uv):
    """
    Select the correct camera pose among 4 possible configurations, meanwhile perform triangulation.
    Parameters:
        pose_mats: [4 x (3, 4) array] list.
        group_K_mat: (2, 3, 3) array.
        group_points_uv: (2, N, 2) array.
    """
    group_pose_mat = np.zeros((2, 3, 4), dtype=np.float64)
    group_pose_mat[0, :, :3] = np.eye(3)

    points_xyz_all = []
    infront_all = []
    # Triangulate with each possible config
    for pose_mat in pose_mats:
        group_pose_mat[1] = pose_mat
        points_xyz = triangulate(group_K_mat, group_pose_mat, group_points_uv)
        points_xyz_all.append(points_xyz)
        # Record the number of points in front of each camera
        infront = 0
        points_xyz_c1 = project_points(group_pose_mat[0], coord_to_hom(points_xyz))
        infront += np.sum(np.where(points_xyz_c1[:, 2]>0, 1, 0).astype(np.float64))
        points_xyz_c2 = project_points(group_pose_mat[1], coord_to_hom(points_xyz))
        infront += np.sum(np.where(points_xyz_c2[:, 2]>0, 1, 0).astype(np.float64))
        infront_all.append(infront)
    # Select the config with most points in front of cameras
    best = np.argmax(np.array(infront_all, dtype=np.float64))
    return pose_mats[best], points_xyz_all[best]


if __name__ == '__main__':
    K_mat = np.array([[1379.74, 0, 760.35],
                      [0, 1382.08, 503.41],
                      [0, 0, 1]], dtype=np.float64)
    points_files = ['data/matches0001.txt', 'data/matches0002.txt']
    save_dir = 'results'
    os.makedirs(save_dir, exist_ok=True)

    # Load the 2D point correspondances
    group_points_uv = []
    for points_file in points_files:
        with open(points_file, 'r') as f:
            points_all = f.readlines()
        points_uv = []
        for points in points_all:
            points = points.split(' ')
            points_uv.append(points)
        points_uv = np.array(points_uv, dtype=np.float64).T
        group_points_uv.append(points_uv)
    group_points_uv = np.stack(group_points_uv, axis=0)

    # Compute the F matrix using eight-point algorithm and extract E matrix
    F_mat = normed_eight_point(group_points_uv)
    E_mat = np.matmul(np.matmul(K_mat.T, F_mat), K_mat)

    # Extract relative camera pose from E matrix and disambiguate
    pose_mats = decompose_E_mat(E_mat)
    group_K_mat = np.stack((K_mat, K_mat), axis=0)
    pose_mat, points_xyz = disambiguate_pose(pose_mats, group_K_mat, group_points_uv)
    print (points_xyz)

    # Visualize the SFM results
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    ax.view_init(90, -90)
    # Transform the world frame to fit the 3D visualization
    transform = np.array([[1, 0, 0],
                          [0, 0, 1],
                          [0, -1, 0]], dtype=np.float64)
    points_xyz = project_points(transform, points_xyz)
    points_x, points_y, points_z = np.split(points_xyz, 3, axis=1)
    ax.scatter(points_x, points_y, points_z)
    # Draw the cameras
    cam_origin_1 = np.zeros(3)
    cam_axis_1 = np.eye(3)
    rot, transl = pose_mat[:, :3], pose_mat[:, 3:]
    c2w = np.concatenate((rot.T, -np.matmul(rot.T, transl)), axis=1)
    cam_origin_2 = c2w[:, 3]
    cam_axis_2 = project_points(c2w[:, :3], cam_axis_1)
    cam_origin_1 = np.matmul(transform, cam_origin_1)
    cam_origin_2 = np.matmul(transform, cam_origin_2)
    cam_axis_1 = project_points(transform, cam_axis_1)
    cam_axis_2 = project_points(transform, cam_axis_2)
    for i, color in enumerate(['red', 'green', 'blue']):
        ax.quiver(cam_origin_1[0], cam_origin_1[1], cam_origin_1[2],
                  cam_axis_1[i, 0], cam_axis_1[i, 1], cam_axis_1[i, 2],
                  color=color, length=0.5, normalize=True)
        ax.quiver(cam_origin_2[0], cam_origin_2[1], cam_origin_2[2],
                  cam_axis_2[i, 0], cam_axis_2[i, 1], cam_axis_2[i, 2],
                  color=color, length=0.5, normalize=True)
    fig.savefig(os.path.join(save_dir, 'sfm_topview.png'))