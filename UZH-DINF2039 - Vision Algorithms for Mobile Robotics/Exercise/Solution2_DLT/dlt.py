import os, sys
currentdir = os.path.dirname(os.path.realpath(__file__))
parentdir = os.path.dirname(currentdir)
sys.path.append(parentdir)

import cv2
import numpy as np
from global_utils.camera import load_K, coord_to_hom, cam_proj


def dlt_equation(point_xyz, point_uv):
    """
    Derive the linear system of equations given a 3D-2D correspondance.
    """
    Q = np.kron(np.array([[1, 0, -1], [0, 1, -1]]), point_xyz)
    point_uv = point_uv[:, :2].T
    Q[:, 8:] = Q[:, 8:] * point_uv
    return Q


def dlt(points_xyz, points_uv, K_mat=None):
    """
    Implement the basic Direct Linear Transform (DLT) algorithm by solving the linear least
    squares with SVD.
    """
    points_xyz = coord_to_hom(points_xyz)
    points_uv = coord_to_hom(points_uv)
    if K_mat is not None:
        points_uv = np.matmul(np.linalg.inv(K_mat), points_uv.T).T

    # Derive the linear system of equations
    n_points = points_xyz.shape[0]
    points_xyz = np.split(points_xyz, n_points)
    points_uv = np.split(points_uv, n_points)
    Q = list(map(dlt_equation, points_xyz, points_uv))
    Q = np.concatenate(Q, axis=0)

    # Solve the linear least square by SVD
    u, sigma, vh = np.linalg.svd(Q)
    pose_mat = vh[-1]

    # Extract rotation and translation
    if pose_mat[-1] < 0:
        pose_mat = -pose_mat
    pose_mat = pose_mat.reshape((3, 4))
    rot, transl = pose_mat[:, :3], pose_mat[:, 3:]
    u, _, vh = np.linalg.svd(rot)
    rot_ = np.matmul(u, vh)
    scale = np.linalg.norm(rot_) / np.linalg.norm(rot)
    transl_ = transl * scale
    return np.concatenate((rot_, transl_), axis=1)


def load_xyz(xyz_file):
    with open(xyz_file, 'r') as f:
        points = f.readlines()

    points_xyz = []
    for point in points:
        point = point.split(', ')
        point[2] = point[2][:-1]
        points_xyz.append(point)
    return np.array(points_xyz, dtype=np.float32)


def load_uv(uv_file):
    with open(uv_file, 'r') as f:
        points = f.readlines()

    points_uv_views = []
    for point in points:
        point = point.split()
        points_uv = np.array(point, dtype=np.float32).reshape(-1, 2)
        points_uv_views.append(points_uv)
    return np.stack(points_uv_views, axis=0)


if __name__ == '__main__':
    K_file = 'data/K.txt'
    xyz_file = 'data/p_W_corners.txt'
    uv_file = 'data/detected_corners.txt'
    img_dir = 'data/images_undistorted'
    save_dir = 'results/images_marked'
    os.makedirs(save_dir, exist_ok=True)

    points_xyz = load_xyz(xyz_file) * 0.01
    points_uv_views = load_uv(uv_file)
    n_views = points_uv_views.shape[0]
    K_mat= load_K(K_file)
    # Estimate camera pose for each view
    for v in range(n_views):
        points_uv = points_uv_views[v]
        pose_mat = dlt(points_xyz, points_uv, K_mat)
        points_uv_reproj = cam_proj(K_mat, pose_mat, points_xyz)
        # Visualize the detected 2D corner points and reprojected 2D corner points
        img_file = 'img_%04d.jpg'%(v+1)
        img = cv2.imread(os.path.join(img_dir, img_file))
        for n in range(points_uv.shape[0]):
            cv2.circle(img, (int(points_uv[n, 0]), int(points_uv[n, 1])), 3, (0, 255, 0), -1)
        for n in range(points_uv_reproj.shape[0]):
            cv2.circle(img, (int(points_uv_reproj[n, 0]), int(points_uv_reproj[n, 1])), 3, (0, 0, 255), -1)
        cv2.imwrite(os.path.join(save_dir, img_file), img)