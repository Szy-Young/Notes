import os
import cv2
import numpy as np


def pose_to_mat(rot, transl, rot_type='rot_vec', eps=1e-6):
    """
    Convert camera pose to a transformation matrix
    """
    if rot_type == 'rot_vec':
        theta = np.linalg.norm(rot)
        k = rot / (theta + eps)
        k_mat = np.array(
            [[0, -k[2], k[1]],
             [k[2], 0, -k[0]],
             [-k[1], k[0], 0]], dtype=float
        )
        # Rodrigues Formula
        rot_mat = np.identity(3) + np.sin(theta) * k_mat + (1-np.cos(theta)) * np.matmul(k_mat, k_mat)
    else:
        raise ValueError('Rotation representation not supported')
    mat = np.concatenate((rot_mat, transl.reshape(3, 1)), axis=1)
    return mat


def project_points(mat, points):
    return np.einsum('ni, ji -> nj', points, mat)


def coord_to_hom(points):
    """
    Coordinates to homogeneous coordinates.
    """
    n_points = points.shape[0]
    return np.concatenate((points, np.ones((n_points, 1))), axis=1)


def hom_to_coord(points, eps=1e-6):
    """
    Homogeneous coordinates to coordinates.
    """
    return points[:, :-1] / (points[:, -1:] + eps)


def cam_proj(K_mat, pose_mat, points_xyz, distortions=None):
    """
    A complete camera perspective projection model.
    """
    cam_mat = np.matmul(K_mat, pose_mat)
    points_hom = coord_to_hom(points_xyz)
    points_hom = project_points(cam_mat, points_hom)
    points_uv = hom_to_coord(points_hom)
    if distortions is not None:
        points_uv = distort_points(points_uv, distortions, center=K_mat[:2, 2])
    return points_uv


def distort_points(points_uv, distortions, center):
    """
    Radial distortion.
    """
    center = center.reshape(1, 2)
    points_uv = points_uv - center
    r_square = np.linalg.norm(points_uv, axis=1, keepdims=True)
    r_square = r_square * r_square
    points_uv = (1 + distortions[0] * r_square + distortions[1] * r_square * r_square) * points_uv + center
    return points_uv


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


def load_cam_pose(pose_file):
    with open(pose_file, 'r') as f:
        poses = f.readlines()

    pose = poses[0].split(' ')
    # Given (rotation, translation) data format
    if len(pose) == 6:
        rot = []
        transl = []
        for pose in poses:
            pose = pose.split(' ')
            rot.append(np.array(pose[:3], dtype=float))
            transl.append(np.array(pose[3:], dtype=float))
        return np.stack(rot, axis=0), np.stack(transl, axis=0)
    # Given complete extrinsic matrix
    elif len(pose) == 12:
        pose_mat = []
        for pose in poses:
            pose = pose.split(' ')
            pose_mat.append(np.array(pose, dtype=float).reshape((3, 4)))
        return np.stack(pose_mat, axis=0)
    else:
        raise ValueError('Unrecognized data format!')


def load_K(K_file):
    with open(K_file, 'r') as f:
        params = f.readlines()

    K_mat = np.zeros((3, 3), dtype=float)
    for i, param in enumerate(params):
        param = param.split()
        K_mat[i] = np.array(param, dtype=float)
    return K_mat


def load_distortions(D_file):
    with open(D_file, 'r') as f:
        params = f.readlines()

    params = params[0].split(' ')
    return np.array(params, dtype=float)


def load_xyz(xyz_file):
    with open(xyz_file, 'r') as f:
        points = f.readlines()

    points_xyz = []
    for point in points:
        point = point.split()
        points_xyz.append(point)
    return np.array(points_xyz, dtype=float)


def load_uv(uv_file):
    with open(uv_file, 'r') as f:
        points = f.readlines()

    points_uv_views = []
    for point in points:
        point = point.split()
        points_uv = np.array(point, dtype=float).reshape(-1, 2)
        points_uv_views.append(points_uv)
    return np.stack(points_uv_views, axis=0)