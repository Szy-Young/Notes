"""
Utils. for multi-view geometry.
"""

import numpy as np


def coord_to_hom(points):
    """
    Coordinates to homogeneous coordinates.
    """
    n_points = points.shape[0]
    return np.concatenate((points, np.ones((n_points, 1))), axis=1)


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