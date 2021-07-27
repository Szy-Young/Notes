import os, sys
currentdir = os.path.dirname(os.path.realpath(__file__))
parentdir = os.path.dirname(currentdir)
sys.path.append(parentdir)

import numpy as np
from global_utils.mvg import triangulate
from global_utils.camera import coord_to_hom, hom_to_coord, project_points, cam_proj


def normalize(points_uv):
    """
    Compute the linear transform to normalize a set of points.
    """
    n_points = points_uv.shape[0]
    center = np.mean(points_uv, axis=0)
    var = np.sum(np.linalg.norm(points_uv-center, axis=1)**2) / n_points
    scale = np.sqrt(2 / var)
    T_mat = np.zeros((3, 3), dtype=np.float64)
    T_mat[0, 0] = T_mat[1, 1] = scale
    T_mat[:2, 2] = - scale * center
    T_mat[2, 2] = 1
    points_uv_normed = project_points(T_mat, coord_to_hom(points_uv))
    points_uv_normed = hom_to_coord(points_uv_normed)
    return T_mat, points_uv_normed


def eight_point(group_points_uv):
    """
    Implement the eight-point algorithm to estimate the fundamental/essential matrix.
    Parameters:
        group_points_uv: (2, N, 2) array.
    """
    n_points = group_points_uv.shape[1]
    Q = np.zeros((n_points, 9), dtype=np.float64)
    # Construct the least squares problem
    for n in range(n_points):
        group_point_uv = coord_to_hom(group_points_uv[:, n])
        Q[n] = np.kron(group_point_uv[0], group_point_uv[1])

    # Solve the linear least square by SVD
    u, sigma, vh = np.linalg.svd(Q)
    F_mat = vh[-1].reshape((3, 3))

    # Add the rank-2 constraint
    u, sigma, vh = np.linalg.svd(F_mat)
    sigma[-1] = 0
    F_mat = np.matmul(np.matmul(u, np.diag(sigma)), vh)
    return F_mat


def normed_eight_point(group_points_uv):
    """
    Implement the normalized eight-point algorithm to estimate the fundamental/essential matrix.
    This algorithm is more robust to noises in point correspondances.
    Parameters:
        group_points_uv: (2, N, 2) array.
    """
    n_points = group_points_uv.shape[1]
    # Compute the linear transform for normalization
    group_T_mat = np.zeros((2, 3, 3), dtype=np.float64)
    group_points_uv_normed = np.zeros_like(group_points_uv)
    for c in range(2):
        group_T_mat[c], group_points_uv_normed[c] = normalize(group_points_uv[c])

    Q = np.zeros((n_points, 9), dtype=np.float64)
    # Construct the least squares problem
    for n in range(n_points):
        group_point_uv = coord_to_hom(group_points_uv_normed[:, n])
        Q[n] = np.kron(group_point_uv[0], group_point_uv[1])

    # Solve the linear least square by SVD
    u, sigma, vh = np.linalg.svd(Q)
    F_mat = vh[-1].reshape((3, 3))

    # Add the rank-2 constraint
    u, sigma, vh = np.linalg.svd(F_mat)
    sigma[-1] = 0
    F_mat = np.matmul(np.matmul(u, np.diag(sigma)), vh)

    # Restore from the linear transform
    F_mat = np.matmul(np.matmul(group_T_mat[1].T, F_mat), group_T_mat[0])
    return F_mat


def algebraic_error(group_points_uv, F_mat):
    n_points = group_points_uv.shape[1]
    Q = np.zeros((n_points, 9), dtype=np.float64)
    for n in range(n_points):
        group_point_uv = coord_to_hom(group_points_uv[:, n])
        Q[n] = np.kron(group_point_uv[0], group_point_uv[1])

    F_mat = F_mat.reshape((9, 1))
    error = np.linalg.norm(np.matmul(Q, F_mat).squeeze(1)**2) / np.sqrt(n_points)
    return error


if __name__ == '__main__':
    # Construct test samples
    n_points = 10
    points_xyz = np.random.randn(n_points, 3) * 5 + 10
    K_mat = np.array([[500, 0, 320],
                      [0, 500, 240],
                      [0, 0, 1]], dtype=np.float64)
    pose_mat_1 = np.concatenate((np.eye(3), np.zeros((3, 1))), axis=1)
    pose_mat_2 = pose_mat_1.copy()
    pose_mat_2[0, 3] = -0.5

    points_uv_1 = cam_proj(K_mat, pose_mat_1, points_xyz)
    points_uv_2 = cam_proj(K_mat, pose_mat_2, points_xyz)
    group_K_mat = np.stack((K_mat, K_mat), axis=0)
    group_pose_mat = np.stack((pose_mat_1, pose_mat_2), axis=0)
    group_points_uv = np.stack((points_uv_1, points_uv_2), axis=0)

    points_uv_1_noised = points_uv_1 + 0.1 * np.random.randn(n_points, 2)
    points_uv_2_noised = points_uv_2 + 0.1 * np.random.randn(n_points, 2)
    group_points_uv_noised = np.stack((points_uv_1_noised, points_uv_2_noised), axis=0)

    # Test the triangulation
    points_xyz_est = triangulate(group_K_mat, group_pose_mat, group_points_uv)
    print ('Triangulation error: ', np.sum((points_xyz_est-points_xyz)**2))

    # Test the eight-point algorithm
    F_mat_est = eight_point(group_points_uv)
    error = algebraic_error(group_points_uv, F_mat_est)
    print ('F matrix estimate -- algebraic error: ', error)

    # Test the eight-point algorithm under noises
    F_mat_est = eight_point(group_points_uv_noised)
    error = algebraic_error(group_points_uv_noised, F_mat_est)
    print('F matrix estimate under noise -- algebraic error: ', error)

    # Test the normalized eight-point algorithm under noises
    F_mat_est = normed_eight_point(group_points_uv_noised)
    error = algebraic_error(group_points_uv_noised, F_mat_est)
    print('F matrix estimate (from normalized 8-point) under noise -- algebraic error: ', error)