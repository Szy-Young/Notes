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
             [-k[1], k[0], 0]], dtype=np.float32
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


def load_cam_pose(pose_file):
    with open(pose_file, 'r') as f:
        poses = f.readlines()

    rot = []
    transl = []
    for pose in poses:
        pose = pose.split(' ')
        rot.append(np.array(pose[:3], dtype=np.float32))
        transl.append(np.array(pose[3:], dtype=np.float32))
    return np.stack(rot, axis=0), np.stack(transl, axis=0)


def load_K(K_file):
    with open(K_file, 'r') as f:
        params = f.readlines()

    K_mat = np.zeros((3, 3), dtype=np.float32)
    for i, param in enumerate(params):
        param = param.split()
        K_mat[i] = np.array(param, dtype=np.float32)
    return K_mat


def load_distortions(D_file):
    with open(D_file, 'r') as f:
        params = f.readlines()

    params = params[0].split(' ')
    return np.array(params, dtype=np.float32)


if __name__ == '__main__':
    pose_file = 'data/poses.txt'
    K_file = 'data/K.txt'
    D_file = 'data/D.txt'
    img_file = 'data/images/img_0001.jpg'
    save_path = 'results'
    os.makedirs(save_path, exist_ok=True)

    board_rows = 6
    board_cols = 9

    # Test the camera model by drawing grid points on checkerboard
    rot, transl = load_cam_pose(pose_file)
    pose_mat = pose_to_mat(rot[0], transl[0])
    K_mat = load_K(K_file)
    distortions = load_distortions(D_file)

    img = cv2.imread(img_file)
    points_xyz = []
    for row in range(board_rows):
        for col in range(board_cols):
            points_xyz.append(np.array([0.04*col, 0.04*row, 0], dtype=np.float32))
    points_xyz = np.stack(points_xyz, axis=0)
    # points_uv = cam_proj(K_mat, pose_mat, points_xyz)
    points_uv = cam_proj(K_mat, pose_mat, points_xyz, distortions)
    for n in range(points_uv.shape[0]):
        cv2.circle(img, (int(points_uv[n, 0]), int(points_uv[n, 1])), 5, (0, 0, 255), -1)
    cv2.imwrite(os.path.join(save_path, 'img_0001.jpg'), img)