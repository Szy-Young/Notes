import os, sys
currentdir = os.path.dirname(os.path.realpath(__file__))
parentdir = os.path.dirname(currentdir)
sys.path.append(parentdir)

import numpy as np
from scipy.linalg import expm, logm
from scipy.optimize import least_squares
from matplotlib import pyplot as plt
from functools import partial

from global_utils.camera import coord_to_hom, project_points, load_cam_pose


def load_observations(data_file):
    with open(data_file, 'r') as f:
        data = f.readlines()

    # Parse the data
    n_frames, n_landmarks = int(data[0]), int(data[1])
    observations = []
    idx = 2
    while idx < len(data):
        n_points = int(data[idx])
        points_uv = np.array(data[(idx+1):(idx+2*n_points+1)], dtype=float)
        points_uv = points_uv.reshape((n_points, 2))[:, ::-1]
        landmark_ids = np.array(data[(idx+2*n_points+1):(idx+3*n_points+1)], dtype=int) - 1
        observation = {'n_points': n_points, 'points_uv': points_uv, 'landmark_ids': landmark_ids}
        observations.append(observation)
        idx += 3*n_points+1
    return n_frames, n_landmarks, observations


def parse_hidden_state(hidden_states, n_frames, n_landmarks):
    poses = hidden_states[:(6*n_frames)].reshape((n_frames, 6))
    landmarks = hidden_states[(6*n_frames):(6*n_frames+3*n_landmarks)].reshape((n_landmarks, 3))
    return poses, landmarks


def load_hidden_state(data_file, n_frames, n_landmarks):
    with open(data_file, 'r') as f:
        data = f.readlines()

    hidden_states = np.array(data, dtype=float)
    poses, landmarks = parse_hidden_state(hidden_states, n_frames, n_landmarks)
    return poses, landmarks


def twist_to_mat(twist):
    """
    Transform twist vectors to 3x4 transform matrices, both representing 3D rotation and translation.
    Parameters:
        twist: (N, 6) array.
    """
    n_mat = twist.shape[0]
    transl, rot = twist[:, :3], twist[:, 3:]
    zero_vec = np.zeros_like(rot[:, 0])
    rot_mat = np.stack((np.stack((zero_vec, -rot[:, 2], rot[:, 1]), axis=1),
                        np.stack((rot[:, 2], zero_vec, -rot[:, 0]), axis=1),
                        np.stack((-rot[:, 1], rot[:, 0], zero_vec), axis=1)), axis=1)
    mat = np.zeros((n_mat, 4, 4), dtype=float)
    mat[:, :3, :3] = rot_mat
    mat[:, :3, 3] = transl
    for n in range(n_mat):
        mat[n] = expm(mat[n])
    return mat[:, :3]


def mat_to_twist(mat):
    """
    Transform 3x4 transform matrices to twist vectors, both representing 3D rotation and translation.
    Parameters:
        mat: (N, 3, 4) array.
    """
    n_mat = mat.shape[0]
    mat = np.concatenate((mat, np.tile(np.array([[0, 0, 0, 1]], dtype=float), (n_mat, 1, 1))), axis=1)
    for n in range(n_mat):
        mat[n] = logm(mat[n])
    rot_mat = mat[:, :3, :3]
    transl = mat[:, :3, 3]
    rot = np.stack((-rot_mat[:, 1, 2], rot_mat[:, 0, 2], -rot_mat[:, 0, 1]), axis=1)
    return np.concatenate((transl, rot), axis=1)


def trajectory_error(transform, transl_gts, transl_ests):
    transform, scale = transform[:6], transform[6]
    transform = twist_to_mat(transform.reshape((1, 6)))[0]
    transform[:, :3] = scale * transform[:, :3]
    transl_ests_transformed = project_points(transform, coord_to_hom(transl_ests))
    return transl_ests_transformed.reshape((-1)) - transl_gts.reshape((-1))


def align_trajectory(transl_gts, transl_ests, transform_init=None, init_scale=1):
    if transform_init is None:
        transform_init = np.eye(4, dtype=float)[:3]
    transform_init = transform_init.reshape((1, 3, 4))
    transform_init = mat_to_twist(transform_init)[0]
    transform_init = np.concatenate((transform_init, np.array([init_scale], dtype=float)), axis=0)
    error_fn = partial(trajectory_error, transl_gts=transl_gts, transl_ests=transl_ests)
    transform = least_squares(fun=error_fn, x0=transform_init).x
    return transform


if __name__ == '__main__':
    pose_file = 'data/poses.txt'
    hs_file = 'data/hidden_state.txt'
    obs_file = 'data/observations.txt'
    crop_frames = 150
    save_dir = 'results'
    os.makedirs(save_dir, exist_ok=True)

    # Load and parse the data
    pose_gts = load_cam_pose(pose_file)
    n_frames, n_landmarks, _ = load_observations(obs_file)
    pose_ests_twist, _ = load_hidden_state(hs_file, n_frames, n_landmarks)

    n_frames = min(n_frames, crop_frames)
    pose_gts = pose_gts[:n_frames]
    pose_ests_twist = pose_ests_twist[:n_frames]

    # Align the estimated trajectory with groundtruth
    pose_ests = twist_to_mat(pose_ests_twist)
    transl_gts = pose_gts[:, :, 3]
    transl_ests = pose_ests[:, :, 3]
    transform = align_trajectory(transl_gts, transl_ests)

    transform, scale = transform[:6], transform[6]
    transform = twist_to_mat(transform.reshape((1, 6)))[0]
    transform[:, :3] = scale * transform[:, :3]
    transl_ests_transformed = project_points(transform, coord_to_hom(transl_ests))

    # Visualize the alignment results
    fig = plt.figure()
    plt.plot(transl_gts[:, 2], transl_gts[:, 0], color='blue', label='GT')
    plt.plot(transl_ests[:, 2], transl_ests[:, 0], color='red', label='original')
    plt.plot(transl_ests_transformed[:, 2], transl_ests_transformed[:, 0], color='green', label='aligned')
    plt.legend()
    fig.savefig(os.path.join(save_dir, 'alignment.png'))