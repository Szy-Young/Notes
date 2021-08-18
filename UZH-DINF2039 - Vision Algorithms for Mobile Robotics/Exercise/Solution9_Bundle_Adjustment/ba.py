import os, sys
currentdir = os.path.dirname(os.path.realpath(__file__))
parentdir = os.path.dirname(currentdir)
sys.path.append(parentdir)

import numpy as np
from itertools import product
from scipy.sparse import coo_matrix
from scipy.optimize import least_squares
from matplotlib import pyplot as plt
from functools import partial
from align import load_observations, load_hidden_state, parse_hidden_state, twist_to_mat

from global_utils.camera import cam_proj, load_K, load_cam_pose


def ba_reproj_error(hidden_states, observations, n_frames, n_landmarks, K_mat):
    poses, landmarks = parse_hidden_state(hidden_states, n_frames, n_landmarks)
    poses = twist_to_mat(poses)
    points_uv_target = []
    points_uv_reproj = []

    for t in range(n_frames):
        pose = poses[t]
        # Camera pose to camera extrinsic parameters
        rot, transl = pose[:, :3], pose[:, 3:]
        pose_mat = np.concatenate((rot.T, -np.matmul(rot.T, transl)), axis=1)
        # Select landmarks observed in the specific frame
        landmark_ids = observations[t]['landmark_ids']
        landmarks_t = landmarks[landmark_ids]
        # Collect reprojections and observations
        points_uv = cam_proj(K_mat, pose_mat, landmarks_t)
        points_uv_reproj.append(points_uv)
        points_uv_target.append(observations[t]['points_uv'])

    points_uv_target = np.concatenate(points_uv_target, axis=0)
    points_uv_reproj = np.concatenate(points_uv_reproj, axis=0)
    return points_uv_reproj.reshape((-1)) - points_uv_target.reshape((-1))


def calc_jac_sparsity(observations, n_frames, n_landmarks):
    """
    Construct a Jocobian sparse mask for efficient BA.
    """
    landmark_idx_bias = 6*n_frames
    row_idxs = []
    col_idxs = []
    reproj_idx_bias = 0

    for t in range(n_frames):
        landmark_ids = observations[t]['landmark_ids']
        n_points_uv = landmark_ids.shape[0]
        for n in range(n_points_uv):
            row_idx = list(range(reproj_idx_bias+2*n, reproj_idx_bias+2*n+2))
            col_idx = []
            # frame - camera pose correspondance
            col_idx.extend(list(range(6*t, 6*t+6)))
            # 2D - 3D correspondance
            landmark_id = landmark_ids[n]
            col_idx.extend(list(range(landmark_idx_bias+3*landmark_id,
                                       landmark_idx_bias+3*landmark_id+3)))
            idxs = [row_idx, col_idx]
            idxs = list(product(*idxs))
            idxs = np.array(idxs, dtype=int)
            row_idxs.append(idxs[:, 0])
            col_idxs.append(idxs[:, 1])
        reproj_idx_bias += (2*n_points_uv)

    row_idxs = np.concatenate(row_idxs, axis=0)
    col_idxs = np.concatenate(col_idxs, axis=0)
    mask_shape = (reproj_idx_bias, 6*n_frames+3*n_landmarks)
    jac_sparsity = coo_matrix((np.ones_like(row_idxs, dtype=float), (row_idxs, col_idxs)), shape=mask_shape)
    return jac_sparsity


def bundle_adjustment(poses_init, landmarks_init, observations, n_frames, n_landmarks, K_mat):
    hidden_states = np.concatenate((poses_init.reshape((-1)), landmarks_init.reshape((-1))), axis=0)
    error_fn = partial(ba_reproj_error, observations=observations,
                       n_frames=n_frames, n_landmarks=n_landmarks, K_mat=K_mat)
    jac_sparsity = calc_jac_sparsity(observations, n_frames, n_landmarks)
    hidden_states = least_squares(fun=error_fn, x0=hidden_states, jac_sparsity=jac_sparsity).x
    poses, landmarks = parse_hidden_state(hidden_states, n_frames, n_landmarks)
    return poses, landmarks


if __name__ == '__main__':
    K_file = 'data/K.txt'
    pose_file = 'data/poses.txt'
    hs_file = 'data/hidden_state.txt'
    obs_file = 'data/observations.txt'
    crop_frames = 250
    save_dir = 'results'
    os.makedirs(save_dir, exist_ok=True)

    # Load and parse the data
    K_mat = load_K(K_file)
    pose_gts = load_cam_pose(pose_file)
    n_frames, n_landmarks, observations = load_observations(obs_file)
    pose_ests_twist, landmarks = load_hidden_state(hs_file, n_frames, n_landmarks)

    n_frames = min(n_frames, crop_frames)
    pose_gts = pose_gts[:n_frames]
    pose_ests_twist = pose_ests_twist[:n_frames]
    observations = observations[:n_frames]

    # Bundle adjustment
    pose_ests_twist_ba, landmarks_ba = bundle_adjustment(pose_ests_twist, landmarks, observations,
                                                         n_frames, n_landmarks, K_mat)

    pose_ests = twist_to_mat(pose_ests_twist)
    transl_ests = pose_ests[:, :, 3]
    pose_ests_ba = twist_to_mat(pose_ests_twist_ba)
    transl_ests_ba = pose_ests_ba[:, :, 3]

    # Visualize camera trajectory and landmarks before BA
    fig, ax = plt.subplots()
    ax.set_xlim(0, 40)
    ax.set_ylim(-10, 10)
    ax.plot(transl_ests[:, 2], transl_ests[:, 0], color='red')
    ax.scatter(landmarks[:, 2], landmarks[:, 0], color='blue', s=0.2, linewidth=0.1)
    ax.title.set_text('before BA')
    fig.savefig(os.path.join(save_dir, 'before_ba.png'))

    # Visualize camera trajectory and landmarks after BA
    fig, ax = plt.subplots()
    ax.set_xlim(0, 40)
    ax.set_ylim(-10, 10)
    ax.plot(transl_ests_ba[:, 2], transl_ests_ba[:, 0], color='red')
    ax.scatter(landmarks_ba[:, 2], landmarks_ba[:, 0], color='blue', s=0.2, linewidth=0.1)
    ax.title.set_text('after BA')
    fig.savefig(os.path.join(save_dir, 'after_ba.png'))