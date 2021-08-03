import os, sys
currentdir = os.path.dirname(os.path.realpath(__file__))
parentdir = os.path.dirname(currentdir)
sys.path.append(parentdir)

import cv2
import numpy as np
from functools import partial

from ransac import ransac, adaptive_ransac
from dlt_ransac import dlt_fit
from global_utils.camera import load_K, load_uv, load_xyz, dlt, cam_proj, pose_to_mat
from global_utils.filters import harris, select_harris_keypoints, describe_keypoints_patch, match_descriptors


def p3p_fit(points, K_mat):
    points_xyz, points_uv = points[0], points[1]
    success, rot, transl = cv2.solvePnP(points_xyz, points_uv, K_mat,
                            distCoeffs=None, flags=cv2.SOLVEPNP_P3P)
    if success:
        pose_mat = pose_to_mat(rot, transl)
    else:
        pose_mat = np.concatenate((np.eye(3), np.zeros((3, 1))), axis=1)
    return pose_mat


def p3p_eval(points, pose_mat, K_mat):
    points_xyz, points_uv = points[0], points[1]
    points_uv_reproj = cam_proj(K_mat, pose_mat, points_xyz)
    reproj_error = np.linalg.norm(points_uv - points_uv_reproj, axis=1)
    return reproj_error


if __name__ == '__main__':
    data_dir = 'data'
    ref_img_file = '000000.png'
    # query_img_files = ['000001.png']
    query_img_files = ['%06d.png'%(i) for i in range(1, 10)]
    K_file = 'data/K.txt'
    save_dir = 'results/tracking'
    os.makedirs(save_dir, exist_ok=True)

    # Params for keypoint matching
    patch_size = 9
    harris_kappa= 0.08
    n_keypoints = 1000
    nms_radius = 8
    descriptor_radius = 9
    match_lambda = 5

    # Params for DLT with RANSAC
    n_iters = 20000
    thresh = 10
    inlier_ratio = 0.63
    points_per_step = 4

    K_mat= load_K(K_file)
    # Load keypoints for the reference image
    ref_img = cv2.imread(os.path.join(data_dir, ref_img_file))
    ref_img = cv2.cvtColor(ref_img, cv2.COLOR_BGR2GRAY)
    ref_keypoints = load_uv(os.path.join(data_dir, 'keypoints.txt'))[:, 0].astype(int)
    ref_descriptors = describe_keypoints_patch(ref_img, ref_keypoints, descriptor_radius)
    # Load 3D keypoints
    points_xyz = load_xyz(os.path.join(data_dir, 'p_W_landmarks.txt'))

    for query_img_file in query_img_files:
        query_img = cv2.imread(os.path.join(data_dir, query_img_file))
        query_img = cv2.cvtColor(query_img, cv2.COLOR_BGR2GRAY)

        # Harris detector
        harris_scores = harris(query_img, patch_size=patch_size, kappa=harris_kappa)
        query_keypoints = select_harris_keypoints(harris_scores, n_keypoints, nms_radius)
        query_descriptors = describe_keypoints_patch(query_img, query_keypoints, descriptor_radius)
        # Match keypoints
        matches = match_descriptors(query_descriptors, ref_descriptors, match_lambda)
        query_matched = np.where(matches>0)[0]
        ref_matched = matches[query_matched]
        query_points_uv = np.array(query_keypoints, dtype=float)[query_matched]
        query_points_uv = query_points_uv[:, np.array([1, 0], dtype=int)]      # (h, w) to (w, h)
        ref_points_xyz = points_xyz[ref_matched]

        print ('--', query_img_file)
        # P3P with RANSAC
        pose_mat_est_ransac, inlier_ids_est, use_iters = ransac([ref_points_xyz, query_points_uv],
                                                                fit_fn=partial(p3p_fit, K_mat=K_mat),
                                                                eval_fn=partial(p3p_eval, K_mat=K_mat),
                                                                refine_fn=partial(dlt_fit, K_mat=K_mat),
                                                                n_iters=n_iters, thresh=thresh,
                                                                inlier_ratio=inlier_ratio, points_per_step=points_per_step)
        print ('Estimated camera pose by RANSAC: \r\n', pose_mat_est_ransac)
        print (use_iters, 'iters used.')

        # P3P with adaptive RANSAC
        pose_mat_est_ransac, inlier_ids_est, use_iters = adaptive_ransac([ref_points_xyz, query_points_uv],
                                                                         fit_fn=partial(p3p_fit, K_mat=K_mat),
                                                                         eval_fn=partial(p3p_eval, K_mat=K_mat),
                                                                         refine_fn=partial(dlt_fit, K_mat=K_mat),
                                                                         n_iters=n_iters, thresh=thresh,
                                                                         points_per_step=points_per_step)
        print ('Estimated camera pose by adaptive RANSAC: \r\n', pose_mat_est_ransac)
        print (use_iters, 'iters used.')

        ref_points_uv = ref_keypoints[ref_matched]
        # Visualize the raw matches
        query_img1 = cv2.imread(os.path.join(data_dir, query_img_file))
        for i in range(query_points_uv.shape[0]):
            query_point = query_points_uv[i, np.array([1, 0])]      # (w, h) to (h, w)
            ref_point = ref_points_uv[i]
            cv2.drawMarker(query_img1, (int(query_point[1]), int(query_point[0])), (0, 0, 255),
                           cv2.MARKER_TILTED_CROSS, markerSize=7, thickness=3)
            cv2.line(query_img1, (int(query_point[1]), int(query_point[0])),
                     (int(ref_point[1]), int(ref_point[0])), (0, 255, 0), 3)
        # Visualize the matches refined by RANSAC
        query_img2 = cv2.imread(os.path.join(data_dir, query_img_file))
        for i in range(query_points_uv.shape[0]):
            query_point = query_points_uv[i, np.array([1, 0])]      # (w, h) to (h, w)
            ref_point = ref_points_uv[i]
            if i in inlier_ids_est:
                cv2.drawMarker(query_img2, (int(query_point[1]), int(query_point[0])), (0, 255, 0),
                           cv2.MARKER_TILTED_CROSS, markerSize=7, thickness=3)
                cv2.line(query_img2, (int(query_point[1]), int(query_point[0])),
                         (int(ref_point[1]), int(ref_point[0])), (0, 255, 0), 3)
            else:
                cv2.drawMarker(query_img2, (int(query_point[1]), int(query_point[0])), (0, 0, 255),
                           cv2.MARKER_TILTED_CROSS, markerSize=7, thickness=3)
        cv2.imwrite(os.path.join(save_dir, query_img_file), np.concatenate((query_img1, query_img2), axis=1))