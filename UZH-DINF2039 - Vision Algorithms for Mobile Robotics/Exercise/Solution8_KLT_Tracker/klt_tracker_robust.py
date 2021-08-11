import os, sys
currentdir = os.path.dirname(os.path.realpath(__file__))
parentdir = os.path.dirname(currentdir)
sys.path.append(parentdir)

import cv2
import numpy as np
import multiprocessing
from functools import partial
from klt_tracker import klt_tracker_keypoint

from global_utils.camera import load_uv


if __name__ == '__main__':
    patch_radius = 15
    num_iters = 50
    ref_img_file = 'data/000000.png'
    ref_img = cv2.imread(ref_img_file)
    ref_img_gray = cv2.cvtColor(ref_img, cv2.COLOR_BGR2GRAY).astype(float)

    # Apply KLT tracker to KITTI with bidirectional error check
    img_files = ['data/%06d.png'%(i) for i in range(1, 21)]
    ref_keypoints = load_uv('data/keypoints.txt')[:, 0][:, ::-1]
    save_dir = 'results/klt_robust'
    os.makedirs(save_dir, exist_ok=True)
    rescale_factor = 0.25
    lambda_robust = 0.1

    # Downsample for KLT convergence
    img_h, img_w = int(rescale_factor * ref_img.shape[0]), int(rescale_factor * ref_img.shape[1])
    ref_img = cv2.resize(ref_img, (img_w, img_h))
    ref_img_gray = cv2.cvtColor(ref_img, cv2.COLOR_BGR2GRAY).astype(float)
    ref_keypoints = ref_keypoints * rescale_factor

    # Parallelization for efficiency
    pool = multiprocessing.Pool(16)
    n_points = ref_keypoints.shape[0]
    ref_keypoints = [ref_keypoints[n] for n in range(n_points)]

    for img_file in img_files:
        warped_img = cv2.imread(img_file)
        warped_img = cv2.resize(warped_img, (img_w, img_h))
        warped_img_gray = cv2.cvtColor(warped_img, cv2.COLOR_BGR2GRAY).astype(float)
        # Tracking
        klt_tracker_fn = partial(klt_tracker_keypoint, ref_img=ref_img_gray, warped_img=warped_img_gray,
                                 patch_radius=patch_radius, num_iters=num_iters)
        warp_ests = pool.map(klt_tracker_fn, ref_keypoints)
        keypoints = []
        for n in range(n_points):
            keypoint = ref_keypoints[n] + warp_ests[n][:, 2]
            keypoints.append((keypoint))
        # Bidirectional error check
        klt_tracker_back_fn = partial(klt_tracker_keypoint, ref_img=warped_img_gray, warped_img=ref_img_gray,
                                      patch_radius=patch_radius, num_iters=num_iters)
        warp_back_ests = pool.map(klt_tracker_back_fn, keypoints)
        valids = []
        for n in range(n_points):
            ref_keypoint = ref_keypoints[n]
            keypoint_back = keypoints[n] + warp_back_ests[n][:, 2]
            valid = False
            if keypoint[0] > 0 and keypoint[0] < img_w and keypoint[1] > 0 and keypoint[1] < img_h:
                if np.linalg.norm(keypoint_back - ref_keypoint) < lambda_robust:
                    valid = True
            valids.append(valid)
        # Visualize the tracking results
        for n in range(n_points):
            if valids[n]:
                ref_keypoint = ref_keypoints[n]
                keypoint = keypoints[n]
                cv2.drawMarker(warped_img, (int(keypoint[0]), int(keypoint[1])), (0, 0, 255),
                               cv2.MARKER_TILTED_CROSS, markerSize=3, thickness=1)
                cv2.line(warped_img, (int(keypoint[0]), int(keypoint[1])),
                         (int(ref_keypoint[0]), int(ref_keypoint[1])), (0, 255, 0), 1)
        cv2.imwrite(os.path.join(save_dir, img_file.split('/')[-1]), warped_img)
        # Update
        ref_img_gray = warped_img_gray
        ref_keypoints = [keypoints[n] for n in range(n_points) if valids[n]]
        n_points = len(ref_keypoints)