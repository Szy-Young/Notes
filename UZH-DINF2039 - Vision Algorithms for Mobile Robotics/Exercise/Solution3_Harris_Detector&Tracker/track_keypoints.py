import os
import cv2
import numpy as np
from scipy.spatial.distance import cdist
from detect_keypoints import harris, select_keypoints


def describe_keypoints(img, keypoints, radius=9):
    """
    Describe keypoints using image patches around them.
    """
    padded_img = np.pad(img, pad_width=((radius, radius), (radius, radius)))
    descriptors = []
    for keypoint in keypoints:
        h, w = keypoint[:2]
        descriptor = padded_img[h:(h+2*radius+1), w:(w+2*radius+1)].reshape(((2*radius+1)*(2*radius+1)))
        descriptors.append(descriptor)
    return descriptors


def match_descriptors(descriptors, last_descriptors, match_lambda=4):
    # Compute the pair-wise distances between two group of descriptors.
    descriptors = np.array(descriptors)
    descriptors = descriptors - np.mean(descriptors, axis=-1, keepdims=True)
    last_descriptors = np.array(last_descriptors)
    last_descriptors = last_descriptors - np.mean(last_descriptors, axis=-1, keepdims=True)
    distances = cdist(descriptors, last_descriptors)

    # Compute the threshold.
    thresh = match_lambda * np.min(distances)

    matches = - np.ones((len(descriptors)), dtype=np.int32)
    while True:
        d_min = np.min(distances)
        if d_min > thresh:
            break
        row, col = divmod(np.argmin(distances), len(last_descriptors))
        matches[row] = col
        distances[row, :] = np.inf
        distances[:, col] = np.inf
    return matches


if __name__ == '__main__':
    patch_size = 9
    harris_kappa= 0.08
    n_keypoints = 200
    nms_radius = 8
    descriptor_radius = 9
    match_lambda = 4
    img_dir = 'data'
    save_dir = 'results/tracker'
    os.makedirs(save_dir, exist_ok=True)

    img_files = sorted(os.listdir(img_dir))
    last_keypoints = None
    last_descriptors = None
    matches = None
    for t, img_file in enumerate(img_files):
        img = cv2.imread(os.path.join(img_dir, img_file))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # Harris detector
        harris_scores = harris(img, patch_size=patch_size, kappa=harris_kappa)
        keypoints = select_keypoints(harris_scores, n_keypoints, nms_radius)
        descriptors = describe_keypoints(img, keypoints, descriptor_radius)
        # Match keypoints
        if last_keypoints is not None and last_descriptors is not None:
            matches = match_descriptors(descriptors, last_descriptors, match_lambda)
        # Visualize detection & tracking
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        for i, keypoint in enumerate(keypoints):
            cv2.drawMarker(img, (int(keypoint[1]), int(keypoint[0])), (0, 0, 255),
                           cv2.MARKER_TILTED_CROSS, markerSize=7, thickness=3)
            if matches is not None:
                if matches[i] > 0:
                    last_keypoint = last_keypoints[matches[i]]
                    cv2.line(img, (int(keypoint[1]), int(keypoint[0])),
                             (int(last_keypoint[1]), int(last_keypoint[0])), (0, 255, 0), 3)
        cv2.imwrite(os.path.join(save_dir, img_file), img)
        # Update
        last_keypoints = keypoints
        last_descriptors = descriptors