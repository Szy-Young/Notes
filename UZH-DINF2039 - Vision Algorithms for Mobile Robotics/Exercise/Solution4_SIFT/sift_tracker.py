import os, sys
currentdir = os.path.dirname(os.path.realpath(__file__))
parentdir = os.path.dirname(currentdir)
sys.path.append(parentdir)

import cv2
import numpy as np
from scipy.ndimage import gaussian_filter
from scipy.spatial.distance import cdist

from sift_detector import DoG_pyramid, select_keypoints
from global_utils.filters import sobel_x, sobel_y


def describe_keypoints(img_pyramid, keypoints):
    """
    Describe keypoints using the SIFT descriptor.
    """
    descriptors = []
    hist_bins = [(i * np.pi / 4) for i in range(-4, 5)]

    for o, points in enumerate(keypoints):
        imgs = img_pyramid[o]
        # Compute the image gradints
        img_grad_x, img_grad_y = [], []
        for img in imgs:
            img_grad_x.append(sobel_x(img))
            img_grad_y.append(sobel_y(img))

        # Extract a descriptor for each keypoint
        for point in points:
            patch_grad_x = img_grad_x[point[0]][(point[1]-7):(point[1]+9), (point[2]-7):(point[2]+9)]
            patch_grad_y = img_grad_y[point[0]][(point[1]-7):(point[1]+9), (point[2]-7):(point[2]+9)]
            patch_grad = np.stack((patch_grad_x, patch_grad_y), axis=-1)
            patch_grad_norm = np.linalg.norm(patch_grad, axis=-1)
            patch_grad_orient = np.arctan2(patch_grad_y, patch_grad_x)
            # Scale the gradient norm by their distances to the keypoint center
            patch_grad_norm = gaussian_filter(patch_grad_norm, 16*1.5)

            # Divide into sub-blocks and construct weighted histogram
            descriptor = []
            for row in range(4):
                for col in range(4):
                    block_grad_norm = patch_grad_norm[(4*row):(4*row+4), (4*col):(4*col+4)].reshape((-1))
                    block_grad_orient = patch_grad_orient[(4*row):(4*row+4), (4*col):(4*col+4)].reshape((-1))
                    block_hist = np.histogram(block_grad_orient, bins=hist_bins, weights=block_grad_norm)[0]
                    descriptor.append(block_hist)
            descriptor = np.concatenate(descriptor)
            descriptor = descriptor / (np.linalg.norm(descriptor) + 1e-12)
            descriptors.append(descriptor)
    return descriptors


def match_descriptors(descriptors, last_descriptors, ratio_test=0.8):
    """
    Feature matching with a robust "Ratio Test" method.
    """
    # Compute the pair-wise distances between two group of descriptors
    descriptors = np.array(descriptors)
    last_descriptors = np.array(last_descriptors)
    distances = cdist(descriptors, last_descriptors)

    # Perform ratio test to find robust matches
    matches = - np.ones((len(descriptors)), dtype=np.int32)
    while True:
        d_min = np.min(distances)
        if d_min == np.inf:
            break
        row, col = divmod(np.argmin(distances), len(last_descriptors))
        row_values = distances[row]
        row_values[col] = np.inf
        d_min2 = np.min(row_values)
        if d_min < ratio_test * d_min2:
            matches[row] = col
            distances[:, col] = np.inf
        distances[row, :] = np.inf
    return matches


if __name__ == '__main__':
    n_scale = 3
    n_octave = 5
    sigma0 = 1.6
    contrast_threshold = 0.04
    nms_size = (3, 3, 3)    # The patch size (num_scale, height, width) to find local maxima
    rescale_factor = 0.2    # Rescale the original image for efficiency

    img_dir = 'images'
    save_dir = 'results'
    os.makedirs(save_dir, exist_ok=True)

    img_files = sorted(os.listdir(img_dir))[:2]
    imgs = []
    img_descriptors = []
    img_keypoints = []
    # Apply detector and descriptor on two images
    for img_file in img_files:
        img = cv2.imread(os.path.join(img_dir, img_file))
        img_h, img_w = img.shape[:2]
        img = cv2.resize(img, (int(rescale_factor*img_w), int(rescale_factor*img_h)))
        imgs.append(img)
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32)

        # SIFT detector and descriptor
        pyramid, dog_pyramid = DoG_pyramid(img_gray, n_scale, n_octave, sigma0)
        keypoints = select_keypoints(dog_pyramid, contrast_threshold, nms_size)
        img_keypoints.append(np.concatenate(keypoints, axis=0)[:, 1:])
        descriptors = describe_keypoints(pyramid, keypoints)
        img_descriptors.append(descriptors)

    # Keypoint matching between two images
    matches = match_descriptors(img_descriptors[1], img_descriptors[0])
    # Visualize matching
    match_vis = np.concatenate(imgs, axis=1)
    for keypoint_id, match in enumerate(matches):
        if match > 0:
            keypoint1 = img_keypoints[0][match]
            keypoint2 = img_keypoints[1][keypoint_id]
            keypoint2[1] += imgs[0].shape[1]
            cv2.drawMarker(match_vis, (int(keypoint1[1]), int(keypoint1[0])), (0, 0, 255),
                           cv2.MARKER_TILTED_CROSS, markerSize=5, thickness=1)
            cv2.drawMarker(match_vis, (int(keypoint2[1]), int(keypoint2[0])), (0, 0, 255),
                           cv2.MARKER_TILTED_CROSS, markerSize=5, thickness=1)
            cv2.line(match_vis, (int(keypoint1[1]), int(keypoint1[0])),
                     (int(keypoint2[1]), int(keypoint2[0])), (0, 255, 0), 1)
    cv2.imwrite(os.path.join(save_dir, 'match_vis.jpg'), match_vis)