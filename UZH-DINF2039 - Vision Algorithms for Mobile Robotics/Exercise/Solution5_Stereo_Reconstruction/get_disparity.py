import os, sys
currentdir = os.path.dirname(os.path.realpath(__file__))
parentdir = os.path.dirname(currentdir)
sys.path.append(parentdir)

import cv2
import numpy as np
from scipy.spatial.distance import cdist
from global_utils.filters import describe_keypoints_patch


def get_disparity(img_left, img_right, patch_radius=5, disp_min=5, disp_max=50):
    """
    Given a pair of images, calculate the pixel-wise disparity for the left image.
    """
    img_h, img_w = img_left.shape[:2]
    img_disp = np.zeros_like(img_left)
    # Process each row (ignore the boundary, same for columns)
    for h in range(patch_radius, img_h-patch_radius):
        for w in range(disp_max+patch_radius, img_w-patch_radius):
            # Compute the similarities between left-image patch and its candidate correspondances on right-image
            left_patch = describe_keypoints_patch(img_left, [[h, w]], radius=patch_radius)
            left_patch = left_patch - np.mean(left_patch, axis=-1, keepdims=True)
            right_cols = np.arange(w-disp_max, w-disp_min+1, dtype=np.int32)
            right_points = np.stack((h*np.ones_like(right_cols), right_cols), axis=1)
            right_patches = describe_keypoints_patch(img_right, right_points, radius=patch_radius)
            right_patches = right_patches - np.mean(right_patches, axis=-1, keepdims=True)
            distances = cdist(left_patch, right_patches)[0]
            # Select the correspondance while rejecting the ambiguities
            match = np.argmin(distances)
            if match == 0 or match == (disp_max-disp_min):
                continue
            min_dist = distances[match]
            min_dist_3 = distances[np.argpartition(distances, 3)[:3]]
            if np.max(min_dist_3) < 1e-10:
                continue
            if np.max(min_dist_3) < 1.5 * min_dist:
                continue
            # Perform sub-pixel refinement to attain continuous disparity
            x = np.array([match-1, match, match+1])
            y = distances[x]
            polynomial = np.polyfit(x, y, deg=2)
            disp = disp_max + 0.5 * polynomial[1] / polynomial[0]
            img_disp[h, w] = disp
    return img_disp


if __name__ == '__main__':
    patch_radius = 5
    disp_min = 5
    disp_max = 50
    rescale_factor = 0.5    # Rescale the original image for efficiency

    img_left_dir = 'data/left'
    img_right_dir = 'data/right'
    save_dir = 'results/disparity'
    os.makedirs(save_dir, exist_ok=True)

    img_files = sorted(os.listdir(img_left_dir))

    # Test on the sequence
    for img_file in img_files:
        img_left = cv2.imread(os.path.join(img_left_dir, img_file))
        img_right = cv2.imread(os.path.join(img_right_dir, img_file))
        img_h, img_w = img_left.shape[:2]
        img_left = cv2.resize(img_left, (int(rescale_factor*img_w), int(rescale_factor*img_h)))
        img_right = cv2.resize(img_right, (int(rescale_factor*img_w), int(rescale_factor*img_h)))
        img_left = cv2.cvtColor(img_left, cv2.COLOR_BGR2GRAY).astype(np.float32)
        img_right = cv2.cvtColor(img_right, cv2.COLOR_BGR2GRAY).astype(np.float32)
        # Calculate the disparity and visualize
        img_disp = get_disparity(img_left, img_right, patch_radius, disp_min, disp_max)
        np.save(os.path.join(save_dir, img_file[:-4]+'.npy'), img_disp)
        heatmap = (255 * img_disp / disp_max).astype(np.uint8)
        heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
        cv2.imwrite(os.path.join(save_dir, img_file), heatmap)