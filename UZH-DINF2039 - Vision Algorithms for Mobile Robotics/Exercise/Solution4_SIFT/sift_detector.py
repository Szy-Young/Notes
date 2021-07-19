import os
import cv2
import numpy as np
from scipy.ndimage import gaussian_filter, maximum_filter


def DoG_pyramid(img, n_scale=3, n_octave=5, sigma0=1.6):
    """
    Construct a Difference of Gaussian (DoG) pyramid.
    """
    img_h, img_w = img.shape[:2]
    pyramid = []
    dog_pyramid = []

    # Process each octave
    for o in range(n_octave):
        img_o = cv2.resize(img, (img_w // (2**o), img_h // (2**o)))
        octave = []
        # Apply Gaussian kernel for each scale
        for s in range(-1, n_scale+2):
            sigma = 2**(s/n_scale) * sigma0
            img_s = gaussian_filter(img_o, sigma)
            octave.append(img_s)
        octave = np.stack(octave)
        pyramid.append(octave[1:(n_scale+1)])
        # Compute the difference
        dog_pyramid.append(octave[1:] - octave[:-1])
    return pyramid, dog_pyramid


def select_keypoints(pyramid, threshold=0.04, nms_size=(3, 3, 3), bound_ratio=0.1):
    keypoints = []
    for octave in pyramid:
        max_values = maximum_filter(octave, size=nms_size)
        local_max = np.logical_and(octave==max_values, octave>threshold)
        # Ignore the points near the boundary
        if bound_ratio > 0:
            octave_h, octave_w = octave.shape[1:]
            bound_h, bound_w = int(bound_ratio*octave_h), int(bound_ratio*octave_w)
        else:
            bound_h, bound_w = 0, 0
        local_max = local_max[1:-1, bound_h:(octave_h-bound_h), bound_w:(octave_w-bound_w)]
        points = np.stack(np.where(local_max), axis=-1)
        points[:, 1:] += np.array([bound_h, bound_w])
        keypoints.append(points)
    return keypoints


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

    img_files = sorted(os.listdir(img_dir))
    # Test on the first image
    img_file = img_files[0]
    img = cv2.imread(os.path.join(img_dir, img_file))
    img_h, img_w = img.shape[:2]
    img = cv2.resize(img, (int(rescale_factor*img_w), int(rescale_factor*img_h)))
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32)

    # SIFT detector
    _, dog_pyramid = DoG_pyramid(img_gray, n_scale, n_octave, sigma0)
    keypoints = select_keypoints(dog_pyramid, contrast_threshold, nms_size)

    # Visualize the detected keypoints in each octave
    for o, points in enumerate(keypoints):
        for point in points:
            point_scale, point_uv = point[0], point[1:]
            point_uv = point_uv * (2**o)
            sigma = int(sigma0 * (2**(o+point_scale/n_scale)))
            cv2.circle(img, (int(point_uv[1]), int(point_uv[0])), sigma, (0, 0, 255), 1)
    cv2.imwrite(os.path.join(save_dir, img_file), img)