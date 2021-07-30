import numpy as np
from scipy.signal import convolve2d


def sobel_x(img):
    """
    Implement the Sobel filter along the x-axis.
    """
    kernel = np.array([[-1, 0, 1],
                       [-2, 0, 2],
                       [-1, 0, 1]], dtype=np.float64)
    return convolve2d(img, kernel, mode='valid')


def sobel_y(img):
    """
    Implement the Sobel filter along the y-axis.
    """
    kernel = np.array([[-1, -2, -1],
                       [0, 0, 0],
                       [1, 2, 1]], dtype=np.float64)
    return convolve2d(img, kernel, mode='valid')


def describe_keypoints_patch(img, keypoints, radius=9):
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