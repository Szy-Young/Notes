import numpy as np
from scipy.signal import convolve2d
from scipy.spatial.distance import cdist


def sobel_x(img):
    """
    Implement the Sobel filter along the x-axis.
    """
    kernel = np.array([[-1, 0, 1],
                       [-2, 0, 2],
                       [-1, 0, 1]], dtype=float)
    return convolve2d(img, kernel, mode='valid')


def sobel_y(img):
    """
    Implement the Sobel filter along the y-axis.
    """
    kernel = np.array([[-1, -2, -1],
                       [0, 0, 0],
                       [1, 2, 1]], dtype=float)
    return convolve2d(img, kernel, mode='valid')


def get_harris_M(img, patch_size=5):
    """
    Construct the matrix M in Harris corner detector.
    """
    # Compute the derivatives using the Sobel filter.
    Ix = sobel_x(img)
    Iy = sobel_y(img)

    # Construct the matrix M.
    Ix_2 = np.power(Ix, 2)
    Iy_2 = np.power(Iy, 2)
    Ix_Iy = np.multiply(Ix, Iy)
    patch_sum_kernel = np.ones((patch_size, patch_size), dtype=float)
    Ix_2_patch = convolve2d(Ix_2, patch_sum_kernel, mode='valid')
    Iy_2_patch = convolve2d(Iy_2, patch_sum_kernel, mode='valid')
    Ix_Iy_patch = convolve2d(Ix_Iy, patch_sum_kernel, mode='valid')
    M = np.stack((Ix_2_patch, Ix_Iy_patch, Ix_Iy_patch, Iy_2_patch), axis=-1)
    return M


def harris(img, patch_size=9, kappa=0.08):
    """
    Implement the Harris corner detector.
    """
    img_h, img_w = img.shape[:2]
    M = get_harris_M(img, patch_size)

    # Compute the corner response scores
    h, w = M.shape[:2]
    M = M.reshape((h*w, 2, 2))
    scores = np.linalg.det(M) - kappa * np.power(np.trace(M, axis1=1, axis2=2), 2)
    scores = np.clip(scores, a_min=0, a_max=None).reshape((h, w))
    scores = np.pad(scores, pad_width=(((img_h-h)//2, (img_h-h)//2), ((img_w-w)//2, (img_w-w)//2)))
    return scores


def shi_tomasi(img, patch_size=9):
    """
    Implement the Shi-Tomasi corner detector.
    """
    img_h, img_w = img.shape[:2]
    M = get_harris_M(img, patch_size)

    # Compute the corner response scores
    h, w = M.shape[:2]
    M = M.reshape((h*w, 2, 2))
    eigv_s, _ = np.linalg.eig(M)
    scores = np.min(eigv_s, axis=-1)
    scores = np.clip(scores, a_min=0, a_max=None).reshape((h, w))
    scores = np.pad(scores, pad_width=(((img_h-h)//2, (img_h-h)//2), ((img_w-w)//2, (img_w-w)//2)))
    return scores


def select_harris_keypoints(scores, n_keypoints=200, nms_radius=8):
    """
    Extract local maximas from the score map as keypoints.
    """
    img_h, img_w = scores.shape[:2]

    keypoints = []
    for i in range(n_keypoints):
        h, w = divmod(np.argmax(scores), img_w)
        keypoints.append((h, w))
        box = (max(0, h-nms_radius), min(img_h, h+nms_radius+1), max(0, w-nms_radius), min(img_w, w+nms_radius+1))
        scores[box[0]:box[1], box[2]:box[3]] = 0
    return keypoints


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


def match_descriptors(descriptors, last_descriptors, match_lambda=4):
    # Compute the pair-wise distances between two group of descriptors.
    descriptors = np.array(descriptors)
    descriptors = descriptors - np.mean(descriptors, axis=-1, keepdims=True)
    last_descriptors = np.array(last_descriptors)
    last_descriptors = last_descriptors - np.mean(last_descriptors, axis=-1, keepdims=True)
    distances = cdist(descriptors, last_descriptors)

    # Compute the threshold.
    thresh = match_lambda * np.min(distances)

    matches = - np.ones((len(descriptors)), dtype=int)
    while True:
        d_min = np.min(distances)
        if d_min > thresh:
            break
        row, col = divmod(np.argmin(distances), len(last_descriptors))
        matches[row] = col
        distances[row, :] = np.inf
        distances[:, col] = np.inf
    return matches