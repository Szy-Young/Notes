import os
import cv2
import numpy as np
from scipy.signal import convolve2d


def sobel_x(img):
    """
    Implement the Sobel filter along the x-axis.
    """
    kernel = np.array([[-1, 0, 1],
                       [-2, 0, 2],
                       [-1, 0, 1]], dtype=np.float32)
    return convolve2d(img, kernel, mode='valid')


def sobel_y(img):
    """
    Implement the Sobel filter along the y-axis.
    """
    kernel = np.array([[-1, -2, -1],
                       [0, 0, 0],
                       [1, 2, 1]], dtype=np.float32)
    return convolve2d(img, kernel, mode='valid')


def get_M(img, patch_size=5):
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
    patch_sum_kernel = np.ones((patch_size, patch_size), dtype=np.float32)
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
    M = get_M(img, patch_size)

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
    M = get_M(img, patch_size)

    # Compute the corner response scores
    h, w = M.shape[:2]
    M = M.reshape((h*w, 2, 2))
    eigv_s, _ = np.linalg.eig(M)
    scores = np.min(eigv_s, axis=-1)
    scores = np.clip(scores, a_min=0, a_max=None).reshape((h, w))
    scores = np.pad(scores, pad_width=(((img_h-h)//2, (img_h-h)//2), ((img_w-w)//2, (img_w-w)//2)))
    return scores


def select_keypoints(scores, n_keypoints=200, nms_radius=8):
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


if __name__ == '__main__':
    patch_size = 9
    harris_kappa= 0.08
    n_keypoints = 200
    nms_radius = 8
    img_dir = 'data'
    save_dir = 'results/detector'
    os.makedirs(save_dir, exist_ok=True)

    img_files = sorted(os.listdir(img_dir))
    # Test on the first frame
    img_file = img_files[0]
    img = cv2.imread(os.path.join(img_dir, img_file))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32)

    # Harris detector
    harris_scores = harris(img, patch_size=patch_size, kappa=harris_kappa)
    heatmap = (255 * harris_scores / np.max(harris_scores)).astype(np.uint8)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    cv2.imwrite(os.path.join(save_dir, img_file[:-4]+'_harris_scores.png'), heatmap)

    # Shi-Tomasi detector
    shi_tomasi_scores = shi_tomasi(img, patch_size=patch_size)
    heatmap = (255 * shi_tomasi_scores / np.max(shi_tomasi_scores)).astype(np.uint8)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    cv2.imwrite(os.path.join(save_dir, img_file[:-4]+'_shi_tomasi_scores.png'), heatmap)

    keypoints = select_keypoints(harris_scores, n_keypoints, nms_radius)
    img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    for keypoint in keypoints:
        cv2.drawMarker(img, (int(keypoint[1]), int(keypoint[0])), (0, 0, 255),
                       cv2.MARKER_TILTED_CROSS, markerSize=7, thickness=3)
    cv2.imwrite(os.path.join(save_dir, img_file[:-4]+'_harris_keypoints.png'), img)