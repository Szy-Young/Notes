import os
import cv2
import numpy as np
from camera import load_K, load_distortions, distort_points


def undistort_image(img_d, distortions, center):
    h, w = img_d.shape[:2]
    grid = np.meshgrid(np.arange(w), np.arange(h))
    points_uv = np.stack(grid, axis=-1).reshape((-1, 2)).astype(np.float32)
    points_uv_d = distort_points(points_uv, distortions, center)

    # Backward warping
    points_uv = points_uv[:, ::-1].astype(np.int64).T.tolist()
    points_uv_d = points_uv_d[:, ::-1].astype(np.int64).T.tolist()
    img = np.zeros_like(img_d)
    img[tuple(points_uv)] = img_d[tuple(points_uv_d)]
    return img


if __name__ == '__main__':
    K_file = 'data/K.txt'
    D_file = 'data/D.txt'
    img_dir = 'data/images'
    save_dir = 'results/images_undistorted'
    os.makedirs(save_dir, exist_ok=True)

    K_mat = load_K(K_file)
    uv_center = K_mat[:2, 2]
    distortions = load_distortions(D_file)
    img_files = sorted(os.listdir(img_dir))

    for img_file in img_files:
        img_d = cv2.imread(os.path.join(img_dir, img_file))
        img = undistort_image(img_d, distortions, uv_center)
        cv2.imwrite(os.path.join(save_dir, img_file), img)