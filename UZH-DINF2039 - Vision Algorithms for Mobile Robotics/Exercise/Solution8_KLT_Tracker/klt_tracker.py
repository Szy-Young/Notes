import os, sys
currentdir = os.path.dirname(os.path.realpath(__file__))
parentdir = os.path.dirname(currentdir)
sys.path.append(parentdir)

from scipy.signal import convolve2d
import cv2
import numpy as np
import multiprocessing
from functools import partial
from image_warp import get_warp, warp_image, get_warped_patch

from global_utils.camera import load_uv


def klt_tracker(ref_img, warped_img, patch_center=(35, 35), patch_radius=15, num_iters=50):
    """
    Implement a KLT tracker to estimate the warp transform.
    Parameters:
        ref_img: The reference image providing the template.
        warped_img: The image to track the template.
    """
    # Extract the template from the reference image
    warp_mat = get_warp()
    template = get_warped_patch(ref_img, warp_mat, patch_center, patch_radius).reshape((-1))

    # Initialize the estimate as identity matrix
    warp_est = warp_mat.copy()
    warp_ests = [warp_est]

    # Compute Jacobians for warp transform
    grid = np.meshgrid(np.arange(-patch_radius, patch_radius+1), np.arange(-patch_radius, patch_radius+1))
    patch_points_uv = np.stack(grid, axis=-1).reshape((-1, 2)).astype(float)
    jacobians = []
    for p in range(patch_points_uv.shape[0]):
        point = patch_points_uv[p]
        jacobian = np.array([[point[0], 0, point[1], 0, 1, 0],
                             [0, point[0], 0, point[1], 0, 1]], dtype=float)
        jacobians.append(jacobian)
    jacobians = np.stack(jacobians, axis=0)

    # Construct image gradient filter
    grad_x_kernel = np.array([[1, 0, -1]], dtype=float)
    grad_y_kernel = grad_x_kernel.T

    # Iterative optimization
    for iter in range(num_iters):
        # Compute I(w(p))-p gradients
        patch_padded = get_warped_patch(warped_img, warp_est, patch_center, patch_radius+1)
        patch_grad_x = convolve2d(patch_padded, grad_x_kernel, mode='valid')[1:-1].reshape((-1))
        patch_grad_y = convolve2d(patch_padded, grad_y_kernel, mode='valid')[:, 1:-1].reshape((-1))
        patch_grad = np.stack((patch_grad_x, patch_grad_y), axis=-1)
        patch_grad_p = np.einsum('mj,mjk->mk', patch_grad, jacobians)
        # Compute patch difference
        patch = patch_padded[1:-1, 1:-1].reshape((-1))
        patch_diff = template - patch
        # Compute Hessian
        hessian = np.matmul(patch_grad_p.T, patch_grad_p)
        # Update
        warp_inc = np.matmul(np.matmul(np.linalg.pinv(hessian), patch_grad_p.T), patch_diff).reshape((3, 2)).T
        warp_est += warp_inc
        warp_ests.append(warp_est)
    return warp_est, warp_ests


def klt_tracker_keypoint(keypoint, ref_img, warped_img, patch_radius=15, num_iters=50):
    return klt_tracker(ref_img, warped_img, keypoint, patch_radius, num_iters)[0]


if __name__ == '__main__':
    patch_radius = 15
    num_iters = 50
    ref_img_file = 'data/000000.png'
    ref_img = cv2.imread(ref_img_file)
    ref_img_gray = cv2.cvtColor(ref_img, cv2.COLOR_BGR2GRAY).astype(float)

    # Test KLT tracker
    patch_center = (900, 291)
    transl = (10, 6)
    
    warp_mat = get_warp(transl=transl)
    warped_img = warp_image(ref_img_gray, warp_mat)
    
    final_warp_est, warp_ests = klt_tracker(ref_img_gray, warped_img, patch_center, patch_radius, num_iters)
    print ('Final estimate of warp transform: \r\n', final_warp_est)

    # Apply KLT tracker to KITTI
    img_files = ['data/%06d.png'%(i) for i in range(1, 21)]
    ref_keypoints = load_uv('data/keypoints.txt')[:, 0][:, ::-1]
    save_dir = 'results/klt'
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
        valids = []
        for n in range(n_points):
            keypoint = ref_keypoints[n] + warp_ests[n][:, 2]
            keypoints.append((keypoint))
            valid = False
            if keypoint[0] > 0 and keypoint[0] < img_w and keypoint[1] > 0 and keypoint[1] < img_h:
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