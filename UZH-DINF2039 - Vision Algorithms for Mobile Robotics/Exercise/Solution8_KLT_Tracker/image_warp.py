import os, sys
currentdir = os.path.dirname(os.path.realpath(__file__))
parentdir = os.path.dirname(currentdir)
sys.path.append(parentdir)

import cv2
import numpy as np
from scipy.spatial.distance import cdist

from global_utils.camera import coord_to_hom, project_points


def get_warp(rot=0, transl=(0, 0), scale=1):
    """
    Construct a warp transform matrix.
    """
    rot_mat = np.array([[np.cos(rot), -np.sin(rot)],
                        [np.sin(rot), np.cos(rot)]], dtype=float)
    transl = np.array(transl, dtype=float).reshape((2, 1))
    warp_mat = np.concatenate((rot_mat, transl), axis=1)
    warp_mat = scale * warp_mat
    return warp_mat


def bilinear_interpolate(ref_img, point):
    # Find nearest-neighbour coordinates in source image
    h, w = point
    h_prev = int(np.floor(h))
    h_next = h_prev + 1
    w_prev = int(np.floor(w))
    w_next = w_prev + 1
    up_left = ref_img[h_prev, w_prev]
    up_right = ref_img[h_prev, w_next]
    down_left = ref_img[h_next, w_prev]
    down_right = ref_img[h_next, w_next]

    # Perform bilinear interpolation
    interp = (h_next-h)*(w_next-w)*up_left + (h_next-h)*(w-w_prev)*up_right\
             + (h-h_prev)*(w_next-w)*down_left + (h-h_prev)*(w-w_prev)*down_right
    return interp


def warp_image(ref_img, warp_mat, interpolate=True):
    # Compute point correspondances
    h, w =  ref_img.shape[:2]
    grid = np.meshgrid(np.arange(w), np.arange(h))
    points_uv = np.stack(grid, axis=-1).reshape((-1, 2)).astype(float)
    points_uv_src = project_points(warp_mat, coord_to_hom(points_uv))

    # Warping
    points_uv = points_uv[:, ::-1].astype(int)
    points_uv_src = points_uv_src[:, ::-1]
    dest_img = np.zeros_like(ref_img)
    for p in range(points_uv.shape[0]):
        point = points_uv[p]
        point_src = points_uv_src[p]
        if point_src[0] < 0 or point_src[0] > (h-1.01) or point_src[1] < 0 or point_src[1] > (w-1.01):
            continue
        if interpolate:
            dest_img[tuple(point)] = bilinear_interpolate(ref_img, point_src)
        else:
            dest_img[tuple(point)] = ref_img[tuple(point_src.astype(int))]
    return dest_img


def get_warped_patch(ref_img, warp_mat, patch_center=(15, 15), patch_radius=15, interpolate=True):
    # Compute point correspondances
    grid = np.meshgrid(np.arange(-patch_radius, patch_radius+1),
                       np.arange(-patch_radius, patch_radius+1))
    points_uv = np.stack(grid, axis=-1).reshape((-1, 2)).astype(float)
    points_uv_src = project_points(warp_mat, coord_to_hom(points_uv)) + patch_center

    # Warping
    points_uv += patch_radius
    points_uv = points_uv[:, ::-1]
    points_uv_src = points_uv_src[:, ::-1]
    patch = np.zeros((2*patch_radius+1, 2*patch_radius+1), dtype=float)
    h, w = ref_img.shape[:2]
    for p in range(points_uv.shape[0]):
        point = points_uv[p]
        point_src = points_uv_src[p]
        if point_src[0] < 0 or point_src[0] > (h - 1.01) or point_src[1] < 0 or point_src[1] > (w - 1.01):
            continue
        if interpolate:
            patch[tuple(point.astype(int))] = bilinear_interpolate(ref_img, point_src)
        else:
            patch[tuple(point.astype(int))] = ref_img[tuple(point_src.astype(int))]
    return patch


def brute_force_tracker(ref_img, warped_img, patch_center=(35, 35), patch_radius=15, search_radius=20):
    """
    Implement a brute-force tracker to estimate the translation only.
    Parameters:
        ref_img: The reference image providing the template.
        warped_img: The image to track the template.
    """
    # Extract the template from the reference image
    warp_mat = get_warp()
    template = get_warped_patch(ref_img, warp_mat, patch_center, patch_radius)

    # Search on the image
    candidates = []
    for i in range(-search_radius, search_radius+1):
        for j in range(-search_radius, search_radius+1):
            warp_mat = get_warp(transl=(i, j))
            patch = get_warped_patch(warped_img, warp_mat, patch_center, patch_radius)
            patch = patch.reshape((-1))
            candidates.append(patch)
    candidates = np.stack(candidates, axis=0)
    template = template.reshape((1, -1))

    distances = cdist(template, candidates)[0]
    match = np.argmin(distances)
    transl_est = np.array([match // (2*search_radius+1) - search_radius, match % (2*search_radius+1) - search_radius], dtype=float)
    return transl_est, distances.reshape(2*search_radius+1, 2*search_radius+1)


if __name__ == '__main__':
    ref_img_file = 'data/000000.png'
    ref_img = cv2.imread(ref_img_file)
    save_dir = 'results/image_warp'
    os.makedirs(save_dir, exist_ok=True)

    # # Test image warping
    # rot = np.pi * 10 / 180
    # transl = (50, -30)
    # scale = 0.5
    #
    # warp_mat = get_warp(rot=rot)
    # dest_img = warp_image(ref_img, warp_mat)
    # cv2.imwrite(os.path.join(save_dir, 'rotation.png'), dest_img)
    # warp_mat = get_warp(transl=transl)
    # dest_img = warp_image(ref_img, warp_mat)
    # cv2.imwrite(os.path.join(save_dir, 'translation.png'), dest_img)
    # warp_mat = get_warp(scale=scale)
    # dest_img = warp_image(ref_img, warp_mat)
    # cv2.imwrite(os.path.join(save_dir, 'scale.png'), dest_img)

    # Test brute-force template tracker
    ref_img_gray = cv2.cvtColor(ref_img, cv2.COLOR_BGR2GRAY)
    patch_center = (900, 291)
    patch_radius = 15
    transl = (10, 6)
    search_radius = 20

    warp_mat = get_warp()
    template = get_warped_patch(ref_img_gray, warp_mat, patch_center, patch_radius)
    cv2.imwrite(os.path.join(save_dir, 'template.png'), template)
    warp_mat = get_warp(transl=transl)
    warped_img = warp_image(ref_img_gray, warp_mat)

    transl_est, distances = brute_force_tracker(ref_img_gray, warped_img, patch_center, patch_radius, search_radius)
    print (transl_est)
    distances = (255 * distances / np.max(distances)).astype(np.uint8)
    distances = cv2.applyColorMap(distances, cv2.COLORMAP_JET)
    cv2.imwrite(os.path.join(save_dir, 'track_scores.png'), distances)