import os
import cv2
import numpy as np
from camera import pose_to_mat, cam_proj, load_cam_pose, load_K


def construct_cube():
    """
    Construct a 3D Cube model.
    """
    verts = np.array(
        [[0.12, 0.04, 0],
         [0.2, 0.04, 0],
         [0.2, 0.12, 0],
         [0.12, 0.12, 0],
         [0.12, 0.04, -0.08],
         [0.2, 0.04, -0.08],
         [0.2, 0.12, -0.08],
         [0.12, 0.12, -0.08]], dtype=np.float32
    )
    edges = [[0, 1], [1, 2], [2, 3], [3, 0],
             [0, 4], [1, 5], [2, 6], [3, 7],
             [4, 5], [5, 6], [6, 7], [7, 4]]
    return verts, edges


if __name__ == '__main__':
    pose_file = 'data/poses.txt'
    K_file = 'data/K.txt'
    img_dir = 'results/images_undistorted'
    save_dir = 'results/images_cube'
    os.makedirs(save_dir, exist_ok=True)

    rot, transl = load_cam_pose(pose_file)
    K_mat = load_K(K_file)

    verts_xyz, edges = construct_cube()
    img_files = sorted(os.listdir(img_dir))
    for t, img_file in enumerate(img_files):
        img = cv2.imread(os.path.join(img_dir, img_file))
        pose_mat = pose_to_mat(rot[t], transl[t])
        # Use the camera model to draw cube on checkerboard
        verts_uv = cam_proj(K_mat, pose_mat, verts_xyz)
        for edge in edges:
            cv2.line(img, (int(verts_uv[edge[0], 0]), int(verts_uv[edge[0], 1])),
                     (int(verts_uv[edge[1], 0]), int(verts_uv[edge[1], 1])), (0, 0, 255), 3)
        cv2.imwrite(os.path.join(save_dir, img_file), img)