"""
Initially written by Ming Hsiao in MATLAB
Redesigned and rewritten by Wei Dong (weidong@andrew.cmu.edu)
"""

import os
import json
import numpy as np
import tyro
from tqdm import tqdm
import quaternion  # pip install numpy-quaternion
from PIL import Image
import transforms


def load_gt_poses(gt_filename):
    indices = []
    Ts = []

    # Camera to world
    # Dirty left 2 right coordinate transform
    # https://github.com/theNded/MeshHashing/blob/master/src/io/config_manager.cc#L88
    T_l2r = np.eye(4)
    T_l2r[1, 1] = -1

    with open(gt_filename) as f:
        content = f.readlines()
        for line in content:
            data = np.array(list(map(float, line.strip().split(" "))))
            indices.append(int(data[0]))

            data = data[1:]

            q = data[3:][[3, 0, 1, 2]]
            q = quaternion.from_float_array(q)
            R = quaternion.as_rotation_matrix(q)

            t = data[:3]
            T = np.eye(4)

            T[0:3, 0:3] = R
            T[0:3, 3] = t

            Ts.append(T_l2r @ T @ np.linalg.inv(T_l2r))

    return indices, Ts


def main(path: str):
    """
    Args:
        path: path to the dataset folder containing rgb/ and depth/
    """
    # Load intrinsics and gt poses for evaluation
    with open("intrinsics.json") as f:
        intrinsic = np.array(json.load(f)["intrinsic_matrix"]).reshape(3, 3, order="F")
    indices, gt_poses = load_gt_poses(os.path.join(path, "livingRoom2.gt.freiburg"))
    depth_scale = 5000.0

    depth_path = os.path.join(path, "depth")
    normal_path = os.path.join(path, "normal")
    os.makedirs(normal_path, exist_ok=True)

    # Generate normal maps
    # WARNING: please start from index 1, as ground truth poses are provided starting from index 1.
    for i in tqdm(indices, desc='preprocessing'):
        depth = np.asarray(Image.open("{}/{}.png".format(depth_path, i))) / depth_scale
        vertex_map = transforms.unproject(depth, intrinsic)

        # Estimate normals from cross products of neighboring vertex map vectors.
        # Respects depth discontinuities and is consistent toward the camera.
        du = np.zeros_like(vertex_map)
        dv = np.zeros_like(vertex_map)
        du[:, :-1] = vertex_map[:, 1:] - vertex_map[:, :-1]
        dv[:-1, :] = vertex_map[1:, :] - vertex_map[:-1, :]
        normal_map = np.cross(du, dv)
        norms = np.linalg.norm(normal_map, axis=2, keepdims=True)
        norms[norms == 0] = 1
        normal_map = normal_map / norms
        np.save("{}/{}.npy".format(normal_path, i), normal_map)


if __name__ == "__main__":
    tyro.cli(main)
