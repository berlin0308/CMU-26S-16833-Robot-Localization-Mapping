"""
Initially written by Ming Hsiao in MATLAB
Redesigned and rewritten by Wei Dong (weidong@andrew.cmu.edu)
"""

import json
import os
from typing import Optional

import numpy as np
import rerun as rr
import transforms
import tyro
import utils
from PIL import Image


def find_projective_correspondence(
    source_points,
    source_normals,
    target_vertex_map,
    target_normal_map,
    intrinsic,
    T_init,
    dist_diff=0.07,
):
    """
    Args:
        source_points: Source point cloud locations, (N, 3)
        source_normals: Source point cloud normals, (N, 3)
        target_vertex_map: Target vertex map, (H, W, 3)
        target_normal_map: Target normal map, (H, W, 3)
        intrinsic: Intrinsic matrix, (3, 3)
        T_init: Initial transformation from source to target, (4, 4)
        dist_diff: Distance difference threshold to filter correspondences

    Returns:
        source_indices: indices of points in the source point cloud with a valid projective correspondence in the target map, (M, 1)
        target_us: associated u coordinate of points in the target map, (M, 1)
        target_vs: associated v coordinate of points in the target map, (M, 1)
    """
    h, w, _ = target_vertex_map.shape

    R = T_init[:3, :3]
    t = T_init[:3, 3:]

    # Transform source points from the source coordinate system to the target coordinate system
    T_source_points = (R @ source_points.T + t).T

    # Set up initial correspondences from source to target
    source_indices = np.arange(len(source_points)).astype(int)
    target_us, target_vs, target_ds = transforms.project(T_source_points, intrinsic)
    target_us = np.round(target_us).astype(int)
    target_vs = np.round(target_vs).astype(int)

    # TODO: first filter: valid projection
    # u, v must fall inside the image; d (depth along optical axis) must be positive.
    mask = (
        (target_us >= 0)
        & (target_us < w)
        & (target_vs >= 0)
        & (target_vs < h)
        & (T_source_points[:, 2] > 0)
    )
    # End of TODO

    source_indices = source_indices[mask]
    target_us = target_us[mask]
    target_vs = target_vs[mask]
    T_source_points = T_source_points[mask]

    # TODO: second filter: apply distance threshold
    target_pts = target_vertex_map[target_vs, target_us]
    dist = np.linalg.norm(T_source_points - target_pts, axis=1)
    n_tgt = target_normal_map[target_vs, target_us]
    n_src = source_normals[source_indices]
    n_src_t = (R @ n_src.T).T
    n_src_len = np.linalg.norm(n_src_t, axis=1, keepdims=True)
    n_tgt_len = np.linalg.norm(n_tgt, axis=1, keepdims=True)
    n_src_t = np.divide(
        n_src_t, n_src_len, out=np.zeros_like(n_src_t), where=n_src_len > 1e-8
    )
    n_tgt = np.divide(n_tgt, n_tgt_len, out=np.zeros_like(n_tgt), where=n_tgt_len > 1e-8)
    cos_angle = np.sum(n_src_t * n_tgt, axis=1)
    mask = (
        (target_pts[:, 2] > 1e-6)
        & (dist < dist_diff)
        & (cos_angle > np.cos(np.deg2rad(30.0)))
    )
    # End of TODO

    source_indices = source_indices[mask]
    target_us = target_us[mask]
    target_vs = target_vs[mask]

    return source_indices, target_us, target_vs


def build_linear_system(source_points, target_points, target_normals, T):
    M = len(source_points)
    assert len(target_points) == M and len(target_normals) == M

    R = T[:3, :3]
    t = T[:3, 3:]

    p_prime = (R @ source_points.T + t).T
    q = target_points
    n_q = target_normals

    A = np.zeros((M, 6))
    b = np.zeros((M,))

    # TODO: build the linear system
    # Point-to-plane linearization: n^T (omega x p' + v) = n^T (q - p')
    # with A_i = [(p' x n)^T, n^T], b_i = n^T (q - p').
    cross_pn = np.cross(p_prime, n_q)
    A[:, :3] = cross_pn
    A[:, 3:] = n_q
    b = np.sum(n_q * (q - p_prime), axis=1)
    # End of TODO

    return A, b


def pose2transformation(delta):
    """
    Args:
        delta: Vector (6, ) in the tangent space with the small angle assumption.

    Returns:
        T: Matrix (4, 4) transformation matrix recovered from delta.
        Reference: https://en.wikipedia.org/wiki/Euler_angles in the ZYX order
    """
    w = delta[:3]
    u = np.expand_dims(delta[3:], axis=1)

    T = np.eye(4)

    # yapf: disable
    R = np.array([[
        np.cos(w[2]) * np.cos(w[1]),
        -np.sin(w[2]) * np.cos(w[0]) + np.cos(w[2]) * np.sin(w[1]) * np.sin(w[0]),
        np.sin(w[2]) * np.sin(w[0]) + np.cos(w[2]) * np.sin(w[1]) * np.cos(w[0])
    ],
    [
        np.sin(w[2]) * np.cos(w[1]),
        np.cos(w[2]) * np.cos(w[0]) + np.sin(w[2]) * np.sin(w[1]) * np.sin(w[0]),
        -np.cos(w[2]) * np.sin(w[0]) + np.sin(w[2]) * np.sin(w[1]) * np.cos(w[0])
    ],
    [
        -np.sin(w[1]),
        np.cos(w[1]) * np.sin(w[0]),
        np.cos(w[1]) * np.cos(w[0])
    ]])
    # yapf: enable

    T[:3, :3] = R
    T[:3, 3:] = u

    return T


def solve(A, b):
    """
    Args:
        A: (6, 6) matrix in the LU formulation, or (N, 6) in the QR formulation
        b: (6, 1) vector in the LU formulation, or (N, 1) in the QR formulation

    Returns:
        delta: (6, 1) vector by solving the linear system. You may directly use dense solvers from numpy.
    """
    # TODO: write your relevant solver
    # Least squares on the overdetermined system A delta = b (QR); equivalent normal
    # equations (LU on 6x6): (A^T A) delta = A^T b.
    delta, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
    return delta


def icp(
    source_points,
    source_normals,
    target_vertex_map,
    target_normal_map,
    intrinsic,
    T_init=None,
    debug_association=False,
    num_iters: int = 10,
):
    """
    Args:
        source_points: Source point cloud locations, (N, 3)
        source_normals: Source point cloud normals, (N, 3)
        target_vertex_map: Target vertex map, (H, W, 3)
        target_normal_map: Target normal map, (H, W, 3)
        intrinsic: Intrinsic matrix, (3, 3)
        T_init: Initial transformation from source to target, (4, 4)
        debug_association: Visualize association between sources and targets for debug
        num_iters: Number of Gauss–Newton ICP iterations.

    Returns:
        T: (4, 4) transformation from source to target
    """

    T = np.eye(4) if T_init is None else T_init

    for i in range(int(num_iters)):
        # TODO: fill in find_projective_correspondences
        source_indices, target_us, target_vs = find_projective_correspondence(
            source_points,
            source_normals,
            target_vertex_map,
            target_normal_map,
            intrinsic,
            T,
        )

        # Select associated source and target points
        corres_source_points = source_points[source_indices]
        corres_target_points = target_vertex_map[target_vs, target_us]
        corres_target_normals = target_normal_map[target_vs, target_us]

        # Debug, if necessary
        if debug_association:
            rr.set_time("icp_iter", sequence=i)
            utils.visualize_correspondences(
                corres_source_points, corres_target_points, T
            )

        # TODO: fill in build_linear_system and solve
        A, b = build_linear_system(
            corres_source_points, corres_target_points, corres_target_normals, T
        )
        delta = solve(A, b)

        # Update and output
        T = pose2transformation(delta) @ T
        loss = np.mean(b**2)
        print(
            "iter {}: avg loss = {:.4e}, inlier count = {}".format(
                i, loss, len(corres_source_points)
            )
        )

    return T


def main(
    path: str,
    source_idx: int = 10,
    target_idx: int = 50,
    headless: bool = False,
    save_figures: Optional[str] = None,
    interactive: bool = False,
    interactive_both: bool = False,
    save_rrd: Optional[str] = None,
):
    """
    Args:
        path: path to the dataset folder containing rgb/ and depth/
        source_idx: index of the source frame
        target_idx: index of the target frame
        headless: If True, do not spawn the Rerun viewer (useful for logging metrics only).
        save_figures: If set, directory to write before/after Matplotlib PNGs (red=source, green=target).
        interactive: If True, open an Open3D window after ICP (rotate / zoom with the mouse).
        interactive_both: If True, open Open3D for BEFORE then AFTER (close first window to see second).
        save_rrd: If set, write Rerun stream to this .rrd (no GUI). Reopen with: rerun <path>
    """

    with open("intrinsics.json") as f:
        intrinsic = np.array(json.load(f)["intrinsic_matrix"]).reshape(3, 3, order="F")

    depth_path = os.path.join(path, "depth")
    normal_path = os.path.join(path, "normal")

    # TUM convention -- uint16 value to float meters
    depth_scale = 5000.0

    # Source: load depth and rescale to meters
    source_depth = (
        np.asarray(Image.open("{}/{}.png".format(depth_path, source_idx))) / depth_scale
    )

    # Unproject depth to vertex map (H, W, 3) and reshape to a point cloud (H*W, 3)
    source_vertex_map = transforms.unproject(source_depth, intrinsic)
    source_points = source_vertex_map.reshape((-1, 3))

    # Load normal map (H, W, 3) and reshape to point cloud normals (H*W, 3)
    source_normal_map = np.load("{}/{}.npy".format(normal_path, source_idx))
    source_normals = source_normal_map.reshape((-1, 3))

    # Similar preparation for target, but keep the image format for projective association
    target_depth = (
        np.asarray(Image.open("{}/{}.png".format(depth_path, target_idx))) / depth_scale
    )
    target_vertex_map = transforms.unproject(target_depth, intrinsic)
    target_normal_map = np.load("{}/{}.npy".format(normal_path, target_idx))
    tgt_flat = target_vertex_map.reshape((-1, 3))

    log_rr = save_rrd is not None or not headless
    if log_rr:
        utils.init_rerun("icp", headless=headless, save_rrd=save_rrd)

    # Visualize before ICP
    if log_rr:
        rr.set_time("step", sequence=0)
        utils.visualize_icp(source_points, tgt_flat, np.eye(4))

    # TODO: fill-in components in ICP
    T = icp(
        source_points,
        source_normals,
        target_vertex_map,
        target_normal_map,
        intrinsic,
        np.eye(4),
        debug_association=False,
    )

    # Visualize after ICP
    if log_rr:
        rr.set_time("step", sequence=1)
        utils.visualize_icp(source_points, tgt_flat, T)

    tag = "s{}_t{}".format(source_idx, target_idx)
    if interactive_both:
        print(
            "Open3D: BEFORE ICP — drag to rotate, scroll to zoom, Shift+drag to pan. Close window to continue."
        )
        utils.interactive_icp_pair(
            source_points,
            tgt_flat,
            np.eye(4),
            "ICP before | source {} → target {}".format(source_idx, target_idx),
        )
        print("Open3D: AFTER ICP — same controls.")
        utils.interactive_icp_pair(
            source_points,
            tgt_flat,
            T,
            "ICP after | source {} → target {}".format(source_idx, target_idx),
        )
    elif interactive:
        print(
            "Open3D: AFTER ICP — drag to rotate, scroll to zoom, Shift+drag to pan. Close window to exit."
        )
        utils.interactive_icp_pair(
            source_points,
            tgt_flat,
            T,
            "ICP after | source {} → target {}".format(source_idx, target_idx),
        )

    if save_figures is not None:
        utils.save_icp_before_after_panel(
            os.path.join(save_figures, "icp_{}_panel.png".format(tag)),
            source_points,
            tgt_flat,
            T,
            also_save_split=True,
            split_dir=save_figures,
            split_tag=tag,
        )
        print("Saved figures under: {}".format(os.path.abspath(save_figures)))


if __name__ == "__main__":
    tyro.cli(main)
