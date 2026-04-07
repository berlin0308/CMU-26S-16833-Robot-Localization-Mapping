'''
    Initially written by Ming Hsiao in MATLAB
    Redesigned and rewritten by Wei Dong (weidong@andrew.cmu.edu)
'''

import os
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import rerun as rr
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 — registers 3d projection


def init_rerun(
    app_id: str,
    *,
    default_blueprint=None,
    headless: bool = False,
    save_rrd: Optional[str] = None,
) -> None:
    """
    If save_rrd is set, log to a .rrd file (no live viewer). Reopen later with:
        rerun path/to/file.rrd
    Per Rerun docs, rr.save() must run before any rr.log().
    """
    if save_rrd:
        parent = os.path.dirname(os.path.abspath(save_rrd))
        if parent:
            os.makedirs(parent, exist_ok=True)
        if default_blueprint is not None:
            rr.init(app_id, spawn=False, default_blueprint=default_blueprint)
            rr.save(save_rrd, default_blueprint=default_blueprint)
        else:
            rr.init(app_id, spawn=False)
            rr.save(save_rrd)
        print("Rerun: streaming to {} — reopen with: rerun {}".format(save_rrd, save_rrd))
    elif default_blueprint is not None:
        rr.init(app_id, spawn=not headless, default_blueprint=default_blueprint)
    else:
        rr.init(app_id, spawn=not headless)


def save_icp_png(
    output_path: str,
    source_points: np.ndarray,
    target_points: np.ndarray,
    T: np.ndarray,
    title: str,
    max_points: int = 65000,
    seed: int = 0,
):
    """
    Save a static 3D scatter (source=red, target=green) for report figures.
    Downsamples dense RGB-D clouds for file size and clarity.
    """
    R = T[:3, :3]
    t = T[:3, 3:]
    transformed_source = (R @ source_points.T + t).T

    rng = np.random.default_rng(seed)
    src = transformed_source
    tgt = target_points
    if len(src) > max_points:
        idx = rng.choice(len(src), size=max_points, replace=False)
        src = src[idx]
    if len(tgt) > max_points:
        idx = rng.choice(len(tgt), size=max_points, replace=False)
        tgt = tgt[idx]

    fig = plt.figure(figsize=(9, 7))
    ax = fig.add_subplot(111, projection="3d")
    ax.scatter(
        src[:, 0],
        src[:, 1],
        src[:, 2],
        c="red",
        s=1,
        alpha=0.35,
        label="source (transformed)",
    )
    ax.scatter(
        tgt[:, 0],
        tgt[:, 1],
        tgt[:, 2],
        c="green",
        s=1,
        alpha=0.35,
        label="target",
    )
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_zlabel("Z (m)")
    ax.set_title(title)
    ax.legend(markerscale=6, loc="upper right")

    all_pts = np.vstack([src, tgt])
    center = all_pts.mean(axis=0)
    max_range = (all_pts.max(axis=0) - all_pts.min(axis=0)).max() / 2.0
    ax.set_xlim(center[0] - max_range, center[0] + max_range)
    ax.set_ylim(center[1] - max_range, center[1] + max_range)
    ax.set_zlim(center[2] - max_range, center[2] + max_range)

    # Top-down view: look along -Z onto the X–Y plane (camera frame: X right, Y down, Z forward).
    ax.view_init(elev=90, azim=-90)
    try:
        ax.set_proj_type("ortho")
    except (AttributeError, ValueError):
        pass

    out_dir = os.path.dirname(os.path.abspath(output_path))
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    plt.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def _prepare_icp_panel_clouds(
    source_points: np.ndarray,
    target_points: np.ndarray,
    T_after: np.ndarray,
    *,
    max_points: int = 100000,
    seed: int = 0,
):
    rng = np.random.default_rng(seed)
    n_src = len(source_points)
    n_tgt = len(target_points)
    if n_src > max_points:
        idx_s = rng.choice(n_src, size=max_points, replace=False)
        src = source_points[idx_s]
    else:
        src = np.asarray(source_points, dtype=np.float64)
    if n_tgt > max_points:
        idx_t = rng.choice(n_tgt, size=max_points, replace=False)
        tgt = target_points[idx_t]
    else:
        tgt = np.asarray(target_points, dtype=np.float64)

    R = T_after[:3, :3]
    t = T_after[:3, 3:]
    src_after = (R @ src.T + t).T
    src_before = src.copy()

    all_pts = np.vstack([src_before, tgt, src_after])
    center = all_pts.mean(axis=0)
    max_range = (all_pts.max(axis=0) - all_pts.min(axis=0)).max() / 2.0
    if max_range < 1e-6:
        max_range = 1.0
    xlim = (center[0] - max_range, center[0] + max_range)
    ylim = (center[1] - max_range, center[1] + max_range)
    zlim = (center[2] - max_range, center[2] + max_range)
    return src_before, tgt, src_after, xlim, ylim, zlim


def _style_icp_axis(ax, xlim, ylim, zlim, elev, azim):
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.set_zlim(zlim)
    try:
        ax.set_box_aspect((1, 1, 1))
    except (AttributeError, ValueError):
        pass
    try:
        ax.set_proj_type("persp")
    except (AttributeError, ValueError):
        pass
    ax.view_init(elev=elev, azim=azim)
    ax.set_xlabel("X (m)", labelpad=-5)
    ax.set_ylabel("Y (m)", labelpad=-5)
    ax.set_zlabel("Z (m)", labelpad=-5)
    ax.grid(False)
    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False


def save_icp_before_after_panel(
    output_path: str,
    source_points: np.ndarray,
    target_points: np.ndarray,
    T_after: np.ndarray,
    *,
    max_points: int = 100000,
    seed: int = 0,
    elev: float = 14.0,
    azim: float = -66.0,
    also_save_split: bool = False,
    split_dir: str = "",
    split_tag: str = "",
):
    """
    Homework-style figure: (a) before ICP, (b) after ICP, side by side.
    Red = source (transformed by current T), green = target. Perspective view.
    If also_save_split, writes {split_dir}/icp_{split_tag}_before.png and _after.png
    with the same camera as (a)/(b).
    """
    src_before, tgt, src_after, xlim, ylim, zlim = _prepare_icp_panel_clouds(
        source_points, target_points, T_after, max_points=max_points, seed=seed
    )

    kw = dict(s=2.2, alpha=0.52, depthshade=True, linewidths=0, edgecolors="none")

    fig, (ax1, ax2) = plt.subplots(
        1,
        2,
        figsize=(15.5, 6.8),
        subplot_kw={"projection": "3d"},
        facecolor="white",
    )

    ax1.scatter(src_before[:, 0], src_before[:, 1], src_before[:, 2], c="red", **kw)
    ax1.scatter(tgt[:, 0], tgt[:, 1], tgt[:, 2], c="green", **kw)
    ax1.set_title("(a) Point clouds before registration", fontsize=12, pad=10)
    _style_icp_axis(ax1, xlim, ylim, zlim, elev, azim)

    ax2.scatter(src_after[:, 0], src_after[:, 1], src_after[:, 2], c="red", **kw)
    ax2.scatter(tgt[:, 0], tgt[:, 1], tgt[:, 2], c="green", **kw)
    ax2.set_title("(b) Point clouds after registration", fontsize=12, pad=10)
    _style_icp_axis(ax2, xlim, ylim, zlim, elev, azim)

    out_dir = os.path.dirname(os.path.abspath(output_path))
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    plt.tight_layout(w_pad=2.0)
    fig.savefig(output_path, dpi=220, bbox_inches="tight", facecolor="white")
    plt.close(fig)

    if also_save_split and split_dir and split_tag:
        base = os.path.join(split_dir, "icp_{}".format(split_tag))
        for path, src, title in (
            (
                "{}_before.png".format(base),
                src_before,
                "(a) Point clouds before registration",
            ),
            (
                "{}_after.png".format(base),
                src_after,
                "(b) Point clouds after registration",
            ),
        ):
            fig = plt.figure(figsize=(7.8, 6.8), facecolor="white")
            ax = fig.add_subplot(111, projection="3d")
            ax.scatter(src[:, 0], src[:, 1], src[:, 2], c="red", **kw)
            ax.scatter(tgt[:, 0], tgt[:, 1], tgt[:, 2], c="green", **kw)
            ax.set_title(title, fontsize=12, pad=10)
            _style_icp_axis(ax, xlim, ylim, zlim, elev, azim)
            sd = os.path.dirname(os.path.abspath(path))
            if sd:
                os.makedirs(sd, exist_ok=True)
            plt.tight_layout()
            fig.savefig(path, dpi=220, bbox_inches="tight", facecolor="white")
            plt.close(fig)


def interactive_icp_pair(
    source_points: np.ndarray,
    target_points: np.ndarray,
    T: np.ndarray,
    window_name: str,
    *,
    max_points: int = 250000,
    seed: int = 0,
    point_size: float = 2.5,
):
    """
    Open3D window: red = transformed source, green = target. Mouse drag to rotate,
    scroll to zoom, shift+drag to pan. Close the window to return.
    """
    import open3d as o3d

    rng = np.random.default_rng(seed)
    src = np.asarray(source_points, dtype=np.float64)
    tgt = np.asarray(target_points, dtype=np.float64)
    if len(src) > max_points:
        src = src[rng.choice(len(src), size=max_points, replace=False)]
    if len(tgt) > max_points:
        tgt = tgt[rng.choice(len(tgt), size=max_points, replace=False)]

    R = T[:3, :3]
    t = T[:3, 3:]
    src_t = (R @ src.T + t).T

    pcd_s = o3d.geometry.PointCloud()
    pcd_s.points = o3d.utility.Vector3dVector(src_t)
    pcd_s.paint_uniform_color([1.0, 0.05, 0.05])

    pcd_t = o3d.geometry.PointCloud()
    pcd_t.points = o3d.utility.Vector3dVector(tgt)
    pcd_t.paint_uniform_color([0.05, 0.85, 0.1])

    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name=window_name, width=1400, height=900)
    ro = vis.get_render_option()
    ro.point_size = float(point_size)
    ro.background_color = np.asarray([1.0, 1.0, 1.0])
    vis.add_geometry(pcd_s)
    vis.add_geometry(pcd_t)
    ctr = vis.get_view_control()
    try:
        ctr.set_zoom(0.55)
    except (AttributeError, RuntimeError):
        pass
    vis.run()
    vis.destroy_window()


def visualize_icp(source_points, target_points, T):
    R = T[:3, :3]
    t = T[:3, 3:]
    transformed_source = (R @ source_points.T + t).T

    rr.log('icp/source', rr.Points3D(transformed_source, colors=[255, 0, 0]))
    rr.log('icp/target', rr.Points3D(target_points, colors=[0, 255, 0]))


def visualize_correspondences(source_points, target_points, T):
    if len(source_points) != len(target_points):
        print(
            'Error! source points and target points has different length {} vs {}'
            .format(len(source_points), len(target_points)))
        return

    R = T[:3, :3]
    t = T[:3, 3:]
    transformed_source = (R @ source_points.T + t).T

    rr.log('icp/source', rr.Points3D(transformed_source, colors=[255, 0, 0]))
    rr.log('icp/target', rr.Points3D(target_points, colors=[0, 255, 0]))

    lines = np.stack([transformed_source, target_points], axis=1)
    rr.log('icp/correspondences', rr.LineStrips3D(lines))
