'''
    Initially written by Ming Hsiao in MATLAB
    Redesigned and rewritten by Wei Dong (weidong@andrew.cmu.edu)
'''

import json
import os
from typing import Optional

import numpy as np
import rerun as rr
import rerun.blueprint as rrb
import transforms
import tyro
import utils
from fusion import Map
from icp import icp
from PIL import Image
from preprocess import load_gt_poses


def main(
    path: str,
    start_idx: int = 1,
    end_idx: int = 200,
    downsample_factor: int = 2,
    headless: bool = False,
    save_rrd: Optional[str] = None,
):
    """
    Args:
        path: path to the dataset folder containing rgb/ and depth/
        start_idx: start frame index
        end_idx: end frame index
        downsample_factor: spatial downsample factor
        headless: If True, do not spawn the Rerun viewer (ignored if save_rrd is set).
        save_rrd: If set, stream Rerun data to this .rrd file (no GUI). Reopen with: rerun <path>
    """
    with open('intrinsics.json') as f:
        intrinsic = np.array(json.load(f)['intrinsic_matrix']).reshape(3, 3, order='F')
    frame_ids, gt_T_list = load_gt_poses(
        os.path.join(path, 'livingRoom2.gt.freiburg'))
    pose_by_frame = {int(fid): T for fid, T in zip(frame_ids, gt_T_list)}

    rgb_path = os.path.join(path, 'rgb')
    depth_path = os.path.join(path, 'depth')
    normal_path = os.path.join(path, 'normal')

    # TUM convention
    depth_scale = 5000.0

    blueprint = rrb.Blueprint(
        rrb.Spatial3DView(
            origin='/',
            overrides={'world/map/normals': rrb.EntityBehavior(visible=False)},
        )
    )
    rr_on = save_rrd is not None or not headless
    if rr_on:
        utils.init_rerun(
            'icp_fusion',
            default_blueprint=blueprint,
            headless=headless,
            save_rrd=save_rrd,
        )
    elif headless:
        print('Rerun disabled (use --save-rrd PATH.rrd to record without a viewer).')

    # Rotate from Y-down/Z-forward (camera convention) to Z-up for visualization
    R_z_up = np.array([[1, 0, 0], [0, 0, 1], [0, -1, 0]], dtype=float)

    m = Map()

    down_factor = downsample_factor
    intrinsic = intrinsic.copy()
    intrinsic /= down_factor
    intrinsic[2, 2] = 1

    # First frame: use GT pose for that frame only (do not use other GT poses here).
    if start_idx not in pose_by_frame:
        raise KeyError(
            'start_idx {} not in ground-truth pose file'.format(start_idx))
    T_cam_to_world = pose_by_frame[start_idx].copy()

    H, W = int(480 / down_factor), int(680 / down_factor)
    if rr_on:
        rr.log('world/camera/image', rr.Pinhole(
            image_from_camera=intrinsic,
            width=W,
            height=H,
            camera_xyz=rr.ViewCoordinates.RDF,
        ), static=True)

    traj_gt = []
    traj_est = []
    trans_errors_m = []

    for i in range(start_idx, end_idx + 1):
        if i not in pose_by_frame:
            print('skip frame {} (no GT entry)'.format(i))
            continue
        print('loading frame {}'.format(i))
        if rr_on:
            rr.set_time('frame', sequence=i)

        depth = np.asarray(Image.open('{}/{}.png'.format(depth_path, i))) / depth_scale
        depth = depth[::down_factor, ::down_factor]
        vertex_map = transforms.unproject(depth, intrinsic)

        color_map = np.asarray(Image.open('{}/{}.png'.format(rgb_path, i))).astype(float) / 255.0
        color_map = color_map[::down_factor, ::down_factor]

        normal_map = np.load('{}/{}.npy'.format(normal_path, i))
        normal_map = normal_map[::down_factor, ::down_factor]

        if i > start_idx:
            print('Frame-to-model ICP')
            T_world_to_cam = np.linalg.inv(T_cam_to_world)
            T_world_to_cam = icp(
                m.points[::down_factor],
                m.normals[::down_factor],
                vertex_map,
                normal_map,
                intrinsic,
                T_world_to_cam,
                debug_association=False,
            )
            T_cam_to_world = np.linalg.inv(T_world_to_cam)

        print('Point-based fusion')
        m.fuse(vertex_map, normal_map, color_map, intrinsic, T_cam_to_world)

        T_gt = pose_by_frame[i]
        p_gt = R_z_up @ T_gt[:3, 3]
        p_est = R_z_up @ T_cam_to_world[:3, 3]
        traj_gt.append(p_gt)
        traj_est.append(p_est)
        trans_errors_m.append(float(np.linalg.norm(p_est - p_gt)))

        if rr_on and len(traj_gt) > 1:
            rr.log('world/trajectory/gt', rr.LineStrips3D([traj_gt], colors=[0, 255, 0]))
            rr.log('world/trajectory/est', rr.LineStrips3D([traj_est], colors=[255, 0, 0]))

        if rr_on:
            points_viz = (R_z_up @ m.points.T).T
            normals_viz = (R_z_up @ m.normals.T).T
            rr.log('world/map', rr.Points3D(
                points_viz,
                colors=(m.colors * 255).astype(np.uint8),
            ))
            rr.log('world/map/normals', rr.Arrows3D(
                vectors=normals_viz * 0.02,
                origins=points_viz,
            ))
            rr.log('world/camera', rr.Transform3D(
                translation=R_z_up @ T_cam_to_world[:3, 3],
                mat3x3=R_z_up @ T_cam_to_world[:3, :3],
            ))

        print(
            '  trans err vs GT (Z-up viz): {:.4f} m | mean so far: {:.4f} m'.format(
                trans_errors_m[-1], float(np.mean(trans_errors_m))))

    if trans_errors_m:
        print(
            'Trajectory summary: mean abs trans error = {:.4f} m, max = {:.4f} m, frames = {}'.format(
                float(np.mean(trans_errors_m)),
                float(np.max(trans_errors_m)),
                len(trans_errors_m),
            ))


if __name__ == '__main__':
    tyro.cli(main)
