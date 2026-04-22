import json
import logging
import os
import sys
from pathlib import Path

import cv2
import numpy as np


CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.dirname(CURRENT_DIR)
ROOT_DIR = os.path.dirname(PARENT_DIR)
sys.path.append(ROOT_DIR)

try:
    import pyrealsense2 as rs
    from grasp_module.config.logging_config import configure_grasp_logger
    from grasp_module.config.predictor_config import build_predictor_arg_parser
    from grasp_module.backend.utils.data_utils import (
        CameraInfo,
        build_ply_output_path,
        create_colored_point_cloud_from_rgbd,
        filter_point_cloud_by_z,
        write_open3d_point_cloud,
    )
except ImportError as e:
    print(f"-> Import Error: {e}")
    print(f"-> Current sys.path: {sys.path}")
    sys.exit(1)


logger = logging.getLogger("vision.grasp")


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)
    return path


def save_json(path, payload):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def normalize_depth_shape(depth_img):
    if depth_img is None:
        return depth_img
    if depth_img.ndim == 3 and depth_img.shape[2] == 1:
        return depth_img[:, :, 0]
    return depth_img


def convert_color_frame_to_rgb(color_frame):
    color_data = np.asanyarray(color_frame.get_data())
    stream_format = color_frame.get_profile().as_video_stream_profile().format()
    if stream_format == rs.format.rgb8:
        return color_data
    return cv2.cvtColor(color_data, cv2.COLOR_BGR2RGB)


def export_depth_metadata(profile, raw_depth_frame, raw_color_frame, output_path):
    depth_intrinsics = raw_depth_frame.get_profile().as_video_stream_profile().get_intrinsics()
    color_intrinsics = raw_color_frame.get_profile().as_video_stream_profile().get_intrinsics()
    depth_scale = profile.get_device().first_depth_sensor().get_depth_scale()
    factor_depth = 1.0 / depth_scale if depth_scale > 0 else 1000.0

    metadata = {
        "camera_type": "realsense_bag",
        "align_mode": "depth",
        "depth": {
            "width": depth_intrinsics.width,
            "height": depth_intrinsics.height,
            "fx": depth_intrinsics.fx,
            "fy": depth_intrinsics.fy,
            "cx": depth_intrinsics.ppx,
            "cy": depth_intrinsics.ppy,
            "model": str(depth_intrinsics.model),
            "coeffs": list(depth_intrinsics.coeffs),
        },
        "color": {
            "width": color_intrinsics.width,
            "height": color_intrinsics.height,
            "fx": color_intrinsics.fx,
            "fy": color_intrinsics.fy,
            "cx": color_intrinsics.ppx,
            "cy": color_intrinsics.ppy,
            "model": str(color_intrinsics.model),
            "coeffs": list(color_intrinsics.coeffs),
        },
        "depth_scale": depth_scale,
        "factor_depth": factor_depth,
        "description": "Camera intrinsics exported from bag for hand-eye debug",
    }
    save_json(output_path, metadata)
    return metadata


def build_camera_info_from_metadata(metadata):
    depth = metadata["depth"]
    factor_depth = metadata.get("factor_depth", 1000.0)
    return CameraInfo(
        width=float(depth["width"]),
        height=float(depth["height"]),
        fx=float(depth["fx"]),
        fy=float(depth["fy"]),
        cx=float(depth["cx"]),
        cy=float(depth["cy"]),
        scale=float(factor_depth),
    )


def compute_scene_stats(points):
    if points.size == 0:
        return None

    min_xyz = points.min(axis=0)
    max_xyz = points.max(axis=0)
    mean_xyz = points.mean(axis=0)
    median_xyz = np.median(points, axis=0)
    bbox_center_xyz = 0.5 * (min_xyz + max_xyz)
    return {
        "point_count": int(points.shape[0]),
        "min_xyz_m": [float(v) for v in min_xyz],
        "max_xyz_m": [float(v) for v in max_xyz],
        "mean_xyz_m": [float(v) for v in mean_xyz],
        "median_xyz_m": [float(v) for v in median_xyz],
        "bbox_center_xyz_m": [float(v) for v in bbox_center_xyz],
    }


def voxel_downsample_points(points, colors=None, voxel_size=0.01):
    if voxel_size <= 0 or points.size == 0:
        return points, colors

    import open3d as o3d

    cloud = o3d.geometry.PointCloud()
    cloud.points = o3d.utility.Vector3dVector(points.astype(np.float32))
    if colors is not None:
        cloud.colors = o3d.utility.Vector3dVector(colors.astype(np.float32))
    cloud = cloud.voxel_down_sample(float(voxel_size))
    down_points = np.asarray(cloud.points, dtype=np.float32)
    down_colors = None
    if colors is not None:
        down_colors = np.asarray(cloud.colors, dtype=np.float32)
    return down_points, down_colors


def collect_bag_candidates(cfgs):
    output_dir = Path(cfgs.output_dir)
    ensure_dir(str(output_dir))

    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_device_from_file(cfgs.bag_file, repeat_playback=False)
    profile = pipeline.start(config)
    playback = profile.get_device().as_playback()
    playback.set_real_time(False)
    align = rs.align(rs.stream.depth)

    metadata = None
    camera_info = None
    total_frames = 0
    sampled_frames = 0
    candidates = []

    try:
        while True:
            try:
                frames = pipeline.wait_for_frames()
            except RuntimeError:
                break

            total_frames += 1
            if cfgs.bag_stride > 1 and (total_frames - 1) % cfgs.bag_stride != 0:
                continue

            raw_depth_frame = frames.get_depth_frame()
            raw_color_frame = frames.get_color_frame()
            aligned_frames = align.process(frames)
            aligned_depth_frame = aligned_frames.get_depth_frame()
            aligned_color_frame = aligned_frames.get_color_frame()
            if not raw_depth_frame or not raw_color_frame or not aligned_depth_frame or not aligned_color_frame:
                continue

            if metadata is None:
                metadata = export_depth_metadata(
                    profile,
                    raw_depth_frame,
                    raw_color_frame,
                    str(output_dir / "camera_metadata.json"),
                )
                camera_info = build_camera_info_from_metadata(metadata)

            color_rgb = convert_color_frame_to_rgb(aligned_color_frame)
            depth_img = normalize_depth_shape(np.asanyarray(aligned_depth_frame.get_data()))
            if depth_img is None or depth_img.ndim != 2:
                continue

            zero_count = int((depth_img == 0).sum())
            valid_ratio = float((depth_img > 0).sum() / depth_img.size)
            sampled_frames += 1
            if valid_ratio < cfgs.bag_min_valid_ratio:
                if cfgs.bag_max_frames > 0 and sampled_frames >= cfgs.bag_max_frames:
                    break
                continue

            points, colors = create_colored_point_cloud_from_rgbd(color_rgb, depth_img, camera_info, mask=None)
            if points.size == 0:
                if cfgs.bag_max_frames > 0 and sampled_frames >= cfgs.bag_max_frames:
                    break
                continue

            filtered_points, filtered_colors = filter_point_cloud_by_z(
                points,
                colors,
                z_min=cfgs.z_min,
                z_max=cfgs.z_max,
            )
            browse_points, browse_colors = voxel_downsample_points(
                filtered_points,
                filtered_colors,
                voxel_size=cfgs.browse_voxel_size,
            )

            candidates.append({
                "frame_index": total_frames - 1,
                "color_rgb": color_rgb,
                "depth_img": depth_img,
                "zero_count": zero_count,
                "valid_ratio": valid_ratio,
                "points": points,
                "colors": colors,
                "scene_stats_raw": compute_scene_stats(points),
                "filtered_points": filtered_points,
                "filtered_colors": filtered_colors,
                "scene_stats_filtered": compute_scene_stats(filtered_points),
                "browse_points": browse_points,
                "browse_colors": browse_colors,
                "scene_stats_browse": compute_scene_stats(browse_points),
            })
            if cfgs.bag_max_frames > 0 and sampled_frames >= cfgs.bag_max_frames:
                break
    finally:
        pipeline.stop()

    if metadata is None or camera_info is None:
        raise RuntimeError(f"Failed to decode any valid frame from bag: {cfgs.bag_file}")

    candidates.sort(
        key=lambda item: (
            item["zero_count"],
            -item["valid_ratio"],
            item["frame_index"],
        )
    )
    return candidates[:cfgs.bag_top_k], metadata, camera_info


def save_frame_outputs(frame_dir, frame_result):
    ensure_dir(frame_dir)
    cv2.imwrite(os.path.join(frame_dir, "color.png"), cv2.cvtColor(frame_result["color_rgb"], cv2.COLOR_RGB2BGR))
    cv2.imwrite(os.path.join(frame_dir, "depth_raw.png"), frame_result["depth_img"])
    raw_cloud_path = build_ply_output_path(frame_dir, "scene_cloud_raw.ply")
    filtered_cloud_path = build_ply_output_path(frame_dir, "scene_cloud_filtered.ply")
    write_open3d_point_cloud(raw_cloud_path, frame_result["points"], frame_result["colors"])
    write_open3d_point_cloud(filtered_cloud_path, frame_result["filtered_points"], frame_result["filtered_colors"])

    browse_cloud_path = None
    if frame_result["browse_points"].size > 0:
        browse_cloud_path = build_ply_output_path(frame_dir, "scene_cloud_browse.ply")
        write_open3d_point_cloud(browse_cloud_path, frame_result["browse_points"], frame_result["browse_colors"])

    return raw_cloud_path, filtered_cloud_path, browse_cloud_path


def build_frame_summary(frame_result, raw_cloud_path, filtered_cloud_path, browse_cloud_path, cfgs):
    summary = {
        "frame_index": int(frame_result["frame_index"]),
        "zero_count": int(frame_result["zero_count"]),
        "valid_ratio": float(frame_result["valid_ratio"]),
        "z_filter_m": {
            "z_min": None if cfgs.z_min is None else float(cfgs.z_min),
            "z_max": None if cfgs.z_max is None else float(cfgs.z_max),
        },
        "browse_voxel_size_m": float(cfgs.browse_voxel_size),
        "scene_stats_raw": frame_result["scene_stats_raw"],
        "scene_stats_filtered": frame_result["scene_stats_filtered"],
        "scene_stats_browse": frame_result["scene_stats_browse"],
        "scene_cloud_raw_ply": raw_cloud_path,
        "scene_cloud_filtered_ply": filtered_cloud_path,
        "scene_cloud_browse_ply": browse_cloud_path,
    }
    return summary


def main():
    configure_grasp_logger(level=logging.INFO)

    default_overrides = {
        "yolo_weights_dir": os.path.join(PARENT_DIR, "weights"),
        "camera_metadata": os.path.join(PARENT_DIR, "config", "realsense_metadata.json"),
    }
    parser = build_predictor_arg_parser(
        description="Export aligned RGB-D frames and full scene point clouds from RealSense bag",
        default_overrides=default_overrides,
    )
    parser.add_argument("--bag_file", type=str, required=True, help="Path to RealSense .bag file")
    parser.add_argument("--output_dir", type=str, default="", help="Output directory for selected frames and summaries")
    parser.add_argument("--bag_top_k", type=int, default=5, help="Number of best bag frames to export")
    parser.add_argument("--bag_stride", type=int, default=5, help="Inspect every Nth frame from the bag")
    parser.add_argument("--bag_max_frames", type=int, default=80, help="Maximum sampled frames to inspect; <=0 means read all")
    parser.add_argument("--bag_min_valid_ratio", type=float, default=0.0, help="Discard frames whose non-zero depth ratio is lower than this threshold")
    parser.add_argument("--z_min", type=float, default=0.15, help="Minimum depth in meters to keep in exported point clouds")
    parser.add_argument("--z_max", type=float, default=2.0, help="Maximum depth in meters to keep in exported point clouds")
    parser.add_argument("--browse_voxel_size", type=float, default=0.01, help="Voxel size in meters for additional browse-friendly downsampled cloud; <=0 disables")
    cfgs = parser.parse_args()

    if not cfgs.output_dir:
        cfgs.output_dir = os.path.join(CURRENT_DIR, "handeye_debug", Path(cfgs.bag_file).stem)
    ensure_dir(cfgs.output_dir)

    selected_frames, metadata, _camera_info = collect_bag_candidates(cfgs)
    if not selected_frames:
        raise RuntimeError("No valid frames found in the bag after depth filtering.")

    summary = {
        "bag_file": cfgs.bag_file,
        "camera_metadata_path": os.path.join(cfgs.output_dir, "camera_metadata.json"),
        "selected_frames": [],
    }

    for rank, frame_result in enumerate(selected_frames, start=1):
        frame_name = f"rank_{rank:02d}_frame_{frame_result['frame_index']:05d}"
        frame_dir = ensure_dir(os.path.join(cfgs.output_dir, frame_name))
        raw_cloud_path, filtered_cloud_path, browse_cloud_path = save_frame_outputs(frame_dir, frame_result)
        frame_summary = build_frame_summary(
            frame_result,
            raw_cloud_path,
            filtered_cloud_path,
            browse_cloud_path,
            cfgs,
        )
        save_json(os.path.join(frame_dir, "summary.json"), frame_summary)
        summary["selected_frames"].append(frame_summary)
        logger.info(
            "[%s] zero_count=%s valid_ratio=%.6f raw_points=%s filtered_points=%s",
            frame_name,
            frame_summary["zero_count"],
            frame_summary["valid_ratio"],
            frame_summary["scene_stats_raw"]["point_count"] if frame_summary["scene_stats_raw"] else 0,
            frame_summary["scene_stats_filtered"]["point_count"] if frame_summary["scene_stats_filtered"] else 0,
        )

    save_json(os.path.join(cfgs.output_dir, "summary.json"), summary)
    logger.info("Bag point-cloud summary saved to: %s", os.path.join(cfgs.output_dir, "summary.json"))


if __name__ == "__main__":
    main()
