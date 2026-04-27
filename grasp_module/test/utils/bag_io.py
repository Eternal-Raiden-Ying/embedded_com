from pathlib import Path

import cv2
import numpy as np

from grasp_module.backend.utils.data_utils import (
    CameraInfo,
    build_ply_output_path,
    create_colored_point_cloud_from_rgbd,
    filter_point_cloud_by_z,
    write_open3d_point_cloud,
)

from .io_utils import ensure_dir, save_json


def normalize_depth_shape(depth_img):
    if depth_img is None:
        return depth_img
    if depth_img.ndim == 3 and depth_img.shape[2] == 1:
        return depth_img[:, :, 0]
    return depth_img


def sanitize_depth_image(depth_img, depth_min_mm=1, depth_max_mm=2000):
    depth_img = normalize_depth_shape(depth_img)
    if depth_img is None:
        return depth_img

    depth = depth_img.astype(np.uint16, copy=True)
    valid_mask = depth > 0
    if depth_min_mm is not None:
        valid_mask &= depth >= int(depth_min_mm)
    if depth_max_mm is not None:
        valid_mask &= depth <= int(depth_max_mm)
    depth[~valid_mask] = 0
    return depth


def _fill_zero_holes_with_median(depth_img, kernel_size=5, iterations=1):
    if kernel_size <= 1 or iterations <= 0:
        return depth_img

    filled = depth_img.astype(np.uint16, copy=True)
    for _ in range(iterations):
        candidate = cv2.medianBlur(filled, kernel_size)
        hole_mask = (filled == 0) & (candidate > 0)
        if not np.any(hole_mask):
            break
        filled[hole_mask] = candidate[hole_mask]
    return filled


def postprocess_depth_image(depth_img, cfgs):
    depth = sanitize_depth_image(
        depth_img,
        depth_min_mm=getattr(cfgs, "depth_min_mm", 1),
        depth_max_mm=getattr(cfgs, "depth_max_mm", 2000),
    )

    if not getattr(cfgs, "depth_postprocess", True):
        return depth

    smooth_kernel = int(getattr(cfgs, "depth_smooth_kernel", 5))
    hole_fill_kernel = int(getattr(cfgs, "depth_hole_fill_kernel", 5))
    hole_fill_iterations = int(getattr(cfgs, "depth_hole_fill_iterations", 2))

    if smooth_kernel % 2 == 0:
        smooth_kernel += 1
    if hole_fill_kernel % 2 == 0:
        hole_fill_kernel += 1

    processed = depth.copy()
    if smooth_kernel > 1:
        smoothed = cv2.medianBlur(processed, smooth_kernel)
        processed = np.where(processed > 0, smoothed, 0).astype(np.uint16)

    processed = _fill_zero_holes_with_median(
        processed,
        kernel_size=hole_fill_kernel,
        iterations=hole_fill_iterations,
    )
    processed = sanitize_depth_image(
        processed,
        depth_min_mm=getattr(cfgs, "depth_min_mm", 1),
        depth_max_mm=getattr(cfgs, "depth_max_mm", 2000),
    )
    return processed


def convert_color_frame_to_rgb(rs_module, color_frame):
    color_data = np.asanyarray(color_frame.get_data())
    stream_format = color_frame.get_profile().as_video_stream_profile().format()
    if stream_format == rs_module.format.rgb8:
        return color_data
    return cv2.cvtColor(color_data, cv2.COLOR_BGR2RGB)


def export_depth_metadata(profile, raw_depth_frame, raw_color_frame, output_path, description):
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
        "description": description,
    }
    save_json(output_path, metadata)
    return output_path, metadata


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


def _collect_aligned_bag_frames(cfgs, output_dir, metadata_description):
    try:
        import pyrealsense2 as rs
    except ImportError as exc:
        raise RuntimeError("pyrealsense2 is required for --bag_file mode") from exc

    output_dir = Path(output_dir)
    ensure_dir(str(output_dir))

    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_device_from_file(cfgs.bag_file, repeat_playback=False)
    profile = pipeline.start(config)
    playback = profile.get_device().as_playback()
    playback.set_real_time(False)
    align = rs.align(rs.stream.depth)

    metadata_path = None
    metadata = None
    total_frames = 0
    sampled_frames = 0
    frames_out = []

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

            if metadata_path is None:
                metadata_path, metadata = export_depth_metadata(
                    profile,
                    raw_depth_frame,
                    raw_color_frame,
                    str(output_dir / "camera_metadata.json"),
                    metadata_description,
                )

            color_img = convert_color_frame_to_rgb(rs, aligned_color_frame)
            raw_depth_img = normalize_depth_shape(np.asanyarray(aligned_depth_frame.get_data()))
            if raw_depth_img is None or raw_depth_img.ndim != 2:
                continue

            filtered_depth_img = sanitize_depth_image(
                raw_depth_img,
                depth_min_mm=getattr(cfgs, "depth_min_mm", 1),
                depth_max_mm=getattr(cfgs, "depth_max_mm", 2000),
            )
            postprocessed_depth_img = postprocess_depth_image(filtered_depth_img, cfgs)

            sampled_frames += 1
            zero_count = int((filtered_depth_img == 0).sum())
            valid_ratio = float((filtered_depth_img > 0).sum() / filtered_depth_img.size)
            if valid_ratio < cfgs.bag_min_valid_ratio:
                if cfgs.bag_max_frames > 0 and sampled_frames >= cfgs.bag_max_frames:
                    break
                continue

            frames_out.append(
                {
                    "frame_index": total_frames - 1,
                    "color_img": color_img.copy(),
                    "depth_raw_img": raw_depth_img.copy(),
                    "depth_filtered_img": filtered_depth_img.copy(),
                    "depth_postprocessed_img": postprocessed_depth_img.copy(),
                    "depth_img": postprocessed_depth_img.copy(),
                    "zero_count": zero_count,
                    "valid_ratio": valid_ratio,
                }
            )
            if cfgs.bag_max_frames > 0 and sampled_frames >= cfgs.bag_max_frames:
                break
    finally:
        pipeline.stop()

    if metadata_path is None or metadata is None:
        raise RuntimeError(f"Failed to read valid RGB-D frames from bag: {cfgs.bag_file}")

    frames_out.sort(key=lambda item: (item["zero_count"], -item["valid_ratio"], item["frame_index"]))
    return frames_out, metadata_path, metadata


def collect_bag_frames(cfgs, output_dir, metadata_description="Camera intrinsics exported from bag for engine debug"):
    frames_out, metadata_path, _ = _collect_aligned_bag_frames(cfgs, output_dir, metadata_description)
    return frames_out[: cfgs.bag_top_k], metadata_path


def collect_bag_candidates(cfgs, output_dir, metadata_description="Camera intrinsics exported from bag for hand-eye debug"):
    frames_out, metadata_path, metadata = _collect_aligned_bag_frames(cfgs, output_dir, metadata_description)
    camera_info = build_camera_info_from_metadata(metadata)
    candidates = []
    for item in frames_out:
        filtered_points, filtered_colors = create_colored_point_cloud_from_rgbd(
            item["color_img"],
            item["depth_filtered_img"],
            camera_info,
            mask=None,
        )
        postprocessed_points, postprocessed_colors = create_colored_point_cloud_from_rgbd(
            item["color_img"],
            item["depth_postprocessed_img"],
            camera_info,
            mask=None,
        )
        if filtered_points.size == 0 and postprocessed_points.size == 0:
            continue

        filtered_points, filtered_colors = filter_point_cloud_by_z(
            filtered_points,
            filtered_colors,
            z_min=cfgs.z_min,
            z_max=cfgs.z_max,
        )
        postprocessed_points, postprocessed_colors = filter_point_cloud_by_z(
            postprocessed_points,
            postprocessed_colors,
            z_min=cfgs.z_min,
            z_max=cfgs.z_max,
        )

        candidates.append(
            {
                **item,
                "filtered_points": filtered_points,
                "filtered_colors": filtered_colors,
                "scene_stats_filtered": compute_scene_stats(filtered_points),
                "postprocessed_points": postprocessed_points,
                "postprocessed_colors": postprocessed_colors,
                "scene_stats_postprocessed": compute_scene_stats(postprocessed_points),
            }
        )
        if len(candidates) >= cfgs.bag_top_k:
            break
    return candidates, metadata_path, camera_info


def save_bag_frame_inputs(frame_dir, color_img, depth_img):
    color_path = str(Path(frame_dir) / "color.png")
    depth_path = str(Path(frame_dir) / "depth_raw.png")
    cv2.imwrite(color_path, cv2.cvtColor(color_img, cv2.COLOR_RGB2BGR))
    cv2.imwrite(depth_path, depth_img)
    return color_path, depth_path


def save_depth_variants(frame_dir, depth_filtered_img, depth_postprocessed_img):
    filtered_path = str(Path(frame_dir) / "depth_filtered.png")
    postprocessed_path = str(Path(frame_dir) / "depth_postprocessed.png")
    cv2.imwrite(filtered_path, depth_filtered_img)
    cv2.imwrite(postprocessed_path, depth_postprocessed_img)
    return filtered_path, postprocessed_path


def save_point_cloud_frame_outputs(frame_dir, frame_result):
    ensure_dir(frame_dir)
    cv2.imwrite(str(Path(frame_dir) / "color.png"), cv2.cvtColor(frame_result["color_img"], cv2.COLOR_RGB2BGR))
    cv2.imwrite(str(Path(frame_dir) / "depth_raw.png"), frame_result["depth_raw_img"])
    cv2.imwrite(str(Path(frame_dir) / "depth_filtered.png"), frame_result["depth_filtered_img"])
    cv2.imwrite(str(Path(frame_dir) / "depth_postprocessed.png"), frame_result["depth_postprocessed_img"])
    filtered_cloud_path = build_ply_output_path(frame_dir, "scene_cloud_filtered.ply")
    postprocessed_cloud_path = build_ply_output_path(frame_dir, "scene_cloud_postprocessed.ply")
    write_open3d_point_cloud(filtered_cloud_path, frame_result["filtered_points"], frame_result["filtered_colors"])
    write_open3d_point_cloud(postprocessed_cloud_path, frame_result["postprocessed_points"], frame_result["postprocessed_colors"])

    return filtered_cloud_path, postprocessed_cloud_path


def build_point_cloud_frame_summary(frame_result, filtered_cloud_path, postprocessed_cloud_path, cfgs):
    return {
        "frame_index": int(frame_result["frame_index"]),
        "zero_count": int(frame_result["zero_count"]),
        "valid_ratio": float(frame_result["valid_ratio"]),
        "depth_filter_mm": {
            "depth_min_mm": None if getattr(cfgs, "depth_min_mm", None) is None else int(cfgs.depth_min_mm),
            "depth_max_mm": None if getattr(cfgs, "depth_max_mm", None) is None else int(cfgs.depth_max_mm),
        },
        "depth_postprocess": {
            "enabled": bool(getattr(cfgs, "depth_postprocess", True)),
            "smooth_kernel": int(getattr(cfgs, "depth_smooth_kernel", 5)),
            "hole_fill_kernel": int(getattr(cfgs, "depth_hole_fill_kernel", 5)),
            "hole_fill_iterations": int(getattr(cfgs, "depth_hole_fill_iterations", 2)),
        },
        "z_filter_m": {
            "z_min": None if cfgs.z_min is None else float(cfgs.z_min),
            "z_max": None if cfgs.z_max is None else float(cfgs.z_max),
        },
        "scene_stats_filtered": frame_result["scene_stats_filtered"],
        "scene_stats_postprocessed": frame_result["scene_stats_postprocessed"],
        "scene_cloud_filtered_ply": filtered_cloud_path,
        "scene_cloud_postprocessed_ply": postprocessed_cloud_path,
    }
