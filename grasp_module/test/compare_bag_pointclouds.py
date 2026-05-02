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
    from grasp_module.backend.utils.data_utils import build_ply_output_path, write_open3d_point_cloud
    from grasp_module.config.logging_config import configure_grasp_logger
    from grasp_module.config.predictor_config import build_predictor_arg_parser
    from grasp_module.test.utils.bag_io import collect_bag_comparison_frames
    from grasp_module.test.utils.io_utils import ensure_dir, log_kv_block, save_json
except ImportError as exc:
    print(f"-> Import Error: {exc}")
    print(f"-> Current sys.path: {sys.path}")
    sys.exit(1)


logger = logging.getLogger("vision.grasp")


def _compute_cloud_stats(points):
    if points.size == 0:
        return {
            "point_count": 0,
            "z_min_m": None,
            "z_max_m": None,
            "z_median_m": None,
            "bbox_extent_xyz_m": None,
        }

    min_xyz = points.min(axis=0)
    max_xyz = points.max(axis=0)
    extent_xyz = max_xyz - min_xyz
    return {
        "point_count": int(points.shape[0]),
        "z_min_m": float(min_xyz[2]),
        "z_max_m": float(max_xyz[2]),
        "z_median_m": float(np.median(points[:, 2])),
        "bbox_extent_xyz_m": [float(v) for v in extent_xyz],
    }


def _compute_valid_ratio(depth_img):
    if depth_img is None or depth_img.size == 0:
        return 0.0
    return float((depth_img > 0).sum() / depth_img.size)


def _nearest_neighbor_errors(source_points, target_points):
    if source_points.size == 0 or target_points.size == 0:
        return {
            "count": int(source_points.shape[0]) if source_points.ndim == 2 else 0,
            "mean_m": None,
            "median_m": None,
            "p95_m": None,
            "max_m": None,
            "ratio_le_2mm": None,
            "ratio_le_5mm": None,
        }

    try:
        from scipy.spatial import cKDTree

        tree = cKDTree(target_points)
        distances, _ = tree.query(source_points, k=1, workers=-1)
    except Exception:
        import open3d as o3d

        target_cloud = o3d.geometry.PointCloud()
        target_cloud.points = o3d.utility.Vector3dVector(target_points.astype(np.float64))
        kdtree = o3d.geometry.KDTreeFlann(target_cloud)
        distances = np.empty(source_points.shape[0], dtype=np.float64)
        for idx, point in enumerate(source_points.astype(np.float64)):
            _, _, dist2 = kdtree.search_knn_vector_3d(point, 1)
            distances[idx] = np.sqrt(dist2[0]) if dist2 else np.nan
        distances = distances[np.isfinite(distances)]

    if distances.size == 0:
        return {
            "count": int(source_points.shape[0]),
            "mean_m": None,
            "median_m": None,
            "p95_m": None,
            "max_m": None,
            "ratio_le_2mm": None,
            "ratio_le_5mm": None,
        }

    return {
        "count": int(distances.shape[0]),
        "mean_m": float(np.mean(distances)),
        "median_m": float(np.median(distances)),
        "p95_m": float(np.percentile(distances, 95)),
        "max_m": float(np.max(distances)),
        "ratio_le_2mm": float(np.mean(distances <= 0.002)),
        "ratio_le_5mm": float(np.mean(distances <= 0.005)),
    }


def _save_depth_and_color(frame_dir, frame_result):
    cv2.imwrite(str(Path(frame_dir) / "color.png"), cv2.cvtColor(frame_result["color_img"], cv2.COLOR_RGB2BGR))
    cv2.imwrite(str(Path(frame_dir) / "depth_raw.png"), frame_result["depth_raw_img"])
    cv2.imwrite(str(Path(frame_dir) / "depth_filtered.png"), frame_result["depth_filtered_img"])
    cv2.imwrite(str(Path(frame_dir) / "depth_postprocessed.png"), frame_result["depth_postprocessed_img"])
    if frame_result.get("depth_official_img") is not None:
        cv2.imwrite(str(Path(frame_dir) / "depth_official.png"), frame_result["depth_official_img"])


def _save_clouds(frame_dir, frame_result):
    reproj_filtered_path = build_ply_output_path(frame_dir, "reproj_filtered.ply")
    reproj_postprocessed_path = build_ply_output_path(frame_dir, "reproj_postprocessed.ply")
    sdk_cloud_path = build_ply_output_path(frame_dir, "sdk_pointcloud.ply")
    reproj_official_path = None
    sdk_official_cloud_path = None

    write_open3d_point_cloud(reproj_filtered_path, frame_result["filtered_points"], frame_result["filtered_colors"])
    write_open3d_point_cloud(
        reproj_postprocessed_path,
        frame_result["postprocessed_points"],
        frame_result["postprocessed_colors"],
    )
    write_open3d_point_cloud(
        sdk_cloud_path,
        frame_result["sdk_points_filtered"],
        frame_result["sdk_colors_filtered"],
    )
    if frame_result.get("official_points") is not None:
        reproj_official_path = build_ply_output_path(frame_dir, "reproj_official_filtered.ply")
        write_open3d_point_cloud(
            reproj_official_path,
            frame_result["official_points"],
            frame_result["official_colors"],
        )
    if frame_result.get("sdk_points_official") is not None:
        sdk_official_cloud_path = build_ply_output_path(frame_dir, "sdk_official_pointcloud.ply")
        write_open3d_point_cloud(
            sdk_official_cloud_path,
            frame_result["sdk_points_official"],
            frame_result["sdk_colors_official"],
        )
    return (
        reproj_filtered_path,
        reproj_postprocessed_path,
        sdk_cloud_path,
        reproj_official_path,
        sdk_official_cloud_path,
    )


def main():
    configure_grasp_logger(level=logging.INFO)

    default_overrides = {
        "yolo_weights_dir": os.path.join(PARENT_DIR, "weights"),
        "camera_metadata": os.path.join(PARENT_DIR, "config", "realsense_metadata.json"),
    }
    parser = build_predictor_arg_parser(
        description="Compare current reprojection point cloud vs RealSense SDK pointcloud on bag replay",
        default_overrides=default_overrides,
    )
    parser.add_argument("--bag_file", type=str, required=True, help="Path to RealSense .bag file")
    parser.add_argument("--output_dir", type=str, default="", help="Output directory for comparison artifacts")
    parser.add_argument("--bag_top_k", type=int, default=1, help="Number of best bag frames to export")
    parser.add_argument("--bag_stride", type=int, default=5, help="Inspect every Nth frame from the bag")
    parser.add_argument("--bag_max_frames", type=int, default=80, help="Maximum sampled frames to inspect; <=0 means read all")
    parser.add_argument("--bag_min_valid_ratio", type=float, default=0.0, help="Discard frames whose non-zero depth ratio is lower than this threshold")
    parser.add_argument("--depth_min_mm", type=int, default=1, help="Minimum valid depth in millimeters; smaller values are zeroed out")
    parser.add_argument("--depth_max_mm", type=int, default=2000, help="Maximum valid depth in millimeters; larger values are zeroed out")
    parser.set_defaults(depth_postprocess=True)
    parser.add_argument("--disable_depth_postprocess", dest="depth_postprocess", action="store_false", help="Disable simple depth postprocessing in the test export chain")
    parser.add_argument("--depth_smooth_kernel", type=int, default=5, help="Median smoothing kernel size for valid depth pixels; odd values only")
    parser.add_argument("--depth_hole_fill_kernel", type=int, default=5, help="Median kernel size used when filling zero-depth holes; odd values only")
    parser.add_argument("--depth_hole_fill_iterations", type=int, default=2, help="Number of small-hole filling iterations")
    parser.add_argument("--rs_threshold_min_m", type=float, default=0.001, help="RealSense threshold filter min distance in meters")
    parser.add_argument("--rs_threshold_max_m", type=float, default=2.0, help="RealSense threshold filter max distance in meters")
    parser.add_argument("--rs_spatial_magnitude", type=int, default=2, help="RealSense spatial filter magnitude")
    parser.add_argument("--rs_spatial_alpha", type=float, default=0.5, help="RealSense spatial filter alpha")
    parser.add_argument("--rs_spatial_delta", type=float, default=20.0, help="RealSense spatial filter delta")
    parser.add_argument("--rs_spatial_holes_fill", type=int, default=1, help="RealSense spatial filter holes_fill option")
    parser.add_argument("--rs_hole_fill_mode", type=int, default=1, help="RealSense hole filling filter mode")
    parser.add_argument("--rs_temporal_enable", action="store_true", help="Enable RealSense temporal filter in the official filter chain")
    parser.add_argument("--rs_temporal_alpha", type=float, default=0.4, help="RealSense temporal filter alpha")
    parser.add_argument("--rs_temporal_delta", type=float, default=20.0, help="RealSense temporal filter delta")
    parser.add_argument("--rs_temporal_holes_fill", type=int, default=3, help="RealSense temporal filter holes_fill option")
    cfgs = parser.parse_args()

    if not cfgs.output_dir:
        cfgs.output_dir = os.path.join(CURRENT_DIR, "pointcloud_compare", Path(cfgs.bag_file).stem)
    ensure_dir(cfgs.output_dir)

    log_kv_block(
        logger,
        "Bag point-cloud comparison settings",
        {
            "bag_file": cfgs.bag_file,
            "bag_top_k": cfgs.bag_top_k,
            "bag_stride": cfgs.bag_stride,
            "bag_max_frames": cfgs.bag_max_frames,
            "bag_min_valid_ratio": cfgs.bag_min_valid_ratio,
            "depth_min_mm": cfgs.depth_min_mm,
            "depth_max_mm": cfgs.depth_max_mm,
            "depth_postprocess": cfgs.depth_postprocess,
            "depth_smooth_kernel": cfgs.depth_smooth_kernel,
            "depth_hole_fill_kernel": cfgs.depth_hole_fill_kernel,
            "depth_hole_fill_iterations": cfgs.depth_hole_fill_iterations,
            "rs_threshold_min_m": cfgs.rs_threshold_min_m,
            "rs_threshold_max_m": cfgs.rs_threshold_max_m,
            "rs_spatial_magnitude": cfgs.rs_spatial_magnitude,
            "rs_spatial_alpha": cfgs.rs_spatial_alpha,
            "rs_spatial_delta": cfgs.rs_spatial_delta,
            "rs_spatial_holes_fill": cfgs.rs_spatial_holes_fill,
            "rs_hole_fill_mode": cfgs.rs_hole_fill_mode,
            "rs_temporal_enable": cfgs.rs_temporal_enable,
            "rs_temporal_alpha": cfgs.rs_temporal_alpha,
            "rs_temporal_delta": cfgs.rs_temporal_delta,
            "rs_temporal_holes_fill": cfgs.rs_temporal_holes_fill,
        },
    )

    selected_frames, metadata_path, _camera_info = collect_bag_comparison_frames(
        cfgs,
        cfgs.output_dir,
        metadata_description="Camera intrinsics exported from bag for reprojection vs SDK comparison",
    )
    if not selected_frames:
        raise RuntimeError("No valid frames found in the bag after depth filtering.")

    summary = {
        "bag_file": cfgs.bag_file,
        "camera_metadata_path": metadata_path,
        "frames": [],
    }

    for rank, frame_result in enumerate(selected_frames, start=1):
        frame_name = f"rank_{rank:02d}_frame_{frame_result['frame_index']:05d}"
        frame_dir = ensure_dir(os.path.join(cfgs.output_dir, frame_name))
        _save_depth_and_color(frame_dir, frame_result)
        (
            reproj_filtered_path,
            reproj_postprocessed_path,
            sdk_cloud_path,
            reproj_official_path,
            sdk_official_cloud_path,
        ) = _save_clouds(frame_dir, frame_result)

        frame_summary = {
            "frame_index": int(frame_result["frame_index"]),
            "zero_count": int(frame_result["zero_count"]),
            "valid_ratio_filtered": _compute_valid_ratio(frame_result["depth_filtered_img"]),
            "valid_ratio_postprocessed": _compute_valid_ratio(frame_result["depth_postprocessed_img"]),
            "valid_ratio_official": _compute_valid_ratio(frame_result.get("depth_official_img")),
            "clouds": {
                "reproj_filtered": _compute_cloud_stats(frame_result["filtered_points"]),
                "reproj_postprocessed": _compute_cloud_stats(frame_result["postprocessed_points"]),
                "sdk_pointcloud": _compute_cloud_stats(frame_result["sdk_points_filtered"]),
                "reproj_official_filtered": _compute_cloud_stats(frame_result["official_points"]) if frame_result.get("official_points") is not None else None,
                "sdk_official_pointcloud": _compute_cloud_stats(frame_result["sdk_points_official"]) if frame_result.get("sdk_points_official") is not None else None,
            },
            "nn_error_to_sdk": {
                "reproj_filtered": _nearest_neighbor_errors(frame_result["filtered_points"], frame_result["sdk_points_filtered"]),
                "reproj_postprocessed": _nearest_neighbor_errors(
                    frame_result["postprocessed_points"],
                    frame_result["sdk_points_filtered"],
                ),
                "reproj_official_filtered": _nearest_neighbor_errors(
                    frame_result["official_points"],
                    frame_result["sdk_points_official"],
                ) if frame_result.get("official_points") is not None and frame_result.get("sdk_points_official") is not None else None,
            },
            "artifacts": {
                "reproj_filtered_ply": reproj_filtered_path,
                "reproj_postprocessed_ply": reproj_postprocessed_path,
                "sdk_pointcloud_ply": sdk_cloud_path,
                "reproj_official_filtered_ply": reproj_official_path,
                "sdk_official_pointcloud_ply": sdk_official_cloud_path,
                "color_png": str(Path(frame_dir) / "color.png"),
                "depth_raw_png": str(Path(frame_dir) / "depth_raw.png"),
                "depth_filtered_png": str(Path(frame_dir) / "depth_filtered.png"),
                "depth_postprocessed_png": str(Path(frame_dir) / "depth_postprocessed.png"),
                "depth_official_png": str(Path(frame_dir) / "depth_official.png") if frame_result.get("depth_official_img") is not None else None,
            },
        }
        save_json(os.path.join(frame_dir, "summary.json"), frame_summary)
        summary["frames"].append(frame_summary)
        official_p95 = frame_summary["nn_error_to_sdk"]["reproj_official_filtered"]["p95_m"] if frame_summary["nn_error_to_sdk"]["reproj_official_filtered"] else -1.0
        logger.info(
            "[%s] filtered_pts=%s postprocessed_pts=%s official_pts=%s sdk_pts=%s filtered_p95=%.6f postprocessed_p95=%.6f official_p95=%.6f",
            frame_name,
            frame_summary["clouds"]["reproj_filtered"]["point_count"],
            frame_summary["clouds"]["reproj_postprocessed"]["point_count"],
            frame_summary["clouds"]["reproj_official_filtered"]["point_count"] if frame_summary["clouds"]["reproj_official_filtered"] else 0,
            frame_summary["clouds"]["sdk_pointcloud"]["point_count"],
            frame_summary["nn_error_to_sdk"]["reproj_filtered"]["p95_m"] or -1.0,
            frame_summary["nn_error_to_sdk"]["reproj_postprocessed"]["p95_m"] or -1.0,
            official_p95,
        )

    save_json(os.path.join(cfgs.output_dir, "summary.json"), summary)
    logger.info("Bag comparison summary saved to: %s", os.path.join(cfgs.output_dir, "summary.json"))


if __name__ == "__main__":
    main()
