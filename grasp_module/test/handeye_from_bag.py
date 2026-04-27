import logging
import os
import sys
from pathlib import Path


CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.dirname(CURRENT_DIR)
ROOT_DIR = os.path.dirname(PARENT_DIR)
sys.path.append(ROOT_DIR)

try:
    from grasp_module.config.logging_config import configure_grasp_logger
    from grasp_module.config.predictor_config import build_predictor_arg_parser
    from grasp_module.test.utils.bag_io import (
        build_point_cloud_frame_summary,
        collect_bag_candidates,
        save_point_cloud_frame_outputs,
    )
    from grasp_module.test.utils.io_utils import ensure_dir, log_kv_block, save_json
except ImportError as e:
    print(f"-> Import Error: {e}")
    print(f"-> Current sys.path: {sys.path}")
    sys.exit(1)


logger = logging.getLogger("vision.grasp")


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
    parser.add_argument("--z_min", type=float, default=0.15, help="Minimum depth in meters to keep in exported point clouds")
    parser.add_argument("--z_max", type=float, default=2.0, help="Maximum depth in meters to keep in exported point clouds")
    cfgs = parser.parse_args()

    if not cfgs.output_dir:
        cfgs.output_dir = os.path.join(CURRENT_DIR, "handeye_debug", Path(cfgs.bag_file).stem)
    ensure_dir(cfgs.output_dir)

    log_kv_block(
        logger,
        "Handeye bag export settings",
        {
            "bag_file": cfgs.bag_file,
            "bag_top_k": cfgs.bag_top_k,
            "bag_stride": cfgs.bag_stride,
            "bag_max_frames": cfgs.bag_max_frames,
            "depth_min_mm": cfgs.depth_min_mm,
            "depth_max_mm": cfgs.depth_max_mm,
            "depth_postprocess": cfgs.depth_postprocess,
            "depth_smooth_kernel": cfgs.depth_smooth_kernel,
            "depth_hole_fill_kernel": cfgs.depth_hole_fill_kernel,
            "depth_hole_fill_iterations": cfgs.depth_hole_fill_iterations,
            "z_min": cfgs.z_min,
            "z_max": cfgs.z_max,
        },
    )

    selected_frames, metadata_path, _camera_info = collect_bag_candidates(
        cfgs,
        cfgs.output_dir,
        metadata_description="Camera intrinsics exported from bag for hand-eye debug",
    )
    if not selected_frames:
        raise RuntimeError("No valid frames found in the bag after depth filtering.")

    summary = {
        "bag_file": cfgs.bag_file,
        "camera_metadata_path": metadata_path,
        "selected_frames": [],
    }

    for rank, frame_result in enumerate(selected_frames, start=1):
        frame_name = f"rank_{rank:02d}_frame_{frame_result['frame_index']:05d}"
        frame_dir = ensure_dir(os.path.join(cfgs.output_dir, frame_name))
        filtered_cloud_path, postprocessed_cloud_path = save_point_cloud_frame_outputs(frame_dir, frame_result)
        frame_summary = build_point_cloud_frame_summary(
            frame_result,
            filtered_cloud_path,
            postprocessed_cloud_path,
            cfgs,
        )
        save_json(os.path.join(frame_dir, "summary.json"), frame_summary)
        summary["selected_frames"].append(frame_summary)
        logger.info(
            "[%s] zero_count=%s valid_ratio=%.6f filtered_points=%s postprocessed_points=%s",
            frame_name,
            frame_summary["zero_count"],
            frame_summary["valid_ratio"],
            frame_summary["scene_stats_filtered"]["point_count"] if frame_summary["scene_stats_filtered"] else 0,
            frame_summary["scene_stats_postprocessed"]["point_count"] if frame_summary["scene_stats_postprocessed"] else 0,
        )

    save_json(os.path.join(cfgs.output_dir, "summary.json"), summary)
    logger.info("Bag point-cloud summary saved to: %s", os.path.join(cfgs.output_dir, "summary.json"))


if __name__ == "__main__":
    main()
