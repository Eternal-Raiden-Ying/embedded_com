import json
import logging
import os
import sys
from pathlib import Path

import cv2
import numpy as np
import open3d as o3d
import torch


CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.dirname(CURRENT_DIR)
ROOT_DIR = os.path.dirname(PARENT_DIR)
sys.path.append(ROOT_DIR)

try:
    from grasp_module.backend.engine import RealSenseGraspPredictor
    from grasp_module.config.logging_config import configure_grasp_logger
    from grasp_module.config.predictor_config import build_predictor_arg_parser
    from grasp_module.app.server_app import build_downstream_response
except ImportError as e:
    print(f"-> Import Error: {e}")
    print(f"-> Current sys.path: {sys.path}")
    sys.exit(1)


logger = logging.getLogger("vision.grasp")


def normalize_depth_shape(depth_img):
    if depth_img is None:
        return depth_img
    if depth_img.ndim == 3 and depth_img.shape[2] == 1:
        return depth_img[:, :, 0]
    return depth_img


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)
    return path


def save_json(path, payload):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def save_best_protocol_grasp_ply(predictor, protocol_grasp_group, dump_dir):
    if protocol_grasp_group is None or len(protocol_grasp_group) == 0:
        return None

    ply_dir = ensure_dir(os.path.join(dump_dir, "ply"))
    best_grasp_group = protocol_grasp_group[:1]
    best_grasp_mesh = predictor._build_grasp_mesh(best_grasp_group)
    output_path = os.path.join(ply_dir, "best_protocol_grasp.ply")
    o3d.io.write_triangle_mesh(output_path, best_grasp_mesh)
    return output_path


def summarize_response(grasp_results, protocol_targets, cfgs):
    response = build_downstream_response(grasp_results, protocol_targets, cfgs)
    if grasp_results is not None and len(grasp_results) > 0:
        top_grasp = grasp_results[0]
        response["top_raw_grasp"] = {
            "score": float(top_grasp.score),
            "translation": [float(v) for v in top_grasp.translation],
            "rotation_matrix": [[float(v) for v in row] for row in top_grasp.rotation_matrix],
            "width": float(top_grasp.width),
            "depth": float(top_grasp.depth),
        }
    else:
        response["top_raw_grasp"] = None
    return response


def run_engine_on_frame(predictor, color_img, depth_img, class_id, frame_label=None):
    grasp_results = predictor.infer(color_img, depth_img, class_id)
    protocol_grasp_group = predictor.build_protocol_grasp_group(grasp_results)
    protocol_targets = predictor.build_protocol_targets(grasp_results)
    response = summarize_response(grasp_results, protocol_targets, predictor.cfgs)
    best_protocol_ply = None
    if getattr(predictor.cfgs, "debug", False):
        best_protocol_ply = save_best_protocol_grasp_ply(
            predictor,
            protocol_grasp_group,
            predictor.cfgs.dump_dir,
        )
        response["best_protocol_grasp_ply"] = best_protocol_ply

    prefix = f"[{frame_label}] " if frame_label else ""
    raw_count = 0 if grasp_results is None else len(grasp_results)
    logger.info("%sFound %s grasps.", prefix, raw_count)
    logger.info("%sFound %s feasible protocol targets.", prefix, len(protocol_targets))
    if response["status"] == "success":
        logger.info("%sDownstream output_count=%s", prefix, response["output_count"])
        logger.info("%sBest Protocol Target: %s", prefix, response["targets"][0])
    else:
        logger.warning("%sNo downstream target. reason=%s", prefix, response.get("reason"))
    if best_protocol_ply:
        logger.info("%sSaved best protocol grasp ply: %s", prefix, best_protocol_ply)
    return grasp_results, protocol_targets, response


def convert_color_frame_to_rgb(rs, color_frame):
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
        "description": "Camera intrinsics exported from bag for engine debug",
    }
    save_json(output_path, metadata)
    return output_path


def collect_bag_frames(cfgs):
    try:
        import pyrealsense2 as rs
    except ImportError as exc:
        raise RuntimeError("pyrealsense2 is required for --bag_file mode") from exc

    bag_output_dir = Path(cfgs.bag_output_dir)
    ensure_dir(str(bag_output_dir))

    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_device_from_file(cfgs.bag_file, repeat_playback=False)
    profile = pipeline.start(config)
    playback = profile.get_device().as_playback()
    playback.set_real_time(False)
    align = rs.align(rs.stream.depth)

    metadata_path = None
    candidates = []
    sampled_frames = 0
    total_frames = 0

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
                metadata_path = export_depth_metadata(
                    profile,
                    raw_depth_frame,
                    raw_color_frame,
                    str(bag_output_dir / "camera_metadata.json"),
                )

            color_img = convert_color_frame_to_rgb(rs, aligned_color_frame)
            depth_img = np.asanyarray(aligned_depth_frame.get_data())
            depth_img = normalize_depth_shape(depth_img)
            if depth_img is None or depth_img.ndim != 2:
                continue

            zero_count = int((depth_img == 0).sum())
            valid_ratio = float((depth_img > 0).sum() / depth_img.size)
            if valid_ratio < cfgs.bag_min_valid_ratio:
                continue

            candidates.append({
                "frame_index": total_frames - 1,
                "color_img": color_img.copy(),
                "depth_img": depth_img.copy(),
                "zero_count": zero_count,
                "valid_ratio": valid_ratio,
            })
            sampled_frames += 1
            if cfgs.bag_max_frames > 0 and sampled_frames >= cfgs.bag_max_frames:
                break
    finally:
        pipeline.stop()

    if metadata_path is None:
        raise RuntimeError(f"Failed to read valid RGB-D frames from bag: {cfgs.bag_file}")

    candidates.sort(key=lambda item: (item["zero_count"], -item["valid_ratio"], item["frame_index"]))
    selected = candidates[:cfgs.bag_top_k]
    logger.info(
        "Collected %s candidate frames from bag, selected top %s by fewest depth holes.",
        len(candidates),
        len(selected),
    )
    for item in selected:
        logger.info(
            "Selected frame=%s zero_count=%s valid_ratio=%.6f",
            item["frame_index"],
            item["zero_count"],
            item["valid_ratio"],
        )
    return selected, metadata_path


def save_bag_frame_inputs(frame_dir, color_img, depth_img):
    color_path = os.path.join(frame_dir, "color.png")
    depth_path = os.path.join(frame_dir, "depth_raw.png")
    cv2.imwrite(color_path, cv2.cvtColor(color_img, cv2.COLOR_RGB2BGR))
    cv2.imwrite(depth_path, depth_img)
    return color_path, depth_path


def run_bag_mode(cfgs):
    bag_name = Path(cfgs.bag_file).stem
    if not cfgs.bag_output_dir:
        cfgs.bag_output_dir = os.path.join(CURRENT_DIR, "bag_debug", bag_name)
    ensure_dir(cfgs.bag_output_dir)

    selected_frames, metadata_path = collect_bag_frames(cfgs)
    cfgs.camera_metadata = metadata_path
    predictor = RealSenseGraspPredictor(cfgs)

    summary = {
        "bag_file": cfgs.bag_file,
        "camera_metadata": metadata_path,
        "class_id": cfgs.yolo_class_id,
        "selected_frames": [],
    }

    for rank, item in enumerate(selected_frames, start=1):
        frame_name = f"rank_{rank:02d}_frame_{item['frame_index']:05d}"
        frame_dir = ensure_dir(os.path.join(cfgs.bag_output_dir, frame_name))
        save_bag_frame_inputs(frame_dir, item["color_img"], item["depth_img"])
        predictor.cfgs.dump_dir = frame_dir

        logger.info("Running predictor on %s", frame_name)
        _, _, response = run_engine_on_frame(
            predictor,
            item["color_img"],
            item["depth_img"],
            cfgs.yolo_class_id,
            frame_label=frame_name,
        )
        frame_summary = {
            "frame_index": item["frame_index"],
            "zero_count": item["zero_count"],
            "valid_ratio": item["valid_ratio"],
            "response": response,
        }
        summary["selected_frames"].append(frame_summary)
        save_json(os.path.join(frame_dir, "summary.json"), frame_summary)

    save_json(os.path.join(cfgs.bag_output_dir, "summary.json"), summary)
    logger.info("Bag debug summary saved to: %s", os.path.join(cfgs.bag_output_dir, "summary.json"))


def run_image_mode(cfgs):
    logger.info("Loading RGB-D data from: %s", cfgs.rgb_path)
    color_bgr = cv2.imread(cfgs.rgb_path, cv2.IMREAD_COLOR)
    color_img = None if color_bgr is None else cv2.cvtColor(color_bgr, cv2.COLOR_BGR2RGB)
    depth_img = normalize_depth_shape(cv2.imread(cfgs.depth_path, cv2.IMREAD_UNCHANGED))

    if depth_img is None:
        raise ValueError(f"Failed to read depth image: {cfgs.depth_path}")
    if color_img is None:
        raise ValueError(f"Failed to read color image: {cfgs.rgb_path}")
    if depth_img.ndim != 2:
        raise ValueError(f"Depth image must be 2D after normalization, got shape {depth_img.shape}")

    logger.info("Using internal YOLO segmentation. target_class_id=%s", cfgs.yolo_class_id)
    logger.info("RGB shape: %s, Depth shape: %s", color_img.shape, depth_img.shape)
    logger.info("Valid depth pixels: %s", int((depth_img > 0).sum()))

    predictor = RealSenseGraspPredictor(cfgs)
    logger.info("Starting inference...")
    _, _, response = run_engine_on_frame(predictor, color_img, depth_img, cfgs.yolo_class_id)
    summary_path = os.path.join(cfgs.dump_dir, "summary.json")
    ensure_dir(cfgs.dump_dir)
    save_json(summary_path, response)
    logger.info("Single-frame summary saved to: %s", summary_path)


def main():
    configure_grasp_logger(level=logging.INFO)
    logger.info("Successfully imported RealSenseGraspPredictor from backend.engine")

    default_overrides = {
        "debug": True,
        "dump_dir": os.path.join(CURRENT_DIR, "debug_res"),
        "rgb_path": os.path.join(CURRENT_DIR, "data", "color", "color_00000.png"),
        "depth_path": os.path.join(CURRENT_DIR, "data", "depth", "depth_raw_00000.png"),
        "camera_metadata": os.path.join(PARENT_DIR, "config", "realsense_metadata.json"),
        "yolo_weights_dir": os.path.join(PARENT_DIR, "weights"),
    }
    parser = build_predictor_arg_parser(
        description="GraspNet inference test script for single RGB-D input or RealSense bag replay",
        default_overrides=default_overrides,
    )
    parser.add_argument("--bag_file", type=str, default="", help="Optional RealSense .bag file for multi-frame debug")
    parser.add_argument("--bag_output_dir", type=str, default="", help="Output folder for bag debug results")
    parser.add_argument("--bag_top_k", type=int, default=5, help="Number of low-hole bag frames to debug")
    parser.add_argument("--bag_stride", type=int, default=1, help="Only inspect every Nth frame from bag")
    parser.add_argument("--bag_max_frames", type=int, default=60, help="Maximum sampled bag frames to inspect; <=0 means read all")
    parser.add_argument("--bag_min_valid_ratio", type=float, default=0.0, help="Discard frames whose non-zero depth ratio is lower than this threshold")
    cfgs = parser.parse_args()

    torch.manual_seed(cfgs.random_seed)
    logger.info("Random seed: %s", cfgs.random_seed)

    if cfgs.bag_file:
        run_bag_mode(cfgs)
    else:
        run_image_mode(cfgs)


if __name__ == "__main__":
    main()
