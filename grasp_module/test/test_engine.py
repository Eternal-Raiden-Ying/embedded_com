import logging
import os
import sys
from pathlib import Path

import cv2
import open3d as o3d
import torch


CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.dirname(CURRENT_DIR)
ROOT_DIR = os.path.dirname(PARENT_DIR)
sys.path.append(ROOT_DIR)

try:
    from grasp_module.config.logging_config import configure_grasp_logger
    from grasp_module.config.predictor_config import build_predictor_arg_parser
    from grasp_module.test.utils.bag_io import (
        collect_bag_frames,
        normalize_depth_shape,
        postprocess_depth_image,
        sanitize_depth_image,
        save_bag_frame_inputs,
        save_depth_variants,
    )
    from grasp_module.test.utils.io_utils import ensure_dir, log_kv_block, parse_int_list_csv, save_json
    from grasp_module.test.utils.reporting import build_downstream_response, summarize_top_raw_grasp
    from grasp_module.test.utils.yolo_probe import choose_detection_frame, load_probe_model, resolve_detection_route
except ImportError as e:
    print(f"-> Import Error: {e}")
    print(f"-> Current sys.path: {sys.path}")
    sys.exit(1)


logger = logging.getLogger("vision.grasp")


def load_runtime_predictor():
    from grasp_module.backend.engine import RealSenseGraspPredictor

    return RealSenseGraspPredictor


def save_best_protocol_grasp_ply(predictor, protocol_grasp_group, dump_dir):
    if protocol_grasp_group is None or len(protocol_grasp_group) == 0:
        return None

    ply_dir = ensure_dir(os.path.join(dump_dir, "ply"))
    best_grasp_group = protocol_grasp_group[:1]
    best_grasp_mesh = predictor._build_grasp_mesh(best_grasp_group)
    output_path = os.path.join(ply_dir, "best_protocol_grasp.ply")
    o3d.io.write_triangle_mesh(output_path, best_grasp_mesh)
    return output_path


def save_yolo_overlay(frame_dir, overlay_bgr):
    output_path = os.path.join(frame_dir, "yolo_overlay.jpg")
    cv2.imwrite(output_path, overlay_bgr)
    return output_path


def summarize_response(grasp_results, protocol_targets, cfgs):
    response = build_downstream_response(grasp_results, protocol_targets, cfgs)
    response["top_raw_grasp"] = summarize_top_raw_grasp(grasp_results)
    return response


def probe_frame(model, cfgs, color_img_rgb):
    bgr_image = cv2.cvtColor(color_img_rgb, cv2.COLOR_RGB2BGR)
    route = resolve_detection_route(
        model,
        bgr_image,
        primary_class_id=cfgs.yolo_class_id,
        fallback_class_ids=parse_int_list_csv(getattr(cfgs, "fallback_class_ids_csv", "")),
        conf=float(getattr(cfgs, "fallback_probe_conf", 0.10)),
        iou=float(getattr(cfgs, "yolo_iou", 0.7)),
        bbox_scale=float(getattr(cfgs, "bbox_expand_scale", 2.0)),
    )
    return route


def build_detection_summary(frame_index, route):
    return {
        "frame_index": frame_index,
        "detection_route": {
            "resolved_class_id": route["resolved_class_id"],
            "resolved_conf": route["resolved_conf"],
            "used_fallback": route["used_fallback"],
            "candidate_inspection": route["candidate_inspection"],
            "multiple_detections": route["multiple_detections"],
            "detection_count": route["detection_count"],
            "best_conf": route["best_conf"],
            "bbox": route["info"].get("bbox"),
            "found": route["info"].get("found"),
        },
    }


def log_detection_route(route, frame_label=None):
    prefix = f"[{frame_label}] " if frame_label else ""
    logger.info(
        "%sYOLO route primary=%s resolved=%s conf=%.3f count=%s fallback=%s",
        prefix,
        route["candidate_inspection"][0]["class_id"] if route["candidate_inspection"] else None,
        route["resolved_class_id"],
        route["resolved_conf"],
        route["detection_count"],
        route["used_fallback"],
    )
    if route["multiple_detections"]:
        logger.warning(
            "%sResolved class_id=%s produced multiple detections: count=%s",
            prefix,
            route["resolved_class_id"],
            route["detection_count"],
        )


def run_grasp_on_frame(cfgs, color_img, depth_img, route, frame_dir, frame_label=None):
    RealSenseGraspPredictor = load_runtime_predictor()
    predictor = RealSenseGraspPredictor(cfgs)
    predictor.cfgs.dump_dir = frame_dir

    original_conf = float(getattr(predictor.cfgs, "yolo_conf", 0.25))
    predictor.cfgs.yolo_conf = float(route["resolved_conf"])
    try:
        grasp_results = predictor.infer(color_img, depth_img, int(route["resolved_class_id"]))
    finally:
        predictor.cfgs.yolo_conf = original_conf

    protocol_grasp_group = predictor.build_protocol_grasp_group(grasp_results)
    protocol_targets = predictor.build_protocol_targets(grasp_results)
    response = summarize_response(grasp_results, protocol_targets, predictor.cfgs)
    response["detection_route"] = {
        "resolved_class_id": route["resolved_class_id"],
        "resolved_conf": route["resolved_conf"],
        "used_fallback": route["used_fallback"],
        "candidate_inspection": route["candidate_inspection"],
        "multiple_detections": route["multiple_detections"],
        "detection_count": route["detection_count"],
        "best_conf": route["best_conf"],
        "bbox": route["info"].get("bbox"),
        "found": route["info"].get("found"),
    }

    best_protocol_ply = None
    if getattr(predictor.cfgs, "debug", False):
        best_protocol_ply = save_best_protocol_grasp_ply(predictor, protocol_grasp_group, predictor.cfgs.dump_dir)
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
    return response


def log_runtime_settings(cfgs, mode):
    log_kv_block(
        logger,
        f"Test engine settings ({mode})",
        {
            "yolo_model": cfgs.yolo_model,
            "yolo_conf": cfgs.yolo_conf,
            "yolo_iou": cfgs.yolo_iou,
            "yolo_class_id": cfgs.yolo_class_id,
            "fallback_class_ids": parse_int_list_csv(getattr(cfgs, "fallback_class_ids_csv", "")),
            "fallback_probe_conf": getattr(cfgs, "fallback_probe_conf", 0.10),
            "protocol_min_score": cfgs.protocol_min_score,
            "response_max_targets": cfgs.response_max_targets,
            "bag_top_k": getattr(cfgs, "bag_top_k", None),
            "bag_stride": getattr(cfgs, "bag_stride", None),
            "bag_max_frames": getattr(cfgs, "bag_max_frames", None),
            "depth_min_mm": getattr(cfgs, "depth_min_mm", None),
            "depth_max_mm": getattr(cfgs, "depth_max_mm", None),
            "depth_postprocess": getattr(cfgs, "depth_postprocess", None),
            "depth_smooth_kernel": getattr(cfgs, "depth_smooth_kernel", None),
            "depth_hole_fill_kernel": getattr(cfgs, "depth_hole_fill_kernel", None),
            "depth_hole_fill_iterations": getattr(cfgs, "depth_hole_fill_iterations", None),
        },
    )


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

    ensure_dir(cfgs.dump_dir)
    filtered_depth_img = sanitize_depth_image(depth_img, cfgs.depth_min_mm, cfgs.depth_max_mm)
    postprocessed_depth_img = postprocess_depth_image(filtered_depth_img, cfgs)
    save_bag_frame_inputs(cfgs.dump_dir, color_img, depth_img)
    save_depth_variants(cfgs.dump_dir, filtered_depth_img, postprocessed_depth_img)

    model = load_probe_model(cfgs.yolo_model, cfgs.yolo_weights_dir)
    route = probe_frame(model, cfgs, color_img)
    log_detection_route(route)
    overlay_path = save_yolo_overlay(cfgs.dump_dir, route["overlay_img"])

    summary = build_detection_summary(frame_index=0, route=route)
    summary["yolo_overlay"] = overlay_path
    if route["detection_count"] == 0:
        summary["status"] = "reposition_required"
        summary["reason"] = "no_detection_after_fallback"
        save_json(os.path.join(cfgs.dump_dir, "summary.json"), summary)
        logger.warning("No detection after fallback routing. Summary saved to: %s", os.path.join(cfgs.dump_dir, "summary.json"))
        return

    response = run_grasp_on_frame(cfgs, color_img, postprocessed_depth_img, route, cfgs.dump_dir)
    summary.update(response)
    save_json(os.path.join(cfgs.dump_dir, "summary.json"), summary)
    logger.info("Single-frame summary saved to: %s", os.path.join(cfgs.dump_dir, "summary.json"))


def run_bag_mode(cfgs):
    bag_name = Path(cfgs.bag_file).stem
    if not cfgs.bag_output_dir:
        cfgs.bag_output_dir = os.path.join(CURRENT_DIR, "bag_debug", bag_name)
    ensure_dir(cfgs.bag_output_dir)

    selected_frames, metadata_path = collect_bag_frames(
        cfgs,
        cfgs.bag_output_dir,
        metadata_description="Camera intrinsics exported from bag for engine debug",
    )
    cfgs.camera_metadata = metadata_path

    logger.info(
        "Collected %s candidate frames from bag, inspecting top %s by fewest depth holes.",
        len(selected_frames),
        len(selected_frames),
    )
    for item in selected_frames:
        logger.info(
            "Candidate frame=%s zero_count=%s valid_ratio=%.6f",
            item["frame_index"],
            item["zero_count"],
            item["valid_ratio"],
        )

    model = load_probe_model(cfgs.yolo_model, cfgs.yolo_weights_dir)
    inspected_frames = []
    for rank, item in enumerate(selected_frames, start=1):
        frame_name = f"rank_{rank:02d}_frame_{item['frame_index']:05d}"
        frame_dir = ensure_dir(os.path.join(cfgs.bag_output_dir, frame_name))
        save_bag_frame_inputs(frame_dir, item["color_img"], item["depth_raw_img"])
        save_depth_variants(frame_dir, item["depth_filtered_img"], item["depth_postprocessed_img"])
        route = probe_frame(model, cfgs, item["color_img"])
        log_detection_route(route, frame_label=frame_name)
        overlay_path = save_yolo_overlay(frame_dir, route["overlay_img"])
        frame_summary = {
            "frame_index": item["frame_index"],
            "zero_count": item["zero_count"],
            "valid_ratio": item["valid_ratio"],
            "yolo_overlay": overlay_path,
            **build_detection_summary(item["frame_index"], route),
        }
        save_json(os.path.join(frame_dir, "summary.json"), frame_summary)
        inspected_frames.append(
            {
                "frame_name": frame_name,
                "frame_dir": frame_dir,
                "item": item,
                "route": route,
                "summary": frame_summary,
            }
        )

    chosen_frame, selection_reason = choose_detection_frame(inspected_frames)
    summary = {
        "bag_file": cfgs.bag_file,
        "camera_metadata": metadata_path,
        "class_id": cfgs.yolo_class_id,
        "fallback_class_ids": parse_int_list_csv(getattr(cfgs, "fallback_class_ids_csv", "")),
        "detection_selection_reason": selection_reason,
        "selected_frames": [item["summary"] for item in inspected_frames],
    }

    if chosen_frame is None:
        summary["status"] = "reposition_required"
        summary["reason"] = "no_detection_after_fallback"
        save_json(os.path.join(cfgs.bag_output_dir, "summary.json"), summary)
        logger.warning("No usable detection found in inspected bag frames.")
        return

    logger.info(
        "Selected %s for grasp inference. reason=%s resolved_class=%s count=%s fallback=%s",
        chosen_frame["frame_name"],
        selection_reason,
        chosen_frame["route"]["resolved_class_id"],
        chosen_frame["route"]["detection_count"],
        chosen_frame["route"]["used_fallback"],
    )
    response = run_grasp_on_frame(
        cfgs,
        chosen_frame["item"]["color_img"],
        chosen_frame["item"]["depth_img"],
        chosen_frame["route"],
        chosen_frame["frame_dir"],
        frame_label=chosen_frame["frame_name"],
    )
    chosen_frame["summary"].update(response)
    save_json(os.path.join(chosen_frame["frame_dir"], "summary.json"), chosen_frame["summary"])
    summary["selected_frame_name"] = chosen_frame["frame_name"]
    summary["selected_frame_index"] = chosen_frame["item"]["frame_index"]
    summary["status"] = response["status"]
    summary["response"] = response
    save_json(os.path.join(cfgs.bag_output_dir, "summary.json"), summary)
    logger.info("Bag debug summary saved to: %s", os.path.join(cfgs.bag_output_dir, "summary.json"))


def main():
    configure_grasp_logger(level=logging.INFO)

    default_overrides = {
        "debug": True,
        "dump_dir": os.path.join(CURRENT_DIR, "debug_res"),
        "rgb_path": os.path.join(CURRENT_DIR, "data", "color", "color_00000.png"),
        "depth_path": os.path.join(CURRENT_DIR, "data", "depth", "depth_raw_00000.png"),
        "camera_metadata": os.path.join(PARENT_DIR, "config", "realsense_metadata.json"),
        "yolo_weights_dir": os.path.join(PARENT_DIR, "weights"),
    }
    parser = build_predictor_arg_parser(
        description="GraspNet debug script with YOLO precheck and optional RealSense bag replay",
        default_overrides=default_overrides,
    )
    parser.add_argument("--bag_file", type=str, default="", help="Optional RealSense .bag file for single-frame grasp debug")
    parser.add_argument("--bag_output_dir", type=str, default="", help="Output folder for bag debug results")
    parser.add_argument("--bag_top_k", type=int, default=1, help="Number of low-hole bag frames to inspect before selecting one for grasp inference")
    parser.add_argument("--bag_stride", type=int, default=1, help="Only inspect every Nth frame from bag")
    parser.add_argument("--bag_max_frames", type=int, default=60, help="Maximum sampled bag frames to inspect; <=0 means read all")
    parser.add_argument("--bag_min_valid_ratio", type=float, default=0.0, help="Discard frames whose non-zero depth ratio is lower than this threshold")
    parser.add_argument("--depth_min_mm", type=int, default=1, help="Minimum valid depth in millimeters; smaller values are zeroed out")
    parser.add_argument("--depth_max_mm", type=int, default=2000, help="Maximum valid depth in millimeters; larger values are zeroed out")
    parser.set_defaults(depth_postprocess=True)
    parser.add_argument("--disable_depth_postprocess", dest="depth_postprocess", action="store_false", help="Disable simple depth postprocessing in the test chain")
    parser.add_argument("--depth_smooth_kernel", type=int, default=5, help="Median smoothing kernel size for valid depth pixels; odd values only")
    parser.add_argument("--depth_hole_fill_kernel", type=int, default=5, help="Median kernel size used when filling zero-depth holes; odd values only")
    parser.add_argument("--depth_hole_fill_iterations", type=int, default=2, help="Number of small-hole filling iterations")
    parser.add_argument("--fallback_class_ids_csv", type=str, default="32,55", help="Comma-separated fallback class ids used when the primary class is not detected")
    parser.add_argument("--fallback_probe_conf", type=float, default=0.10, help="YOLO confidence used for class probing and fallback routing")
    cfgs = parser.parse_args()

    torch.manual_seed(cfgs.random_seed)
    logger.info("Random seed: %s", cfgs.random_seed)
    log_runtime_settings(cfgs, "bag" if cfgs.bag_file else "image")

    if cfgs.bag_file:
        run_bag_mode(cfgs)
    else:
        run_image_mode(cfgs)


if __name__ == "__main__":
    main()
