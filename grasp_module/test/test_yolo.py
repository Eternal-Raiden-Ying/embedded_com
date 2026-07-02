import logging
import os
import sys
from pathlib import Path

import cv2


CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.dirname(CURRENT_DIR)
ROOT_DIR = os.path.dirname(PARENT_DIR)
sys.path.append(ROOT_DIR)

try:
    from grasp_module.config.logging_config import configure_grasp_logger
    from grasp_module.config.predictor_config import build_predictor_defaults
    from grasp_module.test.utils.bag_io import collect_bag_frames, save_bag_frame_inputs
    from grasp_module.test.utils.io_utils import ensure_dir, log_kv_block, save_json
    from grasp_module.test.utils.yolo_probe import (
        choose_best_frame_all_classes,
        load_probe_model,
        probe_all_classes,
    )
except ImportError as e:
    print(f"-> Import Error: {e}")
    print(f"-> Current sys.path: {sys.path}")
    sys.exit(1)


logger = logging.getLogger("vision.grasp")


def build_parser():
    import argparse

    defaults = build_predictor_defaults(
        {
            "rgb_path": os.path.join(CURRENT_DIR, "data", "color", "color_00000.png"),
            "depth_path": os.path.join(CURRENT_DIR, "data", "depth", "depth_raw_00000.png"),
            "yolo_weights_dir": os.path.join(PARENT_DIR, "weights"),
        }
    )
    parser = argparse.ArgumentParser(description="YOLO debug script — detect all classes, save overlay")
    parser.add_argument("--rgb_path", type=str, default=defaults["rgb_path"], help="Path to RGB image")
    parser.add_argument("--depth_path", type=str, default=defaults["depth_path"], help="Path to depth image when saving local debug inputs")
    parser.add_argument("--yolo_model", type=str, default=defaults["yolo_model"], help="YOLO model name or path")
    parser.add_argument("--yolo_weights_dir", type=str, default=defaults["yolo_weights_dir"], help="Ultralytics weights cache directory")
    parser.add_argument("--yolo_conf", type=float, default=0.25, help="YOLO confidence threshold")
    parser.add_argument("--yolo_iou", type=float, default=0.7, help="YOLO NMS IoU threshold")
    parser.add_argument("--dump_dir", type=str, default=os.path.join(CURRENT_DIR, "yolo_debug"), help="Folder to save yolo debug outputs")
    parser.add_argument("--bag_file", type=str, default="", help="Optional RealSense .bag file for multi-frame YOLO replay")
    parser.add_argument("--bag_output_dir", type=str, default="", help="Output folder for bag yolo debug results")
    parser.add_argument("--bag_top_k", type=int, default=1, help="Number of low-hole bag frames to inspect")
    parser.add_argument("--bag_stride", type=int, default=1, help="Only inspect every Nth frame from bag")
    parser.add_argument("--bag_max_frames", type=int, default=60, help="Maximum sampled bag frames to inspect; <=0 means read all")
    parser.add_argument("--bag_min_valid_ratio", type=float, default=0.0, help="Discard frames whose non-zero depth ratio is lower than this threshold")
    parser.add_argument("--enable_depth_align", action="store_true", help="Enable pyrealsense2 depth-to-color alignment during bag replay (default off)")
    return parser


def save_overlay(frame_dir, overlay_bgr):
    output_path = os.path.join(frame_dir, "yolo_overlay.jpg")
    cv2.imwrite(output_path, overlay_bgr)
    return output_path


def log_detection(result, frame_label=None):
    prefix = f"[{frame_label}] " if frame_label else ""
    info = result["info"]
    if info["found"]:
        details = ", ".join(
            f"{name}(id={cid}, conf={conf:.3f})"
            for cid, conf, name in zip(info["class_ids"], info["confs"], info["class_names"])
        )
        logger.info("%sDetected %s objects: %s", prefix, info["count"], details)
    else:
        logger.info("%sNo objects detected.", prefix)


def run_image_mode(cfgs):
    ensure_dir(cfgs.dump_dir)
    # cv2.imread 默认返回 BGR，YOLO 也期望 BGR 输入
    bgr = cv2.imread(cfgs.rgb_path, cv2.IMREAD_COLOR)
    if bgr is None:
        raise ValueError(f"Failed to read color image: {cfgs.rgb_path}")

    model = load_probe_model(cfgs.yolo_model, cfgs.yolo_weights_dir)
    result = probe_all_classes(model, bgr, conf=cfgs.yolo_conf, iou=cfgs.yolo_iou)
    log_detection(result)
    overlay_path = save_overlay(cfgs.dump_dir, result["overlay_img"])
    summary = {
        "mode": "image",
        "rgb_path": cfgs.rgb_path,
        "yolo_overlay": overlay_path,
        "yolo_conf": cfgs.yolo_conf,
        "yolo_iou": cfgs.yolo_iou,
        "detection": result["info"],
    }
    save_json(os.path.join(cfgs.dump_dir, "summary.json"), summary)
    logger.info("YOLO image summary saved to: %s", os.path.join(cfgs.dump_dir, "summary.json"))


def run_bag_mode(cfgs):
    bag_name = Path(cfgs.bag_file).stem
    if not cfgs.bag_output_dir:
        cfgs.bag_output_dir = os.path.join(CURRENT_DIR, "yolo_debug", bag_name)
    ensure_dir(cfgs.bag_output_dir)

    selected_frames, metadata_path = collect_bag_frames(
        cfgs,
        cfgs.bag_output_dir,
        metadata_description="Camera intrinsics exported from bag for yolo replay",
        skip_align=not getattr(cfgs, "enable_depth_align", False),
    )

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
        # color_img 是 RGB（来自 convert_color_frame_to_rgb），YOLO 需要 BGR
        bgr = cv2.cvtColor(item["color_img"], cv2.COLOR_RGB2BGR)
        save_bag_frame_inputs(frame_dir, item["color_img"], item["depth_img"])
        result = probe_all_classes(model, bgr, conf=cfgs.yolo_conf, iou=cfgs.yolo_iou)
        log_detection(result, frame_label=frame_name)
        overlay_path = save_overlay(frame_dir, result["overlay_img"])
        frame_summary = {
            "frame_index": item["frame_index"],
            "zero_count": item["zero_count"],
            "valid_ratio": item["valid_ratio"],
            "yolo_overlay": overlay_path,
            "yolo_conf": cfgs.yolo_conf,
            "yolo_iou": cfgs.yolo_iou,
            "detection": result["info"],
        }
        save_json(os.path.join(frame_dir, "summary.json"), frame_summary)
        inspected_frames.append(
            {
                "frame_name": frame_name,
                "summary": frame_summary,
                "info": result["info"],
            }
        )

    chosen_frame, selection_reason = choose_best_frame_all_classes(inspected_frames)
    summary = {
        "mode": "bag",
        "bag_file": cfgs.bag_file,
        "camera_metadata": metadata_path,
        "yolo_conf": cfgs.yolo_conf,
        "yolo_iou": cfgs.yolo_iou,
        "detection_selection_reason": selection_reason,
        "selected_frames": [item["summary"] for item in inspected_frames],
    }
    if chosen_frame is not None:
        summary["selected_frame_name"] = chosen_frame["frame_name"]
        summary["selected_frame_index"] = chosen_frame["summary"]["frame_index"]
    else:
        summary["reason"] = "no_detection"

    save_json(os.path.join(cfgs.bag_output_dir, "summary.json"), summary)
    logger.info("YOLO bag summary saved to: %s", os.path.join(cfgs.bag_output_dir, "summary.json"))


def main():
    configure_grasp_logger(level=logging.INFO)
    parser = build_parser()
    cfgs = parser.parse_args()

    log_kv_block(
        logger,
        "YOLO replay settings",
        {
            "yolo_model": cfgs.yolo_model,
            "yolo_conf": cfgs.yolo_conf,
            "yolo_iou": cfgs.yolo_iou,
            "bag_top_k": cfgs.bag_top_k,
            "bag_stride": cfgs.bag_stride,
            "bag_max_frames": cfgs.bag_max_frames,
        },
    )

    if cfgs.bag_file:
        run_bag_mode(cfgs)
    else:
        run_image_mode(cfgs)


if __name__ == "__main__":
    main()
