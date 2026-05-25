#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import copy
import json
import math
import os
from pathlib import Path
import sys
import time
from typing import Any, Dict, List, Optional


HERE = Path(__file__).resolve()
REPO_ROOT = HERE.parents[3]
VISTA_ROOT = HERE.parents[2]
for item in (REPO_ROOT, VISTA_ROOT):
    if str(item) not in sys.path:
        sys.path.insert(0, str(item))


FRAME_STRIDE = 30


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate paired full-vs-fast table-plane preview images.")
    parser.add_argument("--bag", type=Path, required=True, help="RealSense .bag file path.")
    parser.add_argument(
        "--output",
        type=Path,
        default=REPO_ROOT / "runs" / "full_fast_stride30_compare",
        help="Output directory. Default: runs/full_fast_stride30_compare.",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=VISTA_ROOT / "configs" / "vision_params.yaml",
        help="Advanced: vision params YAML.",
    )
    return parser


def _fmt(value: Any, digits: int = 3) -> str:
    try:
        out = float(value)
    except Exception:
        return "NA"
    if not math.isfinite(out):
        return "NA"
    return f"{out:.{digits}f}"


def _deg(value: Any) -> str:
    try:
        return _fmt(float(value) * 180.0 / math.pi, 1)
    except Exception:
        return "NA"


def _roi_text(obs: Dict[str, Any]) -> str:
    roi = obs.get("plane_roi") or obs.get("depth_edge_roi") or obs.get("table_edge_roi") or obs.get("edge_roi")
    return str(roi if roi is not None else "NA")


def _obs_label_lines(obs: Dict[str, Any], *, side: str, frame_seq: int) -> List[str]:
    if side == "full":
        yaw = obs.get("yaw_err_rad", obs.get("plane_yaw_err_rad"))
        return [
            "detector_mode=full",
            f"frame_seq={frame_seq}",
            f"roi_source={obs.get('roi_source') or 'NA'} roi={_roi_text(obs)}",
            f"yaw={_fmt(yaw)}rad/{_deg(yaw)}deg dist={_fmt(obs.get('dist_err_m', obs.get('plane_dist_err_m')))}m",
            f"cx={_fmt(obs.get('plane_cx_norm'))} conf={_fmt(obs.get('confidence', obs.get('plane_confidence')))}",
            f"inliers={obs.get('inlier_count', obs.get('edge_inlier_count', 'NA'))} area={_fmt(obs.get('plane_area_ratio'))} width={_fmt(obs.get('plane_width_norm'))}",
            f"x_span={_fmt(obs.get('plane_x_span_m'))} control={obs.get('control_level') or 'NA'} reject={obs.get('reject_reason') or obs.get('control_reject_reason') or 'none'}",
        ]
    yaw = obs.get("fast_raw_yaw_err_rad", obs.get("yaw_err_rad"))
    return [
        "detector_mode=fast_plane_only_v3",
        f"frame_seq={frame_seq}",
        f"roi_source={obs.get('roi_source') or 'NA'} roi={_roi_text(obs)}",
        f"coord={obs.get('fast_coord_frame') or 'NA'} pitch={_fmt(obs.get('fast_camera_pitch_deg'))}deg hcam={_fmt(obs.get('fast_camera_height_m'))} htbl={_fmt(obs.get('fast_table_height_m'))}",
        f"raw_yaw={_fmt(yaw)}rad/{_deg(yaw)}deg raw_dist={_fmt(obs.get('fast_raw_dist_err_m', obs.get('dist_err_m')))}m",
        f"raw_cx={_fmt(obs.get('fast_raw_plane_cx_norm'))} conf={_fmt(obs.get('fast_score_final', obs.get('fast_raw_confidence')))} level={obs.get('fast_control_level') or obs.get('control_level') or 'NA'}",
        f"reject={obs.get('fast_gate_reason') or obs.get('fast_gate_reject_reason') or obs.get('reject_reason') or 'none'} stage={obs.get('fast_distance_stage') or 'NA'}",
        f"sampled={obs.get('fast_raw_sampled_point_count', obs.get('sampled_point_count', 'NA'))} height_candidate={obs.get('fast_candidate_point_count', obs.get('fast_raw_candidate_count', 'NA'))}",
        f"support_pixels={obs.get('fast_support_point_count', obs.get('fast_front_face_support_point_count', 'NA'))} reps={obs.get('fast_rep_count', obs.get('fast_front_face_rep_count', 'NA'))} in={obs.get('fast_rep_inlier_count', obs.get('fast_representative_inlier_count', 'NA'))} out={obs.get('fast_rep_outlier_count', 'NA')}",
        f"clusters={obs.get('fast_rep_cluster_count', 'NA')} selected={obs.get('fast_selected_cluster_index', 'NA')} y={_fmt(obs.get('fast_selected_cluster_y_center'))} score={_fmt(obs.get('fast_selected_cluster_score'))} bg={obs.get('fast_background_rep_count', 'NA')}",
        f"line={obs.get('fast_line_source') or obs.get('fast_fit_line_source') or 'NA'} edge={obs.get('fast_edge_inlier_count', obs.get('fast_edge_candidate_count', 'NA'))} edge_span={_fmt(obs.get('fast_edge_x_span_m'))} bg_blocked={int(bool(obs.get('fast_background_blocked', False)))}",
        f"local_band={obs.get('fast_local_band_support_count', 'NA')}/{_fmt(obs.get('fast_local_band_x_span_m'))}/{_fmt(obs.get('fast_local_band_residual_mean'))} frontness={_fmt(obs.get('fast_frontness_score'))}",
        f"span c/s/r/fit={_fmt(obs.get('fast_candidate_x_span_m'))}/{_fmt(obs.get('fast_support_x_span_m'))}/{_fmt(obs.get('fast_rep_x_span_m'))}/{_fmt(obs.get('fast_fit_inlier_x_span_m', obs.get('fast_raw_plane_x_span_m')))}",
        f"support_mode={obs.get('fast_support_mode') or 'NA'} line={obs.get('fast_fit_line_source') or 'NA'} resid={_fmt(obs.get('fast_residual_mean', obs.get('fast_raw_residual_mean')))}",
    ]


def _draw_label_block(image: Any, lines: List[str]) -> Any:
    import cv2
    import numpy as np

    canvas = np.asarray(image).copy()
    if canvas.ndim != 3 or canvas.shape[2] < 3:
        return canvas
    x0, y0 = 12, 12
    line_h = 22
    block_h = min(canvas.shape[0] - 24, 16 + line_h * len(lines))
    block_w = min(canvas.shape[1] - 24, 690)
    overlay = canvas.copy()
    cv2.rectangle(overlay, (x0, y0), (x0 + block_w, y0 + block_h), (10, 12, 16), -1)
    canvas = cv2.addWeighted(overlay, 0.72, canvas, 0.28, 0)
    for idx, line in enumerate(lines):
        y = y0 + 24 + idx * line_h
        cv2.putText(canvas, str(line), (x0 + 12, y), cv2.FONT_HERSHEY_SIMPLEX, 0.54, (235, 245, 255), 1, cv2.LINE_AA)
    return canvas


def _process_one(processor: Any, pack: Dict[str, Any], frame_seq: int, mode_name: str) -> Dict[str, Any]:
    capture_ts = time.time()
    frames = {
        "rgb": pack.get("rgb"),
        "depth": pack.get("depth"),
        "frame_capture_ts": capture_ts,
        "timestamp_ms": pack.get("timestamp_ms"),
    }
    obs = processor.process_camera_frame(
        frames,
        frame_seq=frame_seq,
        frame_slot={"seq": frame_seq, "ts": capture_ts, "payload": frames},
        local_perception={"has_infer": False, "box_count": 0, "infer_boxes": [], "rgb_shape": getattr(pack.get("rgb"), "shape", None)},
        runtime_status={"stage": "OFFLINE_BAG_COMPARE", "mode": "TABLE_EDGE_PERCEPTION"},
        source_mode="OFFLINE_BAG_COMPARE",
        count_dropped=False,
    )
    obs["source"] = f"full_fast_stride30_compare_{mode_name}"
    obs["bag_timestamp_ms"] = float(pack.get("timestamp_ms", 0.0) or 0.0)
    return obs


def main() -> None:
    args = build_parser().parse_args()
    os.environ["VISION_PARAMS_FILE"] = str(args.config.expanduser().resolve())

    import cv2
    import numpy as np

    from VISTA.vision_module.backend.preview.opencv_sink import OpenCVPreviewSink
    from VISTA.vision_module.backend.table_edge_manager import TableEdgeManager
    from VISTA.vision_module.config.board_config import CONFIG
    from VISTA.vision_module.examples.bag_table_plane import _make_preview_canvas, iter_bag_frames

    output_dir = args.output.expanduser().resolve()
    preview_dir = output_dir / "paired_previews"
    preview_dir.mkdir(parents=True, exist_ok=True)

    full_cfg = copy.deepcopy(CONFIG)
    fast_cfg = copy.deepcopy(CONFIG)
    full_cfg.table_edge.roi_preset = "center_lower"
    fast_cfg.table_edge.roi_preset = "center_lower"
    full_cfg.table_edge.detector_mode = "full"
    fast_cfg.table_edge.detector_mode = "fast_plane_only"

    full_processor = TableEdgeManager(cfg=full_cfg)
    fast_processor = TableEdgeManager(cfg=fast_cfg)
    sink = OpenCVPreviewSink("Full vs Fast Table Plane")

    compared_frames: List[int] = []
    failed_frames: List[Dict[str, Any]] = []
    frames_total = 0
    try:
        for pack in iter_bag_frames(args.bag, 0):
            frame_seq = int(pack["frame"])
            frames_total += 1
            if frame_seq % FRAME_STRIDE != 0:
                continue
            try:
                full_obs = _process_one(full_processor, pack, frame_seq, "full")
                fast_obs = _process_one(fast_processor, pack, frame_seq, "fast_plane_only_v3")
                full_canvas = _make_preview_canvas(sink, pack.get("rgb"), pack.get("depth"), full_obs, frame_seq)
                fast_canvas = _make_preview_canvas(sink, pack.get("rgb"), pack.get("depth"), fast_obs, frame_seq)
                full_canvas = _draw_label_block(full_canvas, _obs_label_lines(full_obs, side="full", frame_seq=frame_seq))
                fast_canvas = _draw_label_block(fast_canvas, _obs_label_lines(fast_obs, side="fast", frame_seq=frame_seq))
                if full_canvas.shape[0] != fast_canvas.shape[0]:
                    h = min(full_canvas.shape[0], fast_canvas.shape[0])
                    full_canvas = cv2.resize(full_canvas, (full_canvas.shape[1], h), interpolation=cv2.INTER_AREA)
                    fast_canvas = cv2.resize(fast_canvas, (fast_canvas.shape[1], h), interpolation=cv2.INTER_AREA)
                paired = np.hstack([full_canvas, fast_canvas])
                out_path = preview_dir / f"compare_{frame_seq:06d}.png"
                if not cv2.imwrite(str(out_path), paired):
                    raise RuntimeError("cv2.imwrite returned false")
                compared_frames.append(frame_seq)
                print(f"[FULL_FAST_COMPARE] wrote {out_path}")
            except Exception as exc:
                failed_frames.append({"frame_seq": int(frame_seq), "error": str(exc)})
                print(f"[FULL_FAST_COMPARE] failed frame_seq={frame_seq} error={exc}")
    finally:
        full_processor.release_all()
        fast_processor.release_all()

    summary = {
        "bag": str(args.bag),
        "frame_stride": FRAME_STRIDE,
        "bag_frame_count": int(frames_total),
        "compared_frame_count": int(len(compared_frames)),
        "compared_frames": compared_frames,
        "detector_modes": ["full", "fast_plane_only_v3"],
        "roi_preset": "center_lower",
        "output_folder": str(output_dir),
        "paired_preview_folder": str(preview_dir),
        "failed_frames": failed_frames,
    }
    (output_dir / "run_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"[FULL_FAST_COMPARE] compared={len(compared_frames)} failed={len(failed_frames)} output={preview_dir}")


if __name__ == "__main__":
    main()
