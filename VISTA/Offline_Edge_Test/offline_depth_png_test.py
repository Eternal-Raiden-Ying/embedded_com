#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Optional

import cv2

HERE = Path(__file__).resolve().parent
ONLINE_DIR = HERE / "Online"
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))
if str(ONLINE_DIR) not in sys.path:
    sys.path.insert(0, str(ONLINE_DIR))

from detector import OnlineTableEdgeDetector, load_calib  # type: ignore
from board_config import CONFIG  # type: ignore


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="使用 Online 目录中的轻量检测器离线验证一张 16-bit depth png，并可拼接 RGB 结果图")
    parser.add_argument("--depth-png", type=Path, required=True)
    parser.add_argument("--color-png", type=Path, default=None, help="可选，对应同一时刻的 RGB png")
    parser.add_argument("--calib-json", type=Path, default=Path("calib.json"))
    parser.add_argument("--out-dir", type=Path, default=Path("test_data/offline_test_output"))
    parser.add_argument("--preview", action="store_true", help="有显示环境时弹出预览")
    return parser


def _resolve_path(base_dir: Path, path: Optional[Path]) -> Optional[Path]:
    if path is None:
        return None
    return (base_dir / path).resolve() if not path.is_absolute() else path.resolve()


def _make_combined_preview(depth_raw, depth_vis, color_bgr):
    if color_bgr is None:
        return depth_vis
    target_h = depth_vis.shape[0]
    target_w = int(color_bgr.shape[1] * target_h / max(1, color_bgr.shape[0]))
    color_panel = cv2.resize(color_bgr, (target_w, target_h))
    gap = 12
    canvas_h = target_h
    canvas_w = depth_vis.shape[1] + gap + color_panel.shape[1]
    canvas = cv2.copyMakeBorder(depth_vis, 0, 0, 0, canvas_w - depth_vis.shape[1], cv2.BORDER_CONSTANT, value=(0, 0, 0))
    canvas[:, depth_vis.shape[1]:depth_vis.shape[1] + gap] = 0
    canvas[:, depth_vis.shape[1] + gap:depth_vis.shape[1] + gap + color_panel.shape[1]] = color_panel
    cv2.putText(canvas, "DEPTH RENDER", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    cv2.putText(canvas, "RGB REFERENCE", (depth_vis.shape[1] + gap + 20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    return canvas


def _require_synced_detector_result(result):
    required = ("image_line_k", "image_line_b", "edge_point_count")
    missing = [name for name in required if not hasattr(result, name)]
    if not missing:
        return
    raise RuntimeError(
        "detector result is missing new fields %s. "
        "Please sync Offline_Edge_Test/Online/detector.py and Offline_Edge_Test/offline_depth_png_test.py "
        "to the board together before rerunning." % (missing,)
    )


def _draw_edge_overlay(depth_vis, result, debug, image_shape):
    edge_uv = debug.get("edge_uv")
    if edge_uv is not None:
        for pt in edge_uv[:: max(1, len(edge_uv) // 120)]:
            cv2.circle(depth_vis, (int(pt[0]), int(pt[1])), 1, (0, 0, 255), -1)

    image_line_k = getattr(result, "image_line_k", None)
    image_line_b = getattr(result, "image_line_b", None)
    if image_line_k is None or image_line_b is None:
        return depth_vis

    x0, y0, x1, y1 = debug["roi_box"]
    xa = int(x0)
    xb = int(x1 - 1)
    ya = int(round(image_line_k * xa + image_line_b))
    yb = int(round(image_line_k * xb + image_line_b))
    clipped, p1, p2 = cv2.clipLine((0, 0, image_shape[1], image_shape[0]), (xa, ya), (xb, yb))
    if clipped:
        cv2.line(depth_vis, p1, p2, (0, 255, 255), 2, cv2.LINE_AA)
        cv2.circle(depth_vis, p1, 4, (255, 255, 0), -1)
        cv2.circle(depth_vis, p2, 4, (255, 255, 0), -1)
    return depth_vis


def _draw_text_block(image, lines, origin=(12, 14), line_h=24):
    x0, y0 = origin
    width = 0
    for line in lines:
        (w, _), _ = cv2.getTextSize(line, cv2.FONT_HERSHEY_SIMPLEX, 0.62, 2)
        width = max(width, w)
    block_w = width + 18
    block_h = line_h * len(lines) + 12
    overlay = image.copy()
    cv2.rectangle(overlay, (x0 - 8, y0 - 10), (x0 - 8 + block_w, y0 - 10 + block_h), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.45, image, 0.55, 0, image)
    for idx, line in enumerate(lines):
        y = y0 + idx * line_h
        cv2.putText(image, line, (x0, y), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (255, 255, 255), 2, cv2.LINE_AA)
    return image


def _project_depth_to_rgb_points(points_uv, depth_shape, color_shape):
    if points_uv is None:
        return None
    dh, dw = depth_shape[:2]
    ch, cw = color_shape[:2]
    sx = float(cw) / float(max(1, dw))
    sy = float(ch) / float(max(1, dh))
    pts = []
    for pt in points_uv:
        pts.append((int(round(float(pt[0]) * sx)), int(round(float(pt[1]) * sy))))
    return pts


def _draw_rgb_overlay(color_bgr, result, debug, depth_shape):
    if color_bgr is None:
        return None
    rgb = color_bgr.copy()
    edge_uv = debug.get("edge_uv")
    edge_pts = _project_depth_to_rgb_points(edge_uv, depth_shape, rgb.shape) if edge_uv is not None else None
    if edge_pts:
        step = max(1, len(edge_pts) // 120)
        for pt in edge_pts[::step]:
            cv2.circle(rgb, pt, 1, (0, 0, 255), -1)

    image_line_k = result.image_line_k
    image_line_b = result.image_line_b
    if image_line_k is not None and image_line_b is not None:
        dh, dw = depth_shape[:2]
        ch, cw = rgb.shape[:2]
        sx = float(cw) / float(max(1, dw))
        sy = float(ch) / float(max(1, dh))
        xa = 0
        xb = dw - 1
        ya = int(round(image_line_k * xa + image_line_b))
        yb = int(round(image_line_k * xb + image_line_b))
        p1 = (int(round(xa * sx)), int(round(ya * sy)))
        p2 = (int(round(xb * sx)), int(round(yb * sy)))
        clipped, cp1, cp2 = cv2.clipLine((0, 0, cw, ch), p1, p2)
        if clipped:
            cv2.line(rgb, cp1, cp2, (0, 255, 255), 2, cv2.LINE_AA)
            cv2.circle(rgb, cp1, 4, (255, 255, 0), -1)
            cv2.circle(rgb, cp2, 4, (255, 255, 0), -1)
    return rgb


def main() -> None:
    args = build_parser().parse_args()
    depth_png = _resolve_path(HERE, args.depth_png)
    color_png = _resolve_path(HERE, args.color_png)
    calib_json = _resolve_path(HERE, args.calib_json)
    out_dir = _resolve_path(HERE, args.out_dir)
    assert depth_png is not None and calib_json is not None and out_dir is not None
    out_dir.mkdir(parents=True, exist_ok=True)

    depth_raw = cv2.imread(str(depth_png), cv2.IMREAD_ANYDEPTH)
    if depth_raw is None:
        raise RuntimeError(f"failed to load depth png: {depth_png}")

    color_bgr = None
    if color_png is not None:
        color_bgr = cv2.imread(str(color_png), cv2.IMREAD_COLOR)
        if color_bgr is None:
            raise RuntimeError(f"failed to load color png: {color_png}")

    calib, target_dist = load_calib(calib_json)
    if float(CONFIG.detector.target_dist_m_override) > 0:
        target_dist = float(CONFIG.detector.target_dist_m_override)
    detector = OnlineTableEdgeDetector(calib, CONFIG.detector, target_dist)
    result, debug = detector.process_depth(depth_raw)
    _require_synced_detector_result(result)

    depth_vis = cv2.convertScaleAbs(depth_raw, alpha=0.06)
    depth_vis = cv2.applyColorMap(depth_vis, cv2.COLORMAP_JET)
    x0, y0, x1, y1 = debug["roi_box"]
    cv2.rectangle(depth_vis, (x0, y0), (x1, y1), (0, 255, 0), 2)
    depth_vis = _draw_edge_overlay(depth_vis, result, debug, depth_raw.shape)
    image_line_k = result.image_line_k
    image_line_b = result.image_line_b
    edge_point_count = int(result.edge_point_count)
    yaw_deg = float(result.yaw_err_rad) * 180.0 / 3.141592653589793
    lines = [
        "DEPTH EDGE RESULT",
        "edge=%d  conf=%.3f" % (int(result.edge_found), float(result.edge_confidence)),
        "yaw_deg=%.2f  dist_m=%.4f" % (yaw_deg, float(result.dist_err_m)),
        ("img_k=%.5f" % image_line_k) if image_line_k is not None else "img_k=None",
        "all=%d table=%d edge=%d" % (int(result.point_count), int(result.table_point_count), edge_point_count),
    ]
    depth_vis = _draw_text_block(depth_vis, lines)

    rgb_overlay = _draw_rgb_overlay(color_bgr, result, debug, depth_raw.shape)
    combined = _make_combined_preview(depth_raw, depth_vis, rgb_overlay if rgb_overlay is not None else color_bgr)

    stamp = time.strftime("%Y%m%d_%H%M%S")
    stem = depth_png.stem
    depth_preview_path = out_dir / f"{stem}_offline_depth_preview_{stamp}.png"
    rgb_overlay_path = out_dir / f"{stem}_offline_rgb_overlay_{stamp}.png"
    combined_preview_path = out_dir / f"{stem}_offline_preview_{stamp}.png"
    result_path = out_dir / f"{stem}_offline_result_{stamp}.json"
    cv2.imwrite(str(depth_preview_path), depth_vis)
    if rgb_overlay is not None:
        cv2.imwrite(str(rgb_overlay_path), rgb_overlay)
    cv2.imwrite(str(combined_preview_path), combined)

    gui_enabled = bool(os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY"))
    preview_enabled = bool(args.preview and gui_enabled)
    if args.preview and not gui_enabled:
        print("preview disabled: no DISPLAY/WAYLAND_DISPLAY detected; preview image was saved to file")
    if preview_enabled:
        cv2.imshow("Offline Edge Preview", combined)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    payload = {
        "depth_png": str(depth_png),
        "color_png": str(color_png) if color_png else None,
        "calib_json": str(calib_json),
        "depth_preview_png": str(depth_preview_path),
        "preview_png": str(combined_preview_path),
        "rgb_overlay_png": str(rgb_overlay_path) if rgb_overlay is not None else None,
        "edge_found": bool(result.edge_found),
        "yaw_err_rad": float(result.yaw_err_rad),
        "dist_err_m": float(result.dist_err_m),
        "edge_confidence": float(result.edge_confidence),
        "line_k": result.line_k,
        "line_b": result.line_b,
        "image_line_k": image_line_k,
        "image_line_b": image_line_b,
        "point_count": int(result.point_count),
        "table_point_count": int(result.table_point_count),
        "edge_point_count": edge_point_count,
        "roi_box": list(debug["roi_box"]),
        "gui_enabled": gui_enabled,
        "preview_enabled": preview_enabled,
    }
    with open(result_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
        f.write("\n")

    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
