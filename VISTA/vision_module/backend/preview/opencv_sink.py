#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import math
import os
import time
from collections import deque
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np

try:
    import aidcv as cv2
except ImportError:
    import cv2

from .base import PreviewFrame, PreviewSink


try:
    from ...utils.table_roi import find_table_bbox as _public_find_table_bbox
except Exception:
    _public_find_table_bbox = None


YELLOW: Tuple[int, int, int] = (0, 255, 255)
CYAN: Tuple[int, int, int] = (255, 255, 0)


class OpenCVPreviewSink(PreviewSink):
    """OpenCV dashboard for local VISTA field debugging."""

    sink_name = "opencv"

    def __init__(self, window_name: str = "VISTA Preview"):
        self.window_name = window_name
        self.window_id = f"{id(self):x}"
        self.layout = "rgb_depth_edge"
        self.scale = 1.0
        self.canvas_w = 1280
        self.canvas_h = 720
        self.show_rgb = True
        self.show_depth = True
        self.show_edge = True
        self.destroy_all_on_close = True
        self.table_bbox_enabled = str(os.getenv("VISTA_TABLE_BBOX_ENABLE", "1")).strip().lower() in {"1", "true", "yes", "on"}
        self.mock_table_bbox: Optional[str] = str(os.getenv("VISTA_MOCK_TABLE_BBOX", "") or "").strip() or None
        self.text_level = "compact"
        self.debug_points_enabled = False
        self.legend_level = "compact"
        self._supported_layouts = {"rgb_minimal", "rgb_depth_edge", "rgb_yolo_edge_overlay", "rgb_yolo_overlay", "rgb_hot_preview"}
        self._opened = False
        self._open_failed = False
        self._open_error = ""
        self._last_frame_ts = 0.0
        self._last_render_ts = 0.0
        self._fps = 0.0
        self._frame_count = 0
        self._timing_frame: Optional[Dict[str, float]] = None
        self._timing_recent = deque(maxlen=240)

    def configure_display(self, **kwargs: Any) -> None:
        """Accept display/table-bbox params pushed from RuntimeSupervisor (CONFIG)."""
        for key in (
            "layout",
            "scale",
            "canvas_w",
            "canvas_h",
            "show_rgb",
            "show_depth",
            "show_edge",
            "destroy_all_on_close",
            "table_bbox_enabled",
            "mock_table_bbox",
        ):
            if key in kwargs:
                setattr(self, key, kwargs[key])

    @staticmethod
    def _env_choice(name: str, default: str, allowed: set) -> str:
        value = str(os.getenv(name, default) or default).strip().lower()
        return value if value in allowed else default

    def _add_timing(self, key: str, ms: float) -> None:
        if self._timing_frame is not None:
            self._timing_frame[key] = float(self._timing_frame.get(key, 0.0) or 0.0) + float(ms)

    def open(self) -> None:
        """Prepare the dashboard window and sink-local resources."""
        if self._open_failed:
            return
        try:
            cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
            cv2.resizeWindow(self.window_name, int(self.canvas_w * self.scale), int(self.canvas_h * self.scale))
            self._opened = True
            self._open_error = ""
            return
        except Exception as first_exc:
            try:
                cv2.namedWindow(self.window_name)
                self._opened = True
                self._open_error = ""
                return
            except Exception as second_exc:
                self._opened = False
                self._open_failed = True
                self._open_error = str(second_exc or first_exc or "preview_open_failed")

    def set_layout(self, layout: str, reason: str = "") -> None:
        """Switch layout without replacing or rebuilding the OpenCV window."""
        value = str(layout or "").strip() or "rgb_minimal"
        self.text_level = "compact"
        self.debug_points_enabled = False
        self.legend_level = "compact"
        self.layout = value if value in self._supported_layouts else "rgb_minimal"

    def render(self, frame: PreviewFrame) -> bool:
        """Render one frame bundle and return False when the user asks to exit."""
        total_start = time.perf_counter()
        self._timing_frame = {
            "preview_compose_ms": 0.0,
            "preview_draw_points_ms": 0.0,
            "preview_draw_text_ms": 0.0,
            "preview_draw_legend_ms": 0.0,
            "preview_imshow_ms": 0.0,
            "preview_waitkey_ms": 0.0,
        }
        if not self._opened:
            self.open()
        if not self._opened:
            self._timing_frame = None
            return True

        compose_start = time.perf_counter()
        frames = frame.image if isinstance(frame.image, dict) else {}
        metadata = dict(getattr(frame.overlay, "metadata", {}) or {})
        mode = str(dict(metadata.get("runtime_status") or {}).get("mode") or frame.mode or "").upper()
        table_edge = metadata.get("table_edge_obs") or {}
        target_obs = metadata.get("target_obs") or {}
        layout = str(metadata.get("preview_layout") or self.layout or "rgb_minimal").strip()
        self.text_level = "compact"
        self.debug_points_enabled = False
        self.legend_level = "compact"
        if layout not in self._supported_layouts:
            layout = "rgb_minimal"
        self.layout = layout

        if layout == "rgb_yolo_edge_overlay":
            rgb_w = max(480, int(self.canvas_w * 0.68))
            info_w = max(280, self.canvas_w - rgb_w)
            target_metadata = dict(metadata)
            target_metadata["preview_layout"] = layout
            target_metadata["window_id"] = self.window_id
            rgb_panel = self._make_rgb_panel(frames.get("rgb") if frames else frame.image, target_metadata, (rgb_w, self.canvas_h))
            info_table_edge = table_edge if bool(metadata.get("show_edge_overlay_in_track_local", True)) else {}
            info_panel = self._make_info_panel(frame, target_metadata, info_table_edge, target_obs, (info_w, self.canvas_h))
            canvas = np.hstack([rgb_panel, info_panel])
        elif layout == "rgb_yolo_overlay":
            table_metadata = dict(metadata)
            table_metadata["preview_layout"] = layout
            table_metadata["window_id"] = self.window_id
            table_metadata["show_yolo_boxes"] = True
            canvas = self._make_rgb_panel(frames.get("rgb") if frames else frame.image, table_metadata, (self.canvas_w, self.canvas_h))
        elif layout == "rgb_depth_edge":
            panel_w = max(320, self.canvas_w // 2)
            panel_h = max(220, self.canvas_h // 2)
            panel_size = (panel_w, panel_h)

            edge_metadata = dict(metadata)
            edge_metadata["preview_layout"] = layout
            edge_metadata["window_id"] = self.window_id
            if mode == "FIND_EDGE":
                edge_metadata["show_yolo_boxes"] = True
            rgb_panel = self._make_rgb_panel(frames.get("rgb") if frames else frame.image, edge_metadata, panel_size)
            depth_panel = self._make_depth_panel(frames.get("depth") if frames else frame.image, table_edge, panel_size)
            edge_panel = self._make_edge_panel(frames.get("depth") if frames else None, table_edge, panel_size)
            info_panel = self._make_info_panel(frame, edge_metadata, table_edge, target_obs, panel_size)

            top = np.hstack([rgb_panel, depth_panel])
            bottom = np.hstack([edge_panel, info_panel])
            canvas = np.vstack([top, bottom])
        else:
            minimal_metadata = dict(metadata)
            minimal_metadata["preview_layout"] = layout
            minimal_metadata["window_id"] = self.window_id
            title = "RGB HOT PREVIEW" if layout == "rgb_hot_preview" else "RGB MINIMAL"
            canvas = self._make_minimal_rgb_panel(frames.get("rgb") if frames else frame.image, minimal_metadata, (self.canvas_w, self.canvas_h), title=title)

        if self.canvas_w > 0 and self.canvas_h > 0 and canvas.shape[:2] != (self.canvas_h, self.canvas_w):
            canvas = cv2.resize(canvas, (self.canvas_w, self.canvas_h), interpolation=cv2.INTER_AREA)
        if self.scale > 0 and abs(self.scale - 1.0) > 1e-3:
            canvas = cv2.resize(canvas, None, fx=self.scale, fy=self.scale, interpolation=cv2.INTER_AREA)
        self._add_timing("preview_compose_ms", (time.perf_counter() - compose_start) * 1000.0)

        imshow_start = time.perf_counter()
        cv2.imshow(self.window_name, canvas)
        self._add_timing("preview_imshow_ms", (time.perf_counter() - imshow_start) * 1000.0)
        self._last_frame_ts = float(frame.ts or 0.0)
        self._update_fps()
        wait_start = time.perf_counter()
        key = cv2.waitKey(1)
        self._add_timing("preview_waitkey_ms", (time.perf_counter() - wait_start) * 1000.0)
        timing = dict(self._timing_frame or {})
        timing["preview_total_ms"] = (time.perf_counter() - total_start) * 1000.0
        timing["preview_fps"] = float(self._fps)
        timing["ts"] = time.time()
        timing["preview_layout"] = self.layout
        timing["preview_text_level"] = self.text_level
        timing["preview_legend_level"] = self.legend_level
        timing["preview_debug_points_enabled"] = bool(self.debug_points_enabled)
        self._timing_recent.append(timing)
        self._timing_frame = None
        if key & 0xFF == 27:
            return False
        return True

    def _update_fps(self) -> None:
        now = time.time()
        self._frame_count += 1
        if self._last_render_ts <= 0.0:
            self._last_render_ts = now
            return
        dt = max(1e-6, now - self._last_render_ts)
        instant = 1.0 / dt
        self._fps = instant if self._fps <= 0.0 else (0.85 * self._fps + 0.15 * instant)
        self._last_render_ts = now

    def _make_rgb_panel(self, image: Any, metadata: Dict[str, Any], size: Tuple[int, int]) -> np.ndarray:
        if not self.show_rgb:
            return self._blank(size, "RGB", "RGB panel disabled")
        if not isinstance(image, np.ndarray) or image.size == 0:
            panel = self._blank(size, "RGB", "RGB unavailable / depth only")
            return panel
        local = dict(metadata.get("local_perception") or {})
        target_obs = dict(metadata.get("target_obs") or {})
        status = dict(metadata.get("runtime_status") or {})
        mode = str(status.get("mode") or "").upper()
        target_name = str(target_obs.get("target") or metadata.get("target") or status.get("target") or "").strip()
        panel, scale, offset = self._fit_with_transform(self._to_bgr(image, prefer_rgb=False), size)
        self._title(panel, "RGB")
        if mode in {"FIND_OBJECT", "FIND_TABLE"} or bool(metadata.get("show_yolo_boxes", False)):
            self._draw_detection_boxes(panel, local, scale, offset, target_name)
        else:
            table_bbox = self._find_table_bbox(local, getattr(image, "shape", None))
            if table_bbox is None:
                self._corner_note(panel, "table_bbox unavailable", fg=YELLOW)
            else:
                self._draw_roi(panel, table_bbox, scale, offset, "table_bbox", YELLOW, dashed=False)
                table_quadrant = self._table_quadrant(local, metadata)
                if table_quadrant:
                    self._corner_note(panel, f"table_quadrant={table_quadrant}", fg=YELLOW)
            search_roi = local.get("rgb_search_roi") or local.get("search_roi") or metadata.get("rgb_search_roi") or metadata.get("search_roi")
            object_roi = local.get("object_search_roi") or metadata.get("object_search_roi")
            quadrant_roi = local.get("quadrant_roi") or metadata.get("quadrant_roi") or self._quadrant_roi(table_bbox, local, metadata)
            if search_roi:
                self._draw_roi(panel, search_roi, scale, offset, "search_roi", YELLOW, dashed=True)
            if quadrant_roi:
                self._draw_roi(panel, quadrant_roi, scale, offset, "quadrant_roi", CYAN, dashed=True)
            if object_roi:
                self._draw_roi(panel, object_roi, scale, offset, "object_roi", CYAN, dashed=True)
        if isinstance(target_obs.get("bbox"), (list, tuple)):
            if mode == "FIND_OBJECT":
                self._draw_roi(panel, target_obs.get("bbox"), scale, offset, f"target:{target_name or 'target'}", (0, 255, 80), dashed=False)
            else:
                self._draw_roi(panel, target_obs.get("bbox"), scale, offset, "target_bbox", (90, 180, 255), dashed=False)
        self._draw_table_edge_line_overlay(panel, dict(metadata.get("table_edge_obs") or {}), scale, offset)
        if metadata.get("frame_stale"):
            self._text_block(panel, [f"frame_stale age={float(metadata.get('frame_age_s') or 0.0):.2f}s"], (16, panel.shape[0] - 24), fg=(80, 210, 255))
        return panel

    def _make_minimal_rgb_panel(self, image: Any, metadata: Dict[str, Any], size: Tuple[int, int], title: str = "RGB MINIMAL") -> np.ndarray:
        if not isinstance(image, np.ndarray) or image.size == 0:
            return self._blank(size, title, "rgb stale/null")
        panel, _scale, _offset = self._fit_with_transform(self._to_bgr(image, prefer_rgb=False), size)
        self._title(panel, title)
        status = dict(metadata.get("runtime_status") or {})
        lines = [
            f"stage={status.get('stage', 'IDLE')}",
            f"mode={status.get('mode', 'IDLE')}",
            f"preview_layout={metadata.get('preview_layout') or self.layout}",
        ]
        frame_age = metadata.get("frame_age_s")
        if frame_age is not None and bool(metadata.get("show_age_ms", True)):
            lines.append(f"frame_age_ms={int(float(frame_age or 0.0) * 1000.0)}")
        if metadata.get("frame_stale"):
            lines.append("rgb stale")
        self._text_block(panel, lines, (16, 64), fg=(230, 235, 240))
        return panel

    def _make_depth_panel(self, image: Any, table_edge: Dict[str, Any], size: Tuple[int, int]) -> np.ndarray:
        if not self.show_depth:
            return self._blank(size, "DEPTH", "depth panel disabled")
        if not isinstance(image, np.ndarray) or image.size == 0:
            return self._blank(size, "DEPTH", "depth stale/null")
        panel, scale, offset = self._fit_with_transform(self._depth_colormap(image), size)
        self._title(panel, "DEPTH COLORMAP")
        for name, color in (
            ("depth_edge_roi", (80, 255, 255)),
            ("table_edge_roi", (80, 190, 255)),
            ("edge_roi", (255, 255, 120)),
        ):
            roi = table_edge.get(name)
            if roi:
                self._draw_roi(panel, roi, scale, offset, name, color, dashed=(name != "depth_edge_roi"))
        preview_roi = table_edge.get("preview_roi_draw") if isinstance(table_edge.get("preview_roi_draw"), dict) else {}
        fixed_roi_enabled = self._boolish(table_edge.get("fixed_roi_enabled") or preview_roi.get("enabled"))
        fixed_roi = table_edge.get("final_fixed_roi_xyxy") or table_edge.get("fixed_roi_xyxy") or preview_roi.get("xyxy")
        if fixed_roi_enabled and fixed_roi:
            mean = self._fmt_na(table_edge.get("final_fixed_roi_depth_mean") or table_edge.get("fixed_roi_depth_mean"))
            p10 = self._fmt_na(table_edge.get("final_fixed_roi_depth_p10") or table_edge.get("fixed_roi_depth_p10"))
            self._draw_roi(panel, fixed_roi, scale, offset, f"final_fixed_roi mean={mean} p10={p10}", (255, 120, 40), dashed=False)
        self._draw_pixel_points_transformed(panel, table_edge.get("fast_sampled_pixels"), scale, offset, (80, 80, 120), radius=1, alpha=0.35)
        self._draw_pixel_points_transformed(panel, table_edge.get("fast_candidate_pixels"), scale, offset, (0, 220, 255), radius=1, alpha=0.78)
        self._draw_pixel_points_transformed(panel, table_edge.get("fast_edge_pixels"), scale, offset, (255, 190, 40), radius=2, alpha=0.95)
        self._draw_pixel_points_transformed(panel, table_edge.get("fast_support_pixels"), scale, offset, (255, 235, 80), radius=1, alpha=0.72)
        self._draw_pixel_points_transformed(panel, table_edge.get("fast_inlier_pixels"), scale, offset, (40, 255, 80), radius=3, alpha=0.95)
        self._draw_pixel_points_transformed(panel, table_edge.get("front_plane_candidate_pixels"), scale, offset, (40, 220, 70), radius=2, alpha=0.55)
        if self.debug_points_enabled:
            self._draw_pixel_points_transformed(panel, table_edge.get("fast_background_pixels"), scale, offset, (120, 160, 255), radius=3, alpha=0.85, marker="cross")
            self._draw_pixel_points_transformed(panel, table_edge.get("fast_weak_pixels"), scale, offset, (210, 80, 210), radius=3, alpha=0.85, marker="cross")
            self._draw_pixel_points_transformed(panel, table_edge.get("fast_outlier_pixels"), scale, offset, (40, 40, 255), radius=4, alpha=0.90, marker="cross")
        if table_edge.get("fast_candidate_pixels") or table_edge.get("fast_edge_pixels") or table_edge.get("fast_support_pixels"):
            self._draw_fast_legend(panel)
        valid = self._valid_depth(image)
        if np.any(valid):
            vals = image[valid].astype(np.float32)
            roi_source = table_edge.get("roi_source") or "n/a"
            quadrant = table_edge.get("table_quadrant") or "n/a"
            self._corner_note(
                panel,
                f"valid={int(valid.sum())} range={float(vals.min()):.0f}..{float(vals.max()):.0f} roi={roi_source} q={quadrant}",
            )
        return panel

    def _make_edge_panel(self, depth: Any, table_edge: Dict[str, Any], size: Tuple[int, int]) -> np.ndarray:
        if not self.show_edge:
            return self._blank(size, "EDGE / TOP VIEW", "edge panel disabled")
        if isinstance(depth, np.ndarray) and depth.size:
            panel, scale, offset = self._fit_with_transform(self._edge_debug_image(depth, table_edge), size)
        else:
            panel = self._blank(size, "EDGE / TOP VIEW", "depth stale/null")
            scale, offset = 1.0, (0, 0)
        self._title(panel, "GEOMETRY / PLANE")
        if not table_edge:
            self._corner_note(panel, "table_edge_obs unavailable", fg=(40, 220, 255))
            return panel
        worker_error = str(table_edge.get("table_edge_worker_error") or table_edge.get("worker_error") or "").strip()
        if worker_error:
            self._corner_note(panel, f"worker_error={worker_error[:64]}", fg=(60, 120, 255))
        roi = table_edge.get("edge_roi") or table_edge.get("table_edge_roi") or table_edge.get("depth_edge_roi")
        if roi:
            self._draw_roi(panel, roi, scale, offset, "ROI", (80, 220, 255), dashed=True)
        fast_sparse = bool(
            table_edge.get("fast_candidate_pixels")
            or table_edge.get("fast_support_pixels")
            or table_edge.get("fast_front_face_rep_pixels")
            or table_edge.get("fast_inlier_pixels")
            or table_edge.get("fast_outlier_pixels")
            or table_edge.get("fast_background_pixels")
            or table_edge.get("fast_weak_pixels")
            or table_edge.get("fast_edge_pixels")
        )
        plane_mask_missing = not bool(
            table_edge.get("plane_mask_status") in ("present", "fast_sparse")
            or table_edge.get("front_plane_inlier_mask")
            or table_edge.get("front_plane_candidate_mask")
            or table_edge.get("front_plane_candidate_pixels")
            or fast_sparse
        )
        if plane_mask_missing:
            self._corner_note(panel, "plane_mask=missing", fg=(40, 220, 255))
        valid_for_control = self._boolish(table_edge.get("valid_for_control", table_edge.get("edge_valid")))
        source = table_edge.get("final_pose_source") or table_edge.get("pose_source") or "none"
        geometry_lines = [
            f"use={source}",
            f"yaw={self._fmt_na(table_edge.get('yaw_err_rad'))}rad/{self._fmt_na(self._deg(table_edge.get('yaw_err_rad')))}deg",
            f"dist={self._fmt_na(table_edge.get('dist_err_m'))}m",
            f"conf={self._fmt_na(table_edge.get('confidence', table_edge.get('edge_conf')))}",
            f"control={table_edge.get('control_level') or 'none'}",
            f"roi={roi or 'NA'}",
            f"calib={table_edge.get('calib_source') or 'NA'}",
            f"reject={table_edge.get('fast_gate_reason') or table_edge.get('fast_gate_reject_reason') or table_edge.get('reject_reason') or table_edge.get('reason') or 'none'}",
        ]
        calib_warning = str(table_edge.get("calib_mismatch_warning") or "").strip()
        if calib_warning:
            geometry_lines.append(calib_warning[:92])
        if fast_sparse:
            yaw = table_edge.get("fast_raw_yaw_err_rad", table_edge.get("yaw_err_rad"))
            yaw_num = self._as_float(yaw)
            yaw_deg = None if yaw_num is None else yaw_num * 180.0 / math.pi
            geometry_lines.extend(
                [
                    f"mode={table_edge.get('detector_mode') or 'fast_plane_only'} frame={table_edge.get('frame_seq', 'NA')}",
                    f"sampled={table_edge.get('fast_raw_sampled_point_count', table_edge.get('sampled_point_count', 'NA'))} height_candidate={table_edge.get('fast_candidate_point_count', table_edge.get('fast_raw_candidate_count', 'NA'))}",
                    f"support_pixels={table_edge.get('fast_support_point_count', table_edge.get('fast_front_face_support_point_count', 'NA'))} reps={table_edge.get('fast_rep_count', table_edge.get('fast_front_face_rep_count', 'NA'))} in={table_edge.get('fast_rep_inlier_count', table_edge.get('fast_representative_inlier_count', 'NA'))} out={table_edge.get('fast_rep_outlier_count', 'NA')}",
                    f"clusters={table_edge.get('fast_rep_cluster_count', 'NA')} selected={table_edge.get('fast_selected_cluster_index', 'NA')} y={self._fmt_na(table_edge.get('fast_selected_cluster_y_center'))} score={self._fmt_na(table_edge.get('fast_selected_cluster_score'))}",
                    f"front_reps={table_edge.get('fast_front_rep_count', 'NA')} front_span={self._fmt_na(table_edge.get('fast_selected_cluster_x_span_m'))} background_reps={table_edge.get('fast_background_rep_count', 'NA')}",
                    f"line_source={table_edge.get('fast_line_source') or table_edge.get('fast_fit_line_source') or 'NA'} edge={table_edge.get('fast_edge_inlier_count', table_edge.get('fast_edge_candidate_count', 'NA'))} edge_span={self._fmt_na(table_edge.get('fast_edge_x_span_m'))} edge_score={self._fmt_na(table_edge.get('fast_edge_support_score'))}",
                    f"frontness={self._fmt_na(table_edge.get('fast_frontness_score'))} bg_blocked={int(bool(table_edge.get('fast_background_blocked', False)))} local_band={table_edge.get('fast_local_band_support_count', 'NA')}/{self._fmt_na(table_edge.get('fast_local_band_x_span_m'))}/{self._fmt_na(table_edge.get('fast_local_band_residual_mean'))}",
                    f"span cand/support/rep/fit={self._fmt_na(table_edge.get('fast_candidate_x_span_m'))}/{self._fmt_na(table_edge.get('fast_support_x_span_m'))}/{self._fmt_na(table_edge.get('fast_rep_x_span_m'))}/{self._fmt_na(table_edge.get('fast_fit_inlier_x_span_m', table_edge.get('fast_raw_plane_x_span_m')))}",
                    f"yaw={self._fmt_na(yaw)}rad/{self._fmt_na(yaw_deg)}deg dist={self._fmt_na(table_edge.get('fast_raw_dist_err_m', table_edge.get('dist_err_m')))}",
                    f"conf={self._fmt_na(table_edge.get('fast_score_final', table_edge.get('fast_raw_confidence')))} control={table_edge.get('fast_control_level') or table_edge.get('control_level') or 'none'} reject={table_edge.get('fast_gate_reason') or table_edge.get('fast_gate_reject_reason') or table_edge.get('reject_reason') or 'none'}",
                    f"resid mean/p90={self._fmt_na(table_edge.get('fast_residual_mean', table_edge.get('fast_raw_residual_mean')))}/{self._fmt_na(table_edge.get('fast_residual_p90', table_edge.get('fast_raw_residual_p90')))}",
                ]
            )
        if self.text_level == "compact":
            geometry_lines = geometry_lines[:9]
        left_y = max(60, panel.shape[0] - (116 + (198 if fast_sparse else 0)))
        self._text_block(panel, geometry_lines, (16, left_y), fg=(210, 255, 210) if valid_for_control else (210, 230, 255), max_width=min(330, panel.shape[1] - 32), font_scale=0.40 if fast_sparse else 0.46, line_h=16 if fast_sparse else 19)
        return panel

    @staticmethod
    def _deg(value: Any) -> Optional[float]:
        try:
            return float(value) * 180.0 / math.pi
        except Exception:
            return None

    @staticmethod
    def _fmt_depth_shape(value: Any) -> str:
        if isinstance(value, (list, tuple)) and len(value) >= 2:
            try:
                return f"{int(value[1])}x{int(value[0])}"
            except Exception:
                return "NA"
        return "NA"

    def _make_info_panel(
        self,
        frame: PreviewFrame,
        metadata: Dict[str, Any],
        table_edge: Dict[str, Any],
        target_obs: Dict[str, Any],
        size: Tuple[int, int],
    ) -> np.ndarray:
        w, h = size
        panel = np.zeros((h, w, 3), dtype=np.uint8)
        panel[:] = (24, 28, 34)
        now = time.time()
        frame_age = now - float(frame.ts or now)
        target_name = target_obs.get("target") or metadata.get("target") or "unavailable"
        source_cameras = metadata.get("source_cameras") or []
        status = dict(metadata.get("runtime_status") or {})
        local = dict(metadata.get("local_perception") or {})
        self._title(panel, "STATUS / TOP VIEW")
        self._draw_status_sections(panel, metadata, status, table_edge, target_obs, local, source_cameras, target_name, frame_age)
        if str(status.get("mode") or "").upper() != "FIND_OBJECT":
            self._draw_top_view(panel, table_edge, graph=(w - 250, h - 170, w - 18, h - 18))
        return panel

    def _to_bgr(self, image: np.ndarray, prefer_rgb: bool = False) -> np.ndarray:
        arr = np.asarray(image)
        if arr.ndim == 2:
            return cv2.cvtColor(self._normalize_gray(arr), cv2.COLOR_GRAY2BGR)
        if arr.ndim == 3 and arr.shape[2] >= 3:
            out = arr[:, :, :3].copy()
            if out.dtype != np.uint8:
                out = self._normalize_gray(out)
            if prefer_rgb:
                try:
                    out = cv2.cvtColor(out, cv2.COLOR_RGB2BGR)
                except Exception:
                    out = out[:, :, ::-1]
            return out
        return np.zeros((1, 1, 3), dtype=np.uint8)

    def _depth_colormap(self, image: np.ndarray) -> np.ndarray:
        arr = np.asarray(image)
        if arr.ndim == 3:
            arr = arr[:, :, 0]
        gray = self._normalize_depth(arr)
        try:
            colored = cv2.applyColorMap(gray, cv2.COLORMAP_TURBO)
        except Exception:
            colored = cv2.applyColorMap(gray, cv2.COLORMAP_JET)
        invalid = ~self._valid_depth(arr)
        if invalid.shape == colored.shape[:2]:
            colored[invalid] = (20, 20, 20)
        return colored

    def _normalize_depth(self, image: np.ndarray) -> np.ndarray:
        arr = np.asarray(image).astype(np.float32)
        valid = self._valid_depth(arr)
        if not np.any(valid):
            return np.zeros(arr.shape[:2], dtype=np.uint8)
        vals = arr[valid]
        lo, hi = np.percentile(vals, [2, 98])
        if hi <= lo:
            hi = lo + 1.0
        gray = np.clip((arr - lo) * 255.0 / (hi - lo), 0, 255).astype(np.uint8)
        gray[~valid] = 0
        return gray

    def _normalize_gray(self, image: np.ndarray) -> np.ndarray:
        arr = np.asarray(image)
        if arr.dtype == np.uint8:
            return arr.copy()
        arr = arr.astype(np.float32)
        lo, hi = float(np.nanmin(arr)), float(np.nanmax(arr))
        if hi <= lo:
            return np.zeros(arr.shape[:2], dtype=np.uint8)
        return np.clip((arr - lo) * 255.0 / (hi - lo), 0, 255).astype(np.uint8)

    def _valid_depth(self, image: np.ndarray) -> np.ndarray:
        arr = np.asarray(image)
        return np.isfinite(arr) & (arr > 0)

    def _blank(self, size: Tuple[int, int], title: str, message: str) -> np.ndarray:
        w, h = size
        panel = np.zeros((h, w, 3), dtype=np.uint8)
        panel[:] = (18, 22, 28)
        self._title(panel, title)
        self._text_block(panel, [message], (16, 70), fg=(80, 210, 255))
        return panel

    def _fit(self, image: np.ndarray, size: Tuple[int, int]) -> np.ndarray:
        return self._fit_with_transform(image, size)[0]

    def _fit_with_transform(self, image: np.ndarray, size: Tuple[int, int]) -> Tuple[np.ndarray, float, Tuple[int, int]]:
        w, h = size
        bgr = self._to_bgr(image)
        ih, iw = bgr.shape[:2]
        if ih <= 0 or iw <= 0:
            return np.zeros((h, w, 3), dtype=np.uint8), 1.0, (0, 0)
        scale = min(w / float(iw), h / float(ih))
        nw, nh = max(1, int(iw * scale)), max(1, int(ih * scale))
        resized = cv2.resize(bgr, (nw, nh), interpolation=cv2.INTER_AREA)
        panel = np.zeros((h, w, 3), dtype=np.uint8)
        panel[:] = (12, 14, 18)
        x0, y0 = (w - nw) // 2, (h - nh) // 2
        panel[y0 : y0 + nh, x0 : x0 + nw] = resized
        return panel, float(scale), (int(x0), int(y0))

    def _edge_debug_image(self, depth: np.ndarray, table_edge: Dict[str, Any]) -> np.ndarray:
        arr = np.asarray(depth)
        if arr.ndim == 3:
            arr = arr[:, :, 0]
        base = self._depth_colormap(arr)
        panel = cv2.addWeighted(base, 0.35, np.zeros_like(base), 0.65, 0)
        roi = self._parse_roi(table_edge.get("edge_roi") or table_edge.get("table_edge_roi") or table_edge.get("depth_edge_roi"))
        z_min = self._as_float(table_edge.get("depth_z_min_m"))
        z_max = self._as_float(table_edge.get("depth_z_max_m"))
        scale = 0.001
        depth_m = arr.astype(np.float32) * scale
        valid = np.isfinite(depth_m) & (depth_m > 0)
        if z_min is not None:
            valid &= depth_m >= z_min
        if z_max is not None:
            valid &= depth_m <= z_max
        if roi is not None:
            x1, y1, x2, y2 = self._clip_roi(roi, arr.shape[1], arr.shape[0])
            roi_mask = np.zeros_like(valid, dtype=bool)
            roi_mask[y1:y2, x1:x2] = True
            valid &= roi_mask
        plane_mask = self._plane_mask_to_frame(table_edge, arr.shape[:2])
        if plane_mask is not None:
            self._overlay_mask(panel, plane_mask, (40, 220, 70), alpha=0.42)
        else:
            self._draw_pixel_points(panel, table_edge.get("fast_sampled_pixels"), (80, 80, 120), radius=1, alpha=0.45)
            self._draw_pixel_points(panel, table_edge.get("fast_candidate_pixels"), (0, 220, 255), radius=1, alpha=0.78)
            self._draw_pixel_points(panel, table_edge.get("fast_edge_pixels"), (255, 190, 40), radius=2, alpha=0.95)
            self._draw_pixel_points(panel, table_edge.get("fast_support_pixels"), (255, 235, 80), radius=1, alpha=0.72)
            self._draw_pixel_points(panel, table_edge.get("fast_front_face_rep_pixels"), (255, 160, 255), radius=4, alpha=0.95, marker="cross")
            self._draw_pixel_points(panel, table_edge.get("fast_inlier_pixels"), (40, 255, 80), radius=4, alpha=0.98, marker="circle")
            if self.debug_points_enabled:
                self._draw_pixel_points(panel, table_edge.get("fast_background_pixels"), (120, 160, 255), radius=4, alpha=0.88, marker="cross")
                self._draw_pixel_points(panel, table_edge.get("fast_weak_pixels"), (210, 80, 210), radius=4, alpha=0.88, marker="cross")
                self._draw_pixel_points(panel, table_edge.get("fast_outlier_pixels"), (40, 40, 255), radius=5, alpha=0.95, marker="cross")
            self._draw_pixel_points(panel, table_edge.get("front_plane_candidate_pixels"), (40, 220, 70), radius=2, alpha=0.55)
            if table_edge.get("fast_candidate_pixels") or table_edge.get("fast_support_pixels"):
                self._draw_fast_legend(panel)
        return panel

    def _plane_mask_to_frame(self, table_edge: Dict[str, Any], frame_shape: Tuple[int, int]) -> Optional[np.ndarray]:
        for key in ("front_plane_inlier_mask", "front_plane_mask", "plane_mask", "front_plane_candidate_mask", "plane_candidate_mask"):
            mask = self._decode_mask_payload(table_edge.get(key), frame_shape)
            if mask is not None and bool(mask.any()):
                return mask
        return None

    def _decode_mask_payload(self, payload: Any, frame_shape: Tuple[int, int]) -> Optional[np.ndarray]:
        try:
            frame_h, frame_w = int(frame_shape[0]), int(frame_shape[1])
        except Exception:
            return None
        if frame_h <= 0 or frame_w <= 0:
            return None
        if isinstance(payload, dict):
            shape_raw = payload.get("shape")
            if not isinstance(shape_raw, (list, tuple)) or len(shape_raw) < 2:
                return None
            try:
                mh, mw = int(shape_raw[0]), int(shape_raw[1])
            except Exception:
                return None
            if mh <= 0 or mw <= 0:
                return None
            mask = np.zeros((mh * mw,), dtype=bool)
            if str(payload.get("encoding") or "").lower() == "rle":
                counts = payload.get("counts")
                if not isinstance(counts, (list, tuple)):
                    return None
                for i in range(0, len(counts) - 1, 2):
                    try:
                        start = max(0, int(counts[i]))
                        length = max(0, int(counts[i + 1]))
                    except Exception:
                        continue
                    end = min(mask.size, start + length)
                    if start < end:
                        mask[start:end] = True
            else:
                data = payload.get("data")
                if data is None:
                    return None
                try:
                    mask = np.asarray(data).reshape((mh * mw,)).astype(bool)
                except Exception:
                    return None
            mask_2d = mask.reshape((mh, mw))
            roi = self._parse_roi(payload.get("roi"))
            coord = str(payload.get("coord") or ("roi" if roi is not None else "full")).lower()
            if coord == "full" or (mh == frame_h and mw == frame_w and roi is None):
                return mask_2d if mask_2d.shape == (frame_h, frame_w) else None
            if roi is None:
                return None
            x1, y1, x2, y2 = self._clip_roi(roi, frame_w, frame_h)
            if x2 <= x1 or y2 <= y1:
                return None
            local_h, local_w = y2 - y1, x2 - x1
            local = mask_2d
            if local.shape != (local_h, local_w):
                local = cv2.resize(local.astype(np.uint8), (local_w, local_h), interpolation=cv2.INTER_NEAREST).astype(bool)
            out = np.zeros((frame_h, frame_w), dtype=bool)
            out[y1:y2, x1:x2] = local[:local_h, :local_w]
            return out
        try:
            arr = np.asarray(payload)
        except Exception:
            return None
        if arr.ndim != 2:
            return None
        mask = arr.astype(bool)
        if mask.shape == (frame_h, frame_w):
            return mask
        return None

    def _overlay_mask(self, panel: np.ndarray, mask: np.ndarray, color: Tuple[int, int, int], alpha: float = 0.4) -> None:
        if mask is None or mask.shape[:2] != panel.shape[:2] or not bool(mask.any()):
            return
        overlay = np.empty_like(panel)
        overlay[:] = color
        panel[mask] = cv2.addWeighted(panel, 1.0 - float(alpha), overlay, float(alpha), 0)[mask]

    def _draw_fast_legend(self, panel: np.ndarray) -> None:
        start = time.perf_counter()
        items = [
            ("sampled", (80, 80, 120), "circle"),
            ("height_candidate", (0, 220, 255), "circle"),
            ("front_edge", (255, 190, 40), "circle"),
            ("support/inlier", (40, 255, 80), "circle"),
            ("final_line", (255, 235, 80), "circle"),
        ]
        if self.legend_level in {"debug", "full"}:
            items.extend(
                [
                    ("reps", (255, 160, 255), "cross"),
                    ("rep_outliers", (40, 40, 255), "cross"),
                    ("rep_background", (120, 160, 255), "cross"),
                    ("rep_weak", (210, 80, 210), "cross"),
                ]
            )
        x = max(12, panel.shape[1] - 150)
        y = max(56, panel.shape[0] - (len(items) * 14 + 18))
        self._rect(panel, (x - 6, y - 14), (panel.shape[1] - 6, y + len(items) * 14 + 3), (0, 0, 0), 0.42)
        for idx, (label, color, marker) in enumerate(items):
            yy = y + idx * 14
            if marker == "cross":
                cv2.drawMarker(panel, (x + 5, yy - 4), color, cv2.MARKER_TILTED_CROSS, 8, 1, line_type=cv2.LINE_AA)
            else:
                cv2.circle(panel, (x + 5, yy - 4), 2, color, -1, lineType=cv2.LINE_AA)
            cv2.putText(panel, label, (x + 15, yy), cv2.FONT_HERSHEY_SIMPLEX, 0.30, (230, 230, 230), 1, cv2.LINE_AA)
        self._add_timing("preview_draw_legend_ms", (time.perf_counter() - start) * 1000.0)

    def _draw_pixel_points(self, panel: np.ndarray, points: Any, color: Tuple[int, int, int], radius: int = 1, alpha: float = 1.0, marker: str = "circle") -> None:
        start = time.perf_counter()
        if not isinstance(points, (list, tuple)):
            return
        h, w = panel.shape[:2]
        target = panel if alpha >= 1.0 else panel.copy()
        for item in points[:2000]:
            if not isinstance(item, (list, tuple)) or len(item) < 2:
                continue
            try:
                x = int(round(float(item[0])))
                y = int(round(float(item[1])))
            except Exception:
                continue
            if 0 <= x < w and 0 <= y < h:
                if marker == "cross":
                    cv2.drawMarker(target, (x, y), color, cv2.MARKER_TILTED_CROSS, max(5, int(radius) * 3), 1, line_type=cv2.LINE_AA)
                else:
                    cv2.circle(target, (x, y), int(radius), color, -1, lineType=cv2.LINE_AA)
        if alpha < 1.0:
            cv2.addWeighted(target, float(alpha), panel, 1.0 - float(alpha), 0, dst=panel)
        self._add_timing("preview_draw_points_ms", (time.perf_counter() - start) * 1000.0)

    def _draw_pixel_points_transformed(
        self,
        panel: np.ndarray,
        points: Any,
        scale: float,
        offset: Tuple[int, int],
        color: Tuple[int, int, int],
        radius: int = 1,
        alpha: float = 1.0,
        marker: str = "circle",
    ) -> None:
        start = time.perf_counter()
        if not isinstance(points, (list, tuple)):
            return
        h, w = panel.shape[:2]
        ox, oy = int(offset[0]), int(offset[1])
        target = panel if alpha >= 1.0 else panel.copy()
        for item in points[:2000]:
            if not isinstance(item, (list, tuple)) or len(item) < 2:
                continue
            try:
                x = int(round(float(item[0]) * float(scale))) + ox
                y = int(round(float(item[1]) * float(scale))) + oy
            except Exception:
                continue
            if 0 <= x < w and 0 <= y < h:
                if marker == "cross":
                    cv2.drawMarker(target, (x, y), color, cv2.MARKER_TILTED_CROSS, max(5, int(radius) * 3), 1, line_type=cv2.LINE_AA)
                else:
                    cv2.circle(target, (x, y), int(radius), color, -1, lineType=cv2.LINE_AA)
        if alpha < 1.0:
            cv2.addWeighted(target, float(alpha), panel, 1.0 - float(alpha), 0, dst=panel)
        self._add_timing("preview_draw_points_ms", (time.perf_counter() - start) * 1000.0)

    def _find_table_bbox(self, local: Dict[str, Any], rgb_shape: Any = None) -> Optional[List[int]]:
        if not self.table_bbox_enabled:
            return None
        mock_raw = self.mock_table_bbox
        if mock_raw:
            mock = self._parse_roi(mock_raw)
            if mock is not None:
                return mock
            if str(mock_raw).strip().lower() in {"1", "true", "yes", "on"}:
                mock = self._mock_table_bbox(local.get("rgb_shape") or rgb_shape)
                if mock is not None:
                    return mock
        if callable(_public_find_table_bbox):
            try:
                roi = self._parse_roi(_public_find_table_bbox(local))
                if roi is not None:
                    return roi
            except Exception:
                pass
        for key in ("table_bbox", "desk_bbox"):
            roi = self._parse_roi(local.get(key))
            if roi is not None:
                return roi
        boxes = local.get("infer_boxes")
        if not isinstance(boxes, list):
            return None
        for row in boxes:
            try:
                if len(row) < 6:
                    continue
                cls_id = int(float(row[5]))
                cls_name = str(row[6]).strip().lower() if len(row) > 6 else ""
                if cls_name in {"table", "desk", "diningtable"} or cls_id == 60:
                    return [int(float(v)) for v in row[:4]]
            except Exception:
                continue
        return None

    @staticmethod
    def _mock_table_bbox(rgb_shape: Any) -> Optional[List[int]]:
        if not isinstance(rgb_shape, (list, tuple)) or len(rgb_shape) < 2:
            return None
        try:
            h = int(rgb_shape[0])
            w = int(rgb_shape[1])
        except Exception:
            return None
        if h <= 0 or w <= 0:
            return None
        return [w // 4, h // 2, (w * 3) // 4, (h * 9) // 10]

    def _table_quadrant(self, local: Dict[str, Any], metadata: Dict[str, Any]) -> str:
        value = local.get("table_quadrant") or local.get("quadrant") or metadata.get("table_quadrant") or metadata.get("quadrant")
        text = str(value or "").strip().upper()
        text = {
            "TOP_LEFT": "LT",
            "TOP_RIGHT": "RT",
            "BOTTOM_LEFT": "LB",
            "BOTTOM_RIGHT": "RB",
        }.get(text, text)
        return text if text in {"LT", "RT", "LB", "RB"} else ""

    def _quadrant_roi(self, table_bbox: Optional[List[int]], local: Dict[str, Any], metadata: Dict[str, Any]) -> Optional[List[int]]:
        if table_bbox is None:
            return None
        quadrant = self._table_quadrant(local, metadata)
        if not quadrant:
            return None
        x1, y1, x2, y2 = table_bbox
        mx = int(round((x1 + x2) / 2.0))
        my = int(round((y1 + y2) / 2.0))
        if quadrant == "LT":
            return [x1, y1, mx, my]
        if quadrant == "RT":
            return [mx, y1, x2, my]
        if quadrant == "LB":
            return [x1, my, mx, y2]
        if quadrant == "RB":
            return [mx, my, x2, y2]
        return None

    def _parse_roi(self, roi: Any) -> Optional[List[int]]:
        if isinstance(roi, str):
            roi = [part.strip() for part in roi.replace(";", ",").split(",") if part.strip()]
        if not isinstance(roi, (list, tuple)) or len(roi) < 4:
            return None
        try:
            return [int(round(float(v))) for v in roi[:4]]
        except Exception:
            return None

    def _clip_roi(self, roi: List[int], width: int, height: int) -> Tuple[int, int, int, int]:
        x1, y1, x2, y2 = roi
        x1 = max(0, min(width - 1, int(x1)))
        y1 = max(0, min(height - 1, int(y1)))
        x2 = max(0, min(width, int(x2)))
        y2 = max(0, min(height, int(y2)))
        if x2 <= x1:
            x2 = min(width, x1 + 1)
        if y2 <= y1:
            y2 = min(height, y1 + 1)
        return x1, y1, x2, y2

    def _draw_roi(
        self,
        panel: np.ndarray,
        roi: Any,
        scale: float,
        offset: Tuple[int, int],
        label: str,
        color: Tuple[int, int, int],
        dashed: bool = False,
    ) -> None:
        parsed = self._parse_roi(roi)
        if parsed is None:
            return
        x1, y1, x2, y2 = parsed
        ox, oy = offset
        p1 = (int(ox + x1 * scale), int(oy + y1 * scale))
        p2 = (int(ox + x2 * scale), int(oy + y2 * scale))
        if dashed:
            self._dashed_rect(panel, p1, p2, color, 2)
        else:
            cv2.rectangle(panel, p1, p2, color, 2)
        self._label(panel, label, (p1[0], max(42, p1[1] - 6)), color)

    def _draw_detection_boxes(
        self,
        panel: np.ndarray,
        local: Dict[str, Any],
        scale: float,
        offset: Tuple[int, int],
        target_name: str,
    ) -> None:
        boxes = local.get("infer_boxes")
        if not isinstance(boxes, list):
            boxes = []
        class_names = local.get("class_names") if isinstance(local.get("class_names"), (list, tuple)) else []
        target_norm = str(target_name or "").strip().lower()
        for row in boxes:
            try:
                if len(row) < 6:
                    continue
                cls_id = int(float(row[5]))
                cls_name = str(row[6]).strip() if len(row) > 6 else ""
                if not cls_name and 0 <= cls_id < len(class_names):
                    cls_name = str(class_names[cls_id])
                conf = float(row[4])
                label = f"{cls_name or cls_id}:{conf:.2f}"
                is_target = bool(target_norm and cls_name.strip().lower() == target_norm)
                color = (0, 255, 80) if is_target else (90, 180, 255)
                self._draw_roi(panel, row[:4], scale, offset, label, color, dashed=False)
            except Exception:
                continue
        if boxes:
            self._corner_note(panel, f"target={target_name or 'n/a'} boxes={len(boxes)}", fg=(0, 255, 80))
        else:
            reason = str(
                local.get("infer_error")
                or local.get("no_boxes_reason")
                or local.get("contract_error")
                or ("predictor_not_ready" if not local.get("has_infer") else "no_boxes")
                or "no_boxes"
            )
            self._corner_note(panel, f"target={target_name or 'n/a'} boxes=0 reason={reason[:44]}", fg=(80, 210, 255))

    def _draw_table_edge_line_overlay(
        self,
        panel: np.ndarray,
        table_edge: Dict[str, Any],
        scale: float,
        offset: Tuple[int, int],
    ) -> None:
        if not table_edge:
            return
        worker_error = str(table_edge.get("table_edge_worker_error") or table_edge.get("worker_error") or "").strip()
        roi = self._parse_roi(table_edge.get("edge_roi") or table_edge.get("table_edge_roi") or table_edge.get("depth_edge_roi"))
        if worker_error:
            self._corner_note(panel, f"worker_error={worker_error[:48]}", fg=(60, 120, 255))
            return
        k = self._as_float(table_edge.get("image_line_k"))
        b = self._as_float(table_edge.get("image_line_b"))
        if k is not None and b is not None:
            h, w = panel.shape[:2]
            ox, oy = offset
            x_min = 0 if roi is None else roi[0]
            x_max = int((w - ox) / max(1e-6, scale)) if roi is None else roi[2]
            pts = []
            for x in np.linspace(float(x_min), float(x_max), 24):
                y = k * float(x) + b
                px = int(round(ox + x * scale))
                py = int(round(oy + y * scale))
                if 0 <= px < w and 0 <= py < h:
                    pts.append((px, py))
            for p1, p2 in zip(pts[:-1], pts[1:]):
                cv2.line(panel, p1, p2, YELLOW, 2, lineType=cv2.LINE_AA)
            current = self._boolish(table_edge.get("preview_line_is_current_frame", not table_edge.get("is_stale", False)))
            source = str(table_edge.get("preview_line_source") or "image_line").strip() or "image_line"
            label = "EDGE_CURRENT" if current else "LAST_EDGE"
            self._corner_note(panel, f"{label} source={source}", fg=YELLOW)
        elif table_edge.get("edge_found"):
            self._corner_note(panel, "edge line image projection unavailable", fg=YELLOW)
        elif table_edge.get("reason") or table_edge.get("reject_reason"):
            self._corner_note(panel, f"edge_missing reason={str(table_edge.get('reject_reason') or table_edge.get('reason'))[:48]}", fg=(80, 210, 255))

    def _dashed_rect(self, panel: np.ndarray, p1: Tuple[int, int], p2: Tuple[int, int], color, thickness: int) -> None:
        x1, y1 = p1
        x2, y2 = p2
        step = 16
        for x in range(x1, x2, step * 2):
            cv2.line(panel, (x, y1), (min(x + step, x2), y1), color, thickness)
            cv2.line(panel, (x, y2), (min(x + step, x2), y2), color, thickness)
        for y in range(y1, y2, step * 2):
            cv2.line(panel, (x1, y), (x1, min(y + step, y2)), color, thickness)
            cv2.line(panel, (x2, y), (x2, min(y + step, y2)), color, thickness)

    def _label(self, panel: np.ndarray, text: str, origin: Tuple[int, int], color: Tuple[int, int, int]) -> None:
        x, y = origin
        self._rect(panel, (x, y - 20), (x + min(220, 12 + len(text) * 9), y + 4), (0, 0, 0), 0.65)
        cv2.putText(panel, text, (x + 4, y - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1)

    def _corner_note(self, panel: np.ndarray, text: str, fg: Tuple[int, int, int] = (230, 230, 230)) -> None:
        self._text_block(panel, [text], (16, 58), fg=fg)

    def _draw_status_sections(
        self,
        panel: np.ndarray,
        metadata: Dict[str, Any],
        status: Dict[str, Any],
        table_edge: Dict[str, Any],
        target_obs: Dict[str, Any],
        local: Dict[str, Any],
        source_cameras: Any,
        target_name: Any,
        frame_age: float,
    ) -> None:
        predictor = "enabled" if local.get("has_infer") else "disabled"
        table_bbox_status = "available" if self._find_table_bbox(local) is not None else "unavailable"
        table_quadrant = self._table_quadrant(local, {}) or table_edge.get("table_quadrant") or "n/a"
        mode_display = {
            "FIND_EDGE": "TABLE_EDGE_PERCEPTION",
            "FIND_OBJECT": "TRACK_LOCAL",
            "FIND_TABLE": "YOLO_TABLE_SEARCH",
        }.get(str(status.get("mode") or "IDLE").strip().upper(), status.get("mode", "IDLE"))
        sections = [
            ("TASK", [
                f"stage={status.get('stage', 'IDLE')}",
                f"mode={mode_display}",
                f"preview_layout={metadata.get('preview_layout') or self.layout or 'rgb_minimal'}",
                f"window_id={metadata.get('window_id') or self.window_id}",
                f"epoch={status.get('epoch', 0)} req={status.get('req_id') or ''}",
            ]),
            ("TABLE", [
                f"worker_error={(table_edge.get('table_edge_worker_error') or table_edge.get('worker_error') or 'none')}",
                f"found={self._boolish(table_edge.get('edge_found'))} control={self._boolish(table_edge.get('valid_for_control', table_edge.get('edge_valid')))} source={table_edge.get('final_pose_source') or table_edge.get('pose_source', 'n/a')}",
                f"level={table_edge.get('control_level', 'none')} geom={self._fmt(table_edge.get('table_geometry_score'))} stable={table_edge.get('stable_count', 'n/a')}",
                f"depth={self._fmt_depth_shape(table_edge.get('depth_shape'))} calib={table_edge.get('calib_source') or 'n/a'}",
                f"fx/fy={self._fmt_na(table_edge.get('fx'))}/{self._fmt_na(table_edge.get('fy'))} cx/cy={self._fmt_na(table_edge.get('cx'))}/{self._fmt_na(table_edge.get('cy'))}",
                f"candidate/support/inliers={table_edge.get('fast_candidate_point_count', table_edge.get('candidate_count', 'n/a'))}/{table_edge.get('fast_support_point_count', table_edge.get('support_point_count', 'n/a'))}/{table_edge.get('inlier_count', table_edge.get('edge_inlier_count', 'n/a'))}",
                f"line={table_edge.get('fast_line_source') or table_edge.get('fast_fit_line_source') or table_edge.get('selected_line_type') or 'n/a'} residual={self._fmt(table_edge.get('fast_residual_mean', table_edge.get('plane_residual_mean')))}",
                f"frontness={self._fmt(table_edge.get('fast_frontness_score'))} bg_blocked={int(bool(table_edge.get('fast_background_blocked', False)))}",
                f"yaw={self._fmt(table_edge.get('yaw_err_rad'))} dist={self._fmt(table_edge.get('dist_err_m'))} target={self._fmt(table_edge.get('target_dist_m'))}",
                f"reject geom={table_edge.get('geometry_reject_reason') or 'ok'} ctrl={table_edge.get('control_reject_reason') or table_edge.get('reject_reason') or table_edge.get('reason') or 'ok'}",
            ]),
            ("ROI", [
                f"rgb_roi={local.get('rgb_search_roi') or local.get('search_roi') or 'n/a'}",
                f"depth_roi={table_edge.get('depth_edge_roi') or 'n/a'}",
                f"edge_roi={table_edge.get('edge_roi') or table_edge.get('table_edge_roi') or 'n/a'}",
                f"table_bbox={table_bbox_status} table_quadrant={table_quadrant}",
                f"roi_source={table_edge.get('roi_source') or 'n/a'}",
                f"final_fixed_roi enabled={self._boolish(table_edge.get('fixed_roi_enabled'))} xyxy={table_edge.get('final_fixed_roi_xyxy') or table_edge.get('fixed_roi_xyxy') or 'n/a'}",
                f"final_fixed_depth mean={self._fmt_na(table_edge.get('final_fixed_roi_depth_mean'))} median={self._fmt_na(table_edge.get('final_fixed_roi_depth_median'))} p10={self._fmt_na(table_edge.get('final_fixed_roi_depth_p10'))}",
            ]),
            ("TARGET", [
                f"target={target_name}",
                f"has_target_obs={bool(target_obs)}",
                f"target_found={self._boolish(target_obs.get('target_found', target_obs.get('found')))} conf={self._fmt(target_obs.get('confidence'))}",
                f"boxes_count={target_obs.get('boxes_count', local.get('box_count', 0))}",
                f"matched_cls={target_obs.get('matched_cls', 'n/a')} matched_conf={self._fmt(target_obs.get('matched_conf'))}",
                f"best_cls={target_obs.get('best_cls', 'n/a')} best_conf={self._fmt(target_obs.get('best_conf'))}",
                f"frame_age_ms={int(frame_age * 1000.0)} infer_age_ms={int(float(target_obs.get('infer_age_ms', -1) or -1))}",
                f"predictor={predictor}",
            ]),
            ("PERF", [
                f"fps={self._fps:.1f} frame_age={frame_age:.2f}s",
                f"frame_stale={self._boolish(frame_age > 1.0)}",
                f"depth_status={'ok' if 'depth' in set(source_cameras or []) else 'stale/null'}",
                f"source_cameras={source_cameras}",
            ]),
        ]
        if self.text_level == "compact":
            sections = [(title, lines[:8]) for title, lines in sections[:4]]
        y = 54
        for title, lines in sections:
            cv2.putText(panel, title, (18, y), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (80, 255, 160), 2)
            y += 20
            for line in lines:
                cv2.putText(panel, str(line)[:58], (26, y), cv2.FONT_HERSHEY_SIMPLEX, 0.43, (230, 235, 240), 1)
                y += 18
            y += 8
            if y > panel.shape[0] - 178:
                break

    def _title(self, panel: np.ndarray, title: str) -> None:
        self._rect(panel, (8, 8), (max(190, 18 + len(title) * 13), 38), (0, 0, 0), 0.62)
        cv2.putText(panel, title, (16, 31), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)

    def _text_block(
        self,
        panel: np.ndarray,
        lines: Iterable[str],
        origin: Tuple[int, int],
        fg: Tuple[int, int, int] = (230, 230, 230),
        bg: Tuple[int, int, int] = (0, 0, 0),
        title_first: bool = False,
        max_width: Optional[int] = None,
        font_scale: float = 0.55,
        line_h: int = 24,
    ) -> None:
        start = time.perf_counter()
        clean = [str(line) for line in lines if line is not None]
        if not clean:
            return
        x, y = origin
        available_w = max(1, panel.shape[1] - x - 10)
        wanted_w = max(220, max(len(line) for line in clean) * 10 + 24)
        if max_width is not None:
            wanted_w = min(wanted_w, max(80, int(max_width)))
        width = min(available_w, wanted_w)
        height = min(panel.shape[0] - y - 8, len(clean) * line_h + 18)
        self._rect(panel, (x - 8, y - 22), (x - 8 + width, y - 22 + height), bg, 0.72)
        max_chars = max(8, int((width - 18) / 9))
        for idx, line in enumerate(clean):
            color = (80, 255, 160) if title_first and idx == 0 else fg
            scale = 0.68 if title_first and idx == 0 else float(font_scale)
            thickness = 2 if title_first and idx == 0 else 1
            yy = y + idx * line_h
            if yy > panel.shape[0] - 12:
                break
            cv2.putText(panel, line[:max_chars], (x, yy), cv2.FONT_HERSHEY_SIMPLEX, scale, color, thickness)
        self._add_timing("preview_draw_text_ms", (time.perf_counter() - start) * 1000.0)

    def _rect(
        self,
        panel: np.ndarray,
        p1: Tuple[int, int],
        p2: Tuple[int, int],
        color: Tuple[int, int, int],
        alpha: float,
    ) -> None:
        x1, y1 = max(0, p1[0]), max(0, p1[1])
        x2, y2 = min(panel.shape[1], p2[0]), min(panel.shape[0], p2[1])
        if x2 <= x1 or y2 <= y1:
            return
        overlay = panel.copy()
        cv2.rectangle(overlay, (x1, y1), (x2, y2), color, -1)
        panel[y1:y2, x1:x2] = cv2.addWeighted(overlay[y1:y2, x1:x2], alpha, panel[y1:y2, x1:x2], 1 - alpha, 0)

    def _draw_top_view(
        self,
        panel: np.ndarray,
        table_edge: Dict[str, Any],
        graph: Optional[Tuple[int, int, int, int]] = None,
    ) -> None:
        k = self._as_float(table_edge.get("edge_k"))
        b = self._as_float(table_edge.get("edge_b"))
        if k is None or b is None:
            k = self._as_float(table_edge.get("plane_k"))
            b = self._as_float(table_edge.get("plane_b"))
        if graph is None:
            graph = (panel.shape[1] - 220, panel.shape[0] - 150, panel.shape[1] - 18, panel.shape[0] - 18)
        x1, y1, x2, y2 = graph
        cv2.rectangle(panel, (x1, y1), (x2, y2), (35, 35, 35), -1)
        cv2.rectangle(panel, (x1, y1), (x2, y2), (110, 110, 110), 1)
        xmin, xmax = -0.7, 0.7
        zmin, zmax = 0.15, 1.4

        def project(x: float, z: float) -> Tuple[int, int]:
            px = int(x1 + (x - xmin) / (xmax - xmin) * (x2 - x1))
            py = int(y2 - (z - zmin) / (zmax - zmin) * (y2 - y1))
            return px, py

        cv2.line(panel, project(0.0, zmin), project(0.0, zmax), (90, 90, 90), 1)
        target_dist = self._as_float(table_edge.get("target_dist_m") or table_edge.get("target_distance_m"))
        if target_dist is not None and zmin <= target_dist <= zmax:
            p_left = project(xmin, target_dist)
            p_right = project(xmax, target_dist)
            self._dashed_line(panel, p_left, p_right, (80, 190, 255), 1)
            cv2.putText(panel, "target dist", (x1 + 8, max(y1 + 38, p_left[1] - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (80, 190, 255), 1)
        safe_dist = self._as_float(table_edge.get("safe_dist_m") or table_edge.get("safety_dist_m"))
        if safe_dist is not None and zmin <= safe_dist <= zmax:
            self._dashed_line(panel, project(xmin, safe_dist), project(xmax, safe_dist), (70, 120, 255), 1)
        def draw_xz_line(line_k: Optional[float], line_b: Optional[float], color: Tuple[int, int, int], thickness: int) -> None:
            if line_k is None or line_b is None:
                return
            pts: List[Tuple[int, int]] = []
            for x in np.linspace(xmin, xmax, 30):
                z = line_k * float(x) + line_b
                if zmin <= z <= zmax:
                    pts.append(project(float(x), float(z)))
            if len(pts) >= 2:
                for a, c in zip(pts[:-1], pts[1:]):
                    cv2.line(panel, a, c, color, thickness)
                mid = pts[len(pts) // 2]
                cv2.arrowedLine(panel, mid, (mid[0], max(y1 + 10, mid[1] - 38)), color, thickness, tipLength=0.35)

        draw_xz_line(k, b, (40, 255, 255), 2)
        dist_err = self._as_float(table_edge.get("dist_err_m"))
        if dist_err is not None:
            base_z = target_dist if target_dist is not None else 0.5
            base_z = min(zmax, max(zmin, base_z))
            start = project(0.0, base_z)
            dz = min(0.28, max(-0.28, dist_err))
            end = project(0.0, min(zmax, max(zmin, base_z + dz)))
            color = (80, 255, 255) if abs(dist_err) <= 0.03 else (80, 120, 255)
            cv2.arrowedLine(panel, start, end, color, 2, tipLength=0.35)
            cv2.putText(panel, f"dist_err={dist_err:+.3f}", (x1 + 8, y2 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.40, color, 1)
        cv2.putText(panel, f"top-view {table_edge.get('final_pose_source') or table_edge.get('pose_source') or 'edge'}", (x1 + 8, y1 + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (230, 230, 230), 1)

    def _dashed_line(
        self,
        panel: np.ndarray,
        p1: Tuple[int, int],
        p2: Tuple[int, int],
        color: Tuple[int, int, int],
        thickness: int,
    ) -> None:
        x1, y1 = p1
        x2, y2 = p2
        length = int(max(1, np.hypot(x2 - x1, y2 - y1)))
        dash = 10
        for start in range(0, length, dash * 2):
            end = min(length, start + dash)
            a = start / float(length)
            b = end / float(length)
            pa = (int(x1 + (x2 - x1) * a), int(y1 + (y2 - y1) * a))
            pb = (int(x1 + (x2 - x1) * b), int(y1 + (y2 - y1) * b))
            cv2.line(panel, pa, pb, color, thickness)

    def _boolish(self, value: Any) -> bool:
        if isinstance(value, str):
            return value.lower() in {"1", "true", "yes", "found", "valid"}
        return bool(value)

    def _as_float(self, value: Any) -> Optional[float]:
        try:
            if value is None:
                return None
            return float(value)
        except (TypeError, ValueError):
            return None

    def _fmt(self, value: Any) -> str:
        num = self._as_float(value)
        if num is None:
            return "n/a"
        return f"{num:.3f}"

    def _fmt_na(self, value: Any) -> str:
        num = self._as_float(value)
        if num is None:
            return "NA"
        return f"{num:.3f}"

    def close(self) -> None:
        """Destroy sink-local resources and close the dashboard window."""
        if self._opened:
            try:
                cv2.destroyWindow(self.window_name)
            except Exception:
                pass
            try:
                if self.destroy_all_on_close:
                    cv2.destroyAllWindows()
            except Exception:
                pass
            try:
                cv2.waitKey(1)
            except Exception:
                pass
        self._opened = False

    def snapshot(self) -> Dict[str, Any]:
        """Expose sink configuration and last-render bookkeeping."""
        snap = super().snapshot()
        timing_summary = self._timing_summary()
        snap.update(
            {
                "window_name": self.window_name,
                "window_id": self.window_id,
                "layout": self.layout,
                "scale": self.scale,
                "canvas_size": [self.canvas_w, self.canvas_h],
                "show_rgb": self.show_rgb,
                "show_depth": self.show_depth,
                "show_edge": self.show_edge,
                "preview_text_level": self.text_level,
                "preview_legend_level": self.legend_level,
                "preview_debug_points_enabled": bool(self.debug_points_enabled),
                "opened": self._opened,
                "open_failed": self._open_failed,
                "open_error": self._open_error,
                "last_frame_ts": self._last_frame_ts,
                "fps": self._fps,
                "timing": timing_summary,
            }
        )
        return snap

    def _timing_summary(self) -> Dict[str, Any]:
        now = time.time()
        recent = [item for item in self._timing_recent if now - float(item.get("ts", now) or now) <= 5.0]
        if not recent:
            return {
                "preview_enabled": bool(self._opened),
                "preview_layout": self.layout,
                "preview_fps": float(self._fps),
                "preview_text_level": self.text_level,
                "preview_legend_level": self.legend_level,
                "preview_debug_points_enabled": bool(self.debug_points_enabled),
                "sample_count": 0,
            }

        def stats(key: str) -> Dict[str, float]:
            vals = sorted(float(item.get(key, 0.0) or 0.0) for item in recent)
            if not vals:
                return {"avg": 0.0, "p95": 0.0, "max": 0.0}
            idx = min(len(vals) - 1, int(round(0.95 * float(len(vals) - 1))))
            return {"avg": sum(vals) / float(len(vals)), "p95": vals[idx], "max": vals[-1]}

        out: Dict[str, Any] = {
            "preview_enabled": bool(self._opened),
            "preview_layout": self.layout,
            "preview_fps": float(self._fps),
            "preview_text_level": self.text_level,
            "preview_legend_level": self.legend_level,
            "preview_debug_points_enabled": bool(self.debug_points_enabled),
            "sample_count": int(len(recent)),
        }
        for key in (
            "preview_compose_ms",
            "preview_draw_points_ms",
            "preview_draw_text_ms",
            "preview_draw_legend_ms",
            "preview_imshow_ms",
            "preview_waitkey_ms",
            "preview_total_ms",
        ):
            s = stats(key)
            out[f"{key}_avg"] = s["avg"]
            out[f"{key}_p95"] = s["p95"]
            out[f"{key}_max"] = s["max"]
        return out
