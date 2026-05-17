#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import time
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
        self.layout = os.getenv("VISTA_PREVIEW_LAYOUT", os.getenv("VISION_PREVIEW_LAYOUT", "rgb_depth_edge"))
        self.scale = self._env_float("VISTA_PREVIEW_SCALE", self._env_float("VISION_PREVIEW_SCALE", 1.0))
        self.canvas_w = self._env_int("VISTA_PREVIEW_WIDTH", 1280)
        self.canvas_h = self._env_int("VISTA_PREVIEW_HEIGHT", 720)
        self.show_rgb = self._env_bool("VISTA_PREVIEW_RGB_PANEL", self._env_bool("VISTA_PREVIEW_RGB", True))
        self.show_depth = self._env_bool("VISTA_PREVIEW_DEPTH", True)
        self.show_edge = self._env_bool("VISTA_PREVIEW_EDGE", True)
        self._supported_layouts = {"rgb_minimal", "rgb_depth_edge", "rgb_yolo_edge_overlay", "rgb_hot_preview"}
        self._opened = False
        self._last_frame_ts = 0.0
        self._last_render_ts = 0.0
        self._fps = 0.0
        self._frame_count = 0

    @staticmethod
    def _env_int(name: str, default: int) -> int:
        try:
            return int(os.getenv(name, str(default)))
        except (TypeError, ValueError):
            return default

    @staticmethod
    def _env_float(name: str, default: float) -> float:
        try:
            return float(os.getenv(name, str(default)))
        except (TypeError, ValueError):
            return default

    @staticmethod
    def _env_bool(name: str, default: bool) -> bool:
        raw = os.getenv(name)
        if raw is None:
            return bool(default)
        return str(raw).strip().lower() in {"1", "true", "yes", "on"}

    def open(self) -> None:
        """Prepare the dashboard window and sink-local resources."""
        try:
            cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
            cv2.resizeWindow(self.window_name, int(self.canvas_w * self.scale), int(self.canvas_h * self.scale))
        except Exception:
            cv2.namedWindow(self.window_name)
        self._opened = True

    def set_layout(self, layout: str, reason: str = "") -> None:
        """Switch layout without replacing or rebuilding the OpenCV window."""
        value = str(layout or "").strip() or "rgb_minimal"
        self.layout = value if value in self._supported_layouts else "rgb_minimal"

    def render(self, frame: PreviewFrame) -> bool:
        """Render one frame bundle and return False when the user asks to exit."""
        if not self._opened:
            self.open()

        frames = frame.image if isinstance(frame.image, dict) else {}
        metadata = dict(getattr(frame.overlay, "metadata", {}) or {})
        mode = str(dict(metadata.get("runtime_status") or {}).get("mode") or frame.mode or "").upper()
        table_edge = metadata.get("table_edge_obs") or {}
        target_obs = metadata.get("target_obs") or {}
        layout = str(metadata.get("preview_layout") or self.layout or "rgb_minimal").strip()
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
        elif layout == "rgb_depth_edge":
            panel_w = max(320, self.canvas_w // 2)
            panel_h = max(220, self.canvas_h // 2)
            panel_size = (panel_w, panel_h)

            edge_metadata = dict(metadata)
            edge_metadata["preview_layout"] = layout
            edge_metadata["window_id"] = self.window_id
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

        cv2.imshow(self.window_name, canvas)
        self._last_frame_ts = float(frame.ts or 0.0)
        self._update_fps()
        if cv2.waitKey(1) & 0xFF == 27:
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
        if mode == "TRACK_LOCAL" or bool(metadata.get("show_yolo_boxes", False)):
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
            if mode == "TRACK_LOCAL":
                self._draw_roi(panel, target_obs.get("bbox"), scale, offset, f"target:{target_name or 'target'}", (0, 255, 80), dashed=False)
            else:
                self._draw_roi(panel, target_obs.get("bbox"), scale, offset, "target_bbox", (90, 180, 255), dashed=False)
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
        for name, color in (
            ("edge_roi", (255, 255, 255)),
            ("table_edge_roi", (80, 220, 255)),
        ):
            roi = table_edge.get(name)
            if roi:
                self._draw_roi(panel, roi, scale, offset, name, color, dashed=(name == "table_edge_roi"))
        valid_for_control = self._boolish(table_edge.get("valid_for_control", table_edge.get("edge_valid")))
        source = table_edge.get("final_pose_source") or table_edge.get("pose_source") or "none"
        geometry_lines = [
            f"use={source}",
            f"plane yaw={self._fmt(table_edge.get('yaw_err_rad'))}",
            f"dist={self._fmt(table_edge.get('dist_err_m'))}",
            f"score={self._fmt(table_edge.get('front_plane_score', table_edge.get('table_geometry_score')))}",
            f"control={table_edge.get('control_level') or 'none'}",
        ]
        left_y = max(76, panel.shape[0] - 116)
        self._text_block(panel, geometry_lines, (16, left_y), fg=(210, 255, 210) if valid_for_control else (210, 230, 255), max_width=min(260, panel.shape[1] - 32), font_scale=0.46, line_h=19)
        return panel

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
        if str(status.get("mode") or "").upper() != "TRACK_LOCAL":
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
        fallback_mask = valid
        has_geometry = bool(
            table_edge.get("front_plane_candidate_pixels")
        )
        if not has_geometry:
            panel[fallback_mask] = (95, 95, 95)
        self._draw_pixel_points(panel, table_edge.get("front_plane_candidate_pixels"), (80, 210, 80), radius=1)
        return panel

    def _draw_pixel_points(self, panel: np.ndarray, points: Any, color: Tuple[int, int, int], radius: int = 1) -> None:
        if not isinstance(points, (list, tuple)):
            return
        h, w = panel.shape[:2]
        for item in points[:2000]:
            if not isinstance(item, (list, tuple)) or len(item) < 2:
                continue
            try:
                x = int(round(float(item[0])))
                y = int(round(float(item[1])))
            except Exception:
                continue
            if 0 <= x < w and 0 <= y < h:
                cv2.circle(panel, (x, y), int(radius), color, -1)

    def _find_table_bbox(self, local: Dict[str, Any], rgb_shape: Any = None) -> Optional[List[int]]:
        if not self._env_bool("VISTA_TABLE_BBOX_ENABLE", True):
            return None
        mock_raw = os.getenv("VISTA_MOCK_TABLE_BBOX")
        mock = self._parse_roi(mock_raw)
        if mock is not None:
            return mock
        if self._env_bool("VISTA_MOCK_TABLE_BBOX", False):
            mock = self._mock_table_bbox(local.get("rgb_shape") or rgb_shape)
            if mock is not None:
                local = dict(local)
                local["mock_table_bbox"] = mock
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
        self._corner_note(panel, f"target={target_name or 'n/a'} boxes={len(boxes)}", fg=(0, 255, 80) if boxes else (80, 210, 255))

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
        sections = [
            ("TABLE", [
                f"found={self._boolish(table_edge.get('edge_found'))} control={self._boolish(table_edge.get('valid_for_control', table_edge.get('edge_valid')))} source={table_edge.get('final_pose_source') or table_edge.get('pose_source', 'n/a')}",
                f"level={table_edge.get('control_level', 'none')} geom={self._fmt(table_edge.get('table_geometry_score'))} stable={table_edge.get('stable_count', 'n/a')}",
                f"usable approach={int(self._boolish(table_edge.get('usable_for_approach')))} align={int(self._boolish(table_edge.get('usable_for_alignment')))} stop={int(self._boolish(table_edge.get('usable_for_stop')))}",
                f"yaw={self._fmt(table_edge.get('yaw_err_rad'))} dist={self._fmt(table_edge.get('dist_err_m'))} target={self._fmt(table_edge.get('target_dist_m'))}",
                f"front_score={self._fmt(table_edge.get('front_plane_score'))} conf={self._fmt(table_edge.get('plane_confidence'))} area={self._fmt(table_edge.get('front_face_area_ratio'))}",
                f"span={self._fmt(table_edge.get('plane_x_span_m'))} residual={self._fmt(table_edge.get('plane_residual_mean'))} inliers={table_edge.get('inlier_count', table_edge.get('edge_inlier_count', 'n/a'))}",
                f"reject geom={table_edge.get('geometry_reject_reason') or 'ok'} ctrl={table_edge.get('control_reject_reason') or table_edge.get('reject_reason') or table_edge.get('reason') or 'ok'}",
            ]),
            ("ROI", [
                f"rgb_roi={local.get('rgb_search_roi') or local.get('search_roi') or 'n/a'}",
                f"depth_roi={table_edge.get('depth_edge_roi') or 'n/a'}",
                f"edge_roi={table_edge.get('edge_roi') or table_edge.get('table_edge_roi') or 'n/a'}",
                f"table_bbox={table_bbox_status} table_quadrant={table_quadrant}",
                f"roi_source={table_edge.get('roi_source') or 'n/a'}",
            ]),
            ("TASK", [
                f"stage={status.get('stage', 'IDLE')}",
                f"mode={status.get('mode', 'IDLE')}",
                f"preview_layout={metadata.get('preview_layout') or self.layout or 'rgb_minimal'}",
                f"epoch={status.get('epoch', 0)} req={status.get('req_id') or ''}",
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

    def close(self) -> None:
        """Destroy sink-local resources and close the dashboard window."""
        if self._opened:
            try:
                cv2.destroyWindow(self.window_name)
            except Exception:
                pass
            try:
                if self._env_bool("VISTA_PREVIEW_DESTROY_ALL_ON_CLOSE", True):
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
                "opened": self._opened,
                "last_frame_ts": self._last_frame_ts,
                "fps": self._fps,
            }
        )
        return snap
