#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import threading
import time
from collections import deque
from typing import Any, Callable, Dict, Optional

from .base import PreviewFrame, PreviewOverlay, PreviewSink

try:
    from ...diagnostics.operator_console import OperatorConsole
    from ...diagnostics.summaries import format_table_edge_summary, format_target_summary
except Exception:  # pragma: no cover
    OperatorConsole = None
    format_table_edge_summary = None
    format_target_summary = None


class PreviewManager:
    """Own preview sink selection and sink lifecycle.

    Stage logic should only emit overlay data. This manager should combine the
    latest frame bundle with overlay metadata and forward the result to the
    configured preview sink.
    """

    def __init__(
        self,
        sink: Optional[PreviewSink] = None,
        logger=None,
        capability_sink: Optional[Callable[[str, Dict[str, Any]], None]] = None,
        operator_console=None,
        cfg=None,
    ):
        self.sink = sink
        self.logger = logger
        self._capability_sink = capability_sink
        self.cfg = cfg
        self.operator_console = operator_console or (OperatorConsole() if OperatorConsole is not None else None)
        self.enabled = False
        self._scheduler = None
        self._generation_getter = lambda: 0
        self._runtime_running = False
        self._worker_thread: Optional[threading.Thread] = None
        self._worker_stop = threading.Event()
        self._worker_interval_s = 0.02
        self._last_frame_generation = 0
        self._last_frame_seq = 0
        self._exit_requested = False
        self._last_source_key = ""
        self._last_preview_seq = 0
        self._last_preview_emit_ts = 0.0
        self._render_times = deque(maxlen=120)
        self._last_render_error_ts = 0.0
        self._stale_warn_s = 1.0
        self._mode_layouts: Dict[str, str] = {
            "IDLE": "rgb_minimal",
            "FIND_EDGE": "rgb_depth_edge",
            "FIND_OBJECT": "rgb_yolo_edge_overlay",
            "FIND_TABLE": "rgb_yolo_overlay",
            "MICRO_ADJUST": "rgb_minimal",
            "GRASP_REMOTE": "rgb_depth_edge",
            "IDLE_HOT": "rgb_hot_preview",
        }
        self._supported_layouts = {"rgb_minimal", "rgb_depth_edge", "rgb_yolo_edge_overlay", "rgb_yolo_overlay", "rgb_hot_preview"}
        self._debug_four_panel_in_track_local = False
        self._show_edge_overlay_in_track_local = True
        self._show_age_ms = True
        self._clear_overlay_on_mode_switch = True
        self._current_mode = "IDLE"
        self._current_layout = self._mode_layouts["IDLE"]

    @staticmethod
    def _display_mode(mode: Any) -> str:
        value = str(mode or "IDLE").strip().upper() or "IDLE"
        return {
            "FIND_EDGE": "TABLE_EDGE_PERCEPTION",
            "FIND_OBJECT": "TRACK_LOCAL",
            "FIND_TABLE": "YOLO_TABLE_SEARCH",
        }.get(value, value)

    def _emit_operator(self, key: str, line: str) -> None:
        if self.operator_console is None:
            return
        try:
            self.operator_console.emit_rate_limited(key, line)
        except Exception:
            pass

    def _warn_operator(self, key: str, line: str) -> None:
        if self.operator_console is None:
            return
        try:
            emitter = getattr(self.operator_console, "emit_error", None)
            if callable(emitter):
                emitter(key, line)
            else:
                self.operator_console.emit_rate_limited(key, line)
        except Exception:
            pass

    def configure_preview_mode(
        self,
        mode: str,
        metadata: Optional[Dict[str, Any]] = None,
        reason: str = "mode_switch",
    ) -> None:
        cfg = dict(metadata or {})
        layouts = cfg.get("mode_layouts")
        if isinstance(layouts, dict):
            self._mode_layouts.update(
                {
                    str(key).strip().upper(): str(value).strip()
                    for key, value in layouts.items()
                    if str(key).strip() and str(value).strip()
                }
            )
        mode_name = str(mode or "IDLE").strip().upper() or "IDLE"
        if cfg.get("layout"):
            self._mode_layouts[mode_name] = str(cfg.get("layout")).strip()
        self._debug_four_panel_in_track_local = bool(cfg.get("debug_four_panel_in_track_local", self._debug_four_panel_in_track_local))
        self._show_edge_overlay_in_track_local = bool(cfg.get("show_edge_overlay_in_track_local", self._show_edge_overlay_in_track_local))
        self._show_age_ms = bool(cfg.get("show_age_ms", self._show_age_ms))
        self._clear_overlay_on_mode_switch = bool(cfg.get("clear_overlay_on_mode_switch", self._clear_overlay_on_mode_switch))

        old_mode = self._current_mode
        old_layout = self._current_layout
        new_layout = self._resolve_layout(mode_name)
        if mode_name == "FIND_OBJECT" and self._debug_four_panel_in_track_local:
            new_layout = "rgb_depth_edge"
        if new_layout not in self._supported_layouts:
            self._warn_operator(
                f"preview:layout_unsupported:{mode_name}:{new_layout}",
                f"[VISTA] WARN PREVIEW layout_unsupported mode={mode_name} layout={new_layout} fallback=rgb_minimal",
            )
            new_layout = "rgb_minimal"

        mode_changed = mode_name != old_mode
        layout_changed = new_layout != old_layout
        if mode_changed or layout_changed:
            if self.logger is not None:
                self.logger.info(
                    "preview layout switch | old_mode=%s new_mode=%s old_layout=%s new_layout=%s reason=%s",
                    old_mode,
                    mode_name,
                    old_layout,
                    new_layout,
                    reason,
                )
            self._emit_operator(
                f"preview:layout_switch:{old_mode}->{mode_name}:{old_layout}->{new_layout}",
                (
                    "[VISTA] PREVIEW_LAYOUT_SWITCH "
                    f"old_mode={self._display_mode(old_mode)} new_mode={self._display_mode(mode_name)} "
                    f"old_layout={old_layout} new_layout={new_layout} reason={reason}"
                ),
            )
            if self._clear_overlay_on_mode_switch:
                self._last_source_key = ""
                self._last_preview_seq = 0
                self._last_preview_emit_ts = 0.0
                self._last_frame_seq = 0
        self._current_mode = mode_name
        self._current_layout = new_layout
        setter = getattr(self.sink, "set_layout", None)
        if callable(setter):
            try:
                setter(new_layout, reason=reason)
            except Exception:
                pass

    def _resolve_layout(self, mode: str) -> str:
        mode_name = str(mode or "IDLE").strip().upper() or "IDLE"
        return str(self._mode_layouts.get(mode_name) or self._mode_layouts.get("IDLE") or "rgb_minimal").strip()

    def _table_edge_summary_line(self, status: Dict[str, Any], table_edge: Dict[str, Any]) -> str:
        if format_table_edge_summary is not None:
            line = format_table_edge_summary(status, table_edge)
            return line.replace("mode=TABLE_EDGE_PERCEPTION", "mode=DEPTH_PERCEPTION alias=TABLE_EDGE_PERCEPTION")
        found = bool(table_edge.get("table_found", table_edge.get("found", False)))
        edge_found = bool(table_edge.get("edge_found", False))
        conf = float(table_edge.get("confidence", 0.0) or 0.0)
        yaw = float(table_edge.get("yaw_err_rad") or 0.0)
        dist = float(table_edge.get("dist_err_m") or 0.0)
        roi = str(table_edge.get("roi_source") or table_edge.get("depth_edge_roi") or table_edge.get("edge_roi") or "n/a").strip()
        pts = int(table_edge.get("point_count", table_edge.get("table_point_count", 0)) or 0)
        reason = str(table_edge.get("reason") or "").strip() or "ok"
        return (
            f"[VISTA] EDGE stage={str(status.get('stage') or 'IDLE').upper()} "
            f"mode={str(status.get('mode') or 'IDLE').upper()} "
            f"found={int(found)} edge={int(edge_found)} conf={conf:.2f} "
            f"yaw={yaw:+.3f} dist={dist:+.3f} roi={roi[:32]} pts={pts} reason={reason[:42]}"
        )

    def _target_summary_line(self, status: Dict[str, Any], target_obs: Dict[str, Any]) -> str:
        found = bool(target_obs.get("target_found", target_obs.get("found", False)))
        conf = float(target_obs.get("confidence", 0.0) or 0.0)
        target = str(target_obs.get("target") or status.get("target") or "target").strip()
        mode = str(status.get("mode") or target_obs.get("mode") or "IDLE").upper()
        boxes = int(target_obs.get("boxes_count", target_obs.get("box_count", 0)) or 0)
        best_cls = str(target_obs.get("best_cls") or target_obs.get("best_class") or "n/a").strip() or "n/a"
        best_conf = float(target_obs.get("best_conf", target_obs.get("best_confidence", 0.0)) or 0.0)
        matched_cls = str(target_obs.get("matched_cls") or target_obs.get("target_cls") or "n/a").strip() or "n/a"
        matched_conf = float(target_obs.get("matched_conf", target_obs.get("confidence", 0.0)) or 0.0)
        frame_age = target_obs.get("frame_age_ms")
        infer_age = target_obs.get("infer_age_ms")
        age_part = ""
        if frame_age is not None:
            age_part += f" frame_age_ms={int(frame_age)}"
        if infer_age is not None:
            age_part += f" infer_age_ms={int(infer_age)}"
        if found:
            full_center = target_obs.get("matched_center_full_norm")
            if not isinstance(full_center, dict):
                full_center = target_obs.get("matched_center") if isinstance(target_obs.get("matched_center"), dict) else {}
            cx = float(full_center.get("cx", full_center.get("x_norm", target_obs.get("x_norm", target_obs.get("cx_norm", 0.0)))) or 0.0)
            cy = float(full_center.get("cy", full_center.get("y_norm", target_obs.get("y_norm", target_obs.get("cy_norm", 0.0)))) or 0.0)
            return (
                f"[VISTA] TARGET mode={mode} target={target[:32]} found=1 boxes={boxes} "
                f"matched_cls={matched_cls[:32]} matched_conf={matched_conf:.2f} "
                f"best_cls={best_cls[:32]} best_conf={best_conf:.2f} conf={conf:.2f} cx={cx:.2f} cy={cy:.2f}{age_part}"
            )
        fps = target_obs.get("fps")
        fps_part = f" fps={float(fps):.1f}" if fps is not None else ""
        reason_part = " reason=no_boxes" if boxes <= 0 else ""
        return (
            f"[VISTA] TARGET mode={mode} target={target[:32]} found=0 "
            f"boxes={boxes} matched_cls={matched_cls[:32]} matched_conf={matched_conf:.2f} "
            f"best_cls={best_cls[:32]} best_conf={best_conf:.2f}{fps_part}{age_part}{reason_part}"
        )

    def _target_overlay(self, status: Dict[str, Any], local: Dict[str, Any], target_obs: Dict[str, Any]) -> Dict[str, Any]:
        target = str(target_obs.get("target") or status.get("target") or local.get("target") or "target").strip() or "target"
        boxes = local.get("infer_boxes")
        if not isinstance(boxes, list):
            boxes = []
        class_names = local.get("class_names") if isinstance(local.get("class_names"), (list, tuple)) else []
        best_cls = "n/a"
        best_conf = 0.0
        for row in boxes:
            try:
                conf = float(row[4])
                cls_id = int(float(row[5]))
                cls_name = str(row[6]).strip() if len(row) > 6 else ""
                if not cls_name and 0 <= cls_id < len(class_names):
                    cls_name = str(class_names[cls_id])
                if conf >= best_conf:
                    best_conf = conf
                    best_cls = cls_name or str(cls_id)
            except Exception:
                continue
        out = dict(target_obs or {})
        out.setdefault("target", target)
        out.setdefault("found", bool(target_obs.get("found", False)))
        if out.get("bbox") and out.get("cy_norm") is None:
            try:
                rgb_shape = local.get("rgb_shape") or {}
                h = float(rgb_shape[0])
                y1, y2 = float(out["bbox"][1]), float(out["bbox"][3])
                out["cy_norm"] = max(0.0, min(1.0, ((y1 + y2) / 2.0) / max(1.0, h)))
            except Exception:
                pass
        out["boxes_count"] = int(local.get("box_count", len(boxes)) or len(boxes))
        out["best_cls"] = best_cls
        out["best_conf"] = float(best_conf)
        out.setdefault("matched_cls", target_obs.get("matched_cls"))
        out.setdefault("matched_conf", target_obs.get("matched_conf", target_obs.get("confidence")))
        out["fps"] = self._fps_snapshot()
        return out

    def _yolo_status_overlay(self, status: Dict[str, Any], local: Dict[str, Any], target_obs: Dict[str, Any]) -> Dict[str, Any]:
        out = self._target_overlay(status, local, target_obs)
        out.setdefault("found", False)
        out.setdefault("reason", "no_boxes" if int(out.get("boxes_count", 0) or 0) <= 0 else "no_target_match")
        out["has_infer"] = bool(local.get("has_infer", False))
        out["contract_ok"] = bool(local.get("contract_ok", True))
        out["contract_error"] = str(local.get("contract_error") or "")
        out["model_name"] = local.get("model_name")
        out["predictor_type"] = local.get("predictor_type")
        return out

    def _fps_snapshot(self) -> Optional[float]:
        sink = self.sink
        value = getattr(sink, "_fps", None)
        try:
            return float(value) if value is not None else None
        except Exception:
            return None

    def _emit(self, action: str, **fields: Any) -> None:
        if self._capability_sink is None:
            return
        try:
            payload = {"action": str(action or "updated").strip().lower()}
            payload.update(dict(fields or {}))
            self._capability_sink("preview", payload)
        except Exception:
            pass

    def bind_runtime(self, scheduler, generation_getter=None) -> None:
        self._scheduler = scheduler
        if callable(generation_getter):
            self._generation_getter = generation_getter

    def start_runtime(self) -> None:
        if self._worker_thread is not None and self._worker_thread.is_alive():
            return
        self._runtime_running = True
        self._exit_requested = False
        self._worker_stop.clear()
        self._last_frame_generation = 0
        self._last_frame_seq = 0
        self._worker_thread = threading.Thread(target=self._worker_loop, name="preview_manager.loop", daemon=True)
        self._worker_thread.start()

    def stop_runtime(self) -> None:
        self._runtime_running = False
        self._worker_stop.set()
        thread = self._worker_thread
        if thread is not None and thread.is_alive():
            thread.join(timeout=1.0)
        self._worker_thread = None

    def _worker_loop(self) -> None:
        while self._runtime_running and not self._worker_stop.is_set():
            if not self.enabled or self.sink is None:
                self._worker_stop.wait(timeout=0.05)
                continue
            scheduler = self._scheduler
            if scheduler is None:
                self._worker_stop.wait(timeout=self._worker_interval_s)
                continue
            frame_slot = scheduler.read_slot("camera_frames")
            if not isinstance(frame_slot, dict):
                self._worker_stop.wait(timeout=self._worker_interval_s)
                continue
            generation = int(frame_slot.get("generation", 0) or 0)
            seq = int(frame_slot.get("seq", 0) or 0)
            frames = frame_slot.get("payload")
            if generation != self._last_frame_generation:
                self._last_frame_generation = generation
                self._last_frame_seq = 0
                self._last_preview_seq = 0
                self._last_preview_emit_ts = 0.0
            if not isinstance(frames, dict):
                self._worker_stop.wait(timeout=self._worker_interval_s)
                continue
            now = time.time()
            stale_age = max(0.0, now - float(frame_slot.get("ts", now) or now))
            is_stale_repeat = seq <= self._last_frame_seq
            if is_stale_repeat:
                if stale_age < self._stale_warn_s or now - self._last_preview_emit_ts < 0.5:
                    self._worker_stop.wait(timeout=self._worker_interval_s)
                    continue
            else:
                self._last_frame_seq = seq
            image = frames.get("rgb")
            if image is None and frames:
                image = next(iter(frames.values()))
            if image is None:
                self._worker_stop.wait(timeout=self._worker_interval_s)
                continue
            status = dict(scheduler.read_result("runtime_status", default={}) or {})
            frame_meta = dict(scheduler.read_result("frame_meta", default={}) or {})
            local_slot = scheduler.read_slot("local_perception")
            local = dict((local_slot or {}).get("payload") or {})
            infer_age = max(0.0, now - float((local_slot or {}).get("ts", now) or now)) if local_slot else None
            table_edge = dict(scheduler.read_result("table_edge_obs", default={}) or {})
            target_obs = dict(scheduler.read_result("target_obs", default={}) or {})
            mode = str(status.get("mode") or "IDLE").upper()
            preview_layout = self._resolve_layout(mode)
            if mode == "FIND_OBJECT" and self._debug_four_panel_in_track_local:
                preview_layout = "rgb_depth_edge"
            if preview_layout not in self._supported_layouts:
                preview_layout = "rgb_minimal"
            if mode in {"FIND_OBJECT", "FIND_TABLE"} or preview_layout in {"rgb_yolo_overlay", "rgb_yolo_edge_overlay"}:
                target_obs = self._target_overlay(status, local, target_obs)
                if mode != "FIND_OBJECT":
                    target_obs = self._yolo_status_overlay(status, local, target_obs)
                if self._show_age_ms:
                    target_obs["frame_age_ms"] = int(round(stale_age * 1000.0))
                if infer_age is not None and self._show_age_ms:
                    target_obs["infer_age_ms"] = int(round(infer_age * 1000.0))
            if stale_age >= self._stale_warn_s:
                self._emit_operator("preview:rgb_stale", f"[VISTA] WARN PREVIEW rgb_frame_stale age={stale_age:.2f}s")
            cameras = sorted(str(k) for k in frames.keys())
            source_key = ",".join(cameras) or "none"
            if source_key != self._last_source_key:
                self._last_source_key = source_key
                if self.logger is not None:
                    self.logger.info(
                        "preview source | cameras=%s layout=%s",
                        source_key,
                        getattr(self.sink, "layout", "default") if self.sink is not None else "none",
                    )
            lines = [
                f"stage={str(status.get('stage') or 'IDLE').upper()}",
                f"mode={str(status.get('mode') or 'IDLE').upper()}",
                f"epoch={int(status.get('epoch', 0) or 0)}",
                f"boxes={int(local.get('box_count', 0) or 0)}",
            ]
            session_id = status.get("session_id")
            if session_id:
                lines.append(f"session={session_id}")
            req_id = status.get("req_id")
            if req_id:
                lines.append(f"req={req_id}")
            if table_edge:
                found = bool(table_edge.get("table_found", table_edge.get("found", False)))
                edge_found = bool(table_edge.get("edge_found", False))
                conf = float(table_edge.get("confidence", 0.0) or 0.0)
                yaw = table_edge.get("yaw_err_rad")
                dist = table_edge.get("dist_err_m")
                reason = str(table_edge.get("reason") or "").strip()
                lines.append(f"table found={int(found)} edge={int(edge_found)} conf={conf:.2f}")
                if yaw is not None or dist is not None:
                    lines.append(f"yaw={float(yaw or 0.0):+.3f} dist={float(dist or 0.0):+.3f}m")
                if reason:
                    lines.append(f"reason={reason[:42]}")
                self._emit_operator("preview:table_edge_obs", self._table_edge_summary_line(status, table_edge))
            if mode in {"FIND_OBJECT", "FIND_TABLE"} and not bool(local.get("has_infer", bool(local.get("infer_boxes")))):
                self._emit_operator("preview:target_predictor_not_ready", f"[VISTA] WARN TARGET predictor_not_ready mode={mode}")
            if mode == "FIND_OBJECT" and bool(target_obs.get("class_not_supported")):
                available = ",".join(str(v) for v in list(target_obs.get("available_classes") or [])[:16])
                self._emit_operator(
                    "preview:target_class_not_supported",
                    f"[VISTA] WARN TARGET class_not_supported target={target_obs.get('target') or status.get('target') or 'target'} available={available or 'n/a'}",
                )
            if target_obs:
                self._emit_operator("preview:target_obs", self._target_summary_line(status, target_obs))
            if not target_obs:
                lines.append("target_obs=unavailable")
            if mode in {"FIND_EDGE", "FIND_OBJECT", "FIND_TABLE"}:
                lines.append(f"preview_layout={preview_layout}")
                boxes_count = int(target_obs.get("boxes_count", local.get("box_count", 0)) or 0)
                if boxes_count <= 0:
                    reason = target_obs.get("reason") or local.get("infer_error") or local.get("no_boxes_reason") or "no_boxes"
                    lines.append(f"yolo=no_boxes reason={reason}")
                if mode == "FIND_EDGE":
                    lines.append(
                        f"yolo26_enabled={int(bool(local.get('yolo26_enabled', False)))} "
                        f"yolo_infer_running={int(bool(local.get('yolo_infer_running', local.get('has_infer', False))))} "
                        f"table_bbox_detected={int(bool(local.get('table_bbox_detected', False)))} "
                        f"table_bbox_used_for_search={int(bool(local.get('table_bbox_used_for_search', False)))} "
                        f"boxes_count={boxes_count}"
                    )
                else:
                    lines.append(
                        f"target_found={int(bool(target_obs.get('target_found', target_obs.get('found', False))))} "
                        f"matched_cls={str(target_obs.get('matched_cls') or 'n/a')[:32]} "
                        f"matched_conf={float(target_obs.get('matched_conf', target_obs.get('confidence', 0.0)) or 0.0):.2f} "
                        f"best_cls={str(target_obs.get('best_cls') or 'n/a')[:32]} "
                        f"best_conf={float(target_obs.get('best_conf', 0.0) or 0.0):.2f} "
                        f"boxes_count={boxes_count} "
                        f"frame_age_ms={int(round(stale_age * 1000.0)) if self._show_age_ms else -1} "
                        f"infer_age_ms={int(round(infer_age * 1000.0)) if infer_age is not None and self._show_age_ms else -1}"
                    )
            if stale_age >= self._stale_warn_s:
                lines.append(f"frame_stale age={stale_age:.2f}s")
            try:
                ok = self.render(
                    PreviewFrame(
                        ts=float(frame_slot.get("ts", now) or now),
                        image=dict(frames),
                        stage=str(status.get("stage") or "IDLE").upper(),
                        mode=str(status.get("mode") or "IDLE").upper(),
                        overlay=PreviewOverlay(
                            title="VISTA Preview",
                            lines=lines,
                            metadata={
                                "frame_seq": seq,
                                "frame_meta": frame_meta,
                                "runtime_status": status,
                                "local_perception": local,
                                "table_edge_obs": table_edge,
                                "target_obs": target_obs,
                                "source_cameras": cameras,
                                "frame_stale": stale_age >= self._stale_warn_s,
                                "frame_age_s": stale_age,
                                "infer_age_s": infer_age,
                                "target": target_obs.get("target") or status.get("target"),
                                "preview_layout": preview_layout,
                                "show_edge_overlay_in_track_local": bool(self._show_edge_overlay_in_track_local),
                                "show_age_ms": bool(self._show_age_ms),
                            },
                        ),
                    )
                )
            except Exception as exc:
                ok = True
                if now - self._last_render_error_ts >= 1.0:
                    self._last_render_error_ts = now
                    if self.logger is not None:
                        self.logger.exception("preview render failed | error=%s", exc)
                    self._emit_operator(
                        "preview:render_failed",
                        f"[VISTA] WARN PREVIEW render_failed error={str(exc)[:96]}",
                    )
            self._last_preview_seq = seq
            self._last_preview_emit_ts = now
            self._render_times.append(now)
            if not ok:
                self._exit_requested = True
                self.disable()
            self._worker_stop.wait(timeout=self._worker_interval_s)

    def set_sink(self, sink: PreviewSink) -> None:
        """Replace the active preview sink implementation."""
        old_name = getattr(self.sink, "sink_name", "unknown") if self.sink is not None else "none"
        if self.enabled and self.sink is not None:
            try:
                self.sink.close()
            except Exception:
                pass
        self.sink = sink
        self._push_display_config(sink)
        self._emit("sink_changed", old_sink=old_name, new_sink=getattr(sink, "sink_name", "unknown"))
        if self.enabled and self.sink is not None:
            try:
                self.sink.open()
                self._warn_if_sink_open_failed()
            except Exception:
                pass

    def _push_display_config(self, sink: PreviewSink) -> None:
        """Push global display config from CONFIG to an opencv sink."""
        configurator = getattr(sink, "configure_display", None)
        if not callable(configurator) or self.cfg is None:
            return
        try:
            preview_cfg = self.cfg.preview
            debug_cfg = self.cfg.debug
            configurator(
                scale=float(getattr(preview_cfg, "scale", 1.0) or 1.0),
                canvas_w=int(getattr(preview_cfg, "canvas_w", 1280) or 1280),
                canvas_h=int(getattr(preview_cfg, "canvas_h", 720) or 720),
                show_rgb=bool(getattr(preview_cfg, "show_rgb", True)),
                show_depth=bool(getattr(preview_cfg, "show_depth", True)),
                show_edge=bool(getattr(preview_cfg, "show_edge", True)),
                destroy_all_on_close=bool(getattr(preview_cfg, "destroy_all_on_close", True)),
                table_bbox_enabled=bool(getattr(debug_cfg, "table_bbox_enabled", True)),
                mock_table_bbox=str(getattr(debug_cfg, "mock_table_bbox", "") or "").strip() or None,
            )
        except Exception:
            pass

    def enable(self) -> bool:
        """Enable preview output for the currently configured sink."""
        if self.enabled:
            return False
        self.enabled = True
        if self.sink is not None:
            self.sink.open()
            self._warn_if_sink_open_failed()
        if self.logger is not None:
            self.logger.info(
                "preview started | sink=%s",
                getattr(self.sink, "sink_name", "unknown") if self.sink is not None else "none",
            )
        self._emit("enabled", enabled=True, sink_name=getattr(self.sink, "sink_name", "unknown"))
        return True

    def _warn_if_sink_open_failed(self) -> None:
        if self.sink is None or not hasattr(self.sink, "snapshot"):
            return
        try:
            snapshot = self.sink.snapshot()
        except Exception:
            return
        if not snapshot.get("open_failed"):
            return
        error = str(snapshot.get("open_error") or "unknown")[:160]
        sink_name = getattr(self.sink, "sink_name", "unknown")
        if self.logger is not None:
            self.logger.warning("preview open failed | sink=%s error=%s", sink_name, error)
        self._emit_operator("preview:open_failed", f"[VISTA] WARN PREVIEW open_failed sink={sink_name} error={error}")

    def disable(self) -> bool:
        """Disable preview output and close sink-local resources."""
        if not self.enabled:
            return False
        self.enabled = False
        if self.sink is not None:
            self.sink.close()
        if self.logger is not None:
            self.logger.info(
                "preview disabled | sink=%s",
                getattr(self.sink, "sink_name", "unknown") if self.sink is not None else "none",
            )
        self._emit("disabled", enabled=False, sink_name=getattr(self.sink, "sink_name", "unknown"))
        return True

    def render(self, frame: PreviewFrame) -> bool:
        """Forward one preview frame to the sink if preview is enabled."""
        if not self.enabled or self.sink is None:
            return True
        return self.sink.render(frame)

    def snapshot(self) -> Dict[str, Any]:
        """Expose preview manager and sink state for diagnostics."""
        now = time.time()
        recent = [float(ts) for ts in self._render_times if now - float(ts) <= 5.0]
        preview_fps = (len(recent) / 5.0) if recent else 0.0
        sink_snapshot = self.sink.snapshot() if self.sink is not None else None
        timing = dict((sink_snapshot or {}).get("timing") or {})
        return {
            "enabled": self.enabled,
            "sink": sink_snapshot,
            "runtime_running": bool(self._runtime_running),
            "last_frame_generation": int(self._last_frame_generation),
            "last_frame_seq": int(self._last_frame_seq),
            "exit_requested": bool(self._exit_requested),
            "current_mode": self._current_mode,
            "current_layout": self._current_layout,
            "mode_layouts": dict(self._mode_layouts),
            "preview_fps": float(preview_fps),
            "preview_timing": timing,
        }
