#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import threading
import time
from typing import Any, Callable, Dict, Optional

from .base import PreviewFrame, PreviewOverlay, PreviewSink

try:
    from common.runtime_logging import OperatorConsole
except Exception:  # pragma: no cover - fallback for direct package execution
    OperatorConsole = None


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
    ):
        self.sink = sink
        self.logger = logger
        self._capability_sink = capability_sink
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

    def _emit_operator(self, key: str, line: str) -> None:
        if self.operator_console is None:
            return
        try:
            self.operator_console.emit_rate_limited(key, line)
        except Exception:
            pass

    def _table_edge_summary_line(self, status: Dict[str, Any], table_edge: Dict[str, Any]) -> str:
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
        found = bool(target_obs.get("found", False))
        conf = float(target_obs.get("confidence", 0.0) or 0.0)
        target = str(target_obs.get("target") or status.get("target") or "target").strip()
        cx = float(target_obs.get("cx_norm", 0.0) or 0.0)
        size = float(target_obs.get("size_norm", target_obs.get("area_norm", 0.0)) or 0.0)
        return (
            f"[VISTA] TARGET stage={str(status.get('stage') or 'IDLE').upper()} "
            f"mode={str(status.get('mode') or 'IDLE').upper()} target={target[:32]} "
            f"found={int(found)} conf={conf:.2f} cx={cx:+.3f} size={size:.3f}"
        )

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
            if seq <= self._last_frame_seq or not isinstance(frames, dict):
                self._worker_stop.wait(timeout=self._worker_interval_s)
                continue
            self._last_frame_seq = seq
            image = frames.get("rgb")
            if image is None and frames:
                image = next(iter(frames.values()))
            if image is None:
                self._worker_stop.wait(timeout=self._worker_interval_s)
                continue
            status = dict(scheduler.read_result("runtime_status", default={}) or {})
            frame_meta = dict(scheduler.read_result("frame_meta", default={}) or {})
            local = dict(scheduler.read_result("local_perception", default={}) or {})
            table_edge = dict(scheduler.read_result("table_edge_obs", default={}) or {})
            target_obs = dict(scheduler.read_result("target_obs", default={}) or {})
            now = time.time()
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
            if target_obs:
                self._emit_operator("preview:target_obs", self._target_summary_line(status, target_obs))
            if not target_obs:
                lines.append("target_obs=unavailable")
            ok = self.render(
                PreviewFrame(
                    ts=now,
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
                        },
                    ),
                )
            )
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
        self._emit("sink_changed", old_sink=old_name, new_sink=getattr(sink, "sink_name", "unknown"))
        if self.enabled and self.sink is not None:
            try:
                self.sink.open()
            except Exception:
                pass

    def enable(self) -> bool:
        """Enable preview output for the currently configured sink."""
        if self.enabled:
            return False
        self.enabled = True
        if self.sink is not None:
            self.sink.open()
        if self.logger is not None:
            self.logger.info(
                "preview started | sink=%s",
                getattr(self.sink, "sink_name", "unknown") if self.sink is not None else "none",
            )
        self._emit("enabled", enabled=True, sink_name=getattr(self.sink, "sink_name", "unknown"))
        return True

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
        return {
            "enabled": self.enabled,
            "sink": self.sink.snapshot() if self.sink is not None else None,
            "runtime_running": bool(self._runtime_running),
            "last_frame_generation": int(self._last_frame_generation),
            "last_frame_seq": int(self._last_frame_seq),
            "exit_requested": bool(self._exit_requested),
        }
