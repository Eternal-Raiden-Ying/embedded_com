#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
import sys
import time
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
VISTA_ROOT = Path(__file__).resolve().parents[1]
HERE = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(VISTA_ROOT) not in sys.path:
    sys.path.insert(0, str(VISTA_ROOT))
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from common.base_module import BaseModule
from common.runtime_logging import RunLogger, ensure_dir
from vision_module.ipc.transport import JsonlClientSender

try:
    from .board_config import CONFIG
    from .detector import load_calib, OnlineTableEdgeDetector
    from .protocol import TableEdgeObsMsg, now_ts
    from .stream_source import RealSenseStreamSource
except ImportError:
    from board_config import CONFIG
    from detector import load_calib, OnlineTableEdgeDetector
    from protocol import TableEdgeObsMsg, now_ts
    from stream_source import RealSenseStreamSource


class OnlineEdgeApp(BaseModule):
    def __init__(self):
        super().__init__("online_edge", CONFIG.runtime.log_enabled, CONFIG.runtime.log_mode)
        ensure_dir(CONFIG.runtime.log_dir)
        ensure_dir(CONFIG.runtime.runs_dir)
        ensure_dir(CONFIG.runtime.pid_dir)
        ensure_dir(CONFIG.runtime.snapshot_dir)
        self.run_logger = RunLogger("online_edge", CONFIG.runtime.runs_dir, CONFIG.runtime.stack_run_id)
        calib, target_dist = load_calib(Path(CONFIG.detector.calib_json))
        if float(CONFIG.detector.target_dist_m_override) > 0:
            target_dist = float(CONFIG.detector.target_dist_m_override)
        self.detector = OnlineTableEdgeDetector(calib, CONFIG.detector, target_dist)
        self.source = RealSenseStreamSource(CONFIG.camera, logger=self.child_logger("source"))
        self.sender = JsonlClientSender(
            mode=CONFIG.output.transport,
            tcp_host=CONFIG.output.host,
            tcp_port=CONFIG.output.port,
            uds_path=CONFIG.output.uds_path,
            name="table_edge_obs_out",
        )
        self._running = False
        self._last_send_ts = 0.0
        self._last_snapshot_ts = 0.0
        self._frame_id = 0

    def _config_dump(self):
        return {
            "runtime": CONFIG.runtime.__dict__,
            "camera": CONFIG.camera.__dict__,
            "detector": CONFIG.detector.__dict__,
            "output": CONFIG.output.__dict__,
        }

    def _build_obs(self, result) -> dict:
        return TableEdgeObsMsg(
            ts=now_ts(),
            table_found=bool(result.table_point_count > 0),
            edge_found=bool(result.edge_found),
            confidence=float(result.edge_confidence),
            yaw_err_rad=float(result.yaw_err_rad) if result.edge_found else None,
            dist_err_m=float(result.dist_err_m) if result.edge_found else None,
            edge_k=result.line_k,
            edge_b=result.line_b,
            edge_valid=bool(getattr(result, "valid_for_control", False)),
            raw_found=bool(getattr(result, "raw_found", False)),
            pose_found=bool(getattr(result, "pose_found", False)),
            valid_for_control=bool(getattr(result, "valid_for_control", False)),
            pose_source=str(getattr(result, "pose_source", "none") or "none"),
            plane_found=bool(getattr(result, "plane_found", False)),
            line_found=bool(getattr(result, "line_found", False)),
            plane_confidence=float(getattr(result, "plane_confidence", 0.0) or 0.0),
            line_confidence=float(getattr(result, "line_confidence", 0.0) or 0.0),
            plane_residual_mean=float(getattr(result, "plane_residual_mean", 0.0) or 0.0),
            line_residual_mean=float(getattr(result, "line_residual_mean", 0.0) or 0.0),
            plane_x_span_m=float(getattr(result, "plane_x_span_m", 0.0) or 0.0),
            line_x_span_m=float(getattr(result, "line_x_span_m", 0.0) or 0.0),
            candidate_count=int(getattr(result, "candidate_count", 0) or 0),
            inlier_count=int(getattr(result, "inlier_count", 0) or 0),
            stable_count=int(getattr(result, "stable_count", 0) or 0),
            front_face_area_ratio=float(getattr(result, "front_face_area_ratio", 0.0) or 0.0),
            reject_reason=str(getattr(result, "reject_reason", "") or ""),
            depth_valid=True,
            point_count=int(result.point_count),
            table_point_count=int(result.table_point_count),
            frame_id=int(self._frame_id),
        ).to_dict()

    def _send_obs_if_needed(self, payload: dict, now: float):
        if CONFIG.output.transport == "disabled":
            return
        if (now - self._last_send_ts) < float(CONFIG.output.send_interval_s):
            return
        self.sender.send(payload)
        self._last_send_ts = now

    def _save_snapshot_if_needed(self, preview_img: np.ndarray, depth_img: np.ndarray, now: float):
        period = float(CONFIG.runtime.save_snapshot_period_s)
        if period <= 0.0:
            return
        if (now - self._last_snapshot_ts) < period:
            return
        self._last_snapshot_ts = now
        stem = f"edge_{int(now)}_{self._frame_id:06d}"
        preview_path = Path(CONFIG.runtime.snapshot_dir) / f"{stem}_preview.png"
        depth_path = Path(CONFIG.runtime.snapshot_dir) / f"{stem}_depth.png"
        cv2.imwrite(str(preview_path), preview_img)
        cv2.imwrite(str(depth_path), depth_img.astype(np.uint16))

    def _render_preview(self, depth_frame: np.ndarray, color_frame: Optional[np.ndarray], result, debug, now: float):
        depth_vis = cv2.convertScaleAbs(depth_frame, alpha=0.03)
        depth_vis = cv2.applyColorMap(depth_vis, cv2.COLORMAP_JET)
        x0, y0, x1, y1 = debug["roi_box"]
        cv2.rectangle(depth_vis, (x0, y0), (x1, y1), (0, 255, 0), 2)
        txt = f"edge={int(result.edge_found)} conf={result.edge_confidence:.2f} yaw={result.yaw_err_rad:.3f} dist={result.dist_err_m:.3f}"
        cv2.putText(depth_vis, txt, (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)
        txt2 = f"points={result.point_count} table_points={result.table_point_count} frame={self._frame_id}"
        cv2.putText(depth_vis, txt2, (20, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.60, (255, 255, 255), 2)

        if color_frame is not None:
            color_bgr = cv2.cvtColor(color_frame, cv2.COLOR_RGB2BGR)
            thumb_h = depth_vis.shape[0] // 3
            thumb_w = int(color_bgr.shape[1] * thumb_h / max(1, color_bgr.shape[0]))
            thumb = cv2.resize(color_bgr, (thumb_w, thumb_h))
            y_start = 10
            x_start = max(0, depth_vis.shape[1] - thumb_w - 10)
            depth_vis[y_start:y_start + thumb_h, x_start:x_start + thumb_w] = thumb
        self._save_snapshot_if_needed(depth_vis, depth_frame, now)
        cv2.imshow("Online Table Edge Detect", depth_vis)
        return cv2.waitKey(1) & 0xFF

    def start(self):
        self.run_logger.write_meta({"service": "online_edge", "config": self._config_dump()})
        self.run_logger.write_service_event("SERVICE_STARTING")
        self.source.start()
        self.run_logger.write_jsonl("stream_profiles", {"profiles": self.source.stream_profiles})
        has_depth = any(item.get("stream_name") == "stream.depth" for item in self.source.stream_profiles)
        if not has_depth:
            raise RuntimeError(f"depth stream not available in source profiles: {self.source.stream_profiles}")
        self._running = True
        self.run_logger.write_service_event("SERVICE_READY")

    def stop(self):
        if not self._running:
            return
        self._running = False
        self.source.stop()
        self.sender.close()
        try:
            cv2.destroyAllWindows()
        except Exception:
            pass
        self.run_logger.write_service_event("SERVICE_STOPPED")
        self.run_logger.close()

    def run(self):
        period_s = 1.0 / max(1.0, float(CONFIG.runtime.loop_hz))
        self.start()
        try:
            while self._running:
                t0 = time.time()
                frame_pack = self.source.read(timeout_ms=3000)
                if frame_pack is None:
                    continue
                depth_frame = frame_pack.get("depth")
                color_frame = frame_pack.get("color")
                if depth_frame is None:
                    self.log_warn("runtime", "depth frame missing; waiting for next frame")
                    continue
                self._frame_id += 1
                result, debug = self.detector.process_depth(depth_frame)
                obs = self._build_obs(result)
                obs["ts_ms"] = frame_pack.get("ts_ms")
                self.run_logger.write_jsonl("table_edge_obs", obs)
                self._send_obs_if_needed(obs, time.time())

                self.log_info(
                    "edge",
                    "frame result",
                    {
                        "frame_id": self._frame_id,
                        "edge_found": result.edge_found,
                        "yaw_err_rad": round(result.yaw_err_rad, 4),
                        "dist_err_m": round(result.dist_err_m, 4),
                        "confidence": round(result.edge_confidence, 4),
                        "table_points": result.table_point_count,
                    },
                )

                if CONFIG.runtime.preview:
                    key = self._render_preview(depth_frame, color_frame, result, debug, time.time())
                    if key == 27:
                        break

                elapsed = time.time() - t0
                if elapsed < period_s:
                    time.sleep(period_s - elapsed)
        finally:
            self.stop()


def main():
    app = OnlineEdgeApp()
    try:
        app.run()
    except Exception as exc:
        try:
            app.log_error("runtime", f"online edge app crashed: {exc}")
        except Exception:
            pass
        try:
            app.stop()
        except Exception:
            pass
        raise


if __name__ == "__main__":
    main()
