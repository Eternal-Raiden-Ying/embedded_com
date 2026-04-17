#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
import os
import sys
import time
from pathlib import Path
from typing import Optional

import cv2

ROOT = Path(__file__).resolve().parents[3]
VISTA_ROOT = Path(__file__).resolve().parents[2]
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

from board_config import CONFIG
from detector import OnlineTableEdgeDetector, load_calib
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
        calib, target_dist = load_calib((HERE / CONFIG.detector.calib_json).resolve())
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
        self._gui_enabled = bool(os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY"))

    @staticmethod
    def _require_synced_detector_result(result):
        required = ("image_line_k", "image_line_b", "edge_point_count")
        missing = [name for name in required if not hasattr(result, name)]
        if not missing:
            return
        raise RuntimeError(
            "detector result is missing new fields %s. "
            "Please sync Offline_Edge_Test/Online/app.py and Offline_Edge_Test/Online/detector.py "
            "to the board together before rerunning." % (missing,)
        )

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

    def _save_snapshot_if_needed(self, preview_img, depth_img, now: float):
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
        cv2.imwrite(str(depth_path), depth_img.astype("uint16"))

    def _render_preview(self, depth_frame, color_frame: Optional[object], result, debug, now: float):
        depth_vis = cv2.convertScaleAbs(depth_frame, alpha=0.03)
        depth_vis = cv2.applyColorMap(depth_vis, cv2.COLORMAP_JET)
        x0, y0, x1, y1 = debug["roi_box"]
        cv2.rectangle(depth_vis, (x0, y0), (x1, y1), (0, 255, 0), 2)
        edge_uv = debug.get("edge_uv")
        if edge_uv is not None:
            for pt in edge_uv[:: max(1, len(edge_uv) // 120)]:
                cv2.circle(depth_vis, (int(pt[0]), int(pt[1])), 1, (0, 0, 255), -1)
        image_line_k = getattr(result, "image_line_k", None)
        image_line_b = getattr(result, "image_line_b", None)
        edge_point_count = int(getattr(result, "edge_point_count", 0))
        if image_line_k is not None and image_line_b is not None:
            xa = int(x0)
            xb = int(x1 - 1)
            ya = int(round(image_line_k * xa + image_line_b))
            yb = int(round(image_line_k * xb + image_line_b))
            clipped, p1, p2 = cv2.clipLine((0, 0, depth_frame.shape[1], depth_frame.shape[0]), (xa, ya), (xb, yb))
            if clipped:
                cv2.line(depth_vis, p1, p2, (0, 255, 255), 2, cv2.LINE_AA)
                cv2.circle(depth_vis, p1, 4, (255, 255, 0), -1)
                cv2.circle(depth_vis, p2, 4, (255, 255, 0), -1)
        text1 = f"edge={int(result.edge_found)} conf={result.edge_confidence:.2f}"
        text2 = f"yaw={result.yaw_err_rad:.4f} dist={result.dist_err_m:.4f}"
        text3 = f"img_k={image_line_k:.4f}" if image_line_k is not None else "img_k=None"
        text4 = f"points={result.point_count} table={result.table_point_count} edge={edge_point_count} frame={self._frame_id}"
        for idx, text in enumerate((text1, text2, text3, text4)):
            cv2.putText(depth_vis, text, (20, 35 + idx * 30), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)
        if color_frame is not None:
            color_bgr = cv2.cvtColor(color_frame, cv2.COLOR_RGB2BGR)
            thumb_h = depth_vis.shape[0] // 3
            thumb_w = int(color_bgr.shape[1] * thumb_h / max(1, color_bgr.shape[0]))
            thumb = cv2.resize(color_bgr, (thumb_w, thumb_h))
            y_start = 10
            x_start = max(0, depth_vis.shape[1] - thumb_w - 10)
            depth_vis[y_start:y_start + thumb_h, x_start:x_start + thumb_w] = thumb
        self._save_snapshot_if_needed(depth_vis, depth_frame, now)
        if CONFIG.runtime.preview and self._gui_enabled:
            cv2.imshow("Online Table Edge Detect", depth_vis)
            return cv2.waitKey(1) & 0xFF
        return -1

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
        if self._gui_enabled:
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
                self._require_synced_detector_result(result)
                obs = self._build_obs(result)
                obs["ts_ms"] = frame_pack.get("ts_ms")
                self.run_logger.write_jsonl("table_edge_obs", obs)
                self._send_obs_if_needed(obs, time.time())
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
    app.run()


if __name__ == "__main__":
    main()
