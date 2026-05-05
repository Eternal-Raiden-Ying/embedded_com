#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
import time
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np

try:
    import aidcv as cv2
except ImportError:
    import cv2

ROOT = Path(__file__).resolve().parents[2]
VISTA_ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(VISTA_ROOT) not in sys.path:
    sys.path.insert(0, str(VISTA_ROOT))

from common.base_module import BaseModule
from common.runtime_logging import RunLogger, ensure_dir

from vision_module.backend.vision_engine import VisionEngine
from vision_module.config.board_config import CONFIG
from vision_module.config.data import ASR_VOCAB_MAP, TARGET_CLASSES
from vision_module.ipc.transport import JsonlClientSender, JsonlInboundServer
from vision_module.utils.plot import draw_detect_res_fast

from .protocol import TableEdgeObs, TargetObs, VisionReq, now_ts


def _center_priority(x1: float, y1: float, x2: float, y2: float, w: int, h: int) -> float:
    cx = (x1 + x2) / 2.0
    cy = (y1 + y2) / 2.0
    dx = abs(cx - (w / 2.0)) / max(1.0, w / 2.0)
    dy = abs(cy - (h / 2.0)) / max(1.0, h / 2.0)
    dist = min(1.0, (dx * dx + dy * dy) ** 0.5)
    return 1.0 - dist


def _mask_bbox(mask: np.ndarray) -> Optional[list]:
    ys, xs = np.nonzero(mask)
    if len(xs) == 0 or len(ys) == 0:
        return None
    return [int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())]


class VistaMaskApp(BaseModule):
    def __init__(self):
        super().__init__("vision_v2", CONFIG.runtime.log_enabled, CONFIG.runtime.log_mode)
        ensure_dir(CONFIG.runtime.log_dir)
        ensure_dir(CONFIG.runtime.runs_dir)
        ensure_dir(CONFIG.runtime.pid_dir)
        self.run_logger = RunLogger("vision_v2", CONFIG.runtime.runs_dir, CONFIG.runtime.stack_run_id)
        self.engine = VisionEngine(CONFIG, logger=self.child_logger("engine"))
        self.req_server = JsonlInboundServer(
            mode=CONFIG.req_in.transport,
            tcp_host=CONFIG.req_in.host,
            tcp_port=CONFIG.req_in.port,
            uds_path=CONFIG.req_in.uds_path,
            name="req_in_v2",
        )
        self.obs_sender = JsonlClientSender(
            mode=CONFIG.obs_out.transport,
            tcp_host=CONFIG.obs_out.host,
            tcp_port=CONFIG.obs_out.port,
            uds_path=CONFIG.obs_out.uds_path,
            name="obs_out_v2",
        )
        self.current_req: Optional[VisionReq] = None
        self._running = False
        self.last_send_ts = 0.0

    def _send_interval_s(self) -> float:
        return 1.0 / max(0.5, CONFIG.runtime.send_hz)

    def _select_target_with_mask(self, frame_shape: Tuple[int, int, int], target: Optional[str], infer_res: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        if infer_res is None:
            return None
        boxes = list(infer_res.get("boxes", []) or [])
        masks = list(infer_res.get("masks", []) or [])
        if not boxes or not target:
            return None
        valid_names = ASR_VOCAB_MAP.get(target, set())
        if not valid_names:
            return None
        h, w = frame_shape[:2]
        candidates = []
        for idx, row in enumerate(boxes):
            x1, y1, x2, y2 = [float(v) for v in row[:4]]
            conf = float(row[4])
            cls_id = int(row[5])
            cls_name = TARGET_CLASSES[cls_id] if 0 <= cls_id < len(TARGET_CLASSES) else str(cls_id)
            if cls_name not in valid_names:
                continue
            area = max(0.0, (x2 - x1)) * max(0.0, (y2 - y1))
            area_norm = area / max(1.0, w * h)
            center_pri = _center_priority(x1, y1, x2, y2, w, h)
            rank_key = (round(conf, 6), round(center_pri, 6), round(area_norm, 6))
            mask = masks[idx] if idx < len(masks) else None
            candidates.append((rank_key, idx, x1, y1, x2, y2, conf, area_norm, mask))
        if not candidates:
            return None
        _, idx, x1, y1, x2, y2, conf, area_norm, mask = max(candidates, key=lambda t: t[0])
        cx = (x1 + x2) / 2.0
        cx_norm = (w / 2.0 - cx) / (w / 2.0)
        mask_meta = {
            "mask_ready": False,
            "mask_shape": None,
            "mask_area_ratio": None,
            "mask_bbox": None,
        }
        if isinstance(mask, np.ndarray):
            mask_u8 = mask.astype(np.uint8)
            mask_meta = {
                "mask_ready": True,
                "mask_shape": list(mask_u8.shape[:2]),
                "mask_area_ratio": float(np.count_nonzero(mask_u8)) / float(mask_u8.shape[0] * mask_u8.shape[1]),
                "mask_bbox": _mask_bbox(mask_u8),
            }
        return {
            "target": target,
            "confidence": float(conf),
            "cx_norm": float(np.clip(cx_norm, -1.0, 1.0)),
            "size_norm": float(np.clip(area_norm, 0.0, 1.0)),
            "bbox": [int(x1), int(y1), int(x2), int(y2)],
            "index": idx,
            **mask_meta,
        }

    def _build_table_edge_obs(self, rgb_raw: Optional[np.ndarray], infer_res: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        table_found = bool(infer_res and len(infer_res.get("boxes", []) or []) > 0)
        table_cx = 0.0
        table_size = 0.0
        conf = 0.0
        if table_found and rgb_raw is not None:
            row = infer_res["boxes"][0]
            h, w = rgb_raw.shape[:2]
            x1, y1, x2, y2 = [float(v) for v in row[:4]]
            conf = float(row[4])
            area = max(0.0, (x2 - x1)) * max(0.0, (y2 - y1))
            cx = (x1 + x2) / 2.0
            table_cx = float(np.clip((w / 2.0 - cx) / (w / 2.0), -1.0, 1.0))
            table_size = float(np.clip(area / max(1.0, w * h), 0.0, 1.0))
        return TableEdgeObs(
            ts=now_ts(),
            table_found=table_found,
            edge_found=False,
            confidence=conf,
            table_cx_norm=table_cx,
            table_size_norm=table_size,
            depth_valid=False,
            session_id=self.current_req.session_id if self.current_req else None,
            req_id=self.current_req.req_id if self.current_req else None,
            epoch=self.current_req.epoch if self.current_req else 0,
        ).to_dict()

    def _build_target_obs(self, rgb_raw: Optional[np.ndarray], infer_res: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        selected = self._select_target_with_mask(rgb_raw.shape, self.current_req.target if self.current_req else None, infer_res) if rgb_raw is not None else None
        if selected is None:
            return TargetObs(
                ts=now_ts(),
                found=False,
                target=self.current_req.target if self.current_req else None,
                session_id=self.current_req.session_id if self.current_req else None,
                req_id=self.current_req.req_id if self.current_req else None,
                epoch=self.current_req.epoch if self.current_req else 0,
            ).to_dict()
        self.run_logger.write_jsonl("mask_meta", selected)
        return TargetObs(
            ts=now_ts(),
            found=True,
            target=selected["target"],
            confidence=selected["confidence"],
            cx_norm=selected["cx_norm"],
            size_norm=selected["size_norm"],
            bbox=selected["bbox"],
            mask_ready=selected["mask_ready"],
            mask_shape=selected["mask_shape"],
            mask_area_ratio=selected["mask_area_ratio"],
            mask_bbox=selected["mask_bbox"],
            session_id=self.current_req.session_id if self.current_req else None,
            req_id=self.current_req.req_id if self.current_req else None,
            epoch=self.current_req.epoch if self.current_req else 0,
        ).to_dict()

    def _handle_request(self, payload: Dict[str, Any]):
        req = VisionReq.from_dict(payload)
        self.current_req = req
        if req.mode in {"IDLE", "STOP"}:
            self.engine.set_inference_enabled(False)
            self.engine.set_camera("rgb", False)
            return
        self.engine.set_camera("rgb", True)
        self.engine.set_model(CONFIG.model.active_model, True)
        self.engine.set_inference_enabled(True)

    def start(self):
        self.run_logger.write_service_event("SERVICE_STARTING")
        self.req_server.start()
        self.engine.init()
        self.engine.start()
        self._running = True
        if CONFIG.debug.preview:
            cv2.namedWindow("VISTA V2")

    def stop(self):
        if not self._running:
            return
        self._running = False
        self.req_server.close()
        self.obs_sender.close()
        self.engine.stop()
        if CONFIG.debug.preview:
            try:
                cv2.destroyAllWindows()
            except Exception:
                pass
        self.run_logger.write_service_event("SERVICE_STOPPED")
        self.run_logger.close()

    def run(self):
        self.start()
        period_s = 1.0 / max(0.5, CONFIG.runtime.loop_hz)
        try:
            while self._running:
                loop_start = time.time()
                for item in self.req_server.drain():
                    self._handle_request(item["payload"])
                frames, infer_res = self.engine.get_new_data()
                rgb_raw = frames.get("rgb") if frames else None
                now = time.time()
                if self.current_req and now - self.last_send_ts >= self._send_interval_s():
                    if self.current_req.mode == "TABLE_EDGE_SEARCH":
                        self.obs_sender.send(self._build_table_edge_obs(rgb_raw, infer_res))
                    elif self.current_req.mode == "EDGE_TARGET_SEARCH":
                        self.obs_sender.send(self._build_target_obs(rgb_raw, infer_res))
                    elif self.current_req.mode == "HOME_TAG_SEARCH":
                        self.obs_sender.send({"ts": now_ts(), "type": "home_tag_obs", "found": False})
                    self.last_send_ts = now

                if CONFIG.debug.preview and rgb_raw is not None:
                    canvas = cv2.cvtColor(rgb_raw, cv2.COLOR_RGB2BGR)
                    if infer_res:
                        canvas = draw_detect_res_fast(canvas, infer_res.get("boxes", []), infer_res.get("masks", []))
                    mode_text = self.current_req.mode if self.current_req else "IDLE"
                    stage_text = self.current_req.stage if self.current_req else "IDLE"
                    cv2.putText(canvas, f"mode={mode_text} stage={stage_text}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                    cv2.imshow("VISTA V2", canvas)
                    if cv2.waitKey(1) & 0xFF == 27:
                        break

                elapsed = time.time() - loop_start
                if elapsed < period_s:
                    time.sleep(period_s - elapsed)
        finally:
            self.stop()


def main():
    VistaMaskApp().run()


if __name__ == "__main__":
    main()
