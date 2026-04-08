#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
import time
from pathlib import Path

import numpy as np

try:
    import aidcv as cv2
except ImportError:
    import cv2

STACK_ROOT = Path(__file__).resolve().parents[3]
if str(STACK_ROOT) not in sys.path:
    sys.path.insert(0, str(STACK_ROOT))

from common.base_module import BaseModule
from common.runtime_logging import RunLogger, ensure_dir

from ..backend.vision_engine import VisionEngine
from ..config.board_config import CONFIG
from ..ipc.protocol import HomeTagObs, TargetObs, VisionObs, VisionReq, now_ts
from ..ipc.transport import JsonlClientSender, JsonlInboundServer
from ..utils.detect import compute_target_obs
from ..utils.plot import draw_detect_res_fast


class VistaApp(BaseModule):
    def __init__(self):
        super().__init__("vision", CONFIG.runtime.log_enabled, CONFIG.runtime.log_mode)
        ensure_dir(CONFIG.runtime.log_dir)
        ensure_dir(CONFIG.runtime.runs_dir)
        ensure_dir(CONFIG.runtime.pid_dir)
        self.run_logger = RunLogger("vision", CONFIG.runtime.runs_dir, CONFIG.runtime.stack_run_id)
        self.engine = VisionEngine(
            CONFIG,
            logger=self.child_logger("engine"),
            event_sink=self._record_engine_event,
        )
        self.req_server = JsonlInboundServer(
            mode=CONFIG.req_in.transport,
            tcp_host=CONFIG.req_in.host,
            tcp_port=CONFIG.req_in.port,
            uds_path=CONFIG.req_in.uds_path,
            name="req_in",
            logger=self._log_ipc_event,
        )
        self.obs_sender = JsonlClientSender(
            mode=CONFIG.obs_out.transport,
            tcp_host=CONFIG.obs_out.host,
            tcp_port=CONFIG.obs_out.port,
            uds_path=CONFIG.obs_out.uds_path,
            name="obs_out",
            logger=self._log_ipc_event,
        )
        self.current_stage = "IDLE"
        self.current_mode = "IDLE"
        self.target_name = None
        self.current_session_id = None
        self.current_req_id = None
        self.current_epoch = 0
        self.active_interaction_id = None
        self.pending_result = None
        self.last_send_ts = 0.0
        self.last_req_receive_ts = 0.0
        self.hot_until_ts = 0.0
        self.active_task_key = None
        self._running = False
        self._stopped = False
        self._last_heartbeat_ts = 0.0

    def _config_dump(self):
        return {
            "stack_run_id": self.run_logger.stack_run_id,
            "project_root": CONFIG.runtime.project_root,
            "log_dir": CONFIG.runtime.log_dir,
            "log_file": CONFIG.runtime.log_file,
            "runs_dir": CONFIG.runtime.runs_dir,
            "pid_dir": CONFIG.runtime.pid_dir,
            "pid_file": CONFIG.runtime.pid_file,
            "loop_hz": CONFIG.runtime.loop_hz,
            "send_hz": CONFIG.runtime.send_hz,
            "hot_standby_s": CONFIG.runtime.hot_standby_s,
            "keep_preview_after_stop": CONFIG.runtime.keep_preview_after_stop,
            "keep_model_hot_in_standby": CONFIG.runtime.keep_model_hot_in_standby,
            "enable_infer_during_hot_standby": CONFIG.runtime.enable_infer_during_hot_standby,
            "req_in": {
                "transport": CONFIG.req_in.transport,
                "host": CONFIG.req_in.host,
                "port": CONFIG.req_in.port,
                "uds_path": CONFIG.req_in.uds_path,
            },
            "obs_out": {
                "transport": CONFIG.obs_out.transport,
                "host": CONFIG.obs_out.host,
                "port": CONFIG.obs_out.port,
                "uds_path": CONFIG.obs_out.uds_path,
            },
        }

    def _record_engine_event(self, event: str, fields):
        payload = {"event": event}
        payload.update(fields or {})
        if event.startswith("camera_"):
            self.run_logger.write_jsonl("camera", payload)
            self.log_info("camera", event, fields or None)
        elif event in {"model_loaded", "model_disabled", "inference_changed", "capture_queue_drop"}:
            self.run_logger.write_jsonl("engine", payload)
            self.log_info("engine", event, fields or None)
        elif event == "pipeline_exception":
            self.run_logger.write_jsonl("engine", payload)
            self.log_error("engine", event, fields or None)
        else:
            self.run_logger.write_jsonl("engine", payload)
            self.log_info("engine", event, fields or None)

    def _ipc_direction_for(self, channel: str) -> str:
        return "RX" if str(channel).endswith("_in") else "TX"

    def _log_ipc_event(self, payload):
        level = payload.get("level", "info")
        channel = payload.get("name", payload.get("src", "ipc"))
        event = payload.get("event", payload.get("msg", "log"))
        details = {k: v for k, v in payload.items() if k not in {"level", "src", "name", "event", "msg"}}
        self.log(level, "ipc", f"{channel} {event}".strip(), details or None)
        self.run_logger.write_ipc(channel, event, direction=self._ipc_direction_for(channel), **details)

    def _safe_mode_text(self, mode: str) -> str:
        return str(mode or "IDLE").strip().upper()

    def _safe_stage_text(self, stage: str) -> str:
        return str(stage or "IDLE").strip().upper()

    def _task_key_for(self, req_stage: str, target, session_id, epoch: int):
        return (req_stage, target, session_id, int(epoch))

    def _new_interaction_id(self) -> str:
        return f"ia_{int(time.time() * 1000)}"

    def _build_target_obs(self, rgb_raw, infer_res):
        obs = None
        if rgb_raw is not None and infer_res is not None:
            obs = compute_target_obs(rgb_raw.shape, self.target_name, infer_res.get("boxes", []))
        if obs is None:
            return TargetObs(found=False, target=self.target_name).to_dict()
        return TargetObs(found=True, **obs).to_dict()

    def _build_grasp_proposal(self, target_obs):
        cx_norm = float(target_obs.get("cx_norm", 0.5))
        offset = 0.5 - cx_norm
        return {
            "motion_delta": {
                "dx_m": 0.0,
                "dy_m": round(offset * 0.10, 4),
                "dyaw_rad": round(offset * 0.35, 4),
            },
            "reason": "align_target_before_remote_grasp",
        }

    def _build_vision_obs(self, status: str, perception=None, proposal=None, result=None, interaction=None):
        return VisionObs(
            ts=now_ts(),
            stage=self.current_stage,
            mode=self.current_mode,
            status=status,
            session_id=self.current_session_id,
            req_id=self.current_req_id,
            epoch=self.current_epoch,
            interaction=interaction,
            perception=perception,
            proposal=proposal,
            result=result,
        ).to_dict()

    def _send_obs(self, out_payload, infer_res=None):
        queued = self.obs_sender.send(out_payload)
        self.run_logger.write_ipc(
            "obs_out",
            "enqueue_ok" if queued else "enqueue_failed",
            direction="TX",
            ok=queued,
            req_id=out_payload.get("req_id"),
            session_id=out_payload.get("session_id"),
            epoch=out_payload.get("epoch"),
            stage=out_payload.get("stage"),
            mode=out_payload.get("mode"),
            msg_type=out_payload.get("type"),
            status=out_payload.get("status"),
        )
        self._record_detection(infer_res, out_payload)
        if not queued:
            self.log_warn("runtime", "obs_out queue busy; skipped enqueue")
        return queued

    def _enter_hot_standby(self, engine: VisionEngine, current_mode: str, target_name, epoch: int):
        engine.set_camera("rgb", True)
        if CONFIG.runtime.keep_model_hot_in_standby:
            engine.set_model(CONFIG.model.active_model, True)
        engine.set_inference_enabled(bool(CONFIG.runtime.enable_infer_during_hot_standby))
        engine.reset_runtime_state()
        new_until = time.time() + float(CONFIG.runtime.hot_standby_s)
        self.log_info(
            "runtime",
            "enter hot standby",
            {"prev_mode": current_mode, "prev_target": target_name, "until_ts": new_until},
        )
        self.run_logger.write_timeline(
            "ENTER_HOT_STANDBY",
            prev_mode=current_mode,
            prev_target=target_name,
            until_ts=new_until,
            epoch=int(epoch),
        )
        return "IDLE_HOT", None, new_until, None, None, None, int(epoch)

    def _enter_cold_idle(self, engine: VisionEngine, epoch: int):
        self.log_info("runtime", "enter cold idle")
        self.run_logger.write_timeline("ENTER_IDLE", epoch=int(epoch))
        engine.set_inference_enabled(False)
        if CONFIG.runtime.keep_model_hot_in_standby:
            engine.set_model(CONFIG.model.active_model, False)
        engine.set_camera("rgb", False)
        engine.reset_runtime_state()
        return "IDLE", None, 0.0, None, None, None, int(epoch)

    def _emit_heartbeat_if_needed(self, force: bool = False):
        now = time.time()
        if not force and (now - self._last_heartbeat_ts) < 1.0:
            return
        self._last_heartbeat_ts = now
        req_snapshot = self.req_server.snapshot()
        obs_snapshot = self.obs_sender.snapshot()
        summary = {
            "ts": now,
            "stage": self.current_stage,
            "mode": self.current_mode,
            "target": self.target_name,
            "session_id": self.current_session_id,
            "req_id": self.current_req_id,
            "epoch": self.current_epoch,
            "last_req_age_s": (now - self.last_req_receive_ts) if self.last_req_receive_ts else None,
            "last_obs_send_age_s": (now - self.last_send_ts) if self.last_send_ts else None,
            "servers": {"req_in": req_snapshot},
            "senders": {"obs_out": obs_snapshot},
            "ready": {
                "req_in_listening": bool(req_snapshot.get("listening")),
                "obs_out_link_state": obs_snapshot.get("link_state"),
            },
        }
        self.run_logger.write_jsonl("heartbeat", summary)
        self.run_logger.write_event(
            f"HB stage={self.current_stage} mode={self.current_mode} target={self.target_name} "
            f"req_age={summary['last_req_age_s']} obs_age={summary['last_obs_send_age_s']}"
        )

    def _record_detection(self, infer_res, out_payload):
        boxes = []
        if infer_res is not None:
            boxes = list(infer_res.get("boxes", []) or [])
        perception = out_payload.get("perception") or {}
        found = None
        for key in ("target_obs", "home_tag_obs"):
            item = perception.get(key)
            if isinstance(item, dict) and "found" in item:
                found = item.get("found")
                break
        self.run_logger.write_jsonl(
            "detections",
            {
                "stage": self.current_stage,
                "mode": self.current_mode,
                "target": self.target_name,
                "found": found,
                "box_count": len(boxes),
                "session_id": out_payload.get("session_id"),
                "req_id": out_payload.get("req_id"),
                "epoch": out_payload.get("epoch"),
                "type": out_payload.get("type"),
            },
        )

    def _send_interval_s(self) -> float:
        return 1.0 / max(0.5, CONFIG.runtime.send_hz)

    def _sync_request_context(self, req: VisionReq):
        if req.session_id:
            self.current_session_id = req.session_id
        if req.req_id:
            self.current_req_id = req.req_id
        self.current_epoch = int(req.epoch)

    def _handle_response_request(self, req: VisionReq, stage: str):
        self.log_info(
            "runtime",
            "receive response",
            {
                "stage": stage,
                "interaction_id": req.interaction_id,
                "response": req.response,
            },
        )
        self.run_logger.write_timeline(
            "VISION_RESPOND",
            stage=stage,
            session_id=req.session_id,
            req_id=req.req_id,
            epoch=req.epoch,
            interaction_id=req.interaction_id,
            response=req.response,
        )
        if self.current_stage != "GRASP":
            return

        decision = self._safe_mode_text((req.response or {}).get("decision", ""))
        self.current_mode = "GRASP_REMOTE" if decision == "ACCEPT" else "MICRO_ADJUST"
        self.pending_result = {
            "accepted": decision == "ACCEPT",
            "response": req.response,
            "feedback": req.payload,
            "note": "grasp_remote placeholder until engine integration",
        }
        self.active_interaction_id = None

    def _handle_stop_request(self, stage: str):
        if CONFIG.runtime.keep_preview_after_stop and float(CONFIG.runtime.hot_standby_s) > 0.0:
            (
                self.current_mode,
                self.target_name,
                self.hot_until_ts,
                self.active_task_key,
                self.current_session_id,
                self.current_req_id,
                self.current_epoch,
            ) = self._enter_hot_standby(
                self.engine,
                self.current_mode,
                self.target_name,
                self.current_epoch,
            )
        else:
            self.log_info("runtime", "enter idle", {"reason": stage})
            (
                self.current_mode,
                self.target_name,
                self.hot_until_ts,
                self.active_task_key,
                self.current_session_id,
                self.current_req_id,
                self.current_epoch,
            ) = self._enter_cold_idle(self.engine, self.current_epoch)
        self.current_stage = "IDLE"
        self.active_interaction_id = None
        self.pending_result = None

    def _start_search_stage(self, req: VisionReq, stage: str):
        next_task_key = self._task_key_for(stage, req.target, req.session_id, req.epoch)
        same_task = self.active_task_key == next_task_key and self.current_stage == "SEARCH"
        if same_task:
            self.log_info(
                "runtime",
                "refresh search target",
                {"target": req.target, "session": req.session_id, "req": req.req_id},
            )
        else:
            self.log_info(
                "runtime",
                "start search target",
                {"target": req.target, "session": req.session_id, "req": req.req_id},
            )
            self.run_logger.write_timeline(
                "SEARCH_TARGET",
                target=req.target,
                session_id=req.session_id,
                req_id=req.req_id,
                epoch=req.epoch,
            )
            self.engine.set_camera("rgb", True)
            self.engine.set_model(CONFIG.model.active_model, True)
            self.engine.set_inference_enabled(True)
            self.engine.reset_runtime_state()

        self.current_stage = "SEARCH"
        self.current_mode = req.mode_hint or "TRACK_LOCAL"
        self.target_name = req.target
        self.current_session_id = req.session_id
        self.current_req_id = req.req_id
        self.current_epoch = int(req.epoch)
        self.active_task_key = next_task_key
        self.active_interaction_id = None
        self.pending_result = None
        self.hot_until_ts = 0.0

    def _start_return_stage(self, req: VisionReq):
        self.log_info(
            "runtime",
            "receive return request",
            {"session": req.session_id, "req": req.req_id},
        )
        self.run_logger.write_timeline(
            "RETURN_REQ",
            session_id=req.session_id,
            req_id=req.req_id,
            epoch=req.epoch,
        )
        self.engine.set_camera("rgb", True)
        if CONFIG.runtime.keep_model_hot_in_standby:
            self.engine.set_model(CONFIG.model.active_model, True)
        self.engine.set_inference_enabled(False)
        self.engine.reset_runtime_state()
        self.current_stage = "RETURN"
        self.current_mode = req.mode_hint or "TRACK_LOCAL"
        self.target_name = None
        self.current_session_id = req.session_id
        self.current_req_id = req.req_id
        self.current_epoch = int(req.epoch)
        self.active_task_key = self._task_key_for("RETURN", None, req.session_id, req.epoch)
        self.active_interaction_id = None
        self.pending_result = None
        self.hot_until_ts = 0.0

    def _start_grasp_stage(self, req: VisionReq):
        target = req.target or self.target_name
        self.log_info(
            "runtime",
            "start grasp stage",
            {"target": target, "session": req.session_id, "req": req.req_id},
        )
        self.run_logger.write_timeline(
            "GRASP_STAGE",
            target=target,
            session_id=req.session_id,
            req_id=req.req_id,
            epoch=req.epoch,
            payload=req.payload,
        )
        self.engine.set_camera("rgb", True)
        self.engine.set_model(CONFIG.model.active_model, True)
        self.engine.set_inference_enabled(True)
        self.engine.reset_runtime_state()
        self.current_stage = "GRASP"
        self.current_mode = req.mode_hint or "MICRO_ADJUST"
        self.target_name = target
        self.current_session_id = req.session_id
        self.current_req_id = req.req_id
        self.current_epoch = int(req.epoch)
        self.active_task_key = self._task_key_for("GRASP", target, req.session_id, req.epoch)
        self.active_interaction_id = None
        self.pending_result = None
        self.hot_until_ts = 0.0

    def _handle_request_payload(self, payload):
        typ = str(payload.get("type", "vision_req")).strip()
        if typ not in {"vision_req", "home_tag_req"}:
            return

        req = VisionReq.from_dict(payload)
        stage = self._safe_stage_text(req.stage)
        self._sync_request_context(req)

        if req.op == "RESPOND":
            self._handle_response_request(req, stage)
            return

        if req.is_stop():
            self._handle_stop_request(stage)
            return

        if stage == "SEARCH" and req.target:
            self._start_search_stage(req, stage)
            return

        if stage == "RETURN":
            self._start_return_stage(req)
            return

        if stage == "GRASP":
            self._start_grasp_stage(req)

    def _tick_search_stage(self, rgb_raw, infer_res, now: float):
        if self.current_stage != "SEARCH" or not self.target_name:
            return
        if now - self.last_send_ts < self._send_interval_s():
            return
        out_payload = self._build_vision_obs(
            "RUNNING",
            perception={"target_obs": self._build_target_obs(rgb_raw, infer_res)},
        )
        self._send_obs(out_payload, infer_res=infer_res)
        self.last_send_ts = now

    def _tick_return_stage(self, now: float):
        if self.current_stage != "RETURN":
            return
        if now - self.last_send_ts < self._send_interval_s():
            return
        out_payload = self._build_vision_obs(
            "RUNNING",
            perception={"home_tag_obs": HomeTagObs(found=False).to_dict()},
        )
        self._send_obs(out_payload, infer_res=None)
        self.last_send_ts = now

    def _tick_grasp_stage(self, rgb_raw, infer_res, now: float):
        if self.current_stage != "GRASP":
            return
        if now - self.last_send_ts < self._send_interval_s():
            return

        target_obs = self._build_target_obs(rgb_raw, infer_res)
        if self.pending_result is not None:
            out_payload = self._build_vision_obs(
                "RESULT_READY",
                perception={"target_obs": target_obs},
                result=self.pending_result,
            )
            self.pending_result = None
        elif target_obs.get("found"):
            if not self.active_interaction_id:
                self.active_interaction_id = self._new_interaction_id()
            self.current_mode = "MICRO_ADJUST"
            out_payload = self._build_vision_obs(
                "WAITING_RESPONSE",
                perception={"target_obs": target_obs},
                proposal=self._build_grasp_proposal(target_obs),
                interaction={
                    "required": True,
                    "interaction_id": self.active_interaction_id,
                    "kind": "MOVE_HINT",
                },
            )
        else:
            out_payload = self._build_vision_obs(
                "RUNNING",
                perception={"target_obs": target_obs},
            )
        self._send_obs(out_payload, infer_res=infer_res)
        self.last_send_ts = now

    def _tick_stage(self, rgb_raw, infer_res, now: float):
        self._tick_search_stage(rgb_raw, infer_res, now)
        self._tick_return_stage(now)
        self._tick_grasp_stage(rgb_raw, infer_res, now)

    def _expire_hot_standby(self, now: float):
        if self.current_mode != "IDLE_HOT" or self.hot_until_ts <= 0 or now < self.hot_until_ts:
            return
        (
            self.current_mode,
            self.target_name,
            self.hot_until_ts,
            self.active_task_key,
            self.current_session_id,
            self.current_req_id,
            self.current_epoch,
        ) = self._enter_cold_idle(self.engine, self.current_epoch)
        self.current_stage = "IDLE"

    def _render_preview(self, rgb_raw, infer_res, now: float) -> str:
        if not CONFIG.debug.preview:
            return "ok"

        if self.current_stage in {"SEARCH", "GRASP", "RETURN"} or self.current_mode == "IDLE_HOT":
            if rgb_raw is None:
                canvas = np.zeros((480, 640, 3), dtype=np.uint8)
                cv2.putText(canvas, "WAITING FOR RGB FRAME", (140, 220), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (120, 120, 120), 2)
                cv2.imshow("VISTA App Dashboard", canvas)
                if cv2.waitKey(1) & 0xFF == 27:
                    self.log_info("runtime", "preview exit requested")
                    return "break"
                return "continue"

            bgr_canvas = cv2.cvtColor(rgb_raw, cv2.COLOR_RGB2BGR)
            if CONFIG.debug.draw_boxes and infer_res is not None:
                bgr_canvas = draw_detect_res_fast(
                    bgr_canvas,
                    infer_res.get("boxes", []),
                    infer_res.get("masks", []),
                )
            cv2.putText(
                bgr_canvas,
                f"Stage: {self.current_stage} | Mode: {self.current_mode}",
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.85,
                (0, 255, 0),
                2,
            )
            if self.current_mode == "IDLE_HOT":
                remain = max(0.0, self.hot_until_ts - now)
                cv2.putText(
                    bgr_canvas,
                    f"Hot standby: {remain:4.1f}s",
                    (20, 80),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0, 255, 255),
                    2,
                )
            else:
                cv2.putText(
                    bgr_canvas,
                    f"AI Sync: {'ON' if infer_res else 'OFF'}",
                    (20, 80),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0, 255, 255),
                    2,
                )
            if self.current_session_id:
                cv2.putText(
                    bgr_canvas,
                    f"session={self.current_session_id} epoch={self.current_epoch}",
                    (20, 115),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (255, 255, 0),
                    2,
                )
            cv2.imshow("VISTA App Dashboard", bgr_canvas)
        else:
            canvas = np.zeros((480, 640, 3), dtype=np.uint8)
            cv2.putText(canvas, "SYSTEM STANDBY", (180, 220), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (150, 150, 150), 2)
            cv2.putText(canvas, "Zero Power Mode Active", (170, 270), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (100, 100, 100), 1)
            cv2.imshow("VISTA App Dashboard", canvas)

        if cv2.waitKey(1) & 0xFF == 27:
            self.log_info("runtime", "preview exit requested")
            return "break"
        return "ok"

    def start(self):
        cfg_dump = self._config_dump()
        self.run_logger.write_meta(
            {
                "service": "vision",
                "run_dir": str(self.run_logger.run_dir),
                "project_root": CONFIG.runtime.project_root,
                "log_file": CONFIG.runtime.log_file,
                "pid_file": CONFIG.runtime.pid_file,
            }
        )
        self.run_logger.write_service_event("SERVICE_STARTING", run_dir=str(self.run_logger.run_dir))
        self.run_logger.write_jsonl("config", cfg_dump)
        self.run_logger.write_timeline("BOOT", run_dir=str(self.run_logger.run_dir), config=cfg_dump)
        self.req_server.start()
        self.engine.init()
        self.engine.start()
        self._running = True
        if CONFIG.debug.preview:
            cv2.namedWindow("VISTA App Dashboard")
        self.run_logger.write_service_event("SERVICE_READY", stage=self.current_stage, mode=self.current_mode)
        self.log_info("runtime", "SERVICE_READY", {"run_dir": str(self.run_logger.run_dir)})
        self._emit_heartbeat_if_needed(force=True)

    def stop(self):
        if self._stopped:
            return
        self._stopped = True
        self._running = False
        self.run_logger.write_service_event("SERVICE_STOPPING", stage=self.current_stage, mode=self.current_mode)
        self.req_server.close()
        self.obs_sender.close()
        self.engine.stop()
        if CONFIG.debug.preview:
            try:
                cv2.destroyAllWindows()
            except Exception:
                pass
        self.run_logger.write_service_event("SERVICE_STOPPED")
        self.log_info("runtime", "SERVICE_STOPPED")
        self.run_logger.close()

    def run(self):
        self.start()
        target_frame_time = 1.0 / max(0.5, CONFIG.runtime.loop_hz)

        try:
            while self._running:
                loop_start = time.time()

                for item in self.req_server.drain():
                    payload = item["payload"]
                    typ = str(payload.get("type", "vision_req")).strip()
                    self.last_req_receive_ts = float(item.get("recv_ts", time.time()))
                    self.run_logger.write_ipc(
                        "req_in",
                        "received",
                        direction="RX",
                        ok=True,
                        msg_type=typ,
                        session_id=payload.get("session_id"),
                        req_id=payload.get("req_id"),
                        epoch=payload.get("epoch"),
                    )
                    self._handle_request_payload(payload)

                frames, infer_res = self.engine.get_new_data()
                rgb_raw = frames.get("rgb") if frames else None
                now = time.time()
                self._tick_stage(rgb_raw, infer_res, now)
                self._expire_hot_standby(now)

                preview_result = self._render_preview(rgb_raw, infer_res, now)
                if preview_result == "break":
                    break
                if preview_result == "continue":
                    continue

                self._emit_heartbeat_if_needed()
                dt = time.time() - loop_start
                if dt < target_frame_time:
                    time.sleep(target_frame_time - dt)

        except KeyboardInterrupt:
            self.log_info("runtime", "keyboard interrupt received")
        except Exception as exc:
            self.run_logger.write_timeline("FATAL", error=str(exc))
            self.log_error("runtime", f"vista main loop crashed: {exc}")
        finally:
            self.stop()


def main():
    app = VistaApp()
    app.run()


if __name__ == "__main__":
    main()
