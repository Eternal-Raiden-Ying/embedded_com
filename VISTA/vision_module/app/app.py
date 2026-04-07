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
from ..ipc.protocol import HomeTagObs, HomeTagReq, TargetObs, VisionReq, now_ts
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
        self.current_mode = "IDLE"
        self.target_name = None
        self.current_session_id = None
        self.current_req_id = None
        self.current_epoch = 0
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

    def _task_key_for(self, req_mode: str, target, session_id, epoch: int):
        return (req_mode, target, session_id, int(epoch))

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
            f"HB mode={self.current_mode} target={self.target_name} "
            f"req_age={summary['last_req_age_s']} obs_age={summary['last_obs_send_age_s']}"
        )

    def _record_detection(self, infer_res, out_payload):
        boxes = []
        if infer_res is not None:
            boxes = list(infer_res.get("boxes", []) or [])
        self.run_logger.write_jsonl(
            "detections",
            {
                "mode": self.current_mode,
                "target": self.target_name,
                "found": out_payload.get("found"),
                "box_count": len(boxes),
                "session_id": out_payload.get("session_id"),
                "req_id": out_payload.get("req_id"),
                "epoch": out_payload.get("epoch"),
                "type": out_payload.get("type"),
            },
        )

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
        self.run_logger.write_service_event("SERVICE_READY", mode=self.current_mode)
        self.log_info("runtime", "SERVICE_READY", {"run_dir": str(self.run_logger.run_dir)})
        self._emit_heartbeat_if_needed(force=True)

    def stop(self):
        if self._stopped:
            return
        self._stopped = True
        self._running = False
        self.run_logger.write_service_event("SERVICE_STOPPING", mode=self.current_mode)
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

                    if typ == "vision_req":
                        req = VisionReq.from_dict(payload)
                        new_mode = self._safe_mode_text(req.mode)

                        if new_mode == "FIND" and req.target:
                            next_task_key = self._task_key_for(new_mode, req.target, req.session_id, req.epoch)
                            same_task = self.active_task_key == next_task_key and self.current_mode == "FIND"
                            if same_task:
                                self.current_req_id = req.req_id
                                self.current_session_id = req.session_id
                                self.current_epoch = int(req.epoch)
                                self.log_info(
                                    "runtime",
                                    "refresh find target",
                                    {"target": req.target, "session": req.session_id, "req": req.req_id},
                                )
                            else:
                                self.log_info(
                                    "runtime",
                                    "start find target",
                                    {"target": req.target, "session": req.session_id, "req": req.req_id},
                                )
                                self.run_logger.write_timeline(
                                    "FIND_TARGET",
                                    target=req.target,
                                    session_id=req.session_id,
                                    req_id=req.req_id,
                                    epoch=req.epoch,
                                )
                                self.engine.set_camera("rgb", True)
                                self.engine.set_model(CONFIG.model.active_model, True)
                                self.engine.set_inference_enabled(True)
                                self.engine.reset_runtime_state()
                                self.current_mode = "FIND"
                                self.target_name = req.target
                                self.current_session_id = req.session_id
                                self.current_req_id = req.req_id
                                self.current_epoch = int(req.epoch)
                                self.active_task_key = next_task_key
                                self.hot_until_ts = 0.0

                        elif new_mode in ["IDLE", "STOP", "CANCEL"]:
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
                                self.log_info("runtime", "enter idle", {"reason": new_mode})
                                (
                                    self.current_mode,
                                    self.target_name,
                                    self.hot_until_ts,
                                    self.active_task_key,
                                    self.current_session_id,
                                    self.current_req_id,
                                    self.current_epoch,
                                ) = self._enter_cold_idle(self.engine, self.current_epoch)

                    elif typ == "home_tag_req":
                        req = HomeTagReq.from_dict(payload)
                        self.log_info(
                            "runtime",
                            "receive return request",
                            {"session": req.session_id, "req": req.req_id},
                        )
                        self.run_logger.write_timeline(
                            "HOME_TAG_REQ",
                            session_id=req.session_id,
                            req_id=req.req_id,
                            epoch=req.epoch,
                        )
                        self.engine.set_camera("rgb", True)
                        if CONFIG.runtime.keep_model_hot_in_standby:
                            self.engine.set_model(CONFIG.model.active_model, True)
                        self.engine.set_inference_enabled(False)
                        self.engine.reset_runtime_state()
                        self.current_mode = "RETURN"
                        self.target_name = None
                        self.current_session_id = req.session_id
                        self.current_req_id = req.req_id
                        self.current_epoch = int(req.epoch)
                        self.active_task_key = self._task_key_for("RETURN", None, req.session_id, req.epoch)
                        self.hot_until_ts = 0.0

                frames, infer_res = self.engine.get_new_data()
                rgb_raw = frames.get("rgb") if frames else None
                now = time.time()

                if self.current_mode == "FIND" and self.target_name:
                    if now - self.last_send_ts >= 1.0 / max(0.5, CONFIG.runtime.send_hz):
                        obs = None
                        if rgb_raw is not None and infer_res is not None:
                            obs = compute_target_obs(rgb_raw.shape, self.target_name, infer_res.get("boxes", []))
                        if obs is None:
                            out_payload = TargetObs(
                                ts=now_ts(),
                                found=False,
                                target=self.target_name,
                                session_id=self.current_session_id,
                                req_id=self.current_req_id,
                                epoch=self.current_epoch,
                            ).to_dict()
                        else:
                            out_payload = TargetObs(
                                ts=now_ts(),
                                found=True,
                                session_id=self.current_session_id,
                                req_id=self.current_req_id,
                                epoch=self.current_epoch,
                                **obs,
                            ).to_dict()
                        queued = self.obs_sender.send(out_payload)
                        self.run_logger.write_ipc(
                            "obs_out",
                            "enqueue_ok" if queued else "enqueue_failed",
                            direction="TX",
                            ok=queued,
                            req_id=out_payload.get("req_id"),
                            session_id=out_payload.get("session_id"),
                            epoch=out_payload.get("epoch"),
                            found=out_payload.get("found"),
                            msg_type=out_payload.get("type"),
                        )
                        self._record_detection(infer_res, out_payload)
                        if not queued:
                            self.log_warn("runtime", "obs_out queue busy; skipped enqueue")
                        self.last_send_ts = now

                elif self.current_mode == "RETURN":
                    if now - self.last_send_ts >= 1.0 / max(0.5, CONFIG.runtime.send_hz):
                        out_payload = HomeTagObs(
                            ts=now_ts(),
                            found=False,
                            session_id=self.current_session_id,
                            req_id=self.current_req_id,
                            epoch=self.current_epoch,
                        ).to_dict()
                        queued = self.obs_sender.send(out_payload)
                        self.run_logger.write_ipc(
                            "obs_out",
                            "enqueue_ok" if queued else "enqueue_failed",
                            direction="TX",
                            ok=queued,
                            req_id=out_payload.get("req_id"),
                            session_id=out_payload.get("session_id"),
                            epoch=out_payload.get("epoch"),
                            found=out_payload.get("found"),
                            msg_type=out_payload.get("type"),
                        )
                        self._record_detection(None, out_payload)
                        self.last_send_ts = now

                if self.current_mode == "IDLE_HOT" and self.hot_until_ts > 0 and now >= self.hot_until_ts:
                    (
                        self.current_mode,
                        self.target_name,
                        self.hot_until_ts,
                        self.active_task_key,
                        self.current_session_id,
                        self.current_req_id,
                        self.current_epoch,
                    ) = self._enter_cold_idle(self.engine, self.current_epoch)

                if CONFIG.debug.preview:
                    if self.current_mode in {"FIND", "RETURN", "IDLE_HOT"} and rgb_raw is not None:
                        bgr_canvas = cv2.cvtColor(rgb_raw, cv2.COLOR_RGB2BGR)
                        if CONFIG.debug.draw_boxes and infer_res is not None:
                            bgr_canvas = draw_detect_res_fast(
                                bgr_canvas,
                                infer_res.get("boxes", []),
                                infer_res.get("masks", []),
                            )
                        cv2.putText(
                            bgr_canvas,
                            f"Mode: {self.current_mode}",
                            (20, 40),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            1.0,
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
                        break

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
