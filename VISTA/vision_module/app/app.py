#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
import time
from pathlib import Path
from typing import Dict

STACK_ROOT = Path(__file__).resolve().parents[3]
if str(STACK_ROOT) not in sys.path:
    sys.path.insert(0, str(STACK_ROOT))

from common.base_module import BaseModule
from common.runtime_logging import RunLogger, ensure_dir

from ..backend.mode_controller import ModeController
from ..backend.vision_engine import VisionEngine
from ..config.mode_defaults import build_default_mode_profiles
from ..config.board_config import CONFIG
from ..ipc.protocol import VisionReq
from ..ipc.transport import JsonlClientSender, JsonlInboundServer
from .stage_controller import StageController
from .stages import GraspStagePlan, ReturnStagePlan, SearchStagePlan


class VistaApp(BaseModule):
    def __init__(self):
        super().__init__("vision", CONFIG.runtime.log_enabled, CONFIG.runtime.log_mode)
        ensure_dir(CONFIG.runtime.log_dir)
        ensure_dir(CONFIG.runtime.runs_dir)
        ensure_dir(CONFIG.runtime.pid_dir)
        self.run_logger = RunLogger(
            "vision",
            CONFIG.runtime.runs_dir,
            CONFIG.runtime.stack_run_id,
            enable_text_events=False,
        )
        self.log_paths = self.run_logger.structured_paths(heartbeat_enabled=CONFIG.runtime.heartbeat_enabled)
        mode_controller = ModeController(
            logger=self.child_logger("mode"),
            backend_event_sink=lambda event, **fields: self._record_backend_event(event, fields),
            preview_allowed=bool(CONFIG.debug.preview),
        )
        mode_controller.register_profiles(build_default_mode_profiles(CONFIG.model.active_model).values())
        self.runtime = VisionEngine(
            CONFIG,
            logger=self.child_logger("engine"),
            event_sink=self._record_backend_event,
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
        self.stage_controller = StageController(
            logger=self.child_logger("stage"),
            event_sink=self._record_stage_event,
            mode_controller=mode_controller,
            runtime_service=self.runtime,
        )
        self.stage_controller.register_default_plans(
            {
                "SEARCH": SearchStagePlan(),
                "GRASP": GraspStagePlan(),
                "RETURN": ReturnStagePlan(),
            }
        )
        self.current_stage = "IDLE"
        self.current_mode = "IDLE"
        self.target_name = None
        self.current_session_id = None
        self.current_req_id = None
        self.current_epoch = 0
        self.active_interaction_id = None
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
            "capability_placeholder": CONFIG.runtime.capability_placeholder,
            "heartbeat_enabled": CONFIG.runtime.heartbeat_enabled,
            "heartbeat_interval_s": CONFIG.runtime.heartbeat_interval_s,
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
            "structured_logs": self.log_paths,
        }

    def _current_event_context(self):
        return {
            "stage": self.current_stage,
            "mode": self.current_mode,
            "session_id": self.current_session_id,
            "req_id": self.current_req_id,
            "epoch": self.current_epoch,
            "interaction_id": self.active_interaction_id,
        }

    def _record_event(self, event: str, level: str = "info", trigger: str = "", data=None, **fields):
        payload = self._current_event_context()
        payload.update(fields or {})
        self.run_logger.write_event_record(
            event=event,
            level=level,
            trigger=trigger,
            data=dict(data or {}),
            **payload,
        )

    def _record_ipc(self, direction: str, channel: str, event: str, level: str = "info", data=None, **fields):
        self.run_logger.write_ipc_record(
            direction=direction,
            channel=channel,
            event=event,
            level=level,
            data=dict(data or {}),
            **fields,
        )

    def _record_stage_event(self, event: str, fields):
        payload = dict(fields or {})
        level = str(payload.pop("level", "info")).strip().lower() or "info"
        trigger = str(payload.pop("trigger", "stage_controller")).strip() or "stage_controller"
        data = dict(payload.pop("data", {}) or {})
        self.log(level, "stage", str(event or "STAGE_EVENT").strip().lower(), data or None)
        self.run_logger.write_event_record(
            event=str(event or "STAGE_EVENT").strip().upper(),
            level=level,
            trigger=trigger,
            data=data,
            **payload,
        )

    def _record_backend_event(self, event: str, fields):
        payload = dict(fields or {})
        event_name = str(event or "BACKEND_EVENT").strip().upper()
        if event_name == "BACKEND_DIAGNOSTIC" and not (
            str(CONFIG.runtime.log_mode).strip().lower() == "full" or bool(CONFIG.runtime.debug)
        ):
            return
        level = str(payload.pop("level", "info")).strip().lower() or "info"
        data = dict(payload or {})

        stage_override = self.current_stage
        mode_override = self.current_mode
        session_override = self.current_session_id
        req_override = self.current_req_id
        epoch_override = self.current_epoch
        interaction_override = self.active_interaction_id

        try:
            ctx = self.stage_controller.context()
            stage_override = self._safe_stage_text(getattr(ctx, "current_stage", stage_override))
            mode_override = self._safe_mode_text(getattr(ctx, "current_mode", mode_override))
            session_override = getattr(ctx, "session_id", session_override)
            req_override = getattr(ctx, "req_id", req_override)
            epoch_override = int(getattr(ctx, "epoch", epoch_override))
            interaction_override = getattr(ctx, "interaction_id", interaction_override)
        except Exception:
            pass

        if event_name == "BACKEND_MODE_CHANGED":
            mode_override = self._safe_mode_text(data.get("current_mode", mode_override))

        self.log(level, "backend", event_name.lower(), data or None)
        self._record_event(
            event_name,
            level=level,
            trigger="backend",
            data=data,
            stage=stage_override,
            mode=mode_override,
            session_id=session_override,
            req_id=req_override,
            epoch=epoch_override,
            interaction_id=interaction_override,
        )

    def _ipc_direction_for(self, channel: str) -> str:
        return "RX" if str(channel).endswith("_in") else "TX"

    def _log_ipc_event(self, payload):
        level = payload.get("level", "info")
        channel = payload.get("name", payload.get("src", "ipc"))
        event = payload.get("event", payload.get("msg", "log"))
        details = {k: v for k, v in payload.items() if k not in {"level", "src", "name", "event", "msg"}}
        self.log(level, "ipc", f"{channel} {event}".strip(), details or None)
        self._record_ipc(
            direction=self._ipc_direction_for(channel),
            channel=channel,
            event=event,
            level=level,
            **details,
        )

    def _safe_mode_text(self, mode: str) -> str:
        return str(mode or "IDLE").strip().upper()

    def _safe_stage_text(self, stage: str) -> str:
        return str(stage or "IDLE").strip().upper()

    def _task_key_for(self, req_stage: str, target, session_id, epoch: int):
        return (req_stage, target, session_id, int(epoch))

    def _send_obs(self, out_payload):
        queued = self.obs_sender.send(out_payload)
        self._record_ipc(
            direction="TX",
            channel="obs_out",
            event="enqueue_ok" if queued else "enqueue_failed",
            level="info" if queued else "warn",
            ok=queued,
            req_id=out_payload.get("req_id"),
            session_id=out_payload.get("session_id"),
            epoch=out_payload.get("epoch"),
            stage=out_payload.get("stage"),
            mode=out_payload.get("mode"),
            msg_type=out_payload.get("type"),
            status=out_payload.get("status"),
        )
        if not queued:
            self.log_warn("runtime", "obs_out queue busy; skipped enqueue")
        return queued

    def _enter_hot_standby(self, current_mode: str, target_name, epoch: int):
        self.stage_controller.set_runtime_mode("IDLE_HOT", reason="enter_hot_standby", force=True)
        new_until = time.time() + float(CONFIG.runtime.hot_standby_s)
        self.log_info(
            "runtime",
            "enter hot standby",
            {"prev_mode": current_mode, "prev_target": target_name, "until_ts": new_until},
        )
        self._record_event(
            "ENTER_HOT_STANDBY",
            trigger="stop_flow",
            data={
                "prev_mode": current_mode,
                "prev_target": target_name,
                "until_ts": new_until,
            },
            epoch=int(epoch),
        )
        return "IDLE_HOT", None, new_until, None, None, None, int(epoch)

    def _enter_cold_idle(self, epoch: int):
        self.log_info("runtime", "enter cold idle")
        self._record_event("ENTER_IDLE", trigger="idle_transition", epoch=int(epoch))
        self.stage_controller.set_runtime_mode("IDLE", reason="enter_cold_idle", force=True)
        return "IDLE", None, 0.0, None, None, None, int(epoch)

    def _emit_heartbeat_if_needed(self, force: bool = False):
        if not CONFIG.runtime.heartbeat_enabled:
            return
        now = time.time()
        interval_s = max(1.0, float(CONFIG.runtime.heartbeat_interval_s))
        if not force and (now - self._last_heartbeat_ts) < interval_s:
            return
        self._last_heartbeat_ts = now
        req_snapshot = self.req_server.snapshot()
        obs_snapshot = self.obs_sender.snapshot()
        runtime_snapshot = self.runtime.runtime_snapshot()
        mode_snapshot = dict((self.stage_controller.snapshot().get("mode_controller") or {}))
        last_req_age_s = (now - self.last_req_receive_ts) if self.last_req_receive_ts else None
        last_obs_send_age_s = (now - self.last_send_ts) if self.last_send_ts else None
        self.run_logger.write_heartbeat_record(
            stage=self.current_stage,
            mode=self.current_mode,
            session_id=self.current_session_id,
            req_id=self.current_req_id,
            epoch=self.current_epoch,
            last_req_age_s=last_req_age_s,
            last_obs_send_age_s=last_obs_send_age_s,
            ready={
                "req_in_listening": bool(req_snapshot.get("listening")),
                "obs_out_link_state": obs_snapshot.get("link_state"),
            },
            data={
                "target": self.target_name,
                "hot_until_ts": self.hot_until_ts,
                "req_in": req_snapshot,
                "obs_out": obs_snapshot,
                "engine": {
                    "current_mode": mode_snapshot.get("current_mode"),
                    "target_mode": mode_snapshot.get("target_mode"),
                    "generation": mode_snapshot.get("generation"),
                    "runtime_running": runtime_snapshot.get("runtime_running"),
                },
            },
        )

    def _send_interval_s(self) -> float:
        return 1.0 / max(0.5, CONFIG.runtime.send_hz)

    def _sync_runtime_request_context(self, req: VisionReq):
        if req.session_id:
            self.current_session_id = req.session_id
        if req.req_id:
            self.current_req_id = req.req_id
        self.current_epoch = int(req.epoch)

    def _sync_runtime_from_stage_context(self, reason: str = ""):
        ctx = self.stage_controller.context()
        prev_stage = self.current_stage
        prev_mode = self.current_mode
        self.current_stage = self._safe_stage_text(ctx.current_stage)
        self.current_mode = self._safe_mode_text(ctx.current_mode)
        self.target_name = ctx.target_name
        self.current_session_id = ctx.session_id
        self.current_req_id = ctx.req_id
        self.current_epoch = int(ctx.epoch)
        self.active_interaction_id = ctx.interaction_id
        if prev_stage != self.current_stage or prev_mode != self.current_mode:
            payload = {
                "reason": reason,
                "prev_stage": prev_stage,
                "stage": self.current_stage,
                "prev_mode": prev_mode,
                "mode": self.current_mode,
                "session_id": self.current_session_id,
                "req_id": self.current_req_id,
                "epoch": self.current_epoch,
            }
            self.log_info("runtime", "stage/mode changed", payload)

    def _apply_stage_output(self, output, now: float, force_send: bool = False) -> bool:
        if output is None:
            return False
        if output.vision_obs is None:
            return False
        if not force_send and (now - self.last_send_ts) < self._send_interval_s():
            return False
        queued = self._send_obs(output.vision_obs)
        if queued:
            self.last_send_ts = now
        return queued

    def _handle_stop_request(self, stage: str, stop_state=None):
        self._record_event("VISION_STOP", trigger="request:STOP", stage=stage)
        state = dict(stop_state or {})
        prev_mode = str(state.get("mode") or self.current_mode or "IDLE").strip().upper()
        prev_target = state.get("target_name", self.target_name)
        stop_epoch = int(state.get("epoch", self.current_epoch))
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
                prev_mode,
                prev_target,
                stop_epoch,
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
            ) = self._enter_cold_idle(stop_epoch)
        self.current_stage = "IDLE"
        self.active_interaction_id = None

    def _handle_request_payload(self, payload):
        typ = str(payload.get("type", "vision_req")).strip()
        if typ not in {"vision_req", "home_tag_req"}:
            return

        req = VisionReq.from_dict(payload)
        request_stage = self._safe_stage_text(req.stage)
        self._sync_runtime_request_context(req)
        req_event_data = {
            "op": req.op,
            "mode_hint": req.mode_hint,
            "payload": req.payload,
            "legacy_type": req.legacy_type,
            "request_stage": request_stage,
        }

        if req.is_stop():
            stop_state = {
                "stage": self.current_stage,
                "mode": self.current_mode,
                "target_name": self.target_name,
                "epoch": self.current_epoch,
            }
            stage_output = self.stage_controller.handle_request(req)
            self._sync_runtime_from_stage_context(reason=f"request:{req.op}")
            self._record_event(
                "VISION_REQ",
                trigger="req_in",
                stage=self.current_stage,
                interaction_id=req.interaction_id,
                data=req_event_data,
            )
            if stage_output is not None and bool(stage_output.signal("mode_apply_failed", False)):
                self.log_warn(
                    "runtime",
                    "skip stop flow due to mode_apply_failed",
                    {
                        "stage": self.current_stage,
                        "mode": self.current_mode,
                        "req_id": req.req_id,
                    },
                )
                self._apply_stage_output(stage_output, now=time.time(), force_send=True)
                return
            self._handle_stop_request(request_stage, stop_state=stop_state)
            return

        self.hot_until_ts = 0.0
        stage_output = self.stage_controller.handle_request(req)
        self._sync_runtime_from_stage_context(reason=f"request:{req.op}")
        self._record_event(
            "VISION_REQ",
            trigger="req_in",
            stage=self.current_stage,
            interaction_id=req.interaction_id,
            data=req_event_data,
        )
        obs_sent = self._apply_stage_output(stage_output, now=time.time(), force_send=bool(stage_output and stage_output.vision_obs))
        if not obs_sent:
            self.last_send_ts = 0.0

    def _tick_stage(self, now: float):
        tick_input = self.runtime.collect_tick_input(ts=now)
        tick_input.snapshot["app"] = {
            "stage": self.current_stage,
            "mode": self.current_mode,
            "session_id": self.current_session_id,
            "req_id": self.current_req_id,
            "epoch": self.current_epoch,
            "hot_until_ts": self.hot_until_ts,
        }
        stage_output = self.stage_controller.tick(tick_input)
        self._sync_runtime_from_stage_context(reason="tick")
        self._apply_stage_output(stage_output, now=now)

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
        ) = self._enter_cold_idle(self.current_epoch)
        self.current_stage = "IDLE"

    def start(self):
        cfg_dump = self._config_dump()
        self.run_logger.write_meta(
            {
                "service": "vision",
                "run_dir": str(self.run_logger.run_dir),
                "project_root": CONFIG.runtime.project_root,
                "log_file": CONFIG.runtime.log_file,
                "pid_file": CONFIG.runtime.pid_file,
                "structured_logs": self.log_paths,
                "config": cfg_dump,
            }
        )
        self._record_event("SERVICE_STARTING", trigger="start", data={"run_dir": str(self.run_logger.run_dir)})
        self.log_info("runtime", "structured logs ready", self.log_paths)
        self.req_server.start()
        self.runtime.init()
        self.runtime.start()
        self.stage_controller.set_runtime_mode("IDLE", reason="service_start", force=True)
        self._sync_runtime_from_stage_context(reason="service_start")
        self._running = True
        self._record_event("SERVICE_READY", trigger="start")
        self.log_info(
            "runtime",
            "SERVICE_READY",
            {
                "run_dir": str(self.run_logger.run_dir),
                "event_file": self.log_paths.get("event"),
                "ipc_file": self.log_paths.get("ipc"),
                "meta_file": self.log_paths.get("meta"),
                "heartbeat_file": self.log_paths.get("heartbeat", "disabled"),
            },
        )
        self._emit_heartbeat_if_needed(force=True)

    def stop(self):
        if self._stopped:
            return
        self._stopped = True
        self._running = False
        self._record_event("SERVICE_STOPPING", trigger="stop")
        self.req_server.close()
        self.obs_sender.close()
        self.runtime.stop()
        self._record_event("SERVICE_STOPPED", trigger="stop")
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
                    self._record_ipc(
                        direction="RX",
                        channel="req_in",
                        event="received",
                        ok=True,
                        msg_type=typ,
                        session_id=payload.get("session_id"),
                        req_id=payload.get("req_id"),
                        epoch=payload.get("epoch"),
                    )
                    self._handle_request_payload(payload)

                now = time.time()
                self._tick_stage(now)
                self._expire_hot_standby(now)

                self._emit_heartbeat_if_needed()
                dt = time.time() - loop_start
                if dt < target_frame_time:
                    time.sleep(target_frame_time - dt)

        except KeyboardInterrupt:
            self.log_info("runtime", "keyboard interrupt received")
        except Exception as exc:
            self._record_event("FATAL", level="error", trigger="main_loop", data={"error": str(exc)})
            self.log_error("runtime", f"vista main loop crashed: {exc}")
        finally:
            self.stop()


def main():
    app = VistaApp()
    app.run()


if __name__ == "__main__":
    main()
