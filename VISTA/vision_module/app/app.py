#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
import time
from collections import deque
from pathlib import Path
from typing import Any, Dict, Optional

STACK_ROOT = Path(__file__).resolve().parents[3]
if str(STACK_ROOT) not in sys.path:
    sys.path.insert(0, str(STACK_ROOT))

from common.base_module import BaseModule
from common.runtime_logging import RunLogger, ensure_dir, env_flag
from common.system_metrics import SystemMetricsSampler

from ..backend.camera_manager import CameraManager
from ..backend.mode_controller import ModeController
from ..backend.predictor_manager import PredictorManager
from ..backend.preview import NullPreviewSink
from ..backend.preview.manager import PreviewManager
from ..backend.remote.client import RemoteGraspClient
from ..backend.remote.manager import RemoteManager
from ..backend.runtime_supervisor import RuntimeSupervisor
from ..backend.scheduler import Scheduler
from ..backend.table_edge_manager import TableEdgeManager
from ..config.mode_defaults import build_default_mode_profiles
from ..config.board_config import CONFIG
from ..diagnostics.operator_console import OperatorConsole
from ..ipc.protocol import VisionReq
from ..ipc.transport import JsonlClientSender, JsonlInboundServer
from .observation import ObservationMetrics, ObservationRouter
from .stage_controller import StageController
from .stages import GraspStagePlan, InitStagePlan, ReturnStagePlan, SearchStagePlan


class VistaApp(BaseModule):
    def __init__(self):
        super().__init__("vision", CONFIG.runtime.log_enabled, CONFIG.runtime.log_mode)
        ensure_dir(CONFIG.runtime.log_dir)
        ensure_dir(CONFIG.runtime.runs_dir)
        ensure_dir(CONFIG.runtime.pid_dir)
        vision_runs_root = Path(CONFIG.runtime.runs_dir)
        self.run_logger = RunLogger(
            "vision",
            str(vision_runs_root),
            CONFIG.runtime.stack_run_id,
            enable_text_events=False,
            module_subdir=True,
        )
        self.operator_console = OperatorConsole(
            mode=CONFIG.runtime.console_mode,
            default_interval_s=CONFIG.runtime.operator_summary_interval_s,
        )
        self.log_paths = self.run_logger.structured_paths(heartbeat_enabled=CONFIG.runtime.heartbeat_enabled)
        self.scheduler = Scheduler()
        camera_manager = CameraManager(cfg=CONFIG, logger=self.child_logger("camera"))
        predictor_manager = PredictorManager(cfg=CONFIG, logger=self.child_logger("predictor"))
        remote_manager = RemoteManager(client=RemoteGraspClient(logger=self.child_logger("remote")),
                                       logger=self.child_logger("remote"))
        table_edge_manager = TableEdgeManager(cfg=CONFIG, logger=self.child_logger("table_edge"))
        preview_manager = PreviewManager(sink=NullPreviewSink(), logger=self.child_logger("preview"), cfg=CONFIG)
        self.supervisor = RuntimeSupervisor(
            scheduler=self.scheduler,
            camera_manager=camera_manager,
            predictor_manager=predictor_manager,
            remote_manager=remote_manager,
            table_edge_manager=table_edge_manager,
            preview_manager=preview_manager,
            logger=self.child_logger("supervisor"),
            backend_event_sink=self._record_backend_event,
        )
        self.mode_controller = ModeController(
            scheduler=self.scheduler,
            supervisor=self.supervisor,
            logger=self.child_logger("mode"),
            backend_event_sink=lambda event, **fields: self._record_backend_event(event, fields),
            preview_allowed=bool(CONFIG.debug.preview),
        )
        self.mode_controller.register_profiles(build_default_mode_profiles(CONFIG.model.active_model, CONFIG).values())
        self.req_server = JsonlInboundServer(
            mode=CONFIG.req_in.transport,
            uds_path=CONFIG.req_in.ipc_socket_path,
            name="req_in",
            logger=self._log_ipc_event,
        )
        self.obs_sender = JsonlClientSender(
            mode=CONFIG.obs_out.transport,
            uds_path=CONFIG.obs_out.ipc_socket_path,
            name="obs_out",
            logger=self._log_ipc_event,
            queue_size=1,
            latest_only=True,
        )
        self.diag_sender = JsonlClientSender(
            mode=CONFIG.obs_out.transport,
            uds_path=CONFIG.obs_out.ipc_socket_path,
            name="obs_diag",
            logger=self._log_ipc_event,
            queue_size=1,
            latest_only=True,
        )
        self.stage_controller = StageController(
            logger=self.child_logger("stage"),
            event_sink=self._record_stage_event,
            mode_controller=self.mode_controller,
            scheduler=self.scheduler,
        )
        self.stage_controller.register_default_plans(
            {
                "INIT": InitStagePlan(),
                "SEARCH": SearchStagePlan(),
                "GRASP": GraspStagePlan(),
                "RETURN": ReturnStagePlan(),
            }
        )
        self.last_send_ts = 0.0
        self.last_req_receive_ts = 0.0
        self.hot_until_ts = 0.0
        self._prev_synced = {}
        self.active_task_key = None
        self._running = False
        self._stopped = False
        self._last_heartbeat_ts = 0.0
        self._last_runtime_reconciled_console = ""
        self._rate_window_s = 10.0
        self._last_rate_emit_ts = 0.0
        self._rate_target_ts = deque(maxlen=256)
        self._rate_edge_ts = deque(maxlen=256)
        self._rate_edge_age_samples = deque(maxlen=256)
        self._rate_request_ts = deque(maxlen=256)
        self._rate_mode_request_ts = deque(maxlen=128)
        self._rate_target_update_ts = deque(maxlen=128)
        self._rate_idempotent_request_ts = deque(maxlen=128)
        self._rate_mode_reset_ts = deque(maxlen=64)
        self._rate_ipc_rx_ts = deque(maxlen=256)
        self._rate_ipc_tx_ts = deque(maxlen=256)
        self._rate_ipc_enqueue_ts = deque(maxlen=256)
        self._rate_obs_out_send_ts = deque(maxlen=256)
        self._last_obs_out_send_ts = 0.0
        self._obs_out_drop_or_skip_count = 0
        self._obs_out_skip_reason = ""
        self._last_rate_target_key = None
        self._last_rate_edge_key = None
        self._last_request_trace_ts = 0.0
        self._last_periodic_log_ts: Dict[str, float] = {}
        self._system_metrics = SystemMetricsSampler("vision", interval_s=float(os.getenv("VISION_SYSTEM_METRICS_INTERVAL_S", "1.0") or 1.0))
        self._last_preview_timing_log_ts = 0.0
        self._last_main_loop_ms = 0.0
        self.obs_skip_count = 0
        self.obs_drop_count = 0
        self.obs_total_age_ms = 0.0
        self.same_frame_reuse_count = 0
        self._last_processed_frame_id = None
        self._last_diag_send_ts = 0.0
        self._rate_diag_send_ts = deque(maxlen=256)
        self.obs_metrics = ObservationMetrics()
        self.obs_router = ObservationRouter(metrics=self.obs_metrics, control_send_interval_s=self._control_send_interval_s())
        self._warn_deprecated_env()

    def _warn_deprecated_env(self):
        deprecated = {
            "VISTA_TRACK_LOCAL_LIGHT_EDGE": "TableEdgeProfile.detector_mode",
            "VISTA_TRACK_LOCAL_EDGE_STRIDE": "TableEdgeProfile.light_stride / fast_plane_stride",
            "VISTA_TRACK_LOCAL_EDGE_UPDATE_HZ": "TableEdgeProfile.update_hz",
        }
        for var, replacement in deprecated.items():
            if os.environ.get(var):
                self.log_warn("deprecation", f"env {var} is deprecated, use {replacement} in ModeProfile instead")

    def _ctx(self):
        return self.stage_controller.context()

    def _config_dump(self):
        min_send_interval_ms = self._send_interval_s() * 1000.0
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
            "track_local_send_hz": CONFIG.runtime.track_local_send_hz,
            "min_send_interval_ms": min_send_interval_ms,
            "camera_max_fps": CONFIG.camera.max_fps,
            "camera_stream_fps": {
                name: getattr(stream, "fps", None)
                for name, stream in dict(CONFIG.camera.streams or {}).items()
            },
            "camera_publish_hz": CONFIG.camera.max_fps,
            "hot_standby_s": CONFIG.runtime.hot_standby_s,
            "keep_preview_after_stop": CONFIG.runtime.keep_preview_after_stop,
            "keep_model_hot_in_standby": CONFIG.runtime.keep_model_hot_in_standby,
            "enable_infer_during_hot_standby": CONFIG.runtime.enable_infer_during_hot_standby,
            "capability_placeholder": CONFIG.runtime.capability_placeholder,
            "heartbeat_enabled": CONFIG.runtime.heartbeat_enabled,
            "heartbeat_interval_s": CONFIG.runtime.heartbeat_interval_s,
            "req_in": {
                "transport": CONFIG.req_in.transport,
                "ipc_socket_path": CONFIG.req_in.ipc_socket_path,
            },
            "obs_out": {
                "transport": CONFIG.obs_out.transport,
                "ipc_socket_path": CONFIG.obs_out.ipc_socket_path,
            },
            "structured_logs": self.log_paths,
            "yolo26_enabled": bool(CONFIG.model.enable_yolo26),
            "yolo_table_search_enabled": bool(CONFIG.model.enable_yolo_table_search),
            "active_model": CONFIG.model.active_model,
            "active_model_path": getattr(CONFIG.model.profiles.get(CONFIG.model.active_model), "target_model", ""),
        }

    def _current_event_context(self):
        ctx = self._ctx()
        return {
            "stage": self._safe_stage_text(ctx.current_stage),
            "mode": self._safe_mode_text(ctx.current_mode),
            "session_id": ctx.session_id,
            "req_id": ctx.req_id,
            "epoch": int(getattr(ctx, "epoch", 0) or 0),
            "interaction_id": ctx.interaction_id,
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
        event_name = str(event or "STAGE_EVENT").strip().upper()
        if self._console_allows_event(event_name, level=level):
            self.log(level, "stage", str(event or "STAGE_EVENT").strip().lower(), data or None)
        self.run_logger.write_event_record(
            event=event_name,
            level=level,
            trigger=trigger,
            data=data,
            **payload,
        )

    def _record_backend_event(self, event: str, fields):
        payload = dict(fields or {})
        event_name = str(event or "BACKEND_EVENT").strip().upper()
        level = str(payload.pop("level", "info")).strip().lower() or "info"
        data = dict(payload or {})

        stage_override = self._safe_stage_text(self._ctx().current_stage)
        mode_override = self._safe_mode_text(self._ctx().current_mode)
        session_override = self._ctx().session_id
        req_override = self._ctx().req_id
        epoch_override = (int(self._ctx().epoch or 0))
        interaction_override = self._ctx().interaction_id

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

        if self._console_allows_event(event_name, level=level, data=data):
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

    def _console_is_full(self) -> bool:
        return (
            self.operator_console.full
            or str(CONFIG.runtime.log_mode).strip().lower() == "full"
            or bool(CONFIG.runtime.debug)
        )

    def _ipc_console_enabled(self) -> bool:
        return self._console_is_full() or bool(CONFIG.runtime.ipc_console) or env_flag("VISION_IPC_CONSOLE", "0")

    def _heartbeat_console_enabled(self) -> bool:
        return self._console_is_full() or bool(CONFIG.runtime.heartbeat_console) or env_flag("VISION_HEARTBEAT_CONSOLE", "0")

    def _verbose_ipc_success_enabled(self) -> bool:
        return self._console_is_full() or env_flag("VISION_IPC_VERBOSE_SUCCESS", "0")

    def _should_log_periodic(self, name: str, interval_s: float, now: Optional[float] = None) -> bool:
        now = time.time() if now is None else float(now)
        interval_s = max(0.0, float(interval_s))
        last_ts = float(self._last_periodic_log_ts.get(name, 0.0) or 0.0)
        if last_ts > 0.0 and (now - last_ts) < interval_s:
            return False
        self._last_periodic_log_ts[name] = now
        return True

    @staticmethod
    def _float_or_none(value) -> Optional[float]:
        try:
            return float(value)
        except (TypeError, ValueError):
            return None

    def _perf_interval_s(self) -> float:
        try:
            return max(0.2, float(os.getenv("VISION_PERF_TIMING_INTERVAL_S", "1.0") or 1.0))
        except Exception:
            return 1.0

    def _perf_slow_threshold_ms(self) -> float:
        try:
            return max(0.0, float(os.getenv("VISION_PERF_TIMING_SLOW_MS", "180.0") or 180.0))
        except Exception:
            return 180.0

    def _console_allows_event(self, event_name: str, level: str = "info", data=None) -> bool:
        if self.operator_console.mode == "silent":
            return False
        if self._console_is_full():
            return True
        level = str(level or "info").strip().lower()
        event_name = str(event_name or "").strip().upper()
        if level in {"error", "fatal", "warning", "warn"}:
            return True
        if event_name in {"BACKEND_DIAGNOSTIC", "HEARTBEAT"}:
            return False
        if event_name == "BACKEND_RUNTIME_RECONCILED":
            data = dict(data or {})
            key = f"{data.get('mode')}:{data.get('ok')}:{data.get('generation')}"
            if key == self._last_runtime_reconciled_console:
                return False
            self._last_runtime_reconciled_console = key
            return True
        return event_name in {
            "SERVICE_STARTING",
            "SERVICE_READY",
            "SERVICE_STOPPED",
            "STAGE_CHANGED",
            "MODE_CHANGED",
            "BACKEND_MODE_CHANGED",
        }

    def _operator_ipc_line(self, channel: str, event: str, details: Dict[str, object]) -> str:
        parts = [f"[VISTA] IPC {channel} {event}"]
        if details.get("transport"):
            parts.append(f"transport={details.get('transport')}")
        if details.get("bind"):
            parts.append(f"bind={details.get('bind')}")
        if details.get("peer"):
            parts.append(f"peer={details.get('peer')}")
        if details.get("error"):
            parts.append(f"error={details.get('error')}")
        if details.get("owner") is not None:
            parts.append(f"owner={details.get('owner')}")
        if details.get("perm") is not None:
            parts.append(f"perm={details.get('perm')}")
        return " ".join(str(p) for p in parts)

    def _log_ipc_event(self, payload):
        level = payload.get("level", "info")
        channel = payload.get("name", payload.get("src", "ipc"))
        event = payload.get("event", payload.get("msg", "log"))
        details = {k: v for k, v in payload.items() if k not in {"level", "src", "name", "event", "msg"}}
        event_norm = str(event).strip().lower()
        now = time.time()
        if event_norm == "recv_ok":
            samples = getattr(self, "_rate_ipc_rx_ts", None)
            if samples is not None:
                samples.append(now)
        elif event_norm == "send_ok":
            samples = getattr(self, "_rate_ipc_tx_ts", None)
            if samples is not None:
                samples.append(now)
        elif event_norm == "enqueue_ok":
            samples = getattr(self, "_rate_ipc_enqueue_ts", None)
            if samples is not None:
                samples.append(now)
        success_events = {"recv_ok", "send_ok", "send_attempt", "enqueue_ok"}
        noisy_success = event_norm in success_events
        if not noisy_success or self._verbose_ipc_success_enabled():
            self._record_ipc(
                direction=self._ipc_direction_for(channel),
                channel=channel,
                event=event,
                level=level,
                **details,
            )
        console_events = {
            "listening",
            "connected",
            "connect_failed",
            "send_failed",
            "invalid_json",
            "queue_drop_oldest",
            "queue_drop_new",
            "queue_drop_failed",
        }
        if noisy_success and not self._ipc_console_enabled():
            return
        if self._ipc_console_enabled():
            self.log(level, "ipc", f"{channel} {event}".strip(), details or None)
            return
        if str(event).strip().lower() in console_events:
            line = self._operator_ipc_line(channel, event, details)
            if str(level).strip().lower() in {"warn", "warning", "error", "fatal"}:
                self.operator_console.emit_error(f"ipc:{channel}:{event}:{details.get('error', '')}", line)
            else:
                self.operator_console.emit_rate_limited(f"ipc:{channel}:{event}", line)

    def _safe_mode_text(self, mode: str) -> str:
        return str(mode or "IDLE").strip().upper()

    def _safe_stage_text(self, stage: str) -> str:
        return str(stage or "IDLE").strip().upper()

    def _task_key_for(self, req_stage: str, target, session_id, epoch: int):
        return (req_stage, target, session_id, int(epoch))

    def _send_obs(self, out_payload, sender=None, obs_class="control", priority=0):
        if sender is None:
            sender = self.obs_sender
        now = time.time()
        min_send_interval_ms = self._send_interval_s() * 1000.0
        send_interval_ms = (
            max(0.0, (now - float(self._last_obs_out_send_ts)) * 1000.0)
            if float(self._last_obs_out_send_ts or 0.0) > 0.0
            else None
        )
        send_hz = 1000.0 / float(send_interval_ms) if send_interval_ms and send_interval_ms > 0.0 else 0.0
        perception = out_payload.get("perception") if isinstance(out_payload, dict) else None
        if isinstance(perception, dict):
            for obs_key in ("table_edge_obs", "target_obs", "home_tag_obs"):
                obs = perception.get(obs_key)
                if isinstance(obs, dict):
                    obs["obs_out_send_ts_ms"] = int(round(now * 1000.0))
                    obs["obs_out_send_interval_ms"] = send_interval_ms
                    obs["obs_out_send_hz"] = float(send_hz)
                    obs["obs_out_drop_or_skip_count"] = int(self._obs_out_drop_or_skip_count)
                    obs["obs_out_skip_reason"] = str(self._obs_out_skip_reason or "")
                    obs["send_hz_config"] = float(CONFIG.runtime.send_hz)
                    obs["track_local_send_hz_config"] = float(CONFIG.runtime.track_local_send_hz)
                    obs["min_send_interval_ms"] = float(min_send_interval_ms)
        queued = sender.send(out_payload)
        if not queued:
            self._obs_out_drop_or_skip_count += 1
            self.obs_drop_count += 1
            self._obs_out_skip_reason = "enqueue_failed"
            self._record_ipc(
                direction="TX",
                channel="obs_out" if sender == self.obs_sender else "obs_diag",
                event="enqueue_failed",
                level="warn",
                ok=queued,
                req_id=out_payload.get("req_id"),
                session_id=out_payload.get("session_id"),
                epoch=out_payload.get("epoch"),
                stage=out_payload.get("stage"),
                mode=out_payload.get("mode"),
                msg_type=out_payload.get("type"),
                status=out_payload.get("status"),
            )
            self.log_warn("runtime", f"{sender.name} queue busy ({obs_class}); skipped enqueue")
        elif self._ipc_console_enabled():
            self.log_info(
                "ipc",
                f"{sender.name} enqueue_ok",
                {
                    "req_id": out_payload.get("req_id"),
                    "epoch": out_payload.get("epoch"),
                    "msg_type": out_payload.get("type"),
                },
            )
        if queued:
            self._last_obs_out_send_ts = now
            if obs_class == "control":
                self._rate_obs_out_send_ts.append(now)
            else:
                self._rate_diag_send_ts.append(now)
            self._obs_out_skip_reason = ""
        return queued

    def _trim_rate_window(self, now: float) -> None:
        cutoff = float(now) - float(self._rate_window_s)
        for samples in (
            self._rate_target_ts,
            self._rate_edge_ts,
            self._rate_edge_age_samples,
            self._rate_request_ts,
            self._rate_mode_request_ts,
            self._rate_target_update_ts,
            self._rate_idempotent_request_ts,
            self._rate_mode_reset_ts,
            getattr(self, "_rate_ipc_rx_ts", ()),
            getattr(self, "_rate_ipc_tx_ts", ()),
            getattr(self, "_rate_ipc_enqueue_ts", ()),
            getattr(self, "_rate_obs_out_send_ts", ()),
        ):
            while samples and float(samples[0][0] if isinstance(samples[0], tuple) else samples[0]) < cutoff:
                samples.popleft()

    @staticmethod
    def _percentile(values, percentile: float) -> Optional[float]:
        nums = sorted(float(v) for v in values if v is not None)
        if not nums:
            return None
        idx = int(round((len(nums) - 1) * max(0.0, min(1.0, float(percentile)))))
        return nums[idx]

    @staticmethod
    def _fmt_rate_value(value: Any, digits: int = 1) -> str:
        if value is None:
            return "n/a"
        try:
            return f"{float(value):.{digits}f}"
        except Exception:
            return "n/a"

    def _hz_for_samples(self, samples, now: float) -> float:
        if not samples:
            return 0.0
        oldest = float(samples[0][0] if isinstance(samples[0], tuple) else samples[0])
        span_s = min(float(self._rate_window_s), max(1.0, float(now) - oldest))
        return float(len(samples)) / span_s

    def _preview_fps_snapshot(self) -> Optional[float]:
        try:
            preview = dict(self.mode_controller.runtime_snapshot().get("capabilities", {}).get("preview") or {})
            value = preview.get("preview_fps")
            return float(value) if value is not None else None
        except Exception:
            return None

    def _preview_timing_snapshot(self) -> Dict[str, Any]:
        try:
            preview = dict(self.mode_controller.runtime_snapshot().get("capabilities", {}).get("preview") or {})
            timing = dict(preview.get("preview_timing") or {})
            return timing
        except Exception:
            return {}

    def _record_rate_sample(self, out_payload: Dict[str, Any], sent_ts: float) -> None:
        if str(out_payload.get("type") or "") != "vision_obs":
            return
        self._record_perf_timing(out_payload, sent_ts)
        mode = str(out_payload.get("mode") or "").strip().upper()
        if mode not in {"FIND_OBJECT", "FIND_EDGE"}:
            return
        perception = out_payload.get("perception")
        if not isinstance(perception, dict):
            return
        target_obs = perception.get("target_obs")
        if isinstance(target_obs, dict):
            target_key = (
                target_obs.get("frame_id"),
                target_obs.get("seq"),
                target_obs.get("obs_ts"),
            )
            if target_key != self._last_rate_target_key:
                self._last_rate_target_key = target_key
                self._rate_target_ts.append(float(sent_ts))
        edge_obs = perception.get("table_edge_obs")
        if isinstance(edge_obs, dict):
            now = time.time()
            profile = edge_obs.get("edge_profile")
            edge_profile_log_ms = None
            edge_profile_every_frame = env_flag("VISION_EDGE_PROFILE_EVERY_FRAME", "0") or self._console_is_full()
            should_log_edge_profile = bool(
                isinstance(profile, dict)
                and (edge_profile_every_frame or self._should_log_periodic("edge_profile", 2.0, now=now))
            )
            if should_log_edge_profile:
                profile_log_start = time.perf_counter()
                self.run_logger.write_jsonl(
                    "edge_profile",
                    {
                        "mode": out_payload.get("mode"),
                        "stage": out_payload.get("stage"),
                        "req_id": out_payload.get("req_id"),
                        "session_id": out_payload.get("session_id"),
                        "epoch": out_payload.get("epoch"),
                        "frame_id": edge_obs.get("frame_id"),
                        "seq": edge_obs.get("seq"),
                        "obs_ts": edge_obs.get("obs_ts"),
                        "edge_process_path": edge_obs.get("edge_process_path"),
                        **profile,
                    },
                )
                edge_profile_log_ms = max(0.0, (time.perf_counter() - profile_log_start) * 1000.0)
            obs_total_age_ms = self._float_or_none(edge_obs.get("obs_total_age_ms"))
            vision_process_ms = self._float_or_none(edge_obs.get("vision_process_ms"))
            abnormal_frame = (
                (obs_total_age_ms is not None and obs_total_age_ms >= 300.0)
                or (vision_process_ms is not None and vision_process_ms >= 180.0)
            )
            should_log_frame_timing = abnormal_frame or self._should_log_periodic("frame_timing", 1.0, now=now)
            frame_record = {
                "mode": out_payload.get("mode"),
                "stage": out_payload.get("stage"),
                "req_id": out_payload.get("req_id"),
                "session_id": out_payload.get("session_id"),
                "epoch": out_payload.get("epoch"),
                "frame_id": edge_obs.get("frame_id"),
                "seq": edge_obs.get("seq"),
                "obs_ts": edge_obs.get("obs_ts"),
                "age_ms": edge_obs.get("age_ms"),
                "table_bbox_xyxy": edge_obs.get("table_bbox_xyxy"),
                "table_cx_norm": edge_obs.get("table_cx_norm"),
                "table_cy_norm": edge_obs.get("table_cy_norm"),
                "table_size_norm": edge_obs.get("table_size_norm"),
                "table_conf": edge_obs.get("table_conf", edge_obs.get("yolo_table_conf")),
                "yolo_table_bbox": edge_obs.get("yolo_table_bbox"),
                "yolo_table_visible": edge_obs.get("yolo_table_visible"),
                "yolo_table_fresh": edge_obs.get("yolo_table_fresh"),
                "yolo_table_age_ms": edge_obs.get("yolo_table_age_ms"),
                "edge_found": edge_obs.get("edge_found"),
                "edge_valid": edge_obs.get("edge_valid"),
                "edge_trusted": edge_obs.get("edge_trusted"),
                "obs_seq": edge_obs.get("obs_seq"),
                "camera_frame_seq": edge_obs.get("camera_frame_seq"),
                "frame_capture_ts": edge_obs.get("frame_capture_ts"),
                "camera_frame_ts_ms": edge_obs.get("camera_frame_ts_ms"),
                "vision_start_ts": edge_obs.get("vision_start_ts"),
                "vision_done_ts": edge_obs.get("vision_done_ts"),
                "obs_publish_ts": edge_obs.get("obs_publish_ts"),
                "vision_process_start_ts_ms": edge_obs.get("vision_process_start_ts_ms"),
                "vision_process_end_ts_ms": edge_obs.get("vision_process_end_ts_ms"),
                "vision_publish_ts_ms": edge_obs.get("vision_publish_ts_ms"),
                "obs_out_send_ts_ms": edge_obs.get("obs_out_send_ts_ms"),
                "obs_send_enqueue_ts": float(sent_ts),
                "log_record_start_ts": time.time(),
                "camera_frame_interval_ms": edge_obs.get("camera_frame_interval_ms"),
                "camera_frame_hz": edge_obs.get("camera_frame_hz", edge_obs.get("camera_frames_hz")),
                "vision_process_interval_ms": edge_obs.get("vision_process_interval_ms"),
                "vision_publish_interval_ms": edge_obs.get("vision_publish_interval_ms", edge_obs.get("table_edge_publish_interval_ms")),
                "table_edge_worker_interval_ms": edge_obs.get("table_edge_worker_interval_ms"),
                "table_edge_no_new_frame_count": edge_obs.get("table_edge_no_new_frame_count"),
                "scheduler_publish_ms": edge_obs.get("scheduler_publish_ms"),
                "obs_out_send_interval_ms": edge_obs.get("obs_out_send_interval_ms"),
                "obs_out_send_hz": edge_obs.get("obs_out_send_hz"),
                "obs_out_drop_or_skip_count": edge_obs.get("obs_out_drop_or_skip_count"),
                "obs_out_skip_reason": edge_obs.get("obs_out_skip_reason"),
                "send_hz_config": edge_obs.get("send_hz_config"),
                "track_local_send_hz_config": edge_obs.get("track_local_send_hz_config"),
                "frame_age_ms": edge_obs.get("frame_age_ms"),
                "vision_process_ms": edge_obs.get("vision_process_ms"),
                "publish_delay_ms": edge_obs.get("publish_delay_ms"),
                "obs_total_age_ms": edge_obs.get("obs_total_age_ms"),
                "latest_frame_lag_ms": edge_obs.get("latest_frame_lag_ms"),
                "depth_frame_fetch_ms": edge_obs.get("depth_frame_fetch_ms"),
                "edge_process_path": edge_obs.get("edge_process_path"),
                "detector_mode": edge_obs.get("detector_mode"),
                "source_mode": edge_obs.get("source_mode"),
                "dropped_frame_count": edge_obs.get("dropped_frame_count"),
                "processed_frame_count": edge_obs.get("processed_frame_count"),
                "edge_profile_log_write_ms": edge_profile_log_ms,
            }
            frame_log_ms = None
            if should_log_frame_timing:
                frame_log_start = time.perf_counter()
                self.run_logger.write_jsonl("frame_timing", frame_record)
                frame_log_ms = max(0.0, (time.perf_counter() - frame_log_start) * 1000.0)
            slow_log_write = any(
                value is not None and float(value) > 10.0
                for value in (edge_profile_log_ms, frame_log_ms)
            )
            if slow_log_write or self._should_log_periodic("log_timing", 5.0, now=now):
                self.run_logger.write_jsonl(
                    "log_timing",
                    {
                        "mode": out_payload.get("mode"),
                        "stage": out_payload.get("stage"),
                        "req_id": out_payload.get("req_id"),
                        "session_id": out_payload.get("session_id"),
                        "epoch": out_payload.get("epoch"),
                        "frame_id": edge_obs.get("frame_id"),
                        "seq": edge_obs.get("seq"),
                        "edge_profile_log_write_ms": edge_profile_log_ms,
                        "frame_timing_log_write_ms": frame_log_ms,
                        "summary": not slow_log_write,
                    },
                )
            edge_key = (
                edge_obs.get("source_mode"),
                edge_obs.get("frame_id"),
                edge_obs.get("seq"),
                edge_obs.get("obs_ts"),
            )
            if edge_key != self._last_rate_edge_key:
                self._last_rate_edge_key = edge_key
                self._rate_edge_ts.append(float(sent_ts))
            age_ms = edge_obs.get("age_ms")
            try:
                self._rate_edge_age_samples.append((float(sent_ts), float(age_ms)))
            except Exception:
                pass

    def _record_perf_timing(self, out_payload: Dict[str, Any], sent_ts: float) -> None:
        now = time.time()
        perception = out_payload.get("perception") if isinstance(out_payload, dict) else None
        edge_obs = perception.get("table_edge_obs") if isinstance(perception, dict) else None
        local = None
        try:
            local = self.scheduler.read_result("local_perception", default=None)
        except Exception:
            local = None
        if not isinstance(local, dict):
            local = perception.get("local_perception") if isinstance(perception, dict) else None
        if not isinstance(local, dict):
            local = {}
        preview_timing = self._preview_timing_snapshot()

        yolo_infer_ms = self._float_or_none(local.get("yolo_infer_ms"))
        yolo_roi_ms = self._float_or_none(local.get("yolo_roi_ms"))
        edge_process_ms = self._float_or_none((edge_obs or {}).get("vision_process_ms")) if isinstance(edge_obs, dict) else None
        obs_total_age_ms = self._float_or_none((edge_obs or {}).get("obs_total_age_ms")) if isinstance(edge_obs, dict) else None
        edge_profile = (edge_obs or {}).get("edge_profile") if isinstance(edge_obs, dict) else None
        edge_profile_ms = {}
        if isinstance(edge_profile, dict):
            edge_profile_ms = {
                str(key): self._float_or_none(value)
                for key, value in edge_profile.items()
                if str(key).endswith("_ms") and self._float_or_none(value) is not None
            }
        obs_enqueue_ms = max(0.0, (now - float(sent_ts)) * 1000.0)
        preview_total_ms = self._float_or_none(preview_timing.get("preview_total_ms_avg"))
        main_loop_ms = self._float_or_none(getattr(self, "_last_main_loop_ms", None))
        edge_frame_seq = (edge_obs or {}).get("camera_frame_seq") if isinstance(edge_obs, dict) else None
        frame_seq = local.get("frame_seq") or edge_frame_seq

        slow_threshold = self._perf_slow_threshold_ms()
        slow = any(
            value is not None and float(value) >= slow_threshold
            for value in (yolo_infer_ms, edge_process_ms, obs_total_age_ms, main_loop_ms)
        )
        if not slow and not self._should_log_periodic("perf_timing", self._perf_interval_s(), now=now):
            return

        record = {
            "ts": now,
            "summary": not slow,
            "slow": bool(slow),
            "stage": out_payload.get("stage"),
            "mode": out_payload.get("mode"),
            "status": out_payload.get("status"),
            "req_id": out_payload.get("req_id"),
            "session_id": out_payload.get("session_id"),
            "epoch": out_payload.get("epoch"),
            "frame_seq": frame_seq,
            "yolo": {
                "enabled": bool(local.get("yolo_infer_running") or local.get("yolo_has_infer")),
                "has_infer": bool(local.get("has_infer")),
                "model_name": local.get("model_name"),
                "predictor_type": local.get("predictor_type"),
                "box_count": local.get("box_count"),
                "infer_ms": yolo_infer_ms,
                "roi_ms": yolo_roi_ms,
                "infer_error": local.get("infer_error"),
            },
            "table_edge": {
                "process_ms": edge_process_ms,
                "table_bbox_xyxy": (edge_obs or {}).get("table_bbox_xyxy") if isinstance(edge_obs, dict) else None,
                "yolo_table_visible": (edge_obs or {}).get("yolo_table_visible") if isinstance(edge_obs, dict) else None,
                "yolo_table_fresh": (edge_obs or {}).get("yolo_table_fresh") if isinstance(edge_obs, dict) else None,
                "yolo_table_age_ms": (edge_obs or {}).get("yolo_table_age_ms") if isinstance(edge_obs, dict) else None,
                "edge_valid": (edge_obs or {}).get("edge_valid") if isinstance(edge_obs, dict) else None,
                "edge_trusted": (edge_obs or {}).get("edge_trusted") if isinstance(edge_obs, dict) else None,
                "latest_frame_lag_ms": (edge_obs or {}).get("latest_frame_lag_ms") if isinstance(edge_obs, dict) else None,
                "obs_total_age_ms": obs_total_age_ms,
                "scheduler_read_ms": (edge_obs or {}).get("scheduler_read_ms") if isinstance(edge_obs, dict) else None,
                "scheduler_publish_ms": (edge_obs or {}).get("scheduler_publish_ms") if isinstance(edge_obs, dict) else None,
                "detector_mode": (edge_obs or {}).get("detector_mode") if isinstance(edge_obs, dict) else None,
                "process_path": (edge_obs or {}).get("edge_process_path") if isinstance(edge_obs, dict) else None,
                "profile_ms": edge_profile_ms,
            },
            "ipc": {
                "obs_enqueue_ms": obs_enqueue_ms,
                "obs_out_send_interval_ms": (edge_obs or {}).get("obs_out_send_interval_ms") if isinstance(edge_obs, dict) else None,
                "obs_out_send_hz": (edge_obs or {}).get("obs_out_send_hz") if isinstance(edge_obs, dict) else None,
                "obs_out_drop_or_skip_count": int(self._obs_out_drop_or_skip_count),
                "obs_out_skip_reason": str(self._obs_out_skip_reason or ""),
                "req_in_hz": self._hz_for_samples(getattr(self, "_rate_ipc_rx_ts", ()), now),
                "obs_out_tx_hz": self._hz_for_samples(getattr(self, "_rate_ipc_tx_ts", ()), now),
                "obs_out_enqueue_hz": self._hz_for_samples(getattr(self, "_rate_ipc_enqueue_ts", ()), now),
            },
            "preview": {
                "enabled": preview_timing.get("preview_enabled"),
                "layout": preview_timing.get("preview_layout"),
                "fps": preview_timing.get("preview_fps"),
                "total_ms_avg": preview_total_ms,
                "total_ms_p95": preview_timing.get("preview_total_ms_p95"),
                "sample_count": preview_timing.get("sample_count"),
            },
            "main_loop_ms": main_loop_ms,
        }
        self.run_logger.write_jsonl("perf_timing", record)

    def _emit_rate_summary_if_needed(self, force: bool = False) -> None:
        mode = self._safe_mode_text(self._ctx().current_mode)
        if mode not in {"FIND_OBJECT", "FIND_EDGE"}:
            return
        now = time.time()
        period_s = max(5.0, float(CONFIG.runtime.operator_summary_interval_s))
        if not force and (now - float(self._last_rate_emit_ts or 0.0)) < period_s:
            return
        self._last_rate_emit_ts = now
        self._trim_rate_window(now)
        window_s = max(1.0, float(self._rate_window_s))
        target_hz = self._hz_for_samples(self._rate_target_ts, now)
        edge_hz = self._hz_for_samples(self._rate_edge_ts, now)
        ages = [sample[1] for sample in self._rate_edge_age_samples]
        p50 = self._percentile(ages, 0.50)
        p95 = self._percentile(ages, 0.95)
        preview_fps = self._preview_fps_snapshot()
        preview_timing = self._preview_timing_snapshot()
        request_hz = self._hz_for_samples(self._rate_request_ts, now)
        mode_request_hz = self._hz_for_samples(self._rate_mode_request_ts, now)
        target_update_hz = self._hz_for_samples(self._rate_target_update_ts, now)
        ipc_rx_hz = self._hz_for_samples(getattr(self, "_rate_ipc_rx_ts", ()), now)
        ipc_tx_hz = self._hz_for_samples(getattr(self, "_rate_ipc_tx_ts", ()), now)
        ipc_enqueue_hz = self._hz_for_samples(getattr(self, "_rate_ipc_enqueue_ts", ()), now)
        obs_out_send_hz = self._hz_for_samples(getattr(self, "_rate_obs_out_send_ts", ()), now)
        record = {
            "mode": self._safe_mode_text(self._ctx().current_mode),
            "request_rate_hz": float(request_hz),
            "mode_request_rate_hz": float(mode_request_hz),
            "target_update_rate_hz": float(target_update_hz),
            "idempotent_request_count": int(len(self._rate_idempotent_request_ts)),
            "mode_reset_count": int(len(self._rate_mode_reset_ts)),
            "target_obs_hz": float(target_hz),
            "table_edge_obs_hz": float(edge_hz),
            "edge_update_hz": float(edge_hz),
            "preview_fps": preview_fps,
            "preview_enabled": preview_timing.get("preview_enabled"),
            "preview_layout": preview_timing.get("preview_layout"),
            "preview_text_level": preview_timing.get("preview_text_level"),
            "preview_debug_points_enabled": preview_timing.get("preview_debug_points_enabled"),
            "preview_total_ms_avg": preview_timing.get("preview_total_ms_avg"),
            "preview_total_ms_p95": preview_timing.get("preview_total_ms_p95"),
            "preview_total_ms_max": preview_timing.get("preview_total_ms_max"),
            "preview_compose_ms_avg": preview_timing.get("preview_compose_ms_avg"),
            "preview_draw_points_ms_avg": preview_timing.get("preview_draw_points_ms_avg"),
            "preview_draw_text_ms_avg": preview_timing.get("preview_draw_text_ms_avg"),
            "preview_draw_legend_ms_avg": preview_timing.get("preview_draw_legend_ms_avg"),
            "preview_imshow_ms_avg": preview_timing.get("preview_imshow_ms_avg"),
            "preview_waitkey_ms_avg": preview_timing.get("preview_waitkey_ms_avg"),
            "edge_age_p50": p50,
            "edge_age_p95": p95,
            "ipc_rx_hz": float(ipc_rx_hz),
            "ipc_tx_hz": float(ipc_tx_hz),
            "ipc_enqueue_hz": float(ipc_enqueue_hz),
            "obs_out_send_hz": float(obs_out_send_hz),
            "obs_out_drop_or_skip_count": int(self._obs_out_drop_or_skip_count),
            "obs_out_skip_reason": str(self._obs_out_skip_reason or ""),
            "send_hz_config": float(CONFIG.runtime.send_hz),
            "track_local_send_hz_config": float(CONFIG.runtime.track_local_send_hz),
            "min_send_interval_ms": float(self._send_interval_s() * 1000.0),
            "camera_publish_hz_config": float(CONFIG.camera.max_fps),
            "table_edge_update_hz_config": None,  # per-mode
            "table_edge_track_local_update_hz_config": None,  # per-mode
            "window_s": float(window_s),
        }
        self.run_logger.write_jsonl("rate", record)
        if preview_timing and (force or now - float(self._last_preview_timing_log_ts or 0.0) >= 1.0):
            self._last_preview_timing_log_ts = now
            self.run_logger.write_jsonl("preview_timing", preview_timing)
        self.operator_console.emit_rate_limited(
            "vision_rate",
            "[VISION][RATE] "
            f"mode={mode} "
            f"request_rate_hz={self._fmt_rate_value(request_hz)} "
            f"mode_request_rate_hz={self._fmt_rate_value(mode_request_hz)} "
            f"target_update_rate_hz={self._fmt_rate_value(target_update_hz)} "
            f"target_obs_hz={self._fmt_rate_value(target_hz)} "
            f"table_edge_obs_hz={self._fmt_rate_value(edge_hz)} "
            f"ipc_rx_hz={self._fmt_rate_value(ipc_rx_hz)} "
            f"ipc_tx_hz={self._fmt_rate_value(ipc_tx_hz)} "
            f"obs_out_send_hz={self._fmt_rate_value(obs_out_send_hz)} "
            f"preview_fps={self._fmt_rate_value(preview_fps)} "
            f"edge_age_p50={self._fmt_rate_value(p50, 0)} "
            f"edge_age_p95={self._fmt_rate_value(p95, 0)}",
            interval_s=period_s,
        )

    def _record_request_trace(self, req: VisionReq) -> None:
        now = time.time()
        trace = dict(getattr(self.stage_controller, "last_request_trace", {}) or {})
        payload = req.payload if isinstance(req.payload, dict) else {}
        req_type = str(trace.get("req_type") or getattr(req, "req_type", "") or payload.get("req_type") or "").strip().lower()
        if req_type not in {"mode_request", "target_update", "keepalive"}:
            req_type = "mode_request" if req.op in {"START", "STOP"} else "target_update"
        previous_ts = float(self._last_request_trace_ts or 0.0)
        elapsed_ms = None if previous_ts <= 0.0 else max(0.0, (now - previous_ts) * 1000.0)
        self._last_request_trace_ts = now
        record = {
            "ts": now,
            "req_id": trace.get("req_id", req.req_id),
            "req_type": req_type,
            "session_id": trace.get("session_id", req.session_id),
            "target": trace.get("target", req.target),
            "requested_mode": trace.get("requested_mode", req.mode_hint),
            "current_mode_before": trace.get("current_mode_before"),
            "current_mode_after": trace.get("current_mode_after", self._safe_mode_text(self._ctx().current_mode)),
            "changed_mode": bool(trace.get("changed_mode", False)),
            "idempotent": bool(trace.get("idempotent", False)),
            "reason": trace.get("reason", ""),
            "time_since_last_request_ms": elapsed_ms,
        }
        self.run_logger.write_jsonl("vision_request_trace", record)
        self._rate_request_ts.append(now)
        if req_type == "mode_request":
            self._rate_mode_request_ts.append(now)
        elif req_type == "target_update":
            self._rate_target_update_ts.append(now)
        if bool(record["idempotent"]):
            self._rate_idempotent_request_ts.append(now)
        if bool(record["changed_mode"]):
            self._rate_mode_reset_ts.append(now)

    def _enter_hot_standby(self, current_mode: str, target_name, epoch: int):
        self.stage_controller.set_runtime_mode("IDLE_HOT", reason="enter_hot_standby", force=True)
        keep_hot = bool(
            getattr(CONFIG.runtime, "keep_vision_alive_after_task", True)
            or getattr(CONFIG.runtime, "keep_preview_alive_after_task", True)
        )
        new_until = 0.0 if keep_hot else time.time() + float(CONFIG.runtime.hot_standby_s)
        self.hot_until_ts = new_until
        self.log_info(
            "runtime",
            "enter hot standby",
            {
                "prev_mode": current_mode,
                "prev_target": target_name,
                "until_ts": new_until,
                "keep_hot": keep_hot,
                "release_model_on_idle": bool(getattr(CONFIG.runtime, "release_model_on_idle", False)),
            },
        )
        self._record_event(
            "ENTER_HOT_STANDBY",
            trigger="stop_flow",
            data={
                "prev_mode": current_mode,
                "prev_target": target_name,
                "until_ts": new_until,
                "keep_hot": keep_hot,
            },
            epoch=int(epoch),
        )

    def _enter_cold_idle(self, epoch: int):
        self.log_info("runtime", "enter cold idle")
        self._record_event("ENTER_IDLE", trigger="idle_transition", epoch=int(epoch))
        self.stage_controller.set_runtime_mode("SILENT", reason="enter_cold_idle", force=True)

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
        runtime_snapshot = self.mode_controller.runtime_snapshot()
        mode_snapshot = dict((self.stage_controller.snapshot().get("mode_controller") or {}))
        last_req_age_s = (now - self.last_req_receive_ts) if self.last_req_receive_ts else None
        last_obs_send_age_s = (now - self.last_send_ts) if self.last_send_ts else None
        self.run_logger.write_heartbeat_record(
            stage=self._safe_stage_text(self._ctx().current_stage),
            mode=self._safe_mode_text(self._ctx().current_mode),
            session_id=self._ctx().session_id,
            req_id=self._ctx().req_id,
            epoch=(int(self._ctx().epoch or 0)),
            last_req_age_s=last_req_age_s,
            last_obs_send_age_s=last_obs_send_age_s,
            ready={
                "req_in_listening": bool(req_snapshot.get("listening")),
                "obs_out_link_state": obs_snapshot.get("link_state"),
            },
            data={
                "target": self._ctx().target_name,
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
        if self._heartbeat_console_enabled():
            self.operator_console.emit_rate_limited(
                "heartbeat",
                f"[VISTA] HEARTBEAT stage={self._safe_stage_text(self._ctx().current_stage)} mode={self._safe_mode_text(self._ctx().current_mode)} "
                f"req={self._ctx().req_id or ''} epoch={(int(self._ctx().epoch or 0))}",
                interval_s=interval_s,
            )

    def _send_interval_s(self) -> float:
        send_hz = float(CONFIG.runtime.send_hz)
        if self._safe_mode_text(self._ctx().current_mode) in {"FIND_OBJECT", "FIND_EDGE"}:
            send_hz = max(send_hz, float(getattr(CONFIG.runtime, "track_local_send_hz", send_hz) or send_hz))
        return 1.0 / max(0.5, send_hz)

    def _control_send_interval_s(self) -> float:
        send_hz = float(CONFIG.runtime.send_hz)
        if self._safe_mode_text(self._ctx().current_mode) in {"FIND_OBJECT", "FIND_EDGE"}:
            send_hz = max(send_hz, float(getattr(CONFIG.runtime, "track_local_send_hz", send_hz) or send_hz))
        # Ensure the control send interval is at most 0.10 to support 8-10 Hz control loop
        send_hz = max(10.0, send_hz)
        return 1.0 / send_hz

    def _should_force_send_stage_output(self, output) -> bool:
        if output is None or not isinstance(getattr(output, "vision_obs", None), dict):
            return False
        obs = output.vision_obs
        status = str(obs.get("status") or "").strip().upper()
        if status in {"START", "STOP", "FAILED", "FATAL"}:
            return True
        signals = getattr(output, "signals", {}) or {}
        request_op = str(signals.get("request_op") or "").strip().upper()
        if request_op in {"START", "STOP"}:
            return True
        if str(signals.get("transition") or "").strip():
            return True
        if bool(signals.get("urgent") or signals.get("force_send") or signals.get("fatal")):
            return True
        if bool(signals.get("mode_apply_failed", False)):
            return True
        result = obs.get("result")
        if isinstance(result, dict) and bool(result.get("urgent") or result.get("fatal")):
            return True
        return False

    def _check_freq_and_reason(self, now: float) -> Optional[str]:
        mode = self._safe_mode_text(self._ctx().current_mode)
        if mode not in {"FIND_OBJECT", "FIND_EDGE"}:
            return None
        if CONFIG.runtime.loop_hz < 8.0:
            return "config loop_hz too low"
        send_hz = float(CONFIG.runtime.send_hz)
        track_send_hz = float(getattr(CONFIG.runtime, "track_local_send_hz", send_hz) or send_hz)
        if track_send_hz < 8.0:
            return "config send_hz too low"
        actual_hz = self._hz_for_samples(self._rate_obs_out_send_ts, now)
        if len(self._rate_obs_out_send_ts) >= 5 and actual_hz < 7.8:
            local = None
            try:
                local = self.scheduler.read_result("local_perception", default=None)
            except Exception:
                pass
            yolo_ms = float(local.get("yolo_infer_ms") or 0.0) if isinstance(local, dict) else 0.0
            edge_obs = None
            try:
                edge_obs = self.scheduler.read_result("table_edge_obs", default=None)
            except Exception:
                pass
            edge_ms = float(edge_obs.get("vision_process_ms") or 0.0) if isinstance(edge_obs, dict) else 0.0
            if yolo_ms > 80.0:
                return f"high inference latency ({yolo_ms:.1f} ms)"
            if edge_ms > 80.0:
                return f"high edge detection latency ({edge_ms:.1f} ms)"
            main_loop_ms = float(getattr(self, "_last_main_loop_ms", 0.0) or 0.0)
            if main_loop_ms > 100.0:
                return f"main loop blocked ({main_loop_ms:.1f} ms)"
            return "camera frame rate or queue delay"
        return None

    def _sync_runtime_from_stage_context(self, reason: str = ""):
        ctx = self._ctx()
        stage = self._safe_stage_text(ctx.current_stage)
        mode = self._safe_mode_text(ctx.current_mode)
        prev = self._prev_synced or {}
        prev_stage = prev.get("stage", "")
        prev_mode = prev.get("mode", "")
        if prev_stage != stage or prev_mode != mode:
            self._prev_synced = {"stage": stage, "mode": mode}
            payload = {
                "reason": reason,
                "prev_stage": prev_stage,
                "stage": stage,
                "prev_mode": prev_mode,
                "mode": mode,
                "session_id": ctx.session_id,
                "req_id": ctx.req_id,
                "epoch": int(getattr(ctx, "epoch", 0) or 0),
            }
            if prev_stage != stage:
                self.operator_console.emit_change(
                    "stage",
                    f"[VISTA] STAGE {prev_stage} -> {stage} reason={reason}",
                )
            if prev_mode != mode:
                self.operator_console.emit_change(
                    "mode",
                    f"[VISTA] MODE {prev_mode} -> {mode} reason={reason}",
                )
                if mode == "FIND_OBJECT" and reason == "target_search":
                    self.operator_console.emit_change(
                        "target_view",
                        f"[VISTA] TARGET_VIEW enter target={ctx.target_name or 'target'}",
                    )
            if self._console_is_full():
                self.log_info("runtime", "stage/mode changed", payload)

    @staticmethod
    def _request_kind(req: VisionReq, request_stage: str) -> str:
        payload = req.payload if isinstance(req.payload, dict) else {}
        kind = (
            payload.get("search_kind")
            or payload.get("kind")
            or ("TABLE_EDGE" if request_stage == "SEARCH" else request_stage)
        )
        return str(kind or "").strip().upper()

    def _request_sync_reason(self, req: VisionReq, request_stage: str, req_kind: str) -> str:
        mode_hint = self._safe_mode_text(req.mode_hint)
        if request_stage == "SEARCH" and req_kind == "TARGET" and mode_hint == "FIND_OBJECT":
            return "target_search"
        return f"request:{req.op}"

    def _apply_stage_output(self, output, now: float, force_send: bool = False) -> bool:
        if output is None:
            return False
        if output.vision_obs is None:
            return False

        frame_meta = {}
        try:
            frame_meta = self.scheduler.read_result("frame_meta") or {}
        except Exception:
            pass

        self.obs_router.update_intervals(control_send_interval_s=self._control_send_interval_s())
        route_result = self.obs_router.route(
            vision_obs=output.vision_obs,
            frame_meta=frame_meta,
            now=now,
            force_send=force_send,
            freq_warning_reason=self._check_freq_and_reason(now) or "",
        )
        self._sync_observation_metrics_from_router()
        if route_result.skipped:
            self._obs_out_drop_or_skip_count += 1
            self._obs_out_skip_reason = route_result.skip_reason
            return False

        control_obs = route_result.control_obs
        if control_obs is None:
            return False

        if isinstance(control_obs, dict):
            perception = control_obs.get("perception") or {}
            edge_obs = perception.get("table_edge_obs")
            if isinstance(edge_obs, dict):
                # Log EDGE_OBS_PAYLOAD_FINAL using self.log_info
                payload_msg = (
                    f"[EDGE_OBS_PAYLOAD_FINAL]\n"
                    f"frame_id={edge_obs.get('frame_id')}\n"
                    f"edge_found={str(bool(edge_obs.get('edge_found', False))).lower()}\n"
                    f"edge_valid={str(bool(edge_obs.get('edge_valid', False))).lower()}\n"
                    f"edge_trusted={str(bool(edge_obs.get('edge_trusted', False))).lower()}\n"
                    f"point_count={edge_obs.get('point_count')}\n"
                    f"table_point_count={edge_obs.get('table_point_count')}\n"
                    f"reason={edge_obs.get('reason')}\n"
                    f"source={edge_obs.get('source')}\n"
                    f"obs_ts={edge_obs.get('obs_ts')}\n"
                    f"send_allowed={str(not route_result.skipped).lower()}\n"
                    f"force_send={str(force_send).lower()}"
                )
                self.log_info("runtime", payload_msg)

                # Check mapping mismatch against the debug_publish fields
                pub_found = bool(edge_obs.get("debug_publish_found", False))
                pub_valid = bool(edge_obs.get("debug_publish_valid", False))
                pub_trusted = bool(edge_obs.get("debug_publish_trusted", False))
                pub_point_count = int(edge_obs.get("debug_publish_point_count", 0) or 0)

                pay_found = bool(edge_obs.get("edge_found", False))
                pay_valid = bool(edge_obs.get("edge_valid", False))
                pay_trusted = bool(edge_obs.get("edge_trusted", False))
                pay_point_count = int(edge_obs.get("point_count", 0) or 0)

                mismatch = (
                    (pub_found and not pay_found) or
                    (pub_valid and not pay_valid) or
                    (pub_trusted and not pay_trusted)
                )
                if mismatch:
                    mismatch_msg = (
                        f"[EDGE_MAPPING_MISMATCH_FINAL]\n"
                        f"frame_id={edge_obs.get('frame_id')}\n"
                        f"publish_found={int(pub_found)}\n"
                        f"publish_valid={int(pub_valid)}\n"
                        f"publish_trusted={int(pub_trusted)}\n"
                        f"publish_point_count={pub_point_count}\n"
                        f"payload_found={int(pay_found)}\n"
                        f"payload_valid={int(pay_valid)}\n"
                        f"payload_trusted={int(pay_trusted)}\n"
                        f"payload_point_count={pay_point_count}\n"
                        f"payload_reason={edge_obs.get('reason')}"
                    )
                    self.log_warn("runtime", mismatch_msg)

        queued = self._send_obs(control_obs, sender=self.obs_sender, obs_class="control")
        if queued:
            self.last_send_ts = now
            self.obs_router.mark_control_sent(now)
            self._sync_observation_metrics_from_router()
            self._record_rate_sample(control_obs, now)
            self._emit_rate_summary_if_needed()
        else:
            self.obs_router.mark_drop()
            self._sync_observation_metrics_from_router()

        diagnostic_obs = route_result.diagnostic_obs
        if diagnostic_obs is not None:
            diag_queued = self._send_obs(diagnostic_obs, sender=self.diag_sender, obs_class="diagnostic")
            if diag_queued:
                self.obs_router.mark_diagnostic_sent(now)
                self._last_diag_send_ts = now
                self._sync_observation_metrics_from_router()
            else:
                self.obs_router.mark_drop()
                self._sync_observation_metrics_from_router()

        return queued

    def _sync_observation_metrics_from_router(self) -> None:
        metrics = self.obs_router.metrics
        self.obs_skip_count = int(metrics.obs_skip_count)
        self.obs_drop_count = int(metrics.obs_drop_count)
        self.obs_total_age_ms = float(metrics.obs_total_age_ms)
        self.same_frame_reuse_count = int(metrics.same_frame_reuse_count)
        self._last_processed_frame_id = metrics.last_processed_frame_id

    def _handle_stop_request(self, stage: str, stop_state=None):
        self._record_event("VISION_STOP", trigger="request:STOP", stage=stage)
        state = dict(stop_state or {})
        ctx = self._ctx()
        prev_mode = str(state.get("mode") or self._safe_mode_text(ctx.current_mode) or "IDLE").strip().upper()
        prev_target = state.get("target_name", ctx.target_name)
        stop_epoch = int(state.get("epoch", int(getattr(ctx, "epoch", 0) or 0)))
        keep_after_stop = bool(
            CONFIG.runtime.keep_preview_after_stop
            or getattr(CONFIG.runtime, "keep_vision_alive_after_task", True)
            or getattr(CONFIG.runtime, "keep_preview_alive_after_task", True)
        )
        if keep_after_stop and float(CONFIG.runtime.hot_standby_s) >= 0.0:
            self._enter_hot_standby(prev_mode, prev_target, stop_epoch)
        else:
            self.log_info("runtime", "enter idle", {"reason": stage})
            self._enter_cold_idle(stop_epoch)

    def _handle_request_payload(self, payload):
        typ = str(payload.get("type", "vision_req")).strip()
        if typ not in {"vision_req", "home_tag_req"}:
            return

        req = VisionReq.from_dict(payload)
        request_stage = self._safe_stage_text(req.stage)
        req_kind = self._request_kind(req, request_stage)
        self.operator_console.emit(
            f"[VISTA] REQ stage={request_stage} kind={req_kind} target={req.target or ''} "
            f"req={req.req_id or ''} epoch={int(req.epoch)}"
        )
        req_event_data = {
            "op": req.op,
            "mode_hint": req.mode_hint,
            "payload": req.payload,
            "legacy_type": req.legacy_type,
            "request_stage": request_stage,
        }

        if req.is_stop():
            stop_state = {
                "stage": self._safe_stage_text(self._ctx().current_stage),
                "mode": self._safe_mode_text(self._ctx().current_mode),
                "target_name": self._ctx().target_name,
                "epoch": (int(self._ctx().epoch or 0)),
            }
            stage_output = self.stage_controller.handle_request(req)
            self._sync_runtime_from_stage_context(reason=f"request:{req.op}")
            self._record_request_trace(req)
            self._record_event(
                "VISION_REQ",
                trigger="req_in",
                stage=self._safe_stage_text(self._ctx().current_stage),
                interaction_id=req.interaction_id,
                data=req_event_data,
            )
            if stage_output is not None and bool(stage_output.signal("mode_apply_failed", False)):
                self.log_warn(
                    "runtime",
                    "skip stop flow due to mode_apply_failed",
                    {
                        "stage": self._safe_stage_text(self._ctx().current_stage),
                        "mode": self._safe_mode_text(self._ctx().current_mode),
                        "req_id": req.req_id,
                    },
                )
                self._apply_stage_output(stage_output, now=time.time(), force_send=True)
                return
            self._handle_stop_request(request_stage, stop_state=stop_state)
            return

        self.hot_until_ts = 0.0
        stage_output = self.stage_controller.handle_request(req)
        sync_reason = self._request_sync_reason(req, request_stage, req_kind)
        self._sync_runtime_from_stage_context(reason=sync_reason)
        self._record_request_trace(req)
        if request_stage == "SEARCH" and req_kind == "TARGET" and self._safe_mode_text(self._ctx().current_mode) == "FIND_OBJECT":
            self.operator_console.emit_change(
                "target_view",
                f"[VISTA] TARGET_VIEW enter target={self._ctx().target_name or req.target or 'target'}",
            )
        self._record_event(
            "VISION_REQ",
            trigger="req_in",
            stage=self._safe_stage_text(self._ctx().current_stage),
            interaction_id=req.interaction_id,
            data=req_event_data,
        )
        obs_sent = self._apply_stage_output(
            stage_output,
            now=time.time(),
            force_send=self._should_force_send_stage_output(stage_output),
        )
        if not obs_sent:
            self.last_send_ts = 0.0

    def _tick_stage(self, now: float):
        plan = self.stage_controller.current_plan()
        mode = self._safe_mode_text(self._ctx().current_mode)
        route_filter = set(plan.subscribed_routes(mode)) if plan else None
        tick_input = self.scheduler.collect_tick_input(ts=now, route_filter=route_filter)
        tick_input.snapshot["app"] = {
            "stage": self._safe_stage_text(self._ctx().current_stage),
            "mode": self._safe_mode_text(self._ctx().current_mode),
            "session_id": self._ctx().session_id,
            "req_id": self._ctx().req_id,
            "epoch": (int(self._ctx().epoch or 0)),
            "hot_until_ts": self.hot_until_ts,
        }
        stage_output = self.stage_controller.tick(tick_input)
        self._sync_runtime_from_stage_context(reason="tick")
        self._apply_stage_output(stage_output, now=now, force_send=self._should_force_send_stage_output(stage_output))

    def _expire_hot_standby(self, now: float):
        if bool(
            getattr(CONFIG.runtime, "keep_vision_alive_after_task", True)
            or getattr(CONFIG.runtime, "keep_preview_alive_after_task", True)
        ):
            return
        if self._safe_mode_text(self._ctx().current_mode) != "IDLE_HOT" or self.hot_until_ts <= 0 or now < self.hot_until_ts:
            return
        self._enter_cold_idle(int(getattr(self._ctx(), "epoch", 0) or 0))

    def start(self):
        print(f"[VISTA_TABLE_EDGE_SCHEDULER_FIX_ACTIVE] version=20260626_table_edge_scheduler_route_v1 file={__file__}", flush=True)
        self.log_info("runtime", f"[VISTA_TABLE_EDGE_SCHEDULER_FIX_ACTIVE] version=20260626_table_edge_scheduler_route_v1 file={__file__}")
        print(f"[VISTA_EDGE_MAPPING_FIX_ACTIVE] version=20260626_edge_payload_path_v2 file={__file__}", flush=True)
        self.log_info("runtime", f"[VISTA_EDGE_MAPPING_FIX_ACTIVE] version=20260626_edge_payload_path_v2 file={__file__}")
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
        self.operator_console.emit(f"[VISTA] SERVICE_STARTING run={self.run_logger.stack_run_id}")
        if self._console_is_full():
            self.log_info("runtime", "structured logs ready", self.log_paths)
        req_mode = str(CONFIG.req_in.transport or "disabled").strip().lower()
        req_path = str(CONFIG.req_in.ipc_socket_path or "")
        req_desc = (
            f"mode=uds path={req_path}"
            if req_mode == "uds"
            else f"mode={req_mode}"
        )
        self.operator_console.emit(f"[VISTA] request server endpoint {req_desc}")
        self.req_server.start()
        self.operator_console.emit(
            f"[VISTA] request server listening mode={self.req_server.mode} ready={int(bool(self.req_server.listening))}"
        )
        self.mode_controller.start_runtime()
        # Activate INIT stage — non-blocking, task worker starts via mode plan
        init_req = VisionReq(ts=time.time(), op="START", stage="INIT", mode_hint="INIT")
        self.stage_controller.activate_stage("INIT", req=init_req)
        self._sync_runtime_from_stage_context(reason="service_start")
        self._running = True
        self._record_event("SERVICE_READY", trigger="start")
        self.operator_console.emit(
            f"[VISTA] READY mode=INIT request_server_ready={int(bool(self.req_server.listening))} run={self.run_logger.stack_run_id}"
        )
        if self._console_is_full():
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
        self._emit_system_metrics_if_needed(force=True)

    def stop(self):
        if self._stopped:
            return
        self._stopped = True
        self._running = False
        self._record_event("SERVICE_STOPPING", trigger="stop")
        self.req_server.close()
        self.obs_sender.close()
        self.diag_sender.close()
        self.mode_controller.stop_runtime()
        self._emit_system_metrics_if_needed(force=True)
        self._record_event("SERVICE_STOPPED", trigger="stop")
        self.operator_console.emit(f"[VISTA] SERVICE_STOPPED run={self.run_logger.stack_run_id}")
        if self._console_is_full():
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
                self._emit_system_metrics_if_needed()
                dt = time.time() - loop_start
                self._last_main_loop_ms = max(0.0, dt * 1000.0)
                if dt < target_frame_time:
                    time.sleep(target_frame_time - dt)

        except KeyboardInterrupt:
            self.log_info("runtime", "keyboard interrupt received")
        except Exception as exc:
            self._record_event("FATAL", level="error", trigger="main_loop", data={"error": str(exc)})
            self.log_error("runtime", f"vista main loop crashed: {exc}")
        finally:
            self.stop()

    def _emit_system_metrics_if_needed(self, force: bool = False) -> None:
        sample = self._system_metrics.sample_if_due(force=force)
        if sample is not None:
            self.run_logger.write_jsonl("system_metrics", sample)


def main():
    app = VistaApp()
    app.run()


if __name__ == "__main__":
    main()
