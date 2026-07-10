#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import logging
import time
from dataclasses import dataclass
from typing import Any, Dict, Optional

from .metrics import ObservationMetrics
from .schema import CONTROL_PERCEPTION_KEYS, DIAGNOSTIC_EXCLUDED_PERCEPTION_KEYS


_LOG = logging.getLogger("vision.observation_router")


@dataclass
class ObservationRouteResult:
    control_obs: Optional[Dict[str, Any]]
    diagnostic_obs: Optional[Dict[str, Any]]
    skipped: bool = False
    skip_reason: str = ""


class ObservationRouter:
    def __init__(
        self,
        *,
        metrics: Optional[ObservationMetrics] = None,
        control_send_interval_s: float = 0.10,
        diagnostic_send_interval_s: float = 1.0,
    ):
        self.metrics = metrics or ObservationMetrics()
        self.control_send_interval_s = float(control_send_interval_s)
        self.diagnostic_send_interval_s = float(diagnostic_send_interval_s)
        self._last_control_identity = None
        self._pending_control_identity = None

    def update_intervals(self, *, control_send_interval_s: float, diagnostic_send_interval_s: float = 1.0) -> None:
        self.control_send_interval_s = float(control_send_interval_s)
        self.diagnostic_send_interval_s = float(diagnostic_send_interval_s)

    def route(
        self,
        *,
        vision_obs: Dict[str, Any],
        frame_meta: Optional[Dict[str, Any]],
        now: float,
        force_send: bool = False,
        freq_warning_reason: str = "",
    ) -> ObservationRouteResult:
        if not isinstance(vision_obs, dict):
            return ObservationRouteResult(control_obs=None, diagnostic_obs=None, skipped=True, skip_reason="empty_obs")

        elapsed = float(now) - self.metrics.last_control_send_ts
        if not force_send and elapsed + 1e-9 < self.control_send_interval_s:
            self.metrics.mark_skip()
            return ObservationRouteResult(
                control_obs=None,
                diagnostic_obs=None,
                skipped=True,
                skip_reason="send_hz_limit",
            )

        frame_id, capture_ts = self._frame_identity(vision_obs, frame_meta or {}, now)
        control_identity = self._control_identity(vision_obs, frame_id)
        if not force_send and control_identity is not None and control_identity == self._last_control_identity:
            self.metrics.mark_skip()
            return ObservationRouteResult(control_obs=None, diagnostic_obs=None, skipped=True, skip_reason="same_observation")
        self.metrics.mark_frame(frame_id)
        self._pending_control_identity = control_identity

        process_done_ts = float(now)
        send_ts = time.time()
        process_latency_ms = max(0.0, (process_done_ts - capture_ts) * 1000.0)
        send_latency_ms = max(0.0, (send_ts - process_done_ts) * 1000.0)
        obs_total_age_ms = max(0.0, (send_ts - capture_ts) * 1000.0)
        self.metrics.obs_total_age_ms = obs_total_age_ms

        perception = vision_obs.get("perception") or {}
        if not isinstance(perception, dict):
            perception = {}

        control_obs = self._base_obs(vision_obs, now, obs_class="control")
        control_obs["perception"] = {
            key: self._slim_control_perception(key, perception[key])
            for key in CONTROL_PERCEPTION_KEYS
            if isinstance(perception.get(key), dict)
        }
        self._inject_latency(
            control_obs,
            frame_id=frame_id,
            capture_ts=capture_ts,
            process_done_ts=process_done_ts,
            send_ts=send_ts,
            process_latency_ms=process_latency_ms,
            send_latency_ms=send_latency_ms,
            obs_total_age_ms=obs_total_age_ms,
        )

        metrics_snapshot = self.metrics.snapshot(float(now), freq_warning_reason=freq_warning_reason)
        control_obs["metrics"] = metrics_snapshot
        self._preserve_grasp_result_for_control(control_obs, vision_obs)
        for key in CONTROL_PERCEPTION_KEYS:
            obs = control_obs["perception"].get(key)
            if isinstance(obs, dict):
                self._inject_latency(
                    obs,
                    frame_id=frame_id,
                    capture_ts=capture_ts,
                    process_done_ts=process_done_ts,
                    send_ts=send_ts,
                    process_latency_ms=process_latency_ms,
                    send_latency_ms=send_latency_ms,
                    obs_total_age_ms=obs_total_age_ms,
                )
                obs.setdefault("camera_frame_seq", frame_id)
                obs["camera_frame_ts_ms"] = int(round(capture_ts * 1000.0))
                obs["vision_process_end_ts_ms"] = int(round(process_done_ts * 1000.0))
                obs["obs_out_send_ts_ms"] = int(round(send_ts * 1000.0))

        diagnostic_obs = None
        if float(now) - self.metrics.last_diag_send_ts >= self.diagnostic_send_interval_s:
            diagnostic_obs = self._base_obs(vision_obs, now, obs_class="diagnostic")
            diagnostic_obs["perception"] = {
                key: value
                for key, value in perception.items()
                if key not in DIAGNOSTIC_EXCLUDED_PERCEPTION_KEYS
            }
            diagnostic_obs["proposal"] = vision_obs.get("proposal")
            diagnostic_obs["result"] = vision_obs.get("result")
            diagnostic_obs["metrics"] = metrics_snapshot

        return ObservationRouteResult(control_obs=control_obs, diagnostic_obs=diagnostic_obs)

    def mark_control_sent(self, now: float) -> None:
        self.metrics.mark_control_sent(now)
        self._last_control_identity = self._pending_control_identity

    @staticmethod
    def _control_identity(vision_obs: Dict[str, Any], fallback_frame_id: Any):
        perception = vision_obs.get("perception") or {}
        identities = []
        if isinstance(perception, dict):
            for key in CONTROL_PERCEPTION_KEYS:
                obs = perception.get(key)
                if not isinstance(obs, dict):
                    continue
                # Runtime producers attach obs_seq to every completed table/target
                # observation.  Legacy payloads without it follow frame_meta so
                # old protocol senders are not accidentally suppressed.
                identity = obs.get("obs_seq")
                if identity is not None:
                    identities.append((key, identity))
        return tuple(identities) if identities else None

    @staticmethod
    def _slim_control_perception(key: str, payload: Dict[str, Any]) -> Dict[str, Any]:
        out = dict(payload or {})
        for field in (
            "edge_profile",
            "profile",
            "fast_debug_pixels",
            "debug_points",
            "candidate_points",
            "all_candidate_classes",
            "target_candidates",
            "candidates",
            "preview",
            "preview_data",
            "preview_roi_draw",
        ):
            out.pop(field, None)
        for field in (
            "vision_process_start_ts_ms",
            "vision_process_end_ts_ms",
            "vision_publish_ts_ms",
            "obs_out_send_ts_ms",
            "camera_frame_ts_ms",
            "process_done_ts",
            "yolo_preprocess_ms",
            "yolo_postprocess_ms",
            "yolo_roi_ms",
        ):
            out.pop(field, None)
        if key == "table_edge_obs":
            for alias in ("seq", "yaw_err", "dist_err", "table_bbox", "yolo_table_bbox"):
                out.pop(alias, None)
        elif key == "target_obs" and out.get("matched_bbox") is not None:
            out.pop("bbox", None)
        return out

    def mark_diagnostic_sent(self, now: float) -> None:
        self.metrics.mark_diag_sent(now)

    def mark_drop(self) -> None:
        self.metrics.mark_drop()

    @staticmethod
    def _base_obs(vision_obs: Dict[str, Any], now: float, *, obs_class: str) -> Dict[str, Any]:
        return {
            "type": "vision_obs",
            "ts": vision_obs.get("ts", now),
            "stage": vision_obs.get("stage"),
            "mode": vision_obs.get("mode"),
            "status": vision_obs.get("status"),
            "session_id": vision_obs.get("session_id"),
            "req_id": vision_obs.get("req_id"),
            "epoch": vision_obs.get("epoch"),
            "interaction": vision_obs.get("interaction"),
            "obs_class": obs_class,
            "perception": {},
        }

    @staticmethod
    def _frame_identity(vision_obs: Dict[str, Any], frame_meta: Dict[str, Any], now: float) -> tuple:
        frame_id = None
        capture_ts = None

        perception = vision_obs.get("perception") or {}
        if isinstance(perception, dict):
            for obs_key in CONTROL_PERCEPTION_KEYS:
                obs = perception.get(obs_key)
                if not isinstance(obs, dict):
                    continue
                if frame_id is None:
                    frame_id = obs.get("source_frame_id") or obs.get("frame_id") or obs.get("camera_frame_seq") or obs.get("seq")
                if capture_ts is None:
                    value = obs.get("frame_capture_ts") or obs.get("capture_ts") or obs.get("obs_ts") or obs.get("ts")
                    if value is not None:
                        capture_ts = float(value)
                if frame_id is not None and capture_ts is not None:
                    break

        if frame_id is None:
            frame_id = frame_meta.get("frame_id") or frame_meta.get("frame_seq") or frame_meta.get("camera_frame_seq")
        if capture_ts is None:
            capture_ts = frame_meta.get("frame_capture_ts")
        if capture_ts is None:
            capture_ts_ms = frame_meta.get("camera_frame_ts_ms")
            if capture_ts_ms is not None:
                capture_ts = float(capture_ts_ms) / 1000.0

        if frame_id is None:
            frame_id = 0
        if capture_ts is None:
            capture_ts = now
        return frame_id, float(capture_ts)

    def _preserve_grasp_result_for_control(self, control_obs: Dict[str, Any], vision_obs: Dict[str, Any]) -> None:
        stage = str((vision_obs or {}).get("stage") or "").strip().upper()
        status = str((vision_obs or {}).get("status") or "").strip().upper()
        if stage != "GRASP" or status not in {"RESULT_READY", "FAILED"}:
            return

        result = vision_obs.get("result")
        if not isinstance(result, dict):
            _LOG.warning(
                "grasp_result_missing_in_control_obs | stage=%s status=%s has_result=%s",
                stage,
                status,
                result is not None,
            )
            return

        control_obs["result"] = result
        grasp = result.get("grasp") if isinstance(result.get("grasp"), dict) else None
        _LOG.info(
            "grasp_result_preserved_in_control_obs | stage=%s status=%s has_result=%s result_keys=%s has_grasp=%s grasp_keys=%s",
            stage,
            status,
            True,
            sorted(str(key) for key in result.keys()),
            grasp is not None,
            sorted(str(key) for key in grasp.keys()) if isinstance(grasp, dict) else [],
        )

    @staticmethod
    def _inject_latency(
        payload: Dict[str, Any],
        *,
        frame_id: object,
        capture_ts: float,
        process_done_ts: float,
        send_ts: float,
        process_latency_ms: float,
        send_latency_ms: float,
        obs_total_age_ms: float,
    ) -> None:
        payload.setdefault("frame_id", frame_id)
        payload["capture_ts"] = capture_ts
        payload["process_done_ts"] = process_done_ts
        payload["send_ts"] = send_ts
        payload["send_mono_ns"] = time.monotonic_ns()
        payload["process_latency_ms"] = process_latency_ms
        payload["send_latency_ms"] = send_latency_ms
        payload["obs_total_age_ms"] = obs_total_age_ms
