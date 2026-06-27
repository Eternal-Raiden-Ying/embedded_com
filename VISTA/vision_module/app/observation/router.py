#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import time
from dataclasses import dataclass
from typing import Any, Dict, Optional

from .metrics import ObservationMetrics
from .schema import CONTROL_PERCEPTION_KEYS, DIAGNOSTIC_EXCLUDED_PERCEPTION_KEYS


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
        self.metrics.mark_frame(frame_id)

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
            key: dict(perception[key])
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
                obs["camera_frame_seq"] = frame_id
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
        frame_id = frame_meta.get("frame_seq") or frame_meta.get("camera_frame_seq")
        capture_ts = frame_meta.get("frame_capture_ts")
        if capture_ts is None:
            capture_ts_ms = frame_meta.get("camera_frame_ts_ms")
            if capture_ts_ms is not None:
                capture_ts = float(capture_ts_ms) / 1000.0

        perception = vision_obs.get("perception") or {}
        if isinstance(perception, dict):
            if frame_id is None:
                for obs_key in CONTROL_PERCEPTION_KEYS:
                    obs = perception.get(obs_key)
                    if isinstance(obs, dict):
                        frame_id = obs.get("frame_id") or obs.get("seq") or obs.get("camera_frame_seq")
                        if frame_id is not None:
                            break
            if capture_ts is None:
                for obs_key in CONTROL_PERCEPTION_KEYS:
                    obs = perception.get(obs_key)
                    if isinstance(obs, dict):
                        obs_ts = obs.get("obs_ts") or obs.get("ts")
                        if obs_ts is not None:
                            capture_ts = float(obs_ts)
                            break

        if frame_id is None:
            frame_id = 0
        if capture_ts is None:
            capture_ts = now
        return frame_id, float(capture_ts)

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
        payload["frame_id"] = frame_id
        payload["capture_ts"] = capture_ts
        payload["process_done_ts"] = process_done_ts
        payload["send_ts"] = send_ts
        payload["process_latency_ms"] = process_latency_ms
        payload["send_latency_ms"] = send_latency_ms
        payload["obs_total_age_ms"] = obs_total_age_ms
