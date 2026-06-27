#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from collections import deque
from dataclasses import dataclass, field
from typing import Deque, Optional


@dataclass
class ObservationMetrics:
    obs_skip_count: int = 0
    obs_drop_count: int = 0
    same_frame_reuse_count: int = 0
    obs_total_age_ms: float = 0.0
    last_control_send_ts: float = 0.0
    last_diag_send_ts: float = 0.0
    last_processed_frame_id: Optional[object] = None
    control_send_samples: Deque[float] = field(default_factory=lambda: deque(maxlen=256))
    diag_send_samples: Deque[float] = field(default_factory=lambda: deque(maxlen=256))

    def mark_skip(self) -> None:
        self.obs_skip_count += 1

    def mark_drop(self) -> None:
        self.obs_drop_count += 1

    def mark_frame(self, frame_id: object) -> None:
        if frame_id == self.last_processed_frame_id:
            self.same_frame_reuse_count += 1
        else:
            self.last_processed_frame_id = frame_id

    def mark_control_sent(self, now: float) -> None:
        self.last_control_send_ts = float(now)
        self.control_send_samples.append(float(now))

    def mark_diag_sent(self, now: float) -> None:
        self.last_diag_send_ts = float(now)
        self.diag_send_samples.append(float(now))

    @staticmethod
    def hz_for_samples(samples: Deque[float], now: float) -> float:
        if len(samples) < 2:
            return 0.0
        window = max(1e-6, float(now) - float(samples[0]))
        return max(0.0, float(len(samples) - 1) / window)

    def snapshot(self, now: float, freq_warning_reason: str = "") -> dict:
        return {
            "obs_skip_count": int(self.obs_skip_count),
            "obs_drop_count": int(self.obs_drop_count),
            "same_frame_reuse_count": int(self.same_frame_reuse_count),
            "control_obs_hz": float(self.hz_for_samples(self.control_send_samples, now)),
            "diag_obs_hz": float(self.hz_for_samples(self.diag_send_samples, now)),
            "obs_total_age_ms": float(self.obs_total_age_ms),
            "freq_warning_reason": str(freq_warning_reason or ""),
        }
