#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import threading
from typing import Any, Dict, List


def start_jog_stop_timer(
    *,
    seq: int,
    duration_ms: int,
    cancel_event: threading.Event,
    active_events: List[threading.Event],
    active_lock: threading.Lock,
    uart: Any,
    stop_line: str,
    tx_meta: Dict[str, Any],
    start_mono: float,
) -> None:
    def _jog_worker() -> None:
        try:
            cancelled = cancel_event.wait(timeout=float(duration_ms) / 1000.0)
            if cancelled:
                return
            if hasattr(uart, "_last_estop_mono") and uart._last_estop_mono >= start_mono:
                return
            if hasattr(uart, "_stop") and uart._stop.is_set():
                return
            uart.send_motion_line(
                stop_line,
                tx_meta=dict(tx_meta, kind="stm32_stop", pulse_stop=True),
                latest_override=False,
            )
        finally:
            with active_lock:
                if cancel_event in active_events:
                    active_events.remove(cancel_event)

    threading.Thread(target=_jog_worker, daemon=True, name=f"jog_worker_{seq}").start()
