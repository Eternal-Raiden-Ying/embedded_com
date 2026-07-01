#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
from pathlib import Path
from types import SimpleNamespace


ROOT = Path(__file__).resolve().parents[2]
ORCH_ROOT = ROOT / "orchestrator"
for path in (ROOT, ORCH_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from orchestrator_service.bridge.arm_serial_bridge import ArmSerialBridge


class _FakeSerial:
    def __init__(self, lines):
        self.lines = list(lines)
        self.writes = []
        self.input_reset_count = 0
        self.output_reset_count = 0
        self.flush_count = 0

    def readline(self):
        if not self.lines:
            return b""
        return (self.lines.pop(0) + "\n").encode("utf-8")

    def reset_input_buffer(self):
        self.input_reset_count += 1

    def reset_output_buffer(self):
        self.output_reset_count += 1

    def write(self, payload):
        self.writes.append(payload)
        return len(payload)

    def flush(self):
        self.flush_count += 1


def test_send_pose_ignores_before_tx_lines_and_mismatched_ok():
    logs = []
    cfg = SimpleNamespace(enabled=True, dry_run=False, readback_enabled=True, response_timeout_s=0.5)
    bridge = ArmSerialBridge(cfg, logger=lambda level, channel, message: logs.append((level, channel, message)))
    fake = _FakeSerial(
        [
            "OK POSE x=10.00 y=0.00 z=10.00 pitch=-45.00 roll=0.00 claw=90.00 t=800",
            "UART1 direct alive",
            "OK POSE x=24.00 y=1.00 z=9.00 pitch=8.00 roll=83.00 claw=90.00 t=800",
        ]
    )
    bridge._ser = fake
    bridge._opened = True
    bridge._drain_pending_lines = lambda duration_s=0.2: [
        "UART1 printf alive",
        "OK POSE x=10.00 y=0.00 z=10.00 pitch=-45.00 roll=0.00 claw=90.00 t=800",
    ]

    result = bridge.send_pose_and_wait("POSE 24 1 9 8 83 90 800", timeout_s=0.5)

    assert result["ok"] is True
    assert result["response"].parsed_status == "OK_POSE"
    assert result["response_pose"] == {"x": 24, "y": 1, "z": 9, "pitch": 8, "roll": 83, "claw": 90, "time_ms": 800}
    assert result["mismatch_lines"][0]["response_pose"]["x"] == 10
    assert fake.writes == [b"POSE 24 1 9 8 83 90 800\r\n"]
    assert fake.flush_count == 1
    assert fake.input_reset_count == 1
    assert fake.output_reset_count == 1
    joined_logs = "\n".join(message for _, _, message in logs)
    assert "arm_serial_buffers_cleared" in joined_logs
    assert "arm_serial_write_done" in joined_logs
    assert "arm_response_pose_mismatch" in joined_logs
    assert "arm_response_pose_matched" in joined_logs
