#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from dataclasses import dataclass
import time
from typing import Any, Dict, List, Optional

from ..ipc.protocol import ArmResponse
from .arm_protocol import encode_grabbed, parse_arm_response_detail, pose_dict_from_command, pose_matches

try:
    import serial  # type: ignore
except Exception:
    serial = None  # type: ignore


@dataclass
class ArmSerialBridgeConfig:
    enabled: bool = True
    dry_run: bool = False
    port: str = "/dev/ttyUSB0"
    baudrate: int = 9600
    timeout_s: float = 0.1
    open_settle_s: float = 3.0
    # After open_settle_s, keep reading boot/status lines until the port is
    # quiet. This prevents late "PC echo mode" / "UART1 ... alive" messages
    # from being interpreted as the response to the next POSE command.
    boot_drain_max_s: float = 5.0
    boot_quiet_s: float = 0.5
    before_tx_drain_s: float = 0.3
    bytesize: int = 8
    parity: str = "N"
    stopbits: int = 1
    rtscts: bool = False
    dsrdtr: bool = False
    set_dtr: bool = False
    set_rts: bool = False
    readback_enabled: bool = True
    response_timeout_s: float = 10.0


class ArmSerialBridge:
    def __init__(self, cfg: Optional[Any] = None, logger=None):
        self.cfg = cfg or ArmSerialBridgeConfig()
        self.log = logger
        self._ser = None
        self._opened = False

    def _emit(self, level: str, event: str, **fields: Any) -> None:
        if self.log is None:
            return
        if callable(self.log):
            try:
                self.log(level, "arm_serial", f"{event} {fields}" if fields else event)
                return
            except Exception:
                pass
        fn = getattr(self.log, level, None)
        if callable(fn):
            fn(f"{event} {fields}" if fields else event)

    def open(self) -> bool:
        if not bool(getattr(self.cfg, "enabled", True)):
            self._emit("info", "arm_serial_disabled")
            return False
        if bool(getattr(self.cfg, "dry_run", False)):
            self._opened = True
            self._emit("info", "arm_serial_opened", dry_run=True, port=getattr(self.cfg, "port", ""))
            return True
        if self._ser is not None and self._opened:
            return True
        if serial is None:
            self._emit("error", "arm_serial_open_failed", error="pyserial_unavailable", port=getattr(self.cfg, "port", ""))
            return False

        try:
            ser = serial.Serial()
            ser.port = str(getattr(self.cfg, "port", "/dev/ttyUSB0") or "/dev/ttyUSB0")
            ser.baudrate = int(getattr(self.cfg, "baudrate", 9600) or 9600)
            ser.bytesize = int(getattr(self.cfg, "bytesize", 8) or 8)
            ser.parity = str(getattr(self.cfg, "parity", "N") or "N")
            ser.stopbits = int(getattr(self.cfg, "stopbits", 1) or 1)
            ser.timeout = float(getattr(self.cfg, "timeout_s", 0.1) or 0.1)
            ser.rtscts = bool(getattr(self.cfg, "rtscts", False))
            ser.dsrdtr = bool(getattr(self.cfg, "dsrdtr", False))
            if hasattr(ser, "exclusive"):
                try:
                    ser.exclusive = True
                except Exception as exc:
                    self._emit("warn", "arm_serial_exclusive_unavailable", error=str(exc))
            else:
                self._emit("warn", "arm_serial_exclusive_unavailable")

            ser.open()
            try:
                ser.setDTR(bool(getattr(self.cfg, "set_dtr", False)))
            except Exception:
                pass
            try:
                ser.setRTS(bool(getattr(self.cfg, "set_rts", False)))
            except Exception:
                pass

            self._ser = ser
            self._opened = True

            settle_s = max(0.0, float(getattr(self.cfg, "open_settle_s", 3.0) or 0.0))
            if settle_s > 0.0:
                time.sleep(settle_s)

            boot_discarded = self._drain_until_quiet(
                max_s=float(getattr(self.cfg, "boot_drain_max_s", 5.0) or 5.0),
                quiet_s=float(getattr(self.cfg, "boot_quiet_s", 0.5) or 0.5),
                phase="after_open_boot_drain",
            )
            self._clear_buffers("after_open_settle", discarded_lines=boot_discarded)
            self._emit(
                "info",
                "arm_serial_opened",
                port=ser.port,
                baudrate=ser.baudrate,
                open_settle_s=settle_s,
                boot_drain_max_s=float(getattr(self.cfg, "boot_drain_max_s", 5.0) or 5.0),
                boot_quiet_s=float(getattr(self.cfg, "boot_quiet_s", 0.5) or 0.5),
                dtr=bool(getattr(self.cfg, "set_dtr", False)),
                rts=bool(getattr(self.cfg, "set_rts", False)),
            )
            return True
        except Exception as exc:
            self._opened = False
            self._ser = None
            self._emit("error", "arm_serial_open_failed", error=str(exc), port=getattr(self.cfg, "port", ""))
            return False

    def _decode_raw_line(self, raw: Any) -> str:
        if isinstance(raw, bytes):
            return raw.decode("utf-8", errors="replace").strip()
        return str(raw or "").strip()

    def _readline_raw(self) -> str:
        ser = self._ser
        if ser is None:
            return ""
        raw = ser.readline()
        return self._decode_raw_line(raw)

    def _drain_until_quiet(self, *, max_s: float, quiet_s: float, phase: str) -> List[str]:
        """Read and discard boot/status lines until the port stays quiet.

        This is stronger than a fixed sleep. The arm firmware may print
        "PC echo mode" slightly after open_settle_s; if we send POSE before
        that line arrives, the following wait loop can confuse boot text with
        command response timing.
        """
        ser = self._ser
        if ser is None:
            return []
        lines: List[str] = []
        max_s = max(0.0, float(max_s or 0.0))
        quiet_s = max(0.0, float(quiet_s or 0.0))
        deadline = time.monotonic() + max_s
        last_activity = time.monotonic()
        read_error = ""

        while time.monotonic() <= deadline:
            try:
                line = self._readline_raw()
            except Exception as exc:
                read_error = str(exc)
                self._emit("warn", "arm_serial_drain_failed", phase=phase, error=read_error)
                break
            now = time.monotonic()
            if line:
                lines.append(line)
                last_activity = now
                continue
            if now - last_activity >= quiet_s:
                break

        self._emit(
            "info",
            "arm_serial_boot_drain_done" if phase == "after_open_boot_drain" else "arm_serial_drain_done",
            phase=phase,
            discarded_lines=list(lines),
            discarded_count=len(lines),
            quiet_s=quiet_s,
            max_s=max_s,
            read_error=read_error,
        )
        return lines

    def _clear_buffers(self, phase: str, discarded_lines: Optional[List[str]] = None) -> None:
        ser = self._ser
        if ser is None:
            return
        input_ok = False
        output_ok = False
        try:
            ser.reset_input_buffer()
            input_ok = True
        except Exception as exc:
            self._emit("warn", "arm_serial_buffer_clear_failed", phase=phase, buffer="input", error=str(exc))
        try:
            ser.reset_output_buffer()
            output_ok = True
        except Exception as exc:
            self._emit("warn", "arm_serial_buffer_clear_failed", phase=phase, buffer="output", error=str(exc))
        self._emit(
            "info",
            "arm_serial_buffers_cleared",
            phase=phase,
            discarded_lines=list(discarded_lines or []),
            input_ok=input_ok,
            output_ok=output_ok,
        )

    def _drain_pending_lines(self, duration_s: float = 0.3) -> List[str]:
        return self._drain_until_quiet(max_s=max(0.0, float(duration_s or 0.0)), quiet_s=max(0.05, min(0.3, float(duration_s or 0.0))), phase="before_tx_drain")

    def _prepare_before_tx(self) -> List[str]:
        drain_s = float(getattr(self.cfg, "before_tx_drain_s", 0.3) or 0.3)
        discarded = self._drain_pending_lines(duration_s=drain_s)
        self._clear_buffers("before_tx", discarded_lines=discarded)
        return discarded

    def close(self) -> None:
        ser = self._ser
        self._ser = None
        self._opened = False
        if ser is not None:
            try:
                ser.close()
            except Exception:
                pass

    def _write_line(self, line: str) -> Dict[str, Any]:
        line = str(line or "").rstrip("\r\n")
        if not line:
            return {"ok": False, "error": "empty_line", "bytes_written": 0, "flush_ok": False}
        if bool(getattr(self.cfg, "dry_run", False)):
            self._emit("info", "arm_serial_write_start", line=line, dry_run=True)
            self._emit("info", "arm_tx_line", line=line, dry_run=True)
            self._emit("info", "arm_serial_write_done", line=line, bytes_written=len(line) + 2, flush_ok=True, tx_ts=time.time(), dry_run=True)
            return {"ok": True, "error": "", "bytes_written": len(line) + 2, "flush_ok": True}
        if not self.open() or self._ser is None:
            return {"ok": False, "error": "arm_serial_open_failed", "bytes_written": 0, "flush_ok": False}
        try:
            payload = (line + "\r\n").encode("utf-8")
            self._emit("info", "arm_serial_write_start", line=line)
            written = self._ser.write(payload)
            self._ser.flush()
            self._emit("info", "arm_tx_line", line=line)
            self._emit("info", "arm_serial_write_done", line=line, bytes_written=int(written or 0), flush_ok=True, tx_ts=time.time())
            return {"ok": True, "error": "", "bytes_written": int(written or 0), "flush_ok": True}
        except Exception as exc:
            self._emit("error", "arm_serial_error", error=str(exc), line=line)
            return {"ok": False, "error": "arm_tx_failed", "bytes_written": 0, "flush_ok": False}

    def send_line(self, line: str) -> bool:
        return bool(self._write_line(line).get("ok", False))

    def read_line(self) -> str:
        if bool(getattr(self.cfg, "dry_run", False)):
            return ""
        if self._ser is None:
            return ""
        try:
            return self._readline_raw()
        except Exception as exc:
            self._emit("error", "arm_serial_error", error=str(exc))
            return ""

    def _failure_response(self, *, parsed_status: str, message: str, raw_line: str, sent_pose: Dict[str, Any], response_pose: Optional[Dict[str, Any]] = None) -> ArmResponse:
        resp = ArmResponse(ok=False, message=message, raw_line=raw_line, ts=time.time(), parsed_status=parsed_status)
        setattr(resp, "sent_pose", dict(sent_pose or {}))
        setattr(resp, "response_pose", dict(response_pose or {}))
        setattr(resp, "response_matches_sent", False)
        return resp

    @staticmethod
    def _command_failure_response(*, parsed_status: str, message: str, raw_line: str) -> ArmResponse:
        return ArmResponse(ok=False, message=message, raw_line=raw_line, ts=time.time(), parsed_status=parsed_status)

    def send_grabbed_and_wait(self, *, timeout_s: Optional[float] = None) -> Dict[str, Any]:
        line = encode_grabbed()
        if not bool(getattr(self.cfg, "enabled", True)):
            resp = self._command_failure_response(parsed_status="ARM_SERIAL_DISABLED", message="arm_serial_disabled", raw_line="arm_serial_disabled")
            return {"ok": False, "error": "arm_serial_disabled", "response": resp, "line": line}
        if bool(getattr(self.cfg, "dry_run", False)):
            write_result = self._write_line(line)
            received_lines = ["OK GRABBED START", "OK KEEP_CLAW dry_run", "OK GRABBED DONE"]
            for raw in received_lines:
                self._emit("info", "arm_rx_line", raw=raw, parsed_status=parse_arm_response_detail(raw).get("status"), dry_run=True)
            resp = ArmResponse(ok=True, message="OK_GRABBED_DONE", raw_line="OK GRABBED DONE", ts=time.time(), parsed_status="OK_GRABBED_DONE")
            return {"ok": True, "error": "", "response": resp, "line": line, "received_lines": received_lines, **write_result}
        if not self.open():
            resp = self._command_failure_response(parsed_status="ARM_SERIAL_OPEN_FAILED", message="arm_serial_open_failed", raw_line="arm_serial_open_failed")
            return {"ok": False, "error": "arm_serial_open_failed", "response": resp, "line": line}

        discarded_lines = self._prepare_before_tx()
        write_result = self._write_line(line)
        if not bool(write_result.get("ok", False)):
            resp = self._command_failure_response(parsed_status="ARM_TX_FAILED", message="arm_tx_failed", raw_line="arm_tx_failed")
            return {"ok": False, "error": "arm_tx_failed", "response": resp, "line": line, "discarded_lines": discarded_lines, **write_result}
        if not bool(getattr(self.cfg, "readback_enabled", True)):
            return {"ok": True, "error": "", "response": None, "line": line, "discarded_lines": discarded_lines, **write_result}

        deadline = time.time() + max(0.1, float(timeout_s if timeout_s is not None else getattr(self.cfg, "response_timeout_s", 10.0)))
        received_lines: List[str] = []
        while time.time() <= deadline:
            raw = self.read_line()
            if not raw:
                continue
            received_lines.append(raw)
            detail = parse_arm_response_detail(raw)
            status = str(detail.get("status") or "UNKNOWN")
            if status == "NOISE":
                self._emit("info", "arm_rx_noise_line", raw=raw)
            else:
                self._emit("info", "arm_rx_line", raw=raw, parsed_status=status)
            self._emit("info", "arm_response_parsed", status=status, raw=raw)

            if status in {"NOISE", "UNKNOWN", "OK_POSE", "ERR_IK"}:
                continue
            if status == "OK_GRABBED_START":
                self._emit("info", "arm_grabbed_started", raw=raw)
                continue
            if status == "OK_KEEP_CLAW":
                self._emit("info", "arm_grabbed_keep_claw", raw=raw)
                continue
            if status == "OK_GRABBED_DONE":
                resp = ArmResponse(ok=True, message="OK_GRABBED_DONE", raw_line=raw, ts=time.time(), parsed_status="OK_GRABBED_DONE")
                return {"ok": True, "error": "", "response": resp, "line": line, "received_lines": received_lines, **write_result}
            if status == "ERR_CMD":
                resp = self._command_failure_response(parsed_status="ERR_CMD", message="ERR_CMD", raw_line=raw)
                return {"ok": False, "error": "err_cmd", "response": resp, "line": line, "received_lines": received_lines, **write_result}

        self._emit(
            "error",
            "arm_response_parsed",
            status="GRABBED_TIMEOUT",
            raw="",
            received_lines_count=len(received_lines),
            last_lines=received_lines[-5:],
        )
        resp = self._command_failure_response(parsed_status="ARM_GRABBED_TIMEOUT", message="arm_grabbed_timeout", raw_line="arm_grabbed_timeout")
        return {
            "ok": False,
            "error": "arm_grabbed_timeout",
            "response": resp,
            "line": line,
            "received_lines": received_lines,
            "received_lines_count": len(received_lines),
            "last_lines": received_lines[-5:],
            **write_result,
        }

    def send_pose_and_wait(self, pose_line: str, *, timeout_s: Optional[float] = None) -> Dict[str, Any]:
        line = str(pose_line or "").rstrip("\r\n")
        sent_pose = pose_dict_from_command(line)
        if not bool(getattr(self.cfg, "enabled", True)):
            resp = self._failure_response(parsed_status="ARM_SERIAL_DISABLED", message="arm_serial_disabled", raw_line="arm_serial_disabled", sent_pose=sent_pose)
            return {"ok": False, "error": "arm_serial_disabled", "response": resp, "line": line, "sent_pose": sent_pose}
        if bool(getattr(self.cfg, "dry_run", False)):
            write_result = self._write_line(line)
            raw = (
                "OK POSE "
                f"x={sent_pose.get('x', 0)} y={sent_pose.get('y', 0)} z={sent_pose.get('z', 0)} "
                f"pitch={sent_pose.get('pitch', 0)} roll={sent_pose.get('roll', 0)} "
                f"claw={sent_pose.get('claw', 0)} t={sent_pose.get('time_ms', 0)}"
            )
            self._emit("info", "arm_rx_line", raw=raw, parsed_status="OK_POSE", dry_run=True)
            self._emit("info", "arm_response_parsed", status="OK_POSE", raw=raw, pose=sent_pose, dry_run=True)
            self._emit("info", "arm_response_pose_matched", sent_pose=sent_pose, response_pose=sent_pose, raw_response=raw, dry_run=True)
            resp = ArmResponse(ok=True, message="OK_POSE", raw_line=raw, ts=time.time(), parsed_status="OK_POSE")
            setattr(resp, "sent_pose", dict(sent_pose))
            setattr(resp, "response_pose", dict(sent_pose))
            setattr(resp, "response_matches_sent", True)
            return {"ok": True, "error": "", "response": resp, "line": line, "sent_pose": sent_pose, **write_result}
        if not self.open():
            resp = self._failure_response(parsed_status="ARM_SERIAL_OPEN_FAILED", message="arm_serial_open_failed", raw_line="arm_serial_open_failed", sent_pose=sent_pose)
            return {"ok": False, "error": "arm_serial_open_failed", "response": resp, "line": line, "sent_pose": sent_pose}

        discarded_lines = self._prepare_before_tx()
        write_result = self._write_line(line)
        if not bool(write_result.get("ok", False)):
            resp = self._failure_response(parsed_status="ARM_TX_FAILED", message="arm_tx_failed", raw_line="arm_tx_failed", sent_pose=sent_pose)
            return {"ok": False, "error": "arm_tx_failed", "response": resp, "line": line, "sent_pose": sent_pose, "discarded_lines": discarded_lines, **write_result}
        if not bool(getattr(self.cfg, "readback_enabled", True)):
            return {"ok": True, "error": "", "response": None, "line": line, "sent_pose": sent_pose, "discarded_lines": discarded_lines, **write_result}

        deadline = time.time() + max(0.1, float(timeout_s if timeout_s is not None else getattr(self.cfg, "response_timeout_s", 10.0)))
        received_lines: List[str] = []
        mismatch_lines: List[Dict[str, Any]] = []
        while time.time() <= deadline:
            raw = self.read_line()
            if not raw:
                continue
            received_lines.append(raw)
            detail = parse_arm_response_detail(raw)
            status = str(detail.get("status") or "UNKNOWN")
            pose = detail.get("pose") if isinstance(detail.get("pose"), dict) else {}
            if status == "NOISE":
                self._emit("info", "arm_rx_noise_line", raw=raw)
            else:
                self._emit("info", "arm_rx_line", raw=raw, parsed_status=status)
            self._emit("info", "arm_response_parsed", status=status, raw=raw, pose=pose)

            if status in {"NOISE", "UNKNOWN"}:
                continue
            if status == "ERR_IK":
                resp = self._failure_response(parsed_status="ERR_IK", message="ERR_IK", raw_line=raw, sent_pose=sent_pose, response_pose=pose)
                return {"ok": False, "error": "err_ik", "response": resp, "line": line, "sent_pose": sent_pose, "received_lines": received_lines, "mismatch_lines": mismatch_lines, **write_result}
            if status == "ERR_CMD":
                resp = self._failure_response(parsed_status="ERR_CMD", message="ERR_CMD", raw_line=raw, sent_pose=sent_pose, response_pose=pose)
                return {"ok": False, "error": "err_cmd", "response": resp, "line": line, "sent_pose": sent_pose, "received_lines": received_lines, "mismatch_lines": mismatch_lines, **write_result}
            if status == "OK_POSE":
                if pose_matches(sent_pose, pose):
                    self._emit("info", "arm_response_pose_matched", sent_pose=sent_pose, response_pose=pose, raw_response=raw)
                    resp = ArmResponse(ok=True, message="OK_POSE", raw_line=raw, ts=time.time(), parsed_status="OK_POSE")
                    setattr(resp, "sent_pose", dict(sent_pose))
                    setattr(resp, "response_pose", dict(pose))
                    setattr(resp, "response_matches_sent", True)
                    return {"ok": True, "error": "", "response": resp, "line": line, "sent_pose": sent_pose, "response_pose": pose, "received_lines": received_lines, "mismatch_lines": mismatch_lines, **write_result}
                mismatch = {"raw_response": raw, "response_pose": dict(pose), "action": "ignore_and_continue_waiting"}
                mismatch_lines.append(mismatch)
                self._emit("warn", "arm_response_pose_mismatch", sent_pose=sent_pose, response_pose=pose, raw_response=raw, action="ignore_and_continue_waiting")
                continue

        self._emit(
            "error",
            "arm_response_parsed",
            status="TIMEOUT",
            raw="",
            received_lines_count=len(received_lines),
            mismatch_count=len(mismatch_lines),
            last_lines=received_lines[-5:],
        )
        resp = self._failure_response(parsed_status="ARM_RESPONSE_TIMEOUT", message="arm_response_timeout", raw_line="arm_response_timeout", sent_pose=sent_pose)
        return {
            "ok": False,
            "error": "arm_response_timeout",
            "response": resp,
            "line": line,
            "sent_pose": sent_pose,
            "received_lines": received_lines,
            "mismatch_lines": mismatch_lines,
            "received_lines_count": len(received_lines),
            "mismatch_count": len(mismatch_lines),
            "last_lines": received_lines[-5:],
            **write_result,
        }
