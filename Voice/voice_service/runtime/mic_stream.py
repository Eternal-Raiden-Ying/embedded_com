#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import subprocess
import time
from typing import List, Optional

from .common import jlog, FRAME_BYTES


class RawMicStream:
    def __init__(self, device: str, sr: int, channels: int = 1,
                frame_bytes: int = FRAME_BYTES,
                read_timeout_sec: float = 2.0,
                startup_delay_sec: float = 0.15,
                mic_debug: bool = False,
                mic_debug_every: int = 50):
        self.device = device
        self.sr = sr
        self.channels = channels
        self.frame_bytes = frame_bytes
        self.read_timeout_sec = read_timeout_sec
        self.startup_delay_sec = startup_delay_sec
        self.mic_debug = mic_debug
        self.mic_debug_every = max(1, int(mic_debug_every))
        self.proc: Optional[subprocess.Popen] = None
        self.restart_count = 0
        self.frames_ok = 0
        self.partial_events = 0
        self.timeout_events = 0
        self.eof_events = 0
        self.last_chunk_sizes: List[int] = []
        self.last_restart_reason = ""
        self.start()

    def start(self):
        self.close()
        cmd = [
            "arecord", "-D", self.device,
            "-q", "-t", "raw",
            "-f", "S16_LE", "-r", str(self.sr), "-c", str(self.channels)
        ]
        self.proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, bufsize=0)
        self.restart_count += 1
        if self.startup_delay_sec > 0:
            time.sleep(self.startup_delay_sec)
        jlog({"level": "info", "src": "mic", "msg": "arecord started", "device": self.device, "restart": self.restart_count})

    def _read_stderr_nonblocking(self) -> str:
        if self.proc is None or self.proc.stderr is None:
            return ""
        try:
            import fcntl
            fd = self.proc.stderr.fileno()
            flags = fcntl.fcntl(fd, fcntl.F_GETFL)
            fcntl.fcntl(fd, fcntl.F_SETFL, flags | os.O_NONBLOCK)
            data = self.proc.stderr.read()
            if not data:
                return ""
            return data.decode("utf-8", errors="ignore")
        except Exception:
            return ""

    def _restart(self, reason: str, got_bytes: int = 0, extra: Optional[dict] = None):
        self.last_restart_reason = reason
        err = self._read_stderr_nonblocking().strip()
        code = None if self.proc is None else self.proc.poll()
        payload = {
            "level": "warn", "src": "mic", "msg": "arecord restarting",
            "reason": reason, "got_bytes": got_bytes, "frame_bytes": self.frame_bytes,
            "returncode": code, "stderr": err[:400], "last_chunk_sizes": self.last_chunk_sizes[-8:],
        }
        if extra:
            payload.update(extra)
        jlog(payload)
        time.sleep(0.2)
        self.start()

    def _read_exact(self, nbytes: int) -> Optional[bytes]:
        if self.proc is None:
            self.start()
        assert self.proc is not None
        if self.proc.stdout is None:
            raise RuntimeError("arecord stdout is None")
        import select
        fd = self.proc.stdout.fileno()
        buf = bytearray()
        chunks: List[int] = []
        t0 = time.monotonic()
        while len(buf) < nbytes:
            remain_timeout = max(0.0, self.read_timeout_sec - (time.monotonic() - t0))
            if remain_timeout <= 0:
                self.timeout_events += 1
                self.last_chunk_sizes = chunks[-16:]
                self._restart("timeout_wait_full_frame", got_bytes=len(buf), extra={"elapsed_ms": int((time.monotonic()-t0)*1000)})
                return None
            rlist, _, _ = select.select([fd], [], [], remain_timeout)
            if not rlist:
                self.timeout_events += 1
                self.last_chunk_sizes = chunks[-16:]
                self._restart("select_timeout_wait_data", got_bytes=len(buf), extra={"elapsed_ms": int((time.monotonic()-t0)*1000)})
                return None
            chunk = os.read(fd, nbytes - len(buf))
            if not chunk:
                self.eof_events += 1
                self.last_chunk_sizes = chunks[-16:]
                self._restart("stdout_eof", got_bytes=len(buf), extra={"elapsed_ms": int((time.monotonic()-t0)*1000)})
                return None
            buf.extend(chunk)
            chunks.append(len(chunk))
            if len(chunk) < (nbytes - len(buf) + len(chunk)):
                self.partial_events += 1
            if self.mic_debug and ((self.frames_ok + 1) % self.mic_debug_every == 0 or len(chunks) > 1):
                jlog({"level": "debug", "src": "mic", "msg": "chunked frame read", "chunks": chunks[-8:], "sum_bytes": len(buf), "target_bytes": nbytes})
        self.last_chunk_sizes = chunks[-16:]
        self.frames_ok += 1
        return bytes(buf)

    def read_frame(self) -> Optional[bytes]:
        return self._read_exact(self.frame_bytes)

    def stats(self) -> dict:
        return {
            "restarts": self.restart_count,
            "frames_ok": self.frames_ok,
            "partial_events": self.partial_events,
            "timeout_events": self.timeout_events,
            "eof_events": self.eof_events,
            "last_restart_reason": self.last_restart_reason,
            "last_chunk_sizes": self.last_chunk_sizes[-8:],
        }

    def close(self):
        if self.proc is not None:
            try:
                self.proc.terminate()
            except Exception:
                pass
            try:
                self.proc.wait(timeout=1.0)
            except Exception:
                try:
                    self.proc.kill()
                except Exception:
                    pass
        self.proc = None
