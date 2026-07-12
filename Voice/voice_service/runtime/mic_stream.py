#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import errno
import subprocess
import time
import wave
from pathlib import Path
from typing import List, Optional

from .common import jlog, FRAME_BYTES


class WavReplayAudioSource:
    """Deterministic PCM source with the same read_frame contract as arecord."""
    def __init__(self, manifest_path: str, frame_bytes: int = FRAME_BYTES, realtime: bool = True, repeat: int = 1):
        if not manifest_path:
            raise ValueError("wav_replay requires replay_manifest")
        self.manifest_path = Path(manifest_path)
        if not self.manifest_path.is_file():
            raise FileNotFoundError("replay manifest not found: {}".format(self.manifest_path))
        self.frame_bytes = int(frame_bytes)
        self.realtime = bool(realtime)
        self.repeat = max(1, int(repeat))
        self.frames = self._load_frames()
        self.index = 0
        self.completed = False
        self.started_mono = None
        self.max_schedule_lag_ms = 0.0
        self.scenario = self.manifest_path.stem

    def _load_frames(self) -> List[bytes]:
        try:
            import yaml
        except ImportError as exc:
            raise RuntimeError("PyYAML is required for replay manifest: {}".format(exc))
        data = yaml.safe_load(self.manifest_path.read_text(encoding="utf-8")) or {}
        if not isinstance(data, dict) or not isinstance(data.get("steps"), list):
            raise ValueError("replay manifest requires a steps list")
        frames: List[bytes] = []
        root = Path(__file__).resolve().parents[3]
        silence = b"\x00" * self.frame_bytes
        for step in data["steps"]:
            if not isinstance(step, dict):
                raise ValueError("replay step must be a mapping")
            if "silence_s" in step:
                count = int(round(float(step["silence_s"]) * 16000 * 2 / self.frame_bytes))
                frames.extend([silence] * max(0, count))
                continue
            rel = step.get("wav")
            if not rel:
                raise ValueError("replay step requires silence_s or wav")
            path = Path(str(rel))
            if not path.is_absolute():
                path = root / path
            if not path.is_file():
                raise FileNotFoundError("replay WAV not found: {}".format(path))
            with wave.open(str(path), "rb") as wav:
                if wav.getframerate() != 16000 or wav.getnchannels() != 1 or wav.getsampwidth() != 2:
                    raise ValueError("replay WAV must be 16kHz mono S16_LE: {}".format(path))
                data_bytes = wav.readframes(wav.getnframes())
            for offset in range(0, len(data_bytes), self.frame_bytes):
                frame = data_bytes[offset:offset + self.frame_bytes]
                if len(frame) < self.frame_bytes:
                    frame += b"\x00" * (self.frame_bytes - len(frame))
                frames.append(frame)
        if not frames:
            raise ValueError("replay manifest produced no PCM frames")
        return frames * self.repeat

    def read_frame(self) -> Optional[bytes]:
        if self.index >= len(self.frames):
            self.completed = True
            return None
        if self.started_mono is None:
            self.started_mono = time.monotonic()
        if self.realtime:
            due = self.started_mono + (self.index * self.frame_bytes / 2.0 / 16000.0)
            now = time.monotonic()
            if due > now:
                time.sleep(due - now)
            else:
                self.max_schedule_lag_ms = max(self.max_schedule_lag_ms, (now - due) * 1000.0)
        frame = self.frames[self.index]
        self.index += 1
        return frame

    def stats(self) -> dict:
        return {
            "replay": True,
            "scenario": self.scenario,
            "frames_total": len(self.frames),
            "frames_emitted": self.index,
            "completed": self.completed,
            "max_schedule_lag_ms": round(self.max_schedule_lag_ms, 3),
        }

    def close(self):
        self.completed = True

try:
    import fcntl
except ImportError:
    fcntl = None

try:
    import select
except ImportError:
    select = None


class RawMicStream:
    def __init__(self, device: str, sr: int, channels: int = 1,
                 frame_bytes: int = FRAME_BYTES,
                 read_timeout_sec: float = 2.0,
                 startup_delay_sec: float = 0.15,
                 mic_debug: bool = False,
                 mic_debug_every: int = 50,
                 dry_run_text: bool = False):
        self.device = device
        self.sr = sr
        self.channels = channels
        self.frame_bytes = frame_bytes
        self.read_timeout_sec = read_timeout_sec
        self.startup_delay_sec = startup_delay_sec
        self.mic_debug = mic_debug
        self.mic_debug_every = max(1, int(mic_debug_every))
        self.dry_run_text = dry_run_text
        self.proc: Optional[subprocess.Popen] = None
        self.restart_count = 0
        self.frames_ok = 0
        self.partial_events = 0
        self.timeout_events = 0
        self.eof_events = 0
        self.last_chunk_sizes: List[int] = []
        self.last_restart_reason = ""

        if not self.dry_run_text:
            self.start()

    def start(self):
        self.close()
        if os.name == "nt" or self.dry_run_text:
            return

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
        if self.proc is None or self.proc.stderr is None or fcntl is None:
            return ""
        try:
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
        if reason == "interrupted" or (self.proc is not None and self.proc.poll() is not None and self.last_restart_reason == "closing"):
            jlog({"level": "info", "src": "mic", "msg": "arecord stopped", "reason": reason})
            return
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
        if os.name == "nt" or self.dry_run_text:
            time.sleep(0.08)  # simulate frame capture timing
            return bytes(self.frame_bytes)

        if self.proc is None:
            self.start()
        assert self.proc is not None
        if self.proc.stdout is None:
            raise RuntimeError("arecord stdout is None")

        if select is None:
            raise RuntimeError("select module not available on this platform")

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
            try:
                chunk = os.read(fd, nbytes - len(buf))
            except OSError as exc:
                if exc.errno == errno.EINTR:
                    self._restart("interrupted", got_bytes=len(buf))
                    return None
                raise
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
            self.last_restart_reason = "closing"
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
