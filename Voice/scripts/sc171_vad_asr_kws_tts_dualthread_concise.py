#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import json
import os
import re
import sys
import time
import signal
import queue
import threading
import subprocess
import tempfile
import wave
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import soundfile as sf

from funasr_onnx import Fsmn_vad, Paraformer
import openwakeword
from openwakeword.utils import AudioFeatures

try:
    import tflite_runtime.interpreter as tflite
except Exception:
    tflite = None

try:
    import onnxruntime as ort
except Exception:
    ort = None

try:
    from piper.voice import PiperVoice
except Exception:
    PiperVoice = None

try:
    import serial
except Exception:
    serial = None


SR = 16000
FRAME_MS = 80
FRAME_SAMPLES = SR * FRAME_MS // 1000
FRAME_BYTES = FRAME_SAMPLES * 2
MIN_UTT_MS = 200

LOG_MODE = "concise"
LOG_ENABLED = True
QUIET_MIC_INFO = True


def configure_logging(mode: str = "concise", quiet_mic_info: bool = True, log_enabled: bool = True):
    global LOG_MODE, QUIET_MIC_INFO, LOG_ENABLED
    LOG_MODE = mode
    QUIET_MIC_INFO = quiet_mic_info
    LOG_ENABLED = log_enabled


def should_emit(payload: Dict[str, Any]) -> bool:
    if not LOG_ENABLED:
        return False
    level = str(payload.get("level", "info"))
    src = str(payload.get("src", ""))
    msg = str(payload.get("msg", ""))

    if LOG_MODE == "full":
        return True

    if level in {"error", "warn"}:
        return True

    # concise mode: keep only milestone logs
    if level == "debug":
        return False

    keep_info_src = {"boot", "loop", "oww", "seg", "decision", "tts", "heartbeat", "signal", "queue"}
    if src == "mic":
        # keep restart warnings via level=warn; suppress frequent start/info chatter
        return not QUIET_MIC_INFO and level == "info"
    if src in keep_info_src:
        return True
    return False


def clamp01(x: float) -> float:
    return float(max(0.0, min(1.0, x)))


def jlog(payload: Dict[str, Any], *, stderr: bool = False):
    if not should_emit(payload):
        return
    s = json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
    print(s, file=sys.stderr if stderr else sys.stdout, flush=True)


def rms_int16(x: np.ndarray) -> float:
    xf = x.astype(np.float32)
    return float(np.sqrt(np.mean(xf * xf) + 1e-12))


def normalize_text(s: str) -> str:
    return re.sub(r"\s+", "", s or "").strip()


def infer_oww_key(model_path: str) -> str:
    return Path(model_path).stem


def kws_trigger(pred: Dict[str, float], key: str, th: float) -> bool:
    return float(pred.get(key, 0.0)) >= th


def model_file_exists(model_dir: str, filename: str) -> bool:
    return Path(model_dir, filename).exists()


def auto_quant_flag(model_dir: str, want_quant: bool, role: str) -> bool:
    has_model = model_file_exists(model_dir, "model.onnx")
    has_quant = model_file_exists(model_dir, "model_quant.onnx")
    if want_quant:
        if has_quant:
            return True
        if has_model:
            print(f"[{role}] NOTE: model_quant.onnx missing -> use model.onnx", file=sys.stderr)
            return False
    else:
        if has_model:
            return False
        if has_quant:
            print(f"[{role}] NOTE: model.onnx missing -> use model_quant.onnx", file=sys.stderr)
            return True
    raise RuntimeError(f"[{role}] need model.onnx or model_quant.onnx in {model_dir}")


def normalize_vad_segments(raw: Any) -> List[Tuple[float, float]]:
    if isinstance(raw, list) and len(raw) == 1 and isinstance(raw[0], list):
        raw = raw[0]
    segs: List[Tuple[float, float]] = []
    if not isinstance(raw, list):
        return segs
    for s in raw:
        if isinstance(s, (list, tuple)) and len(s) >= 2:
            try:
                beg, end = float(s[0]), float(s[1])
            except Exception:
                continue
            if end > beg:
                segs.append((beg, end))
    return segs


def pick_best_segment(segs: List[Tuple[float, float]]) -> Optional[Tuple[float, float]]:
    if not segs:
        return None
    return max(segs, key=lambda x: x[1] - x[0])


def to_sample_index_ms(x_ms: float) -> int:
    return int(x_ms * SR / 1000.0)


def parse_asr_output(asr_out: Any) -> Tuple[str, Optional[float]]:
    obj = asr_out
    if isinstance(obj, list) and obj:
        obj = obj[0]
    text = ""
    conf = None
    if isinstance(obj, dict):
        if "preds" in obj:
            preds = obj["preds"]
            if isinstance(preds, (list, tuple)) and len(preds) >= 1:
                text = str(preds[0])
            else:
                text = str(preds)
        else:
            text = str(obj.get("text", obj.get("pred", obj.get("result", ""))))
        for k in ("confidence", "conf", "score", "prob"):
            if k in obj:
                try:
                    conf = float(obj[k])
                except Exception:
                    conf = None
                break
    else:
        text = str(obj)
    return text, conf



DEFAULT_COMMAND_RULES = {
    "stop": ["停止", "停下", "别动", "取消", "危险", "stop", "停"],
    "return": ["回来", "返回", "回去", "return"],
    "find": {
        "cup": ["水杯", "杯子", "杯"],
        "apple": ["苹果"],
        "keys": ["钥匙", "钥", "要是", "钥石"],
    },
}

COMMAND_RULES = DEFAULT_COMMAND_RULES.copy()


def load_command_rules(json_path: str) -> Dict[str, Any]:
    if not json_path:
        return DEFAULT_COMMAND_RULES.copy()
    p = Path(json_path)
    data = json.loads(p.read_text(encoding="utf-8"))
    rules = {
        "stop": list(data.get("stop", DEFAULT_COMMAND_RULES["stop"])),
        "return": list(data.get("return", DEFAULT_COMMAND_RULES["return"])),
        "find": dict(data.get("find", DEFAULT_COMMAND_RULES["find"])),
    }
    return rules


def infer_intent_and_target(text: str) -> Tuple[str, Optional[str], float]:
    t_raw = normalize_text(text)
    t = t_raw.lower()
    if not t_raw:
        return "REJECT", None, 0.0

    stop_kw = COMMAND_RULES.get("stop", [])
    if any(k in t or k in t_raw for k in stop_kw):
        return "STOP", None, 0.90

    return_kw = COMMAND_RULES.get("return", [])
    if any(k in t or k in t_raw for k in return_kw):
        return "RETURN", None, 0.85

    target_map = COMMAND_RULES.get("find", {})
    for target, kws in target_map.items():
        if any(k in t_raw for k in kws):
            return "FIND", target, 0.75

    return "REJECT", None, 0.0


class SerialCommander:
    def __init__(self, port: str, baud: int):
        if serial is None:
            raise RuntimeError("pyserial not installed. pip install pyserial")
        self.ser = serial.Serial(port, baudrate=baud, timeout=0)

    def send_jsonl(self, payload: Dict[str, Any]):
        line = json.dumps(payload, ensure_ascii=False, separators=(",", ":")) + "\n"
        self.ser.write(line.encode("utf-8", errors="ignore"))

    def close(self):
        try:
            self.ser.close()
        except Exception:
            pass


class PiperTTS:
    def __init__(self, model_onnx: str, cache_dir: str, out_dir: str,
                 mode: str = "save", play_cmd: str = "aplay -q"):
        self.model_onnx = model_onnx
        self.cache_dir = Path(cache_dir)
        self.out_dir = Path(out_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.mode = mode
        self.play_cmd = play_cmd
        self.voice = None
        if PiperVoice is not None:
            try:
                self.voice = PiperVoice.load(model_onnx)
            except Exception as e:
                jlog({"level": "warn", "src": "tts", "msg": f"Piper load failed: {e}"}, stderr=True)
                self.voice = None

    def _safe_name(self, text: str) -> str:
        return re.sub(r"[^0-9A-Za-z\u4e00-\u9fff]+", "_", text).strip("_")[:40] or "tts"

    def _wav_path(self, text: str) -> Path:
        return self.cache_dir / f"tts_{self._safe_name(text)}.wav"

    def synth_to_wav(self, text: str, wav_path: Path):
        if self.voice is not None:
            with wave.open(str(wav_path), "wb") as wf:
                wf.setnchannels(1)
                wf.setsampwidth(2)
                wf.setframerate(SR)
                self.voice.synthesize(text, wf)
        else:
            cmd = f'echo "{text}" | piper --model "{self.model_onnx}" --output_file "{wav_path}"'
            subprocess.run(cmd, shell=True, check=True)

    def say(self, text: str) -> Optional[Path]:
        wav_path = self._wav_path(text)
        if not wav_path.exists():
            self.synth_to_wav(text, wav_path)
        stamp = int(time.time() * 1000)
        out_path = self.out_dir / f"{stamp}_{wav_path.name}"
        if not out_path.exists():
            out_path.write_bytes(wav_path.read_bytes())
        if self.mode == "play":
            subprocess.run(f'{self.play_cmd} "{wav_path}"', shell=True, check=False)
        return out_path

    def warmup_phrases(self, phrases: List[str]):
        for t in phrases:
            wav_path = self._wav_path(t)
            if not wav_path.exists():
                try:
                    self.synth_to_wav(t, wav_path)
                except Exception as e:
                    jlog({"level": "warn", "src": "tts", "msg": f"warmup failed: {e}", "text": t}, stderr=True)


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
        jlog({"level": "info", "src": "mic", "msg": "arecord started", "device": self.device, "restart": self.restart_count}, stderr=True)

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
        jlog(payload, stderr=True)
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
                jlog({"level": "debug", "src": "mic", "msg": "chunked frame read", "chunks": chunks[-8:], "sum_bytes": len(buf), "target_bytes": nbytes}, stderr=True)
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


class FlexibleWakeWord:
    def __init__(self, wakeword_models: List[str], vad_threshold: float = 0.0, ncpu: int = 1):
        self.pre = AudioFeatures(inference_framework="tflite", ncpu=ncpu)
        self.models: Dict[str, Dict[str, Any]] = {}
        self.vad_threshold = float(vad_threshold)
        self.vad = None
        if self.vad_threshold > 0:
            try:
                self.vad = openwakeword.VAD()
            except Exception as e:
                jlog({"level": "warn", "src": "oww", "msg": f"VAD init failed, disabled: {e}"}, stderr=True)
                self.vad = None
                self.vad_threshold = 0.0

        for mdl_path in wakeword_models:
            name = Path(mdl_path).stem
            ext = Path(mdl_path).suffix.lower()
            meta: Dict[str, Any] = {"path": mdl_path, "name": name, "ext": ext, "n_calls": 0}
            if ext == ".tflite":
                if tflite is None:
                    raise RuntimeError("tflite_runtime is required for .tflite wake models")
                itp = tflite.Interpreter(model_path=mdl_path, num_threads=1)
                itp.allocate_tensors()
                inp = itp.get_input_details()[0]
                out = itp.get_output_details()[0]
                shape = tuple(int(i) for i in inp["shape"])
                meta.update({
                    "runner": itp, "input_index": inp["index"], "output_index": out["index"],
                    "input_name": None, "output_name": None, "io_kind": "tflite", "input_shape": shape,
                })
            elif ext == ".onnx":
                if ort is None:
                    raise RuntimeError("onnxruntime is required for .onnx wake models")
                sess = ort.InferenceSession(mdl_path, providers=["CPUExecutionProvider"])
                inp = sess.get_inputs()[0]
                out = sess.get_outputs()[0]
                shape = tuple(int(i) if isinstance(i, (int, np.integer)) else -1 for i in inp.shape)
                meta.update({
                    "runner": sess, "input_index": None, "output_index": None,
                    "input_name": inp.name, "output_name": out.name, "io_kind": "onnx", "input_shape": shape,
                })
            else:
                raise RuntimeError(f"Unsupported wake model extension for {mdl_path}: {ext}")

            shape = meta["input_shape"]
            if len(shape) != 3 or shape[0] != 1:
                raise RuntimeError(f"Unsupported wake model input shape for {mdl_path}: {shape}")
            if shape[2] == 96:
                layout = "standard"
                n_feature_frames = shape[1]
            elif shape[1] == 96:
                layout = "transposed"
                n_feature_frames = shape[2]
            else:
                raise RuntimeError(f"Wake model {mdl_path} not compatible: {shape}")
            meta.update({"layout": layout, "n_feature_frames": int(n_feature_frames)})
            self.models[name] = meta
            jlog({
                "level": "info", "src": "oww", "model": name, "ext": ext,
                "input_shape": list(shape), "layout": layout, "n_feature_frames": int(n_feature_frames),
                "backend": meta["io_kind"],
            }, stderr=True)

    def reset(self):
        try:
            self.pre.reset()
        except Exception:
            pass
        if self.vad is not None:
            try:
                self.vad.reset()
            except Exception:
                pass
        for meta in self.models.values():
            meta["n_calls"] = 0

    def _vad_gate_ok(self, x: np.ndarray) -> bool:
        if self.vad is None or self.vad_threshold <= 0:
            return True
        try:
            self.vad(x)
            vad_frames = list(self.vad.prediction_buffer)[-7:-4]
            vad_max_score = float(np.max(vad_frames)) if len(vad_frames) > 0 else 0.0
            return vad_max_score >= self.vad_threshold
        except Exception:
            return True

    def predict(self, x: np.ndarray) -> Dict[str, float]:
        if not isinstance(x, np.ndarray):
            x = np.asarray(x, dtype=np.int16)
        if x.dtype != np.int16:
            x = x.astype(np.int16)
        self.pre(x)
        preds: Dict[str, float] = {}
        vad_ok = self._vad_gate_ok(x)
        for name, meta in self.models.items():
            meta["n_calls"] += 1
            feat = self.pre.get_features(meta["n_feature_frames"])
            if feat.shape[1] != meta["n_feature_frames"]:
                preds[name] = 0.0
                continue
            if meta["layout"] == "transposed":
                feat = np.transpose(feat, (0, 2, 1))
            feat = feat.astype(np.float32)
            if meta["io_kind"] == "tflite":
                meta["runner"].set_tensor(meta["input_index"], feat)
                meta["runner"].invoke()
                out = meta["runner"].get_tensor(meta["output_index"])
            else:
                out = meta["runner"].run([meta["output_name"]], {meta["input_name"]: feat})[0]
            score = float(np.array(out).reshape(-1)[0])
            if meta["n_calls"] < 5:
                score = 0.0
            if not vad_ok:
                score = 0.0
            preds[name] = clamp01(score)
        return preds


@dataclass
class AudioConfig:
    wake_key: str
    stop_key: str
    wake_th: float
    stop_th: float
    armed_secs: float
    energy_th: float
    start_frames: int
    end_frames: int
    pre_frames: int
    max_frames: int
    post_wake_mute_secs: float
    heartbeat_secs: float
    debug: bool


@dataclass
class RuntimeState:
    lock: threading.Lock = field(default_factory=threading.Lock)
    current_state: str = "IDLE"
    armed_until: float = 0.0
    mute_until: float = 0.0
    busy: bool = False
    last_rms: float = 0.0
    last_text: str = ""

    def set_state(self, state: str):
        with self.lock:
            self.current_state = state

    def arm(self, secs: float):
        with self.lock:
            self.armed_until = time.time() + secs
            self.current_state = "ARMED_WAIT"

    def disarm(self):
        with self.lock:
            self.armed_until = 0.0
            if not self.busy:
                self.current_state = "IDLE"

    def is_armed(self) -> bool:
        with self.lock:
            return time.time() < self.armed_until

    def set_busy(self, v: bool):
        with self.lock:
            self.busy = v
            if v:
                self.current_state = "BUSY"
            elif time.time() < self.armed_until:
                self.current_state = "ARMED_WAIT"
            else:
                self.current_state = "IDLE"

    def set_mute(self, secs: float):
        with self.lock:
            self.mute_until = max(self.mute_until, time.time() + secs)

    def is_muted(self) -> bool:
        with self.lock:
            return time.time() < self.mute_until

    def snapshot(self) -> dict:
        with self.lock:
            return {
                "state": self.current_state,
                "armed": time.time() < self.armed_until,
                "mute": time.time() < self.mute_until,
                "busy": self.busy,
                "rms": round(self.last_rms, 2),
                "last_text": self.last_text,
            }

    def set_rms(self, rms: float):
        with self.lock:
            self.last_rms = rms

    def set_last_text(self, text: str):
        with self.lock:
            self.last_text = text


class AudioKWSWorker(threading.Thread):
    def __init__(self, args, rt: RuntimeState, stop_event: threading.Event, utter_q: queue.Queue):
        super().__init__(daemon=True)
        self.args = args
        self.rt = rt
        self.stop_event = stop_event
        self.utter_q = utter_q

        ww_models = [args.wake_model]
        if args.stop_model:
            ww_models.append(args.stop_model)
        self.oww = FlexibleWakeWord(wakeword_models=ww_models, vad_threshold=args.oww_vad_th)
        self.mic = RawMicStream(
            args.arecord_device, SR, channels=1,
            read_timeout_sec=args.mic_read_timeout,
            startup_delay_sec=args.mic_startup_delay,
            mic_debug=(args.mic_debug or args.debug),
            mic_debug_every=args.mic_debug_every,
        )
        self.cfg = AudioConfig(
            wake_key=args.wake_key,
            stop_key=args.stop_key,
            wake_th=args.wake_th,
            stop_th=args.stop_th,
            armed_secs=args.armed_secs,
            energy_th=args.energy_th,
            start_frames=args.start_frames,
            end_frames=args.end_frames,
            pre_frames=args.pre_frames,
            max_frames=args.max_frames,
            post_wake_mute_secs=args.post_wake_mute_secs,
            heartbeat_secs=args.heartbeat_secs,
            debug=args.debug,
        )
        self.last_hb = 0.0
        self.prebuf: deque[np.ndarray] = deque(maxlen=args.pre_frames)
        self.state = "IDLE"
        self.speech_up = 0
        self.speech_down = 0
        self.captured: List[np.ndarray] = []

    def _emit_heartbeat(self):
        now = time.time()
        if self.cfg.heartbeat_secs <= 0 or now - self.last_hb < self.cfg.heartbeat_secs:
            return
        hb = self.rt.snapshot()
        hb.update({"level": "info", "src": "heartbeat", "mic": self.mic.stats()})
        jlog(hb, stderr=True)
        self.last_hb = now

    def _reset_recording(self, next_state: str = "IDLE"):
        self.state = next_state
        self.speech_up = 0
        self.speech_down = 0
        self.captured = []
        self.rt.set_state(next_state)

    def _enqueue_utterance(self, frames: List[np.ndarray], rms: float):
        if not frames:
            return
        audio = np.concatenate(frames, axis=0).astype(np.int16)
        item = {
            "ts": time.time(),
            "audio": audio,
            "rms": float(rms),
        }
        try:
            self.utter_q.put_nowait(item)
            self.rt.set_busy(True)
            self.rt.set_state("BUSY")
        except queue.Full:
            try:
                _ = self.utter_q.get_nowait()
            except Exception:
                pass
            try:
                self.utter_q.put_nowait(item)
                self.rt.set_busy(True)
                self.rt.set_state("BUSY")
                jlog({"level": "warn", "src": "queue", "msg": "utterance queue full, dropped oldest"}, stderr=True)
            except Exception:
                jlog({"level": "warn", "src": "queue", "msg": "utterance queue full, drop new utterance"}, stderr=True)

    def run(self):
        jlog({"level": "info", "src": "loop", "msg": "audio/kws thread started"}, stderr=True)
        try:
            while not self.stop_event.is_set():
                b = self.mic.read_frame()
                if b is None:
                    continue
                x = np.frombuffer(b, dtype=np.int16)
                r = rms_int16(x)
                self.rt.set_rms(r)
                self.prebuf.append(x.copy())
                self._emit_heartbeat()

                muted = self.rt.is_muted()
                armed = self.rt.is_armed()
                busy = self.rt.snapshot()["busy"]

                if not muted:
                    pred = self.oww.predict(x)
                    if self.cfg.debug:
                        jlog({"level": "debug", "src": "oww", "pred": pred}, stderr=True)
                    if self.cfg.stop_key and kws_trigger(pred, self.cfg.stop_key, self.cfg.stop_th):
                        payload = {"ts": float(time.time()), "intent": "STOP", "confidence": 0.98, "source": "stop_kws"}
                        jlog(payload)
                        self.rt.disarm()
                        self.rt.set_mute(0.8)
                        self._reset_recording("IDLE")
                        continue
                    if not armed and not busy and kws_trigger(pred, self.cfg.wake_key, self.cfg.wake_th):
                        self.rt.arm(self.cfg.armed_secs)
                        self.rt.set_mute(self.cfg.post_wake_mute_secs)
                        self.state = "ARMED_WAIT"
                        self.speech_up = 0
                        self.speech_down = 0
                        self.captured = []
                        jlog({"level": "info", "src": "oww", "msg": "WAKE triggered -> armed"}, stderr=True)
                        continue

                armed = self.rt.is_armed()
                if not armed:
                    if self.state != "IDLE":
                        self._reset_recording("IDLE")
                    continue
                if busy:
                    continue

                if self.state == "ARMED_WAIT":
                    if r >= self.cfg.energy_th:
                        self.speech_up += 1
                    else:
                        self.speech_up = 0
                    if self.speech_up >= self.cfg.start_frames:
                        self.captured = list(self.prebuf)
                        self.captured.append(x.copy())
                        self.speech_down = 0
                        self.state = "REC"
                        self.rt.set_state("REC")
                        jlog({"level": "info", "src": "seg", "msg": "REC start", "rms": round(r, 2)}, stderr=True)
                    continue

                if self.state == "REC":
                    self.captured.append(x.copy())
                    if r < self.cfg.energy_th:
                        self.speech_down += 1
                    else:
                        self.speech_down = 0

                    enough = len(self.captured) * FRAME_MS >= MIN_UTT_MS
                    end_now = enough and self.speech_down >= self.cfg.end_frames
                    too_long = len(self.captured) >= self.cfg.max_frames
                    if end_now or too_long:
                        self._enqueue_utterance(self.captured, r)
                        jlog({
                            "level": "info", "src": "seg", "msg": "REC end",
                            "frames": len(self.captured), "too_long": too_long,
                        }, stderr=True)
                        self._reset_recording("IDLE")
                        self.rt.disarm()
                        self.rt.set_busy(True)
        finally:
            try:
                self.mic.close()
            except Exception:
                pass


class ASRDecisionWorker(threading.Thread):
    def __init__(self, args, rt: RuntimeState, stop_event: threading.Event, utter_q: queue.Queue):
        super().__init__(daemon=True)
        self.args = args
        self.rt = rt
        self.stop_event = stop_event
        self.utter_q = utter_q
        asr_quant = auto_quant_flag(args.asr_dir, args.asr_quant, "ASR")
        vad_quant = auto_quant_flag(args.vad_dir, args.vad_quant, "VAD")
        self.vad = Fsmn_vad(args.vad_dir, quantize=vad_quant)
        self.asr = Paraformer(args.asr_dir, batch_size=1, quantize=asr_quant, device_id=-1)
        self.ser_cmd: Optional[SerialCommander] = SerialCommander(args.serial, args.baud) if args.serial else None
        self.tts: Optional[PiperTTS] = None
        if not args.disable_tts:
            self.tts = PiperTTS(args.piper_model, args.tts_cache, args.tts_out_dir, mode=args.tts_mode, play_cmd=args.play_cmd)
            self.tts.warmup_phrases(["已停止", "开始执行", "未识别到物品", "返回中"])
        self.wake_phrases = [normalize_text(x) for x in args.wake_phrases.split(",") if normalize_text(x)]

    def say_text(self, text: str):
        if self.tts:
            out = self.tts.say(text)
            if out is not None and self.args.debug:
                jlog({"level": "info", "src": "tts", "saved": str(out)}, stderr=True)

    def emit_action(self, intent: str, target: Optional[str], conf: float, text: str):
        payload = {"ts": float(time.time()), "intent": intent, "confidence": clamp01(conf), "text": text}
        if intent == "FIND":
            payload["target"] = target or "unknown"
        jlog(payload)
        if self.ser_cmd:
            self.ser_cmd.send_jsonl(payload)

    def _is_wake_text(self, text: str) -> bool:
        nt = normalize_text(text)
        return any(p and p in nt for p in self.wake_phrases)

    def _handle_result(self, text: str, latency_ms: float):
        if self._is_wake_text(text):
            jlog({"level": "info", "src": "decision", "msg": "ignore wake phrase as command", "text": text}, stderr=True)
            return
        intent, target, conf = infer_intent_and_target(text)
        jlog({"level": "info", "src": "decision", "text": text, "intent": intent, "target": target, "confidence": conf, "latency_ms": round(latency_ms, 2)}, stderr=True)
        if intent == "REJECT":
            jlog({"level": "info", "src": "decision", "msg": "reject / no action", "text": text}, stderr=True)
            return
        self.emit_action(intent, target, conf, text)
        if intent == "FIND":
            if (target or "") == "unknown":
                self.say_text("未识别到物品")
            else:
                self.say_text("开始执行")
        elif intent == "RETURN":
            self.say_text("返回中")
        elif intent == "STOP":
            self.say_text("已停止")

    def process_utterance(self, item: dict):
        audio = item["audio"]
        t0 = time.perf_counter()
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            tmp_wav = f.name
        try:
            sf.write(tmp_wav, audio.astype(np.int16), SR, subtype="PCM_16")
            raw_vad = self.vad(tmp_wav)
            segs = normalize_vad_segments(raw_vad)
            best = pick_best_segment(segs)
            cut = audio.astype(np.float32)
            if best is not None:
                s0 = max(0, to_sample_index_ms(best[0]))
                s1 = min(len(audio), to_sample_index_ms(best[1]))
                if s1 > s0:
                    cut = audio[s0:s1].astype(np.float32)
            if len(cut) < int(MIN_UTT_MS * SR / 1000):
                jlog({"level": "info", "src": "decision", "msg": "drop too-short utterance", "samples": int(len(cut))}, stderr=True)
                return
            asr_out = self.asr(cut)
            if self.args.debug:
                jlog({"level": "debug", "src": "asr", "raw": str(asr_out)}, stderr=True)
            text, _ = parse_asr_output(asr_out)
            text = text.strip()
            self.rt.set_last_text(text)
            latency_ms = (time.perf_counter() - t0) * 1000.0
            self._handle_result(text, latency_ms)
        finally:
            try:
                os.unlink(tmp_wav)
            except Exception:
                pass

    def run(self):
        jlog({"level": "info", "src": "loop", "msg": "asr/decision thread started"}, stderr=True)
        while not self.stop_event.is_set():
            try:
                item = self.utter_q.get(timeout=0.2)
            except queue.Empty:
                continue
            try:
                self.process_utterance(item)
            except Exception as e:
                jlog({"level": "error", "src": "worker", "msg": f"process utterance failed: {e}"}, stderr=True)
            finally:
                self.rt.set_busy(False)
                self.rt.disarm()
                self.rt.set_mute(self.args.post_tts_mute_secs)
                self.utter_q.task_done()
        if self.ser_cmd:
            self.ser_cmd.close()


def list_audio_devices() -> int:
    print("===== arecord -l =====")
    subprocess.run(["arecord", "-l"], check=False)
    print("\n===== arecord -L =====")
    subprocess.run(["arecord", "-L"], check=False)
    print("\n===== aplay -l =====")
    subprocess.run(["aplay", "-l"], check=False)
    print("\n===== aplay -L =====")
    subprocess.run(["aplay", "-L"], check=False)
    return 0


def build_argparser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser()
    ap.add_argument("--asr_dir", required=True)
    ap.add_argument("--vad_dir", required=True)
    ap.add_argument("--asr_quant", action="store_true")
    ap.add_argument("--vad_quant", action="store_true")
    ap.add_argument("--wake_model", default="")
    ap.add_argument("--stop_model", default="")
    ap.add_argument("--wake_tflite", default="")  # legacy alias
    ap.add_argument("--stop_tflite", default="")  # legacy alias
    ap.add_argument("--wake_key", default="")
    ap.add_argument("--stop_key", default="")
    ap.add_argument("--wake_th", type=float, default=0.60)
    ap.add_argument("--stop_th", type=float, default=0.60)
    ap.add_argument("--armed_secs", type=float, default=6.0)
    ap.add_argument("--oww_vad_th", type=float, default=0.0)
    ap.add_argument("--wake_phrases", default="你好小车,你好 小车")
    ap.add_argument("--energy_th", type=float, default=450.0)
    ap.add_argument("--start_frames", type=int, default=2)
    ap.add_argument("--end_frames", type=int, default=4)
    ap.add_argument("--pre_frames", type=int, default=3)
    ap.add_argument("--max_frames", type=int, default=80)
    ap.add_argument("--piper_model", required=True)
    ap.add_argument("--tts_cache", default="/home/aidlux/2026/Voice/tts_cache")
    ap.add_argument("--tts_out_dir", default="/home/aidlux/2026/Voice/tts_out")
    ap.add_argument("--tts_mode", choices=["save", "play"], default="save")
    ap.add_argument("--disable_tts", action="store_true")
    ap.add_argument("--play_cmd", default="aplay -q")
    ap.add_argument("--serial", default="")
    ap.add_argument("--baud", type=int, default=115200)
    ap.add_argument("--arecord_device", default="default")
    ap.add_argument("--list_audio_devices", action="store_true")
    ap.add_argument("--mic_read_timeout", type=float, default=2.0)
    ap.add_argument("--mic_startup_delay", type=float, default=0.15)
    ap.add_argument("--mic_debug", action="store_true")
    ap.add_argument("--mic_debug_every", type=int, default=50)
    ap.add_argument("--post_tts_mute_secs", type=float, default=1.2)
    ap.add_argument("--post_wake_mute_secs", type=float, default=0.5)
    ap.add_argument("--heartbeat_secs", type=float, default=10.0)
    ap.add_argument("--debug", action="store_true")
    ap.add_argument("--log_mode", choices=["concise", "full"], default="concise")
    ap.add_argument("--show_mic_info", action="store_true")
    ap.add_argument("--commands_json", default="")
    return ap


def main():
    ap = build_argparser()
    args = ap.parse_args()

    if args.list_audio_devices:
        raise SystemExit(list_audio_devices())

    configure_logging("full" if (args.debug or args.mic_debug or args.log_mode == "full") else "concise",
                      quiet_mic_info=not args.show_mic_info)
    global COMMAND_RULES
    COMMAND_RULES = load_command_rules(args.commands_json) if args.commands_json else DEFAULT_COMMAND_RULES.copy()

    args.wake_model = args.wake_model or args.wake_tflite
    args.stop_model = args.stop_model or args.stop_tflite
    if not args.wake_model:
        raise SystemExit("--wake_model (or legacy --wake_tflite) is required")
    args.wake_key = args.wake_key or infer_oww_key(args.wake_model)
    args.stop_key = args.stop_key or (infer_oww_key(args.stop_model) if args.stop_model else "")

    jlog({
        "level": "info", "src": "boot",
        "wake_model": args.wake_model,
        "stop_model": args.stop_model,
        "wake_key": args.wake_key,
        "stop_key": args.stop_key,
        "tts_mode": args.tts_mode,
        "arecord_device": args.arecord_device,
        "threads": ["audio_kws", "asr_decision"],
        "log_mode": args.log_mode,
        "commands_json": args.commands_json,
    }, stderr=True)

    stop_event = threading.Event()
    rt = RuntimeState()
    utter_q: queue.Queue = queue.Queue(maxsize=2)

    def handle_sig(signum, frame):
        jlog({"level": "info", "src": "signal", "msg": f"got signal {signum}, stopping"}, stderr=True)
        stop_event.set()

    signal.signal(signal.SIGINT, handle_sig)
    signal.signal(signal.SIGTERM, handle_sig)

    audio_thread = AudioKWSWorker(args, rt, stop_event, utter_q)
    worker_thread = ASRDecisionWorker(args, rt, stop_event, utter_q)
    audio_thread.start()
    worker_thread.start()

    try:
        while not stop_event.is_set():
            time.sleep(0.5)
    finally:
        stop_event.set()
        audio_thread.join(timeout=3.0)
        worker_thread.join(timeout=3.0)


if __name__ == "__main__":
    main()
