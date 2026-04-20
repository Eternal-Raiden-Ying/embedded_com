#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import inspect
import os
import tempfile
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import soundfile as sf

from .common import (
    MIN_UTT_MS,
    SR,
    auto_quant_flag,
    clean_asr_text,
    jlog,
    normalize_text,
    normalize_vad_segments,
    parse_asr_output,
    pick_best_segment,
    to_sample_index_ms,
)
from .commands import CommandInterpreter


try:
    from funasr_onnx import Fsmn_vad, Paraformer as OfflineParaformer  # type: ignore
except Exception:  # pragma: no cover - board env specific
    Fsmn_vad = None
    OfflineParaformer = None

try:
    from funasr_onnx.paraformer_online_bin import Paraformer as OnlineParaformerImpl  # type: ignore
except Exception:  # pragma: no cover - board env specific
    OnlineParaformerImpl = None


class VADProcessor:
    def __init__(self, vad_dir: str, quantize: bool):
        if Fsmn_vad is None:
            raise RuntimeError("funasr_onnx.Fsmn_vad not available")
        self.vad = Fsmn_vad(vad_dir, quantize=quantize)

    def cut(self, audio):
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            tmp_wav = f.name
        try:
            sf.write(tmp_wav, audio.astype("int16"), SR, subtype="PCM_16")
            raw_vad = self.vad(tmp_wav)
            segs = normalize_vad_segments(raw_vad)
            best = pick_best_segment(segs)
            cut = audio.astype("float32")
            if best is not None:
                s0 = max(0, to_sample_index_ms(best[0]))
                s1 = min(len(audio), to_sample_index_ms(best[1]))
                if s1 > s0:
                    cut = audio[s0:s1].astype("float32")
            return cut
        finally:
            try:
                os.unlink(tmp_wav)
            except Exception:
                pass


class OfflineASREngine:
    def __init__(self, asr_dir: str, quantize: bool):
        if OfflineParaformer is None:
            raise RuntimeError("funasr_onnx.Paraformer not available")
        self.asr = OfflineParaformer(asr_dir, batch_size=1, quantize=quantize, device_id=-1)

    def transcribe(self, audio, debug: bool = False) -> Tuple[str, Optional[float]]:
        asr_out = self.asr(audio)
        if debug:
            jlog({"level": "debug", "src": "asr", "raw": str(asr_out)})
        text, conf = parse_asr_output(asr_out)
        return text.strip(), conf


@dataclass
class OnlineStreamSession:
    backend: Any
    started_at: float = field(default_factory=time.perf_counter)
    samples: int = 0
    partial_text: str = ""
    final_text: str = ""
    last_conf: Optional[float] = None
    feed_calls: int = 0
    last_partial_at: float = 0.0
    cache: Dict[str, Any] = field(default_factory=dict)
    debug: Dict[str, Any] = field(default_factory=dict)


def _safe_kwargs(func: Any, kwargs: Dict[str, Any]) -> Dict[str, Any]:
    try:
        sig = inspect.signature(func)
        params = sig.parameters
        if any(p.kind == inspect.Parameter.VAR_KEYWORD for p in params.values()):
            return kwargs
        return {k: v for k, v in kwargs.items() if k in params}
    except Exception:
        return kwargs


class OnlineASREngine:
    def __init__(
        self,
        asr_dir: str,
        quantize: bool,
        chunk_size: List[int],
        encoder_chunk_look_back: int,
        decoder_chunk_look_back: int,
    ):
        self.asr_dir = asr_dir
        self.quantize = quantize
        self.chunk_size = list(chunk_size)
        self.encoder_chunk_look_back = int(encoder_chunk_look_back)
        self.decoder_chunk_look_back = int(decoder_chunk_look_back)
        self.backend_ctor, self.backend_ctor_name = self._pick_backend_ctor()

    def _pick_backend_ctor(self):
        if OnlineParaformerImpl is not None:
            return OnlineParaformerImpl, "funasr_onnx.paraformer_online_bin.Paraformer"
        raise RuntimeError("funasr_onnx online backend not available")

    def _build_backend(self):
        ctor = self.backend_ctor
        attempts = [
            {
                "batch_size": 1,
                "quantize": self.quantize,
                "chunk_size": self.chunk_size,
                "intra_op_num_threads": 1,
            },
            {
                "batch_size": 1,
                "quantize": self.quantize,
                "chunk_size": self.chunk_size,
            },
            {
                "batch_size": 1,
                "quantize": self.quantize,
            },
            {},
        ]
        last_err = None
        for extra in attempts:
            try:
                kwargs = _safe_kwargs(ctor, {"model_dir": self.asr_dir, **extra})
                try:
                    return ctor(**kwargs)
                except TypeError:
                    kwargs = _safe_kwargs(ctor, {"model_path": self.asr_dir, **extra})
                    return ctor(**kwargs)
            except Exception as e:  # pragma: no cover - backend specific
                last_err = e
                continue
        raise RuntimeError(f"failed to init online ASR backend {self.backend_ctor_name}: {last_err}")

    def create_session(self) -> OnlineStreamSession:
        backend = self._build_backend()
        return OnlineStreamSession(backend=backend, debug={"backend": self.backend_ctor_name})

    @staticmethod
    def _merge_text(prev: str, new: str) -> str:
        prev = str(prev or "")
        new = str(new or "")
        if not prev:
            return new
        if not new:
            return prev
        if new.startswith(prev) or prev in new:
            return new
        if prev.startswith(new) or new in prev:
            return prev
        max_overlap = 0
        limit = min(len(prev), len(new))
        for i in range(1, limit + 1):
            if prev[-i:] == new[:i]:
                max_overlap = i
        return prev + new[max_overlap:]

    def _call_backend(self, session: OnlineStreamSession, audio, is_final: bool):
        backend = session.backend
        audio = np.asarray(audio, dtype=np.float32).reshape(-1)
        if audio.size:
            peak = float(np.max(np.abs(audio)))
            if peak > 1.5:
                audio = audio / 32768.0
        audio = np.ascontiguousarray(audio, dtype=np.float32)

        session.debug["audio_dtype"] = str(audio.dtype)
        session.debug["audio_samples"] = int(audio.size)
        session.debug["audio_min"] = float(audio.min()) if audio.size else 0.0
        session.debug["audio_max"] = float(audio.max()) if audio.size else 0.0
        session.debug["is_final"] = bool(is_final)
        session.debug["call_style"] = "audio_in+param_dict"

        out = backend(
            audio_in=audio,
            param_dict={
                "cache": session.cache,
                "is_final": bool(is_final),
            },
        )
        return out

    def feed(self, session: OnlineStreamSession, audio, is_final: bool = False, debug: bool = False) -> Dict[str, Any]:
        t0 = time.perf_counter()
        audio_arr = np.asarray(audio)
        session.samples += int(len(audio_arr))
        session.feed_calls += 1
        asr_out = self._call_backend(session, audio_arr, is_final=is_final)
        if debug:
            jlog({
                "level": "debug",
                "src": "asr_stream",
                "raw": str(asr_out),
                "is_final": bool(is_final),
                "backend": self.backend_ctor_name,
                "call_style": session.debug.get("call_style"),
                "audio_dtype": session.debug.get("audio_dtype"),
                "audio_samples": session.debug.get("audio_samples"),
                "audio_min": session.debug.get("audio_min"),
                "audio_max": session.debug.get("audio_max"),
                "chunk_size": list(self.chunk_size),
            })
        text, conf = parse_asr_output(asr_out)
        text = text.strip()
        if conf is not None:
            session.last_conf = conf
        if text:
            if is_final:
                session.final_text = self._merge_text(session.partial_text or session.final_text, text)
            else:
                session.partial_text = self._merge_text(session.partial_text, text)
                session.last_partial_at = time.time()
        if is_final and not session.final_text:
            session.final_text = session.partial_text
        return {
            "text": text,
            "merged_text": session.final_text if is_final else session.partial_text,
            "confidence": conf,
            "feed_latency_ms": (time.perf_counter() - t0) * 1000.0,
            "samples": session.samples,
            "is_final": bool(is_final),
            "backend": self.backend_ctor_name,
            "call_style": session.debug.get("call_style"),
        }


class AudioCommandPipeline:
    def __init__(self, cfg, interpreter: CommandInterpreter):
        self.asr_mode = str(getattr(cfg, "asr_mode", "offline") or "offline").lower()
        asr_quant = auto_quant_flag(cfg.asr_dir, cfg.asr_quant, "ASR")
        self.interpreter = interpreter
        self.wake_phrases = [normalize_text(x) for x in cfg.wake_phrases.split(",") if normalize_text(x)]
        self.debug = cfg.debug

        self.vad = None
        self.asr = None
        if self.asr_mode == "online":
            online_chunk_size = list(getattr(cfg, "asr_online_chunk_size", [5, 10, 5]))
            if len(online_chunk_size) < 3 or online_chunk_size == [0, 8, 4]:
                online_chunk_size = [5, 10, 5]
            self.asr = OnlineASREngine(
                cfg.asr_dir,
                asr_quant,
                chunk_size=online_chunk_size,
                encoder_chunk_look_back=int(getattr(cfg, "asr_online_encoder_chunk_look_back", online_chunk_size[0] if len(online_chunk_size) >= 1 else 5)),
                decoder_chunk_look_back=int(getattr(cfg, "asr_online_decoder_chunk_look_back", online_chunk_size[2] if len(online_chunk_size) >= 3 else 5)),
            )
        else:
            vad_quant = auto_quant_flag(cfg.vad_dir, cfg.vad_quant, "VAD")
            self.vad = VADProcessor(cfg.vad_dir, vad_quant)
            self.asr = OfflineASREngine(cfg.asr_dir, asr_quant)

    def is_online(self) -> bool:
        return self.asr_mode == "online"

    def is_wake_text(self, text: str) -> bool:
        nt = normalize_text(text)
        return any(p and p in nt for p in self.wake_phrases)

    def strip_leading_wake_phrase(self, text: str) -> Tuple[str, Optional[str]]:
        nt = normalize_text(text)
        if not nt:
            return "", None
        phrases = sorted([p for p in self.wake_phrases if p], key=len, reverse=True)
        for p in phrases:
            if nt == p:
                return "", p
            if nt.startswith(p):
                return nt[len(p):].strip(), p
        return nt, None

    def _interpret_text(self, text: str, asr_conf: Optional[float], latency_ms: float, samples: int) -> Dict[str, Any]:
        text = clean_asr_text(text)
        if samples < int(MIN_UTT_MS * SR / 1000):
            return {
                "status": "DROP_SHORT",
                "samples": int(samples),
                "latency_ms": latency_ms,
            }
        if not text:
            return {
                "status": "IGNORE_NOISE",
                "text": "",
                "asr_confidence": asr_conf,
                "latency_ms": latency_ms,
            }
        stripped_text, matched_wake = self.strip_leading_wake_phrase(text)
        if matched_wake is not None:
            if not stripped_text:
                return {
                    "status": "IGNORE_WAKE",
                    "text": text,
                    "asr_confidence": asr_conf,
                    "latency_ms": latency_ms,
                }
            text = stripped_text
        if self.interpreter.is_residual_text(text):
            return {
                "status": "IGNORE_NOISE",
                "text": text,
                "asr_confidence": asr_conf,
                "latency_ms": latency_ms,
            }
        intent, target, rule_conf = self.interpreter.infer_intent_and_target(text)
        return {
            "status": "OK" if intent != "REJECT" else "REJECT",
            "text": text,
            "intent": intent,
            "target": target,
            "confidence": rule_conf,
            "asr_confidence": asr_conf,
            "latency_ms": latency_ms,
        }

    def process_audio(self, audio):
        t0 = time.perf_counter()
        if self.is_online():
            raise RuntimeError("process_audio() called while asr_mode=online")
        cut = self.vad.cut(audio)
        if len(cut) < int(MIN_UTT_MS * SR / 1000):
            return {
                "status": "DROP_SHORT",
                "samples": int(len(cut)),
                "latency_ms": (time.perf_counter() - t0) * 1000.0,
            }
        text, asr_conf = self.asr.transcribe(cut, debug=self.debug)
        return self._interpret_text(text, asr_conf, (time.perf_counter() - t0) * 1000.0, len(cut))

    def start_stream_session(self) -> OnlineStreamSession:
        if not self.is_online():
            raise RuntimeError("start_stream_session() called while asr_mode!=online")
        return self.asr.create_session()

    def stream_feed(self, session: OnlineStreamSession, audio, is_final: bool = False) -> Dict[str, Any]:
        if not self.is_online():
            raise RuntimeError("stream_feed() called while asr_mode!=online")
        return self.asr.feed(session, audio, is_final=is_final, debug=self.debug)

    def finalize_stream_result(self, session: OnlineStreamSession) -> Dict[str, Any]:
        if not self.is_online():
            raise RuntimeError("finalize_stream_result() called while asr_mode!=online")
        text = clean_asr_text((session.final_text or session.partial_text or "").strip())
        latency_ms = (time.perf_counter() - session.started_at) * 1000.0
        return self._interpret_text(text, session.last_conf, latency_ms, session.samples)
