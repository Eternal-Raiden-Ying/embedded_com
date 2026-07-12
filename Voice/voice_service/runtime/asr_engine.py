#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import tempfile
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

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


class VADProcessor:
    def __init__(self, vad_dir: str, quantize: bool, dry_run_text: bool = False):
        self.dry_run_text = dry_run_text
        self.vad = None
        if self.dry_run_text:
            return

        from funasr_onnx import Fsmn_vad
        self.vad = Fsmn_vad(vad_dir, quantize=quantize)

    def cut(self, audio) -> np.ndarray:
        if self.dry_run_text:
            return np.asarray(audio, dtype=np.float32)

        import soundfile as sf
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
    def __init__(self, asr_dir: str, quantize: bool, dry_run_text: bool = False):
        self.dry_run_text = dry_run_text
        self.asr = None
        if self.dry_run_text:
            return

        from funasr_onnx import Paraformer as OfflineParaformer
        self.asr = OfflineParaformer(asr_dir, batch_size=1, quantize=quantize, device_id=-1)

    def transcribe(self, audio, debug: bool = False) -> Tuple[str, Optional[float]]:
        if self.dry_run_text:
            return "", None

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


class OnlineASREngine:
    def __init__(
        self,
        asr_dir: str,
        quantize: bool,
        chunk_size: List[int],
        encoder_chunk_look_back: int,
        decoder_chunk_look_back: int,
        dry_run_text: bool = False,
    ):
        self.asr_dir = asr_dir
        self.quantize = quantize
        self.chunk_size = list(chunk_size)
        self.encoder_chunk_look_back = int(encoder_chunk_look_back)
        self.decoder_chunk_look_back = int(decoder_chunk_look_back)
        self.dry_run_text = dry_run_text

    def create_session(self) -> OnlineStreamSession:
        if self.dry_run_text:
            return OnlineStreamSession(backend=None, debug={"backend": "mock"})

        from funasr_onnx.paraformer_online_bin import Paraformer as OnlineParaformerImpl
        attempts = [
            {"batch_size": 1, "quantize": self.quantize, "chunk_size": self.chunk_size, "intra_op_num_threads": 1},
            {"batch_size": 1, "quantize": self.quantize, "chunk_size": self.chunk_size},
            {"batch_size": 1, "quantize": self.quantize},
            {},
        ]
        last_err = None
        for extra in attempts:
            try:
                # Basic signature helper (inline logic to avoid inspect parameters on built-ins)
                kwargs = {"model_dir": self.asr_dir}
                kwargs.update(extra)
                try:
                    backend = OnlineParaformerImpl(**kwargs)
                except TypeError:
                    kwargs = {"model_path": self.asr_dir}
                    kwargs.update(extra)
                    backend = OnlineParaformerImpl(**kwargs)
                return OnlineStreamSession(backend=backend, debug={"backend": "funasr_onnx"})
            except Exception as e:
                last_err = e
                continue
        raise RuntimeError(f"failed to init online ASR backend: {last_err}")

    def feed(self, session: OnlineStreamSession, audio, is_final: bool = False, debug: bool = False) -> Dict[str, Any]:
        if self.dry_run_text:
            return {
                "text": "",
                "merged_text": "",
                "confidence": 1.0,
                "feed_latency_ms": 0.0,
                "samples": session.samples,
                "is_final": bool(is_final),
                "backend": "mock",
            }

        t0 = time.perf_counter()
        audio_arr = np.asarray(audio)
        session.samples += int(len(audio_arr))
        session.feed_calls += 1

        # Audio normalization and contiguous check
        audio_feed = np.asarray(audio_arr, dtype=np.float32).reshape(-1)
        if audio_feed.size:
            peak = float(np.max(np.abs(audio_feed)))
            if peak > 1.5:
                audio_feed = audio_feed / 32768.0
        audio_feed = np.ascontiguousarray(audio_feed, dtype=np.float32)

        asr_out = session.backend(
            audio_in=audio_feed,
            param_dict={
                "cache": session.cache,
                "is_final": bool(is_final),
            },
        )

        text, conf = parse_asr_output(asr_out)
        text = text.strip()
        if conf is not None:
            session.last_conf = conf
        if text:
            # Inline text merging helper
            prev = session.final_text if is_final else session.partial_text
            if not prev:
                merged = text
            elif not text:
                merged = prev
            elif text.startswith(prev) or prev in text:
                merged = text
            elif prev.startswith(text) or text in prev:
                merged = prev
            else:
                max_overlap = 0
                limit = min(len(prev), len(text))
                for i in range(1, limit + 1):
                    if prev[-i:] == text[:i]:
                        max_overlap = i
                merged = prev + text[max_overlap:]

            if is_final:
                session.final_text = merged
            else:
                session.partial_text = merged
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
            "backend": "funasr_onnx",
        }


class AudioCommandPipeline:
    def __init__(self, cfg, interpreter: CommandInterpreter):
        self.asr_mode = str(getattr(cfg, "asr_mode", "offline") or "offline").lower()
        self.interpreter = interpreter
        self.wake_phrases = [normalize_text(x) for x in cfg.wake_phrases.split(",") if normalize_text(x)]
        self.debug = cfg.debug
        self.dry_run_text = getattr(cfg, "dry_run_text", False)

        self.vad = None
        self.asr = None
        if self.dry_run_text:
            self.asr = OnlineASREngine(
                "", False, [5, 10, 5], 5, 5, dry_run_text=True
            )
            self.vad = VADProcessor("", False, dry_run_text=True)
            return

        asr_quant = auto_quant_flag(cfg.asr_dir, cfg.asr_quant, "ASR")
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
                dry_run_text=False,
            )
        else:
            vad_quant = auto_quant_flag(cfg.vad_dir, cfg.vad_quant, "VAD")
            self.vad = VADProcessor(cfg.vad_dir, vad_quant, dry_run_text=False)
            self.asr = OfflineASREngine(cfg.asr_dir, asr_quant, dry_run_text=False)

        # Check hotwords capability
        hotwords = getattr(getattr(cfg, "lexicon", None), "asr_hotwords", [])
        if hotwords:
            jlog({"level": "warn", "src": "boot", "msg": "backend capability warning: ASR engine does not support runtime hotwords."})

    def is_online(self) -> bool:
        return self.asr_mode == "online"

    def warmup(self, samples: int = 1600) -> float:
        """Run one discarded silence inference without creating a Voice session."""
        silence = np.zeros((max(1, int(samples)),), dtype=np.int16)
        t0 = time.perf_counter()
        if self.is_online():
            session = self.start_stream_session()
            self.stream_feed(session, silence, is_final=True)
            self.finalize_stream_result(session)
        else:
            self.process_audio(silence)
        return (time.perf_counter() - t0) * 1000.0

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
        t_intent_start = time.perf_counter()
        intent, target, rule_conf = self.interpreter.infer_intent_and_target(text)
        intent_latency_ms = (time.perf_counter() - t_intent_start) * 1000.0
        return {
            "status": "OK" if intent != "REJECT" else "REJECT",
            "text": text,
            "intent": intent,
            "target": target,
            "confidence": rule_conf,
            "asr_confidence": asr_conf,
            "latency_ms": latency_ms,
            "intent_latency_ms": intent_latency_ms,
        }

    def process_audio(self, audio) -> Dict[str, Any]:
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
        return self.asr.create_session()

    def stream_feed(self, session: OnlineStreamSession, audio, is_final: bool = False) -> Dict[str, Any]:
        return self.asr.feed(session, audio, is_final=is_final, debug=self.debug)

    def finalize_stream_result(self, session: OnlineStreamSession) -> Dict[str, Any]:
        text = clean_asr_text((session.final_text or session.partial_text or "").strip())
        latency_ms = (time.perf_counter() - session.started_at) * 1000.0
        return self._interpret_text(text, session.last_conf, latency_ms, session.samples)
