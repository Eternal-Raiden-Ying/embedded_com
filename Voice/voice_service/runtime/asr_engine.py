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


def merge_online_hypothesis(previous_text: str, raw_text: str) -> str:
    """Merge one Paraformer streaming hypothesis into the session hypothesis.

    Online Paraformer may return either a cumulative hypothesis or only its
    latest increment.  This is intentionally the sole merge point: capture and
    decision workers consume the resulting ``merged_text`` but never merge text
    themselves.
    """
    previous = str(previous_text or "").strip()
    incoming = str(raw_text or "").strip()
    if not incoming:
        return previous
    if not previous:
        return incoming
    if incoming.startswith(previous):
        return incoming
    if previous.startswith(incoming) or incoming in previous:
        return previous
    limit = min(len(previous), len(incoming))
    for overlap in range(limit, 0, -1):
        if previous[-overlap:] == incoming[:overlap]:
            return previous + incoming[overlap:]
    return previous + incoming


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


class AsrBackend:
    """Small common contract; capture/VAD/NLU stay outside the backend."""

    mode = "offline"
    name = ""

    def close_session(self, session: Any) -> None:
        """Release per-command state. Offline backends have no session state."""


class OfflineASREngine(AsrBackend):
    mode = "offline"
    name = "funasr_onnx_offline"
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
    merged_text: str = ""
    last_conf: Optional[float] = None
    feed_calls: int = 0
    last_partial_at: float = 0.0
    cache: Dict[str, Any] = field(default_factory=dict)
    debug: Dict[str, Any] = field(default_factory=dict)
    finalized: bool = False


@dataclass
class OnlineVADSession:
    """Per-command state for the single shared FSMN online VAD model."""

    in_cache: List[Any] = field(default_factory=list)
    speech_active: bool = False
    closed: bool = False


class OnlineVADProcessor:
    """Incremental FSMN-VAD adapter used only by the online ASR capture path.

    ``Fsmn_vad_online`` keeps frontend and detector state internally, so the
    voice service deliberately permits only one active capture session.  Each
    new session resets that state and owns its ONNX FSMN cache.
    """

    def __init__(self, vad_dir: str, quantize: bool, dry_run_text: bool = False):
        self.dry_run_text = dry_run_text
        self.backend = None
        self._active_session: Optional[OnlineVADSession] = None
        if not self.dry_run_text:
            from funasr_onnx import Fsmn_vad_online
            self.backend = Fsmn_vad_online(vad_dir, quantize=quantize, intra_op_num_threads=1)

    def create_session(self) -> OnlineVADSession:
        if self._active_session is not None:
            self.close_session(self._active_session)
        session = OnlineVADSession()
        if self.backend is not None:
            # These two state holders live in funasr_onnx rather than in its
            # public param_dict.  Resetting them here prevents a new wake turn
            # from inheriting the previous turn's speech state or frontend tail.
            self.backend.frontend.cache_reset()
            self.backend.vad_scorer.AllResetDetection()
        self._active_session = session
        return session

    def close_session(self, session: Optional[OnlineVADSession]) -> None:
        if session is None:
            return
        session.in_cache.clear()
        session.speech_active = False
        session.closed = True
        if self._active_session is session:
            self._active_session = None
            if self.backend is not None:
                self.backend.frontend.cache_reset()
                self.backend.vad_scorer.AllResetDetection()

    def feed(self, session: OnlineVADSession, audio, is_final: bool = False) -> Dict[str, Any]:
        if session.closed or self.backend is None:
            return {"speech_started": False, "speech_ended": False, "segments": []}
        audio_arr = np.asarray(audio, dtype=np.float32).reshape(-1)
        if audio_arr.size and float(np.max(np.abs(audio_arr))) > 1.5:
            audio_arr = audio_arr / 32768.0
        raw_segments = self.backend(
            np.ascontiguousarray(audio_arr, dtype=np.float32),
            param_dict={"in_cache": session.in_cache, "is_final": bool(is_final)},
        )
        # The online FunASR VAD emits an open segment through its detector
        # state, then reports ``[-1, end_ms]`` when it closes.  Inspecting the
        # active detector buffer lets capture begin at VAD onset rather than
        # waiting until the end-of-speech event.
        buffers = getattr(self.backend.vad_scorer, "output_data_buf", [])
        active = any(
            bool(getattr(buf, "contain_seg_start_point", False))
            and not bool(getattr(buf, "contain_seg_end_point", False))
            for buf in buffers
        )
        segments = normalize_vad_segments(raw_segments)
        reported_end = any(
            isinstance(segment, (list, tuple)) and len(segment) >= 2 and float(segment[1]) >= 0
            for batch in (raw_segments if isinstance(raw_segments, list) else [])
            for segment in (batch if isinstance(batch, list) else [])
        )
        speech_started = active and not session.speech_active
        speech_ended = session.speech_active and (reported_end or not active or bool(is_final))
        session.speech_active = active and not bool(is_final)
        return {
            "speech_started": speech_started,
            "speech_ended": speech_ended,
            "segments": segments,
        }


class OnlineASREngine(AsrBackend):
    mode = "online"
    name = "funasr_onnx_online"
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
        self.backend = None
        if not self.dry_run_text:
            # Load once at boot. Cache remains strictly session-owned.
            self.backend = self._load_backend()

    @property
    def step_samples(self) -> int:
        return max(1, int(self.chunk_size[1]) * 960)

    def _load_backend(self) -> Any:
        from funasr_onnx.paraformer_online_bin import Paraformer as OnlineParaformerImpl
        attempts = [
            {"batch_size": 1, "quantize": self.quantize, "chunk_size": self.chunk_size, "intra_op_num_threads": 1},
            {"batch_size": 1, "quantize": self.quantize, "chunk_size": self.chunk_size},
        ]
        last_err = None
        for extra in attempts:
            try:
                kwargs = {"model_dir": self.asr_dir}
                kwargs.update(extra)
                try:
                    return OnlineParaformerImpl(**kwargs)
                except TypeError:
                    kwargs = {"model_path": self.asr_dir}
                    kwargs.update(extra)
                    return OnlineParaformerImpl(**kwargs)
            except Exception as exc:
                last_err = exc
        raise RuntimeError("online Paraformer load failed model_path={!r}: {}".format(self.asr_dir, last_err))

    def create_session(self) -> OnlineStreamSession:
        if self.dry_run_text:
            return OnlineStreamSession(backend=None, debug={"backend": "mock"})
        return OnlineStreamSession(backend=self.backend, debug={"backend": self.name})

    def close_session(self, session: OnlineStreamSession) -> None:
        session.cache.clear()
        session.backend = None
        session.merged_text = ""
        session.finalized = True

    def feed(self, session: OnlineStreamSession, audio, is_final: bool = False, debug: bool = False) -> Dict[str, Any]:
        if session.finalized:
            return {"text": "", "raw_text": "", "previous_text": session.merged_text,
                    "merged_text": session.merged_text, "confidence": None, "feed_latency_ms": 0.0,
                    "samples": session.samples, "is_final": bool(is_final), "backend": self.name}
        if self.dry_run_text:
            return {
                "text": "",
                "raw_text": "",
                "previous_text": session.merged_text,
                "merged_text": session.merged_text,
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
                "chunk_size": self.chunk_size,
                "encoder_chunk_look_back": self.encoder_chunk_look_back,
                "decoder_chunk_look_back": self.decoder_chunk_look_back,
            },
        )

        text, conf = parse_asr_output(asr_out)
        raw_text = text.strip()
        if conf is not None:
            session.last_conf = conf
        previous_text = session.merged_text
        session.merged_text = merge_online_hypothesis(previous_text, raw_text)
        if not is_final:
            session.last_partial_at = time.time()
        if is_final:
            session.finalized = True

        return {
            "text": raw_text,
            "raw_text": raw_text,
            "previous_text": previous_text,
            "merged_text": session.merged_text,
            "confidence": conf,
            "feed_latency_ms": (time.perf_counter() - t0) * 1000.0,
            "samples": session.samples,
            "is_final": bool(is_final),
            "backend": "funasr_onnx",
        }


def create_asr_backend(cfg: Any, *, dry_run_text: bool = False) -> AsrBackend:
    """The only ASR mode switch. No capture/VAD/NLU code is duplicated."""
    mode = str(getattr(cfg, "asr_mode", "offline") or "offline").lower()
    asr_quant = auto_quant_flag(cfg.asr_dir, cfg.asr_quant, "ASR") if not dry_run_text else bool(cfg.asr_quant)
    if mode == "online":
        chunk_size = list(getattr(cfg, "asr_online_chunk_size", [5, 10, 5]))
        if len(chunk_size) != 3:
            raise ValueError("asr_online_chunk_size must contain exactly three integers")
        return OnlineASREngine(
            cfg.asr_dir, asr_quant, chunk_size,
            int(getattr(cfg, "asr_online_encoder_chunk_look_back", 4)),
            int(getattr(cfg, "asr_online_decoder_chunk_look_back", 1)),
            dry_run_text=dry_run_text,
        )
    if mode != "offline":
        raise ValueError("unsupported asr_mode={!r}".format(mode))
    return OfflineASREngine(cfg.asr_dir, asr_quant, dry_run_text=dry_run_text)


class AudioCommandPipeline:
    def __init__(self, cfg, interpreter: CommandInterpreter):
        self.asr_mode = str(getattr(cfg, "asr_mode", "offline") or "offline").lower()
        self.interpreter = interpreter
        self.wake_phrases = [normalize_text(x) for x in cfg.wake_phrases.split(",") if normalize_text(x)]
        self.debug = cfg.debug
        self.dry_run_text = getattr(cfg, "dry_run_text", False)

        self.asr = create_asr_backend(cfg, dry_run_text=self.dry_run_text)
        self.vad = None
        vad_quant = auto_quant_flag(cfg.vad_dir, cfg.vad_quant, "VAD") if not self.dry_run_text else bool(cfg.vad_quant)
        if self.is_online():
            self.vad = OnlineVADProcessor(cfg.vad_dir, vad_quant, dry_run_text=self.dry_run_text)
        else:
            self.vad = VADProcessor(cfg.vad_dir, vad_quant, dry_run_text=self.dry_run_text)

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
        if not self.is_online():
            raise RuntimeError("start_stream_session() requires online backend")
        return self.asr.create_session()

    def start_vad_stream_session(self) -> OnlineVADSession:
        if not self.is_online():
            raise RuntimeError("start_vad_stream_session() requires online backend")
        return self.vad.create_session()

    def stream_vad_feed(self, session: OnlineVADSession, audio, is_final: bool = False) -> Dict[str, Any]:
        if not self.is_online():
            raise RuntimeError("stream_vad_feed() requires online backend")
        return self.vad.feed(session, audio, is_final=is_final)

    def abort_vad_stream_session(self, session: Optional[OnlineVADSession]) -> None:
        if self.is_online():
            self.vad.close_session(session)

    def stream_feed(self, session: OnlineStreamSession, audio, is_final: bool = False) -> Dict[str, Any]:
        return self.asr.feed(session, audio, is_final=is_final, debug=self.debug)

    def finalize_stream_result(self, session: OnlineStreamSession) -> Dict[str, Any]:
        text = clean_asr_text(session.merged_text.strip())
        latency_ms = (time.perf_counter() - session.started_at) * 1000.0
        return self._interpret_text(text, session.last_conf, latency_ms, session.samples)

    def abort_stream_session(self, session: Optional[OnlineStreamSession]) -> None:
        if session is not None and self.is_online():
            self.asr.close_session(session)
