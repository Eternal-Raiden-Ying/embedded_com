#!/usr/bin/env python3
"""Offline/Online Paraformer comparison; does not start any robot service."""
import argparse
import time
import wave
from pathlib import Path

import numpy as np

from Voice.voice_service.runtime.asr_engine import OfflineASREngine, OnlineASREngine
from Voice.voice_service.runtime.commands import CommandInterpreter
from Voice.voice_service.runtime.common import SR, parse_asr_output


def read_wav(path: Path) -> np.ndarray:
    with wave.open(str(path), "rb") as src:
        if src.getframerate() != SR or src.getnchannels() != 1 or src.getsampwidth() != 2:
            raise ValueError("--wav must be mono 16 kHz PCM16")
        return np.frombuffer(src.readframes(src.getnframes()), dtype=np.int16)


def intent(interpreter, text):
    return interpreter.infer_intent_and_target(text)[:2]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--wav", required=True)
    ap.add_argument("--offline-model", required=True)
    ap.add_argument("--online-model", required=True)
    ap.add_argument("--chunk-size", default="5,10,5")
    ap.add_argument("--quantized", action="store_true", default=False)
    args = ap.parse_args()
    chunk = [int(v) for v in args.chunk_size.split(",")]
    if len(chunk) != 3:
        ap.error("--chunk-size requires three comma-separated integers")
    audio = read_wav(Path(args.wav))
    interpreter = CommandInterpreter.from_json("Voice/config/commands.json")

    start = time.perf_counter()
    offline = OfflineASREngine(args.offline_model, args.quantized)
    offline_text, _ = offline.transcribe(audio)
    offline_ms = (time.perf_counter() - start) * 1000.0
    print("offline_final={!r} latency_ms={:.2f} intent_target={}".format(offline_text, offline_ms, intent(interpreter, offline_text)))

    online = OnlineASREngine(args.online_model, args.quantized, chunk, 4, 1)
    session = online.create_session()
    start = time.perf_counter()
    first_partial_ms = None
    partials = []
    for offset in range(0, len(audio), online.step_samples):
        result = online.feed(session, audio[offset:offset + online.step_samples], is_final=False)
        text = result.get("merged_text", "")
        if text and (not partials or partials[-1] != text):
            partials.append(text)
            if first_partial_ms is None:
                first_partial_ms = (time.perf_counter() - start) * 1000.0
            print("online_partial={!r}".format(text))
    final_meta = online.feed(session, np.zeros((0,), dtype=np.int16), is_final=True)
    final_text = final_meta.get("merged_text", "")
    final_ms = (time.perf_counter() - start) * 1000.0
    rtf = final_ms / 1000.0 / max(len(audio) / float(SR), 1e-9)
    print("online_final={!r} first_partial_ms={} final_latency_ms={:.2f} chunks={} samples={} rtf={:.3f} intent_target={}".format(
        final_text, "n/a" if first_partial_ms is None else "{:.2f}".format(first_partial_ms), final_ms,
        session.feed_calls, session.samples, rtf, intent(interpreter, final_text)))
    online.close_session(session)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
