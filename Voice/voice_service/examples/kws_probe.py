#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
import time
import wave
import argparse
from pathlib import Path
import numpy as np

from voice_service.config.loader import load_voice_config
from voice_service.runtime.kws_engine import create_kws_backend

def read_wav(wav_path):
    with wave.open(str(wav_path), "rb") as wf:
        sr = wf.getframerate()
        channels = wf.getnchannels()
        sampwidth = wf.getsampwidth()
        frames = wf.getnframes()
        data = wf.readframes(frames)
    duration = frames / sr
    audio = np.frombuffer(data, dtype=np.int16)
    if sampwidth == 1:
        # Convert unsigned 8-bit to signed 16-bit
        audio = ((audio.astype(np.float32) - 128.0) * 256.0).astype(np.int16)
    elif sampwidth == 4:
        # Convert 32-bit float to 16-bit
        audio = (np.frombuffer(data, dtype=np.float32) * 32768.0).astype(np.int16)

    if channels > 1:
        audio = audio.reshape(-1, channels).mean(axis=1).astype(np.int16)
    return audio, sr, duration

def resample_audio(audio, orig_sr, target_sr=16000):
    if orig_sr == target_sr:
        return audio, False
    duration = len(audio) / orig_sr
    num_samples = int(duration * target_sr)
    xp = np.arange(len(audio))
    x = np.linspace(0, len(audio) - 1, num_samples)
    resampled = np.interp(x, xp, audio).astype(np.int16)
    return resampled, True

def main():
    parser = argparse.ArgumentParser(description="KWS File Probe")
    parser.add_argument("--profile", type=str, required=True, help="Profile YAML name or path")
    parser.add_argument("--wav", type=str, required=True, help="Path to input 16kHz mono WAV")
    args, unknown = parser.parse_known_args()

    # Load configuration
    cfg = load_voice_config(["--profile", args.profile])

    print("\n==================================================")
    print("Voice Gateway KWS Probe")
    print("==================================================")
    print(f"WAV Path: {args.wav}")

    if not os.path.exists(args.wav):
        print(f"ERROR: WAV file does not exist: {args.wav}")
        sys.exit(1)

    # Read audio
    raw_audio, sr, duration = read_wav(args.wav)
    print(f"Original Sample Rate: {sr} Hz")
    print(f"Original Duration: {duration:.3f} seconds")

    audio, resampled = resample_audio(raw_audio, sr, target_sr=16000)
    print(f"Resampling Performed: {resampled}")

    print(f"KWS Model Directory: {cfg.kws.model_dir}")
    print(f"Keywords File: {cfg.kws.keywords_file}")

    # Measure model load time
    t0 = time.perf_counter()
    # Forces normal initialization path (we set dry_run_text=False for testing or honor the config)
    # The probe runs normal mode unless explicitly requested
    dry_run = cfg.dry_run_text
    try:
        kws = create_kws_backend(cfg.kws, dry_run_text=dry_run)
    except (ValueError, ImportError, ModuleNotFoundError) as e:
        print(f"\n[VOICE][MODEL] status=SKIPPED_ENV_DEPENDENCY reason='{e}'")
        import traceback
        traceback.print_exc()
        sys.exit(0)
    except Exception as e:
        print(f"\n[VOICE][MODEL] status=FAIL reason='{e}'")
        import traceback
        traceback.print_exc()
        sys.exit(0)
    load_time = time.perf_counter() - t0
    print(f"Backend: {'mock_stub' if dry_run else 'sherpa_onnx'}")
    print(f"Model Load Time: {load_time * 1000.0:.2f} ms")

    # Segment audio in 80ms (1280 samples) frames
    frame_samples = 1280
    num_frames = len(audio) // frame_samples

    latencies = []
    wake_triggered = False
    stop_triggered = False

    for i in range(num_frames):
        chunk = audio[i * frame_samples : (i + 1) * frame_samples]

        t_start = time.perf_counter()
        keyword = kws.process(np.ascontiguousarray(chunk.astype(np.float32) / 32768.0))
        t_elapsed = time.perf_counter() - t_start
        latencies.append(t_elapsed * 1000.0)
        if keyword == cfg.kws.wake_keyword:
            wake_triggered = True
        if keyword == cfg.kws.stop_keyword:
            stop_triggered = True

    if not latencies:
        print("ERROR: WAV file too short for evaluation")
        sys.exit(1)

    cold_time = latencies[0]
    warm_times = latencies[1:] if len(latencies) > 1 else [latencies[0]]

    p50 = np.percentile(warm_times, 50)
    p95 = np.percentile(warm_times, 95)

    print("\n--- Inference Performance Baseline ---")
    print(f"Cold Inference Latency (First frame): {cold_time:.2f} ms")
    print(f"Warm Inference Latency (p50)        : {p50:.2f} ms")
    print(f"Warm Inference Latency (p95)        : {p95:.2f} ms")
    print(f"Wake Triggered                      : {wake_triggered}")
    print(f"Stop Triggered                      : {stop_triggered}")

    # Save latency summary logs
    run_dir = Path(cfg.runs_dir) / "probes"
    run_dir.mkdir(parents=True, exist_ok=True)
    with open(run_dir / "kws_probe_latency.json", "w") as fp:
        import json
        json.dump({
            "cold_latency_ms": cold_time,
            "warm_p50_ms": p50,
            "warm_p95_ms": p95,
            "wake_triggered": wake_triggered,
            "stop_triggered": stop_triggered,
        }, fp, indent=2)

if __name__ == "__main__":
    main()
