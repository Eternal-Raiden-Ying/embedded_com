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
from voice_service.runtime.asr_engine import AudioCommandPipeline
from voice_service.runtime.commands import CommandInterpreter
from voice_service.runtime.service import build_task_sender

def read_wav_mono_16k(wav_path):
    with wave.open(str(wav_path), "rb") as wf:
        sr = wf.getframerate()
        channels = wf.getnchannels()
        sampwidth = wf.getsampwidth()
        frames = wf.getnframes()
        data = wf.readframes(frames)
    duration = frames / sr
    audio = np.frombuffer(data, dtype=np.int16)
    if channels > 1:
        audio = audio.reshape(-1, channels).mean(axis=1).astype(np.int16)
    if sr != 16000:
        # Simple linear interpolation
        num_samples = int(duration * 16000)
        xp = np.arange(len(audio))
        x = np.linspace(0, len(audio) - 1, num_samples)
        audio = np.interp(x, xp, audio).astype(np.int16)
    return audio, duration

def get_process_metrics():
    cpu_time = "n/a"
    rss_mb = "n/a"
    try:
        import psutil
        p = psutil.Process(os.getpid())
        rss_mb = f"{p.memory_info().rss / (1024 * 1024):.2f} MB"
        cpu_time = f"{p.cpu_times().user + p.cpu_times().system:.3f} s"
    except ImportError:
        # Fall back to standard resource module on Unix
        try:
            import resource
            usage = resource.getrusage(resource.RUSAGE_SELF)
            cpu_time = f"{usage.ru_utime + usage.ru_stime:.3f} s"
            # on Linux, ru_maxrss is in KB
            rss_mb = f"{usage.ru_maxrss / 1024:.2f} MB"
        except Exception:
            pass
    return rss_mb, cpu_time


def percentile_ms(values, percentile):
    if not values:
        return 0.0
    return float(np.percentile(np.asarray(values, dtype=np.float64), percentile))

def main():
    parser = argparse.ArgumentParser(description="ASR File Probe")
    parser.add_argument("--profile", type=str, required=True, help="Profile name or YAML path")
    parser.add_argument("--wav", type=str, required=True, help="Input WAV path")
    parser.add_argument("--mode", type=str, choices=["online", "offline"], default="offline", help="ASR Mode")
    parser.add_argument("--send-task", action="store_true", default=False, help="Connect and send TaskCmd to Orchestrator")
    parser.add_argument("--repeat", type=int, default=1, help="Warm inference repetitions on one loaded pipeline")
    parser.add_argument("--warmup", type=int, default=0, help="Discarded silence warmups on one loaded pipeline")
    args = parser.parse_known_args()[0]

    # Keeps numba/librosa compilation cache outside the repository on SC171.
    os.environ.setdefault("NUMBA_CACHE_DIR", "/tmp/voice_numba_cache")
    # Overrides config's asr_mode
    overrides = ["--profile", args.profile, "--asr_mode", args.mode]
    cfg = load_voice_config(overrides)

    print("\n==================================================")
    print("Voice Gateway ASR Probe")
    print("==================================================")
    print(f"ASR Mode: {args.mode}")
    print(f"WAV Path: {args.wav}")

    if not os.path.exists(args.wav):
        print(f"ERROR: WAV file does not exist: {args.wav}")
        sys.exit(1)

    audio, duration = read_wav_mono_16k(args.wav)
    print(f"Audio Duration: {duration:.3f} seconds")

    interpreter = CommandInterpreter.from_json(cfg.commands_json)

    # Check companion files
    from voice_service.examples.inspect_models import check_companion_files

    vad_missing = check_companion_files(Path(cfg.vad_dir), ["config.yaml", "configuration.json", "am.mvn"]) if cfg.vad_dir else []
    asr_missing = check_companion_files(Path(cfg.asr_dir), ["tokens.json", "config.yaml", "configuration.json", "am.mvn"]) if cfg.asr_dir else []

    if (cfg.vad_dir and vad_missing) or (cfg.asr_dir and asr_missing):
        missing_all = vad_missing + asr_missing
        print(f"\n[VOICE][MODEL] status=BLOCKED_MODEL_LAYOUT reason='Missing required companion files: {missing_all}'")
        sys.exit(0)

    # Measure ASR pipeline load time
    t0 = time.perf_counter()
    try:
        pipeline = AudioCommandPipeline(cfg, interpreter)
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

    print(f"Backend ASR Model: {cfg.asr_dir}")
    print(f"Backend VAD Model: {cfg.vad_dir}")
    print(f"model_load_ms    : {load_time * 1000.0:.2f}")

    def infer_once():
        if args.mode == "online":
            session = pipeline.start_stream_session()
            frame_samples = 1280
            num_frames = max(1, len(audio) // frame_samples)
            for i in range(num_frames):
                chunk = audio[i * frame_samples : (i + 1) * frame_samples]
                pipeline.stream_feed(session, chunk, is_final=(i == num_frames - 1))
            return pipeline.finalize_stream_result(session)
        return pipeline.process_audio(audio)

    try:
        for _ in range(max(0, int(args.warmup))):
            pipeline.warmup()
    except Exception as e:
        print("[VOICE][WARMUP] status=FAIL reason={!r}".format(e))
        sys.exit(1)

    t_start = time.perf_counter()
    result = infer_once()
    cold_inference_ms = (time.perf_counter() - t_start) * 1000.0
    warm_latencies_ms = []
    for _ in range(max(1, int(args.repeat))):
        t_start = time.perf_counter()
        result = infer_once()
        warm_latencies_ms.append((time.perf_counter() - t_start) * 1000.0)
    inference_time = warm_latencies_ms[-1] / 1000.0
    rtf = inference_time / duration if duration > 0 else 0.0

    print("\n--- Transcription Result ---")
    print(f"Status        : {result.get('status')}")
    print(f"Raw ASR       : {result.get('text')}")
    print(f"Intent        : {result.get('intent')}")
    print(f"Target        : {result.get('target')}")
    print(f"Confidence    : {result.get('confidence')}")
    print(f"ASR Conf      : {result.get('asr_confidence')}")

    print("\n--- Inference Performance Baseline ---")
    print(f"cold_inference_ms: {cold_inference_ms:.2f}")
    print(f"warm_p50_ms      : {percentile_ms(warm_latencies_ms, 50):.2f}")
    print(f"warm_p95_ms      : {percentile_ms(warm_latencies_ms, 95):.2f}")
    print(f"RTF              : {rtf:.4f}")

    rss, cpu = get_process_metrics()
    print(f"peak_rss        : {rss}")
    print(f"CPU Time      : {cpu}")

    # Send task cmd to Orchestrator if send_task is active
    if args.send_task:
        print("\nSending TaskCmd to Orchestrator...")
        from voice_service.ipc.protocol import build_task_cmd
        task_sender = build_task_sender(cfg)
        payload = {
            "intent": result.get("intent"),
            "target": result.get("target"),
            "confidence": result.get("confidence", 0.0),
            "text": result.get("text", "")
        }
        cmd_msg = build_task_cmd(payload)
        sent = task_sender.send(cmd_msg)
        print(f"TaskCmd Sent Status: {sent}")
        task_sender.close()
    else:
        print("\nNote: Orchestrator dispatch skipped (--send-task flag not active)")

    # Save latency summary logs
    run_dir = Path(cfg.runs_dir) / "probes"
    run_dir.mkdir(parents=True, exist_ok=True)
    with open(run_dir / "asr_probe_latency.json", "w") as fp:
        import json
        json.dump({
            "model_load_ms": load_time * 1000.0,
            "cold_inference_ms": cold_inference_ms,
            "warm_latencies_ms": warm_latencies_ms,
            "rtf": rtf,
            "raw_text": result.get("text"),
            "intent": result.get("intent"),
            "target": result.get("target"),
            "status": result.get("status"),
        }, fp, indent=2)

if __name__ == "__main__":
    main()
