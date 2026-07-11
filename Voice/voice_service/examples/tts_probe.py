#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
import time
import wave
import argparse
from pathlib import Path

from voice_service.config.loader import load_voice_config
from voice_service.config.paths import resolve_path
from voice_service.runtime.tts_engine import PiperTTS

def get_wav_duration_and_sr(wav_path):
    try:
        with wave.open(str(wav_path), "rb") as wf:
            sr = wf.getframerate()
            frames = wf.getnframes()
            return frames / sr, sr
    except Exception:
        return 0.0, 16000

def main():
    parser = argparse.ArgumentParser(description="TTS Synthesis Probe")
    parser.add_argument("--profile", type=str, required=True, help="Profile name or YAML path")
    parser.add_argument("--text", type=str, default="我在，请说出要拿的物品。", help="Text to synthesize")
    parser.add_argument("--output", type=str, default="Voice/runs/tts_probe.wav", help="Output WAV path")
    parser.add_argument("--play", action="store_true", default=False, help="Play audio after synthesis")
    parser.add_argument("--no-play", action="store_true", default=False, help="Explicitly disable playback")
    args = parser.parse_known_args()[0]

    # Honor explicit --no-play flag
    play_audio = args.play and not args.no_play

    # Load configuration
    cfg = load_voice_config(["--profile", args.profile])

    print("\n==================================================")
    print("Voice Gateway TTS Probe")
    print("==================================================")
    print(f"Text to Synthesize: {args.text}")

    output_path = resolve_path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Check companion file
    if cfg.piper_model:
        tts_model_path = Path(cfg.piper_model)
        tts_json_path = tts_model_path.parent / (tts_model_path.name + ".json")
        if tts_model_path.exists() and not tts_json_path.exists():
            print(f"\n[VOICE][MODEL] status=BLOCKED_MODEL_LAYOUT reason='Missing TTS companion JSON config: {tts_json_path.name}'")
            sys.exit(0)

    # Measure model load time
    t0 = time.perf_counter()
    try:
        tts = PiperTTS(
            model_path=cfg.piper_model,
            cache_dir=cfg.tts_cache,
            output_dir=str(output_path.parent),
            mode="save",  # Save to output file
            play_cmd=cfg.play_cmd,
            dry_run_text=cfg.dry_run_text
        )
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

    print(f"Backend            : {'mock_stub' if cfg.dry_run_text else 'piper'}")
    print(f"Model Path         : {cfg.piper_model}")
    print(f"Config Path        : {cfg.piper_config}")
    print(f"Model Load Time    : {load_time * 1000.0:.2f} ms")

    # Synthesis
    t_synth = time.perf_counter()
    # In piper mode "save", say() synthesizes to a file and returns path
    # Let's ensure we synthesize exactly to the requested output file!
    # Our PiperTTS.say() uses cache or returns filename. To force output name:
    # let's call synthesize_to_file helper or mock
    if cfg.dry_run_text:
        # Create a blank/dummy wav file representing output
        with wave.open(str(output_path), "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(16000)
            wf.writeframes(bytes(32000))  # 1 second of silence
        res_file = str(output_path)
    else:
        # PiperTTS has standard say synthesis
        res_file = tts.say(args.text)
        if res_file and os.path.exists(res_file) and res_file != str(output_path):
            # Rename or copy to output path
            import shutil
            shutil.copy(res_file, str(output_path))
            res_file = str(output_path)

    synth_time = time.perf_counter() - t_synth

    duration, sr = get_wav_duration_and_sr(output_path)
    print(f"Synthesis Time     : {synth_time * 1000.0:.2f} ms")
    print(f"WAV Output Path    : {output_path}")
    print(f"Output Sample Rate : {sr} Hz")
    print(f"Audio Duration     : {duration:.3f} seconds")

    playback_backend = cfg.play_cmd
    playback_result = "SKIPPED"

    if play_audio:
        print(f"Playback started using backend: {playback_backend}")
        t_play = time.perf_counter()
        if cfg.dry_run_text:
            print(f"[TTS MOCK PLAYBACK] {args.text}")
            playback_result = "SUCCESS"
        else:
            # Run command aplay
            import subprocess
            cmd = playback_backend.split() + [str(output_path)]
            try:
                res = subprocess.run(cmd, check=False)
                playback_result = "SUCCESS" if res.returncode == 0 else f"ERROR: returncode={res.returncode}"
            except Exception as e:
                playback_result = f"ERROR: {e}"
        print(f"Playback Result    : {playback_result} (started in {(time.perf_counter() - t_play)*1000.0:.2f} ms)")
    else:
        print("Playback Result    : SKIPPED (use --play flag to enable)")

    # Save latency summary logs
    run_dir = Path(cfg.runs_dir) / "probes"
    run_dir.mkdir(parents=True, exist_ok=True)
    with open(run_dir / "tts_probe_latency.json", "w") as fp:
        import json
        json.dump({
            "load_time_ms": load_time * 1000.0,
            "synthesis_time_ms": synth_time * 1000.0,
            "audio_duration_s": duration,
            "sample_rate_hz": sr,
            "playback_backend": playback_backend,
            "playback_result": playback_result,
        }, fp, indent=2)

if __name__ == "__main__":
    main()
