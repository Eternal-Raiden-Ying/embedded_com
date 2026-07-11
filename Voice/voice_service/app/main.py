#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
import queue
import threading
import time
from pathlib import Path

from voice_service.config.loader import load_voice_config
from voice_service.runtime.service import run_voice_service, list_audio_devices


def check_models(cfg) -> None:
    # "Normal vs. dry-run models: If model binaries are missing during dry-run-text, the gateway must print a warning and run successfully. In normal mode, it must crash with a structured error."
    missing = []
    
    # We check: KWS wake, KWS stop, VAD, ASR, TTS model files
    to_check = {
        "Wake Word Model": cfg.wake_tflite,
        "Stop Word Model": cfg.stop_tflite,
        "ASR Model Directory": cfg.asr_dir,
    }
    
    if cfg.asr_mode != "online":
        to_check["VAD Model Directory"] = cfg.vad_dir
    if not cfg.disable_tts:
        to_check["TTS Model File"] = cfg.piper_model
        if cfg.piper_config:
            to_check["TTS Config File"] = cfg.piper_config

    for label, path_str in to_check.items():
        if path_str:
            p = Path(path_str)
            if not p.exists():
                missing.append((label, path_str))
        else:
            missing.append((label, "<not configured>"))
                
    if missing:
        lines = [f" - {lbl}: {p_str}" for lbl, p_str in missing]
        msg = "CRITICAL ERROR: Configuration references missing model files or directories:\n" + "\n".join(lines)
        if cfg.dry_run_text:
            # Under dry-run-text mode, print warning and continue
            print("\n" + "=" * 60, file=sys.stderr)
            print("WARNING: Model binaries are missing, but dry-run-text mode is active.", file=sys.stderr)
            print("The service will start using stub mocks.", file=sys.stderr)
            print(msg, file=sys.stderr)
            print("=" * 60 + "\n", file=sys.stderr)
        else:
            # Normal mode: halt execution
            print("\n" + "=" * 60, file=sys.stderr)
            print(msg, file=sys.stderr)
            print("=" * 60 + "\n", file=sys.stderr)
            sys.exit(1)


def run_audio_replay(cfg, replay_path_str: str, send_task: bool) -> None:
    import csv
    from voice_service.runtime.commands import CommandInterpreter
    from voice_service.runtime.asr_engine import AudioCommandPipeline
    from voice_service.runtime.kws_engine import FlexibleWakeWord
    
    interpreter = CommandInterpreter.from_json(cfg.commands_json)
    pipeline = AudioCommandPipeline(cfg, interpreter)
    
    models = []
    if cfg.wake_tflite:
        models.append(cfg.wake_tflite)
    if cfg.stop_tflite:
        models.append(cfg.stop_tflite)
    oww = FlexibleWakeWord(models, vad_threshold=cfg.oww_vad_th, ncpu=1, dry_run_text=cfg.dry_run_text)
    
    replay_path = Path(replay_path_str)
    wav_files = sorted(list(replay_path.rglob("*.wav"))) if replay_path.is_dir() else [replay_path]
    
    print(f"\n--- Audio Replay Mode started: {len(wav_files)} files ---")
    
    results = []
    run_dir = Path(cfg.runs_dir) / f"replay_{int(time.time())}"
    run_dir.mkdir(parents=True, exist_ok=True)
    
    for wf_path in wav_files:
        print(f"\nProcessing file: {wf_path.name}")
        try:
            from voice_service.examples.kws_probe import read_wav, resample_audio
            raw_audio, sr, duration = read_wav(wf_path)
            audio, _ = resample_audio(raw_audio, sr, 16000)
            
            t0 = time.perf_counter()
            frame_samples = 1280
            num_frames = len(audio) // frame_samples
            wake_score, stop_score = 0.0, 0.0
            wake_key = cfg.wake_key or (Path(cfg.wake_tflite).stem if cfg.wake_tflite else "")
            stop_key = cfg.stop_key or (Path(cfg.stop_tflite).stem if cfg.stop_tflite else "")
            
            for i in range(num_frames):
                chunk = audio[i * frame_samples : (i + 1) * frame_samples]
                preds = oww.predict(chunk)
                ws = preds.get(wake_key, 0.0)
                ss = preds.get(stop_key, 0.0)
                if ws > wake_score: wake_score = ws
                if ss > stop_score: stop_score = ss
                
            kws_latency = (time.perf_counter() - t0) * 1000.0
            
            t_asr = time.perf_counter()
            asr_result = pipeline.process_audio(audio)
            asr_latency = (time.perf_counter() - t_asr) * 1000.0
            total_latency = (time.perf_counter() - t0) * 1000.0
            
            print(f"  KWS Wake Score: {wake_score:.3f}, Stop Score: {stop_score:.3f}")
            print(f"  Raw ASR: {asr_result.get('text')}")
            print(f"  Intent: {asr_result.get('intent')}, Target: {asr_result.get('target')}")
            
            if send_task and asr_result.get("status") == "OK":
                print("  Sending TaskCmd to Orchestrator...")
                from voice_service.runtime.service import build_task_sender
                from voice_service.ipc.protocol import build_task_cmd
                task_sender = build_task_sender(cfg)
                cmd_msg = build_task_cmd({
                    "intent": asr_result.get("intent"),
                    "target": asr_result.get("target"),
                    "confidence": asr_result.get("confidence", 0.0),
                    "text": asr_result.get("text", "")
                })
                task_sender.send(cmd_msg)
                task_sender.close()
                
            results.append({
                "file": wf_path.name,
                "expected_wake": "yes",
                "expected_text": wf_path.stem,
                "expected_intent": asr_result.get("intent"),
                "expected_target": asr_result.get("target"),
                "wake_score": f"{wake_score:.3f}",
                "stop_score": f"{stop_score:.3f}",
                "raw_asr": asr_result.get("text"),
                "normalized_asr": asr_result.get("text"),
                "actual_intent": asr_result.get("intent"),
                "actual_target": asr_result.get("target"),
                "kws_latency_ms": f"{kws_latency:.2f}",
                "asr_latency_ms": f"{asr_latency:.2f}",
                "total_latency_ms": f"{total_latency:.2f}",
                "result": asr_result.get("status"),
                "error": ""
            })
        except Exception as e:
            print(f"  Error processing {wf_path.name}: {e}")
            results.append({
                "file": wf_path.name,
                "expected_wake": "unknown",
                "expected_text": "",
                "expected_intent": "",
                "expected_target": "",
                "wake_score": "0.0",
                "stop_score": "0.0",
                "raw_asr": "",
                "normalized_asr": "",
                "actual_intent": "",
                "actual_target": "",
                "kws_latency_ms": "0.0",
                "asr_latency_ms": "0.0",
                "total_latency_ms": "0.0",
                "result": "ERROR",
                "error": str(e)
            })
            
    if replay_path.is_dir():
        csv_file = run_dir / "audio_replay_results.csv"
        print(f"\nWriting replay results to CSV: {csv_file}")
        fields = ["file", "expected_wake", "expected_text", "expected_intent", "expected_target",
                  "wake_score", "stop_score", "raw_asr", "normalized_asr", "actual_intent",
                  "actual_target", "kws_latency_ms", "asr_latency_ms", "total_latency_ms", "result", "error"]
        with open(csv_file, "w", newline="", encoding="utf-8") as fp:
            writer = csv.DictWriter(fp, fieldnames=fields)
            writer.writeheader()
            writer.writerows(results)

def main():
    # Force UTF-8 input/output streams for cross-platform compatibility
    if sys.platform == "win32":
        import io
        try:
            sys.stdin = io.TextIOWrapper(sys.stdin.buffer, encoding="utf-8")
            sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
            sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8")
        except Exception:
            pass
    elif hasattr(sys.stdin, "reconfigure"):
        try:
            sys.stdin.reconfigure(encoding="utf-8")
            sys.stdout.reconfigure(encoding="utf-8")
            sys.stderr.reconfigure(encoding="utf-8")
        except Exception:
            pass

    # If first argument is 'list-devices', run list_audio_devices and exit
    if len(sys.argv) > 1 and sys.argv[1] == "list-devices":
        sys.exit(list_audio_devices())

    # Check CLI flags before loading models or opening audio devices.
    import argparse
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--profile", type=str, default=None)
    parser.add_argument("--inspect-config", action="store_true", default=False)
    parser.add_argument("--list-audio-devices", action="store_true", default=False)
    parser.add_argument("--audio-replay", type=str, default=None)
    parser.add_argument("--send-task", action="store_true", default=False)
    parser.add_argument("--no-task-send", action="store_true", default=False)
    cli_args, remaining = parser.parse_known_args()

    if cli_args.list_audio_devices:
        sys.exit(list_audio_devices())

    cfg = load_voice_config()
    if cli_args.inspect_config:
        for key, value in sorted(vars(cfg).items()):
            print("{}={}".format(key, value))
        return
    check_models(cfg)

    if cli_args.audio_replay:
        run_audio_replay(cfg, cli_args.audio_replay, cli_args.send_task and not cli_args.no_task_send)
        sys.exit(0)

    stop_event = threading.Event()
    utter_q = queue.Queue(maxsize=128)

    if cfg.dry_run_text:
        print("\n" + "*" * 60)
        print("VOICE GATEWAY STARTING IN DRY-RUN INTERACTIVE TEXT MODE")
        print("You can type simulated voice command sentences below.")
        print("Type 'exit', 'quit', or send EOF to shutdown.")
        print("*" * 60 + "\n")

        # Start the runtime service in a daemon thread
        svc_thread = threading.Thread(
            target=run_voice_service,
            args=(cfg, stop_event, utter_q),
            daemon=True,
            name="voice_service_backend"
        )
        svc_thread.start()

        # Stdin reading loop in the main thread
        try:
            while not stop_event.is_set():
                # Print prompt
                sys.stdout.write("Voice Command > ")
                sys.stdout.flush()
                
                # Check for input or shutdown
                line = sys.stdin.readline()
                if not line:
                    # EOF
                    break
                line = line.strip()
                if line.lower() in ("exit", "quit"):
                    break
                if not line:
                    continue
                    
                # Put the command text into the queue with wildcard epoch
                utter_q.put({"kind": "TEXT", "text": line, "epoch": -1})
                # Pause briefly to allow the worker thread to process and print logs
                time.sleep(0.4)
        except (KeyboardInterrupt, SystemExit):
            pass
        finally:
            print("\nStopping Voice Gateway...")
            stop_event.set()
            svc_thread.join(timeout=2.0)
            print("Goodbye.")
    else:
        # Run in foreground synchronously
        run_voice_service(cfg, stop_event, utter_q)


if __name__ == "__main__":
    main()
