#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
import time
import wave
import subprocess
import shutil
import argparse
from pathlib import Path
import numpy as np

# Fix python path dynamically if run directly
current = Path(__file__).resolve().parent
for parent in [current] + list(current.parents):
    if (parent / "voice_service").exists():
        sys.path.insert(0, str(parent))
        break

from voice_service.config.loader import load_voice_config

def check_command_exists(cmd):
    return shutil.which(cmd) is not None

def print_subprocess_output(cmd):
    try:
        res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, check=False)
        print(res.stdout)
        if res.stderr:
            print("  Stderr:", res.stderr.strip())
    except Exception as e:
        print(f"  Error executing {cmd}: {e}")

def run_5sec_record_test(device, out_path):
    print(f"\n--- Starting 5-Second Recording Test using {device} ---")
    if not check_command_exists("arecord"):
        print("ERROR: arecord binary not available on this system.")
        return False
        
    cmd = [
        "arecord", "-D", device,
        "-d", "5", "-f", "S16_LE", "-r", "16000", "-c", "1",
        str(out_path)
    ]
    print(f"Executing: {' '.join(cmd)}")
    t0 = time.time()
    try:
        res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=8.0, check=False)
        duration = time.time() - t0
        print(f"Recording finished in {duration:.2f} seconds.")
        if res.returncode != 0:
            print(f"ERROR: arecord returned non-zero exit code: {res.returncode}")
            print(res.stderr.decode("utf-8", errors="ignore"))
            return False
    except subprocess.TimeoutExpired:
        print("ERROR: arecord command timed out.")
        return False
    except Exception as e:
        print(f"ERROR: failed to execute arecord: {e}")
        return False
        
    # Read recording and analyze stats
    if not out_path.exists():
        print("ERROR: Output recording WAV file was not created.")
        return False
        
    try:
        with wave.open(str(out_path), "rb") as wf:
            frames = wf.getnframes()
            data = wf.readframes(frames)
        audio = np.frombuffer(data, dtype=np.int16)
        if len(audio) == 0:
            print("ERROR: Recorded file is empty.")
            return False
            
        peak = np.max(np.abs(audio))
        rms = np.sqrt(np.mean(audio.astype(np.float64) ** 2))
        
        # Judge if silent or clipped
        silent = rms < 10.0
        clipped = peak >= 32767
        
        print("\n--- Audio Stats ---")
        print(f"  Total samples  : {len(audio)}")
        print(f"  Peak Amplitude : {peak}")
        print(f"  RMS Amplitude  : {rms:.2f}")
        print(f"  Silent         : {silent} (RMS < 10.0)")
        print(f"  Clipped        : {clipped} (Peak >= 32767)")
        return True
    except Exception as e:
        print(f"ERROR: Failed to analyze recording stats: {e}")
        return False

def run_play_test(play_cmd, test_wav):
    print(f"\n--- Starting Playback Test ---")
    if not os.path.exists(test_wav):
        print(f"Creating a 1-second dummy beep wav at {test_wav}...")
        # Create a simple 440Hz sine wave
        sr = 16000
        t = np.linspace(0, 1.0, sr, endpoint=False)
        data = (np.sin(2 * np.pi * 440 * t) * 10000).astype(np.int16)
        with wave.open(test_wav, "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(sr)
            wf.writeframes(data.tobytes())
            
    print(f"Playing WAV: {test_wav}")
    cmd = play_cmd.split() + [test_wav]
    print(f"Executing: {' '.join(cmd)}")
    try:
        res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)
        if res.returncode == 0:
            print("Playback succeeded.")
            return True
        else:
            print(f"ERROR: Playback command returned non-zero code: {res.returncode}")
            return False
    except Exception as e:
        print(f"ERROR: failed to execute playback: {e}")
        return False

def main():
    parser = argparse.ArgumentParser(description="Audio Device Probe")
    parser.add_argument("--profile", type=str, required=True, help="Profile name or YAML path")
    parser.add_argument("--record-test", action="store_true", default=False, help="Perform a 5-second record test")
    parser.add_argument("--play-test", action="store_true", default=False, help="Perform a playback test")
    args = parser.parse_known_args()[0]
    
    cfg = load_voice_config(["--profile", args.profile])
    
    print("\n==================================================")
    print("Voice Gateway Audio Device Probe")
    print("==================================================")
    print(f"arecord path: {shutil.which('arecord') or 'not found'}")
    print(f"aplay path: {shutil.which('aplay') or 'not found'}")
    
    print("\n[ arecord -l ]")
    print_subprocess_output(["arecord", "-l"])
    
    print("\n[ arecord -L ]")
    print_subprocess_output(["arecord", "-L"])
    
    print("\n[ aplay -l ]")
    print_subprocess_output(["aplay", "-l"])
    
    print("\n[ aplay -L ]")
    print_subprocess_output(["aplay", "-L"])
    
    print("\n[ Configuration Settings ]")
    print(f"  Configured Input Device  : {cfg.arecord_device}")
    print(f"  Configured Playback Cmd  : {cfg.play_cmd}")
    
    # 5-second record test
    if args.record_test:
        out_wav = Path(cfg.runs_dir) / "probes" / "record_test.wav"
        out_wav.parent.mkdir(parents=True, exist_ok=True)
        run_5sec_record_test(cfg.arecord_device, out_wav)
    else:
        print("\nNote: Recording test skipped (use --record-test to run)")
        
    # Play test
    if args.play_test:
        test_wav = str(Path(cfg.runs_dir) / "probes" / "playback_test_beep.wav")
        run_play_test(cfg.play_cmd, test_wav)
    else:
        print("Note: Playback test skipped (use --play-test to run)")

if __name__ == "__main__":
    main()
