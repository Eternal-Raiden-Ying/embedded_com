#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import re
import subprocess
import threading
import time
import wave
from pathlib import Path
from typing import List, Optional

from .common import SR, jlog

try:
    from piper.voice import PiperVoice
except Exception:
    PiperVoice = None


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
                jlog({"level": "warn", "src": "tts", "msg": f"Piper load failed: {e}"})
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
                    jlog({"level": "warn", "src": "tts", "msg": f"warmup failed: {e}", "text": t})


class ThreadSafeTTS:
    def __init__(self, inner: PiperTTS):
        self.inner = inner
        self._lock = threading.Lock()

    def say(self, text: str) -> Optional[Path]:
        with self._lock:
            return self.inner.say(text)

    def warmup_phrases(self, phrases: List[str]):
        with self._lock:
            return self.inner.warmup_phrases(phrases)
