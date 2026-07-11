#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
import re
import sys
import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

SR = 16000
FRAME_MS = 80
FRAME_SAMPLES = SR * FRAME_MS // 1000
FRAME_BYTES = FRAME_SAMPLES * 2
MIN_UTT_MS = 200

LOG_MODE = "concise"
QUIET_MIC_INFO = True
_RUN_DIR: Optional[Path] = None
_LOCK = threading.Lock()


def configure_logging(mode: str = "concise", quiet_mic_info: bool = True):
    global LOG_MODE, QUIET_MIC_INFO
    LOG_MODE = mode
    QUIET_MIC_INFO = quiet_mic_info


def configure_artifact_logging(runs_root: str) -> str:
    global _RUN_DIR
    ts = time.strftime("run_%Y%m%d_%H%M%S")
    _RUN_DIR = Path(runs_root) / ts
    _RUN_DIR.mkdir(parents=True, exist_ok=True)
    return str(_RUN_DIR)


def current_run_dir() -> str:
    return str(_RUN_DIR) if _RUN_DIR is not None else ""


def _append_jsonl(name: str, payload: Dict[str, Any]):
    if _RUN_DIR is None:
        return
    path = _RUN_DIR / f"{name}.jsonl"
    with _LOCK, open(path, "a", encoding="utf-8") as fp:
        fp.write(json.dumps(payload, ensure_ascii=False, separators=(",", ":")) + "\n")


def write_named_jsonl(name: str, payload: Dict[str, Any]):
    _append_jsonl(name, payload)


def write_timeline(event: str, **fields):
    payload = {"ts": time.time(), "event": event}
    payload.update(fields)
    _append_jsonl("timeline", payload)


def write_ipc_event(event: str, **fields):
    payload = {"ts": time.time(), "event": event}
    payload.update(fields)
    _append_jsonl("ipc", payload)


def write_state_block(block: Dict[str, Any]):
    _append_jsonl("state_blocks", block)


def write_stop_trace(event: str, **fields):
    payload = {"ts": time.time(), "event": event}
    payload.update(fields)
    _append_jsonl("stop_trace", payload)


def write_config_snapshot(payload: Dict[str, Any]):
    _append_jsonl("config", payload)


def should_emit(payload: Dict[str, Any]) -> bool:
    level = str(payload.get("level", "info"))
    src = str(payload.get("src", ""))
    if LOG_MODE == "full":
        return True
    if level in {"error", "warn"}:
        return True
    if level == "debug":
        return False
    keep_info_src = {
        "boot", "loop", "oww", "seg", "decision", "tts", "tts_event",
        "heartbeat", "signal", "queue", "ipc", "state", "stop", "asr_partial",
    }
    if src == "mic":
        return not QUIET_MIC_INFO and level == "info"
    return src in keep_info_src


def jlog(payload: Dict[str, Any]):
    _append_jsonl("console", dict(payload))
    if not should_emit(payload):
        return
    s = json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
    print(s, file=sys.stderr, flush=True)


def clamp01(x: float) -> float:
    return float(max(0.0, min(1.0, x)))


def rms_int16(x: np.ndarray) -> float:
    xf = x.astype(np.float32)
    return float(np.sqrt(np.mean(xf * xf) + 1e-12))


_EMPTY_BRACKET_RE = re.compile(r"(?:\[\s*\]|（\s*）|\(\s*\)|\{\s*\}|【\s*】|<\s*>)")
_FILLER_PUNCT_RE = re.compile(r"^[,，.。!！?？;；、~～]+|[,，.。!！?？;；、~～]+$")


def clean_asr_text(s: str) -> str:
    t = str(s or "")
    t = _EMPTY_BRACKET_RE.sub("", t)
    t = t.replace("[", "").replace("]", "")
    t = t.replace("【", "").replace("】", "")
    t = t.replace("（", "").replace("）", "")
    t = t.replace("(", "").replace(")", "")
    t = re.sub(r"\s+", "", t)
    t = _FILLER_PUNCT_RE.sub("", t)
    return t.strip()


def normalize_text(s: str) -> str:
    return clean_asr_text(s)


def infer_oww_key(model_path: str) -> str:
    return Path(model_path).stem


def kws_trigger(pred: Dict[str, float], key: str, th: float) -> bool:
    return float(pred.get(key, 0.0)) >= th


def model_file_exists(model_dir: str, filename: str) -> bool:
    return Path(model_dir, filename).exists()


def auto_quant_flag(model_dir: str, want_quant: bool, role: str) -> bool:
    has_model = model_file_exists(model_dir, "model.onnx")
    has_quant = model_file_exists(model_dir, "model_quant.onnx")
    if want_quant:
        if has_quant:
            return True
        if has_model:
            print(f"[{role}] NOTE: model_quant.onnx missing -> use model.onnx", file=sys.stderr)
            return False
    else:
        if has_model:
            return False
        if has_quant:
            print(f"[{role}] NOTE: model.onnx missing -> use model_quant.onnx", file=sys.stderr)
            return True
    raise RuntimeError(f"[{role}] need model.onnx or model_quant.onnx in {model_dir}")


def normalize_vad_segments(raw: Any) -> List[Tuple[float, float]]:
    if isinstance(raw, list) and len(raw) == 1 and isinstance(raw[0], list):
        raw = raw[0]
    segs: List[Tuple[float, float]] = []
    if not isinstance(raw, list):
        return segs
    for s in raw:
        if isinstance(s, (list, tuple)) and len(s) >= 2:
            try:
                beg, end = float(s[0]), float(s[1])
            except Exception:
                continue
            if end > beg:
                segs.append((beg, end))
    return segs


def pick_best_segment(segs: List[Tuple[float, float]]) -> Optional[Tuple[float, float]]:
    if not segs:
        return None
    return max(segs, key=lambda x: x[1] - x[0])


def to_sample_index_ms(x_ms: float) -> int:
    return int(x_ms * SR / 1000.0)


def parse_asr_output(asr_out: Any) -> Tuple[str, Optional[float]]:
    obj = asr_out
    if isinstance(obj, list) and obj:
        obj = obj[0]
    text = ""
    conf = None
    if isinstance(obj, dict):
        if "preds" in obj:
            preds = obj["preds"]
            if isinstance(preds, (list, tuple)) and len(preds) >= 1:
                text = str(preds[0])
            else:
                text = str(preds)
        else:
            text = str(obj.get("text", obj.get("pred", obj.get("result", ""))))
        for k in ("confidence", "conf", "score", "prob"):
            if k in obj:
                try:
                    conf = float(obj[k])
                except Exception:
                    conf = None
                break
    else:
        text = str(obj)
    return clean_asr_text(text), conf
