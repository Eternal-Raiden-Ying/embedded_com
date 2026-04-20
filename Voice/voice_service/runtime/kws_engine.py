#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import numpy as np
import openwakeword
from openwakeword.utils import AudioFeatures

from .common import clamp01, jlog

try:
    import tflite_runtime.interpreter as tflite
except Exception:
    tflite = None

try:
    import onnxruntime as ort
except Exception:
    ort = None


class FlexibleWakeWord:
    def __init__(self, wakeword_models: List[str], vad_threshold: float = 0.0, ncpu: int = 1):
        self.pre = AudioFeatures(inference_framework="tflite", ncpu=ncpu)
        self.models: Dict[str, Dict[str, Any]] = {}
        self.vad_threshold = float(vad_threshold)
        self.vad = None
        if self.vad_threshold > 0:
            try:
                self.vad = openwakeword.VAD()
            except Exception as e:
                jlog({"level": "warn", "src": "oww", "msg": f"VAD init failed, disabled: {e}"})
                self.vad = None
                self.vad_threshold = 0.0

        for mdl_path in wakeword_models:
            name = Path(mdl_path).stem
            ext = Path(mdl_path).suffix.lower()
            meta: Dict[str, Any] = {"path": mdl_path, "name": name, "ext": ext, "n_calls": 0}
            if ext == ".tflite":
                if tflite is None:
                    raise RuntimeError("tflite_runtime is required for .tflite wake models")
                itp = tflite.Interpreter(model_path=mdl_path, num_threads=1)
                itp.allocate_tensors()
                inp = itp.get_input_details()[0]
                out = itp.get_output_details()[0]
                shape = tuple(int(i) for i in inp["shape"])
                meta.update({
                    "runner": itp, "input_index": inp["index"], "output_index": out["index"],
                    "input_name": None, "output_name": None, "io_kind": "tflite", "input_shape": shape,
                })
            elif ext == ".onnx":
                if ort is None:
                    raise RuntimeError("onnxruntime is required for .onnx wake models")
                sess = ort.InferenceSession(mdl_path, providers=["CPUExecutionProvider"])
                inp = sess.get_inputs()[0]
                out = sess.get_outputs()[0]
                shape = tuple(int(i) if isinstance(i, (int, np.integer)) else -1 for i in inp.shape)
                meta.update({
                    "runner": sess, "input_index": None, "output_index": None,
                    "input_name": inp.name, "output_name": out.name, "io_kind": "onnx", "input_shape": shape,
                })
            else:
                raise RuntimeError(f"Unsupported wake model extension for {mdl_path}: {ext}")

            shape = meta["input_shape"]
            if len(shape) != 3 or shape[0] != 1:
                raise RuntimeError(f"Unsupported wake model input shape for {mdl_path}: {shape}")
            if shape[2] == 96:
                layout = "standard"
                n_feature_frames = shape[1]
            elif shape[1] == 96:
                layout = "transposed"
                n_feature_frames = shape[2]
            else:
                raise RuntimeError(f"Wake model {mdl_path} not compatible: {shape}")
            meta.update({"layout": layout, "n_feature_frames": int(n_feature_frames)})
            self.models[name] = meta
            jlog({
                "level": "info", "src": "oww", "model": name, "ext": ext,
                "input_shape": list(shape), "layout": layout, "n_feature_frames": int(n_feature_frames),
                "backend": meta["io_kind"],
            })

    def reset(self):
        try:
            self.pre.reset()
        except Exception:
            pass
        if self.vad is not None:
            try:
                self.vad.reset()
            except Exception:
                pass
        for meta in self.models.values():
            meta["n_calls"] = 0

    def _vad_gate_ok(self, x: np.ndarray) -> bool:
        if self.vad is None or self.vad_threshold <= 0:
            return True
        try:
            self.vad(x)
            vad_frames = list(self.vad.prediction_buffer)[-7:-4]
            vad_max_score = float(np.max(vad_frames)) if len(vad_frames) > 0 else 0.0
            return vad_max_score >= self.vad_threshold
        except Exception:
            return True

    def predict(self, x: np.ndarray, only: Optional[Iterable[str]] = None) -> Dict[str, float]:
        if not isinstance(x, np.ndarray):
            x = np.asarray(x, dtype=np.int16)
        if x.dtype != np.int16:
            x = x.astype(np.int16)
        self.pre(x)
        preds: Dict[str, float] = {}
        vad_ok = self._vad_gate_ok(x)
        wanted = set(only) if only is not None else None
        for name, meta in self.models.items():
            if wanted is not None and name not in wanted:
                continue
            meta["n_calls"] += 1
            feat = self.pre.get_features(meta["n_feature_frames"])
            if feat.shape[1] != meta["n_feature_frames"]:
                preds[name] = 0.0
                continue
            if meta["layout"] == "transposed":
                feat = np.transpose(feat, (0, 2, 1))
            feat = feat.astype(np.float32)
            if meta["io_kind"] == "tflite":
                meta["runner"].set_tensor(meta["input_index"], feat)
                meta["runner"].invoke()
                out = meta["runner"].get_tensor(meta["output_index"])
            else:
                out = meta["runner"].run([meta["output_name"]], {meta["input_name"]: feat})[0]
            score = float(np.array(out).reshape(-1)[0])
            if meta["n_calls"] < 5:
                score = 0.0
            if not vad_ok:
                score = 0.0
            preds[name] = clamp01(score)
        return preds
