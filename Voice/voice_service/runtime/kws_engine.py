#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Protocol

import numpy as np

from .common import clamp01, jlog


class KwsBackendError(RuntimeError):
    pass


class KwsBackendFatalError(KwsBackendError):
    pass


class KwsBackend(Protocol):
    """One streaming KWS instance fed by the Voice Gateway capture worker."""

    def process(self, audio_float32: np.ndarray) -> str:
        ...

    def reset(self, reason: str) -> None:
        ...


class SherpaOnnxKwsBackend:
    """Streaming Sherpa-ONNX transducer KWS with one long-lived stream."""

    _ENCODER = "encoder-epoch-13-avg-2-chunk-16-left-64.int8.onnx"
    _DECODER = "decoder-epoch-13-avg-2-chunk-16-left-64.onnx"
    _JOINER = "joiner-epoch-13-avg-2-chunk-16-left-64.int8.onnx"
    _TOKENS = "tokens.txt"

    def __init__(self, cfg: Any, spotter_factory: Any = None):
        model_dir = Path(str(cfg.model_dir))
        keywords_file = Path(str(cfg.keywords_file))
        required = {
            "encoder": model_dir / self._ENCODER,
            "decoder": model_dir / self._DECODER,
            "joiner": model_dir / self._JOINER,
            "tokens": model_dir / self._TOKENS,
            "keywords_file": keywords_file,
        }
        missing = ["{}={}".format(name, path) for name, path in required.items() if not path.is_file()]
        if missing:
            message = "missing Sherpa KWS resources: " + ", ".join(missing)
            jlog({"level": "error", "src": "kws", "event": "KWS_BACKEND_ERROR", "stage": "validate_resources", "exception": message})
            raise KwsBackendError(message)

        try:
            if spotter_factory is None:
                import sherpa_onnx
                spotter_factory = sherpa_onnx.KeywordSpotter
            # Per-keyword score/threshold values are read by Sherpa directly
            # from keywords_file. Do not pass competing global overrides here.
            self.spotter = spotter_factory(
                tokens=str(required["tokens"]),
                encoder=str(required["encoder"]),
                decoder=str(required["decoder"]),
                joiner=str(required["joiner"]),
                keywords_file=str(keywords_file),
                provider=str(cfg.provider),
                num_threads=int(cfg.num_threads),
                max_active_paths=int(cfg.max_active_paths),
                num_trailing_blanks=int(cfg.num_trailing_blanks),
            )
            self.stream = self.spotter.create_stream()
        except Exception as exc:
            jlog({"level": "error", "src": "kws", "event": "KWS_BACKEND_ERROR", "stage": "initialize", "exception": str(exc)})
            raise KwsBackendError("Sherpa KWS initialization failed: {}".format(exc)) from exc

        self.max_consecutive_errors = max(1, int(cfg.max_consecutive_errors))
        self.consecutive_errors = 0
        jlog({
            "level": "info", "src": "kws", "event": "KWS_BACKEND_READY",
            "backend": "sherpa_onnx", "model_dir": str(model_dir),
            "keywords_file": str(keywords_file),
        })

    def reset(self, reason: str) -> None:
        try:
            self.spotter.reset_stream(self.stream)
        except Exception as exc:
            jlog({"level": "error", "src": "kws", "event": "KWS_BACKEND_ERROR", "stage": "reset", "exception": str(exc)})
            try:
                self.stream = self.spotter.create_stream()
            except Exception as recreate_exc:
                raise KwsBackendFatalError("Sherpa KWS stream recreation failed: {}".format(recreate_exc)) from recreate_exc
        jlog({"level": "info", "src": "kws", "event": "KWS_RESET", "reason": str(reason)})

    def process(self, audio_float32: np.ndarray) -> str:
        stage = "accept_waveform"
        try:
            audio = np.ascontiguousarray(audio_float32, dtype=np.float32).reshape(-1)
            self.stream.accept_waveform(16000, audio)
            stage = "decode_stream"
            while self.spotter.is_ready(self.stream):
                self.spotter.decode_stream(self.stream)
            stage = "get_result"
            keyword = str(self.spotter.get_result(self.stream) or "").strip()
            self.consecutive_errors = 0
            if keyword:
                self.reset("keyword_detected")
            return keyword
        except KwsBackendFatalError:
            raise
        except Exception as exc:
            self.consecutive_errors += 1
            jlog({
                "level": "error", "src": "kws", "event": "KWS_BACKEND_ERROR",
                "stage": stage, "exception": str(exc),
                "consecutive_errors": self.consecutive_errors,
            })
            try:
                self.reset("decode_exception")
            except KwsBackendFatalError:
                raise
            if self.consecutive_errors >= self.max_consecutive_errors:
                raise KwsBackendFatalError(
                    "Sherpa KWS failed {} consecutive frames".format(self.consecutive_errors)
                ) from exc
            return ""


class DryRunKwsBackend:
    def process(self, audio_float32: np.ndarray) -> str:
        return ""

    def reset(self, reason: str) -> None:
        return None


def create_kws_backend(cfg: Any, dry_run_text: bool = False) -> KwsBackend:
    backend = str(getattr(cfg, "backend", "") or "").strip().lower()
    if backend != "sherpa_onnx":
        raise KwsBackendError("unsupported production KWS backend: {!r}".format(backend))
    if dry_run_text:
        return DryRunKwsBackend()
    return SherpaOnnxKwsBackend(cfg)

class FlexibleWakeWord:
    def __init__(self, wakeword_models: List[str], vad_threshold: float = 0.0, ncpu: int = 1, dry_run_text: bool = False,
                 frontend_backend: str = "tflite", classifier_backend: str = "onnx"):
        self.dry_run_text = dry_run_text
        self.models: Dict[str, Dict[str, Any]] = {}
        self.vad_threshold = float(vad_threshold)
        self.vad = None
        self.pre = None

        if self.dry_run_text:
            # Create stubs for models in dry-run mode
            for mdl_path in wakeword_models:
                name = Path(mdl_path).stem
                ext = Path(mdl_path).suffix.lower()
                self.models[name] = {
                    "path": mdl_path, "name": name, "ext": ext, "n_calls": 0,
                    "io_kind": "mock", "input_shape": (1, 1, 96), "layout": "standard",
                    "n_feature_frames": 1,
                }
            jlog({"level": "info", "src": "oww", "msg": "FlexibleWakeWord initialized in dry-run-text mode with mock stubs."})
            return

        # Lazy imports for libraries
        import openwakeword
        from openwakeword.utils import AudioFeatures

        try:
            import tflite_runtime.interpreter as tflite
        except Exception:
            tflite = None

        try:
            import onnxruntime as ort
        except Exception:
            ort = None

        if frontend_backend == "tflite":
            if tflite is None:
                raise ImportError("tflite_runtime is required but not installed")
            self.pre = AudioFeatures(inference_framework="tflite", ncpu=ncpu)
        elif frontend_backend == "onnx":
            if ort is None:
                raise ImportError("onnxruntime is required but not installed")
            self.pre = AudioFeatures(inference_framework="onnx", ncpu=ncpu)
        else:
            raise ValueError(f"Unsupported frontend_backend: {frontend_backend}")
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
        if self.dry_run_text:
            for meta in self.models.values():
                meta["n_calls"] = 0
            return

        try:
            if self.pre is not None:
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
        preds: Dict[str, float] = {}
        wanted = set(only) if only is not None else None

        if self.dry_run_text:
            for name in self.models.keys():
                if wanted is not None and name not in wanted:
                    continue
                preds[name] = 0.0
            return preds

        if not isinstance(x, np.ndarray):
            x = np.asarray(x, dtype=np.int16)
        if x.dtype != np.int16:
            x = x.astype(np.int16)

        if self.pre is None:
            return {name: 0.0 for name in self.models.keys()}

        self.pre(x)
        vad_ok = self._vad_gate_ok(x)
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
