#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from pathlib import Path
from typing import Dict, Any, List

from voice_service.config.loader import load_voice_config
from voice_service.config.paths import resolve_path

def check_companion_files(base_path: Path, expected: List[str]) -> List[str]:
    missing = []
    if not base_path.exists():
        return expected
    for f in expected:
        # If ASR/VAD directory
        if base_path.is_dir():
            target = base_path / f
        else:
            # If TTS file, companion is in the same folder with .json appended
            target = base_path.parent / f
        if not target.exists():
            missing.append(f)
    return missing

def inspect_onnx_metadata(onnx_path: Path) -> Dict[str, Any]:
    meta = {
        "opset": "unknown",
        "inputs": [],
        "outputs": [],
        "providers": [],
        "selected_provider": "none",
        "status": "PASS",
        "error": None
    }
    
    try:
        import onnxruntime as ort
    except ImportError:
        meta["status"] = "SKIPPED_ENV_DEPENDENCY"
        meta["error"] = "onnxruntime not installed"
        return meta
        
    try:
        meta["providers"] = ort.get_available_providers()
        sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
        meta["selected_provider"] = sess.get_providers()[0] if sess.get_providers() else "none"
        
        for inp in sess.get_inputs():
            meta["inputs"].append({
                "name": inp.name,
                "shape": inp.shape,
                "type": inp.type
            })
        for out in sess.get_outputs():
            meta["outputs"].append(out.name)
            
        try:
            import onnx
            model = onnx.load(str(onnx_path), load_external_data=False)
            opset_info = [op.version for op in model.op_import]
            if opset_info:
                meta["opset"] = str(opset_info[0])
        except Exception:
            pass
            
    except Exception as e:
        meta["status"] = "FAIL"
        meta["error"] = str(e)
        
    return meta

def inspect_models():
    print("==================================================")
    print("Voice Gateway Model Inspection Tool")
    print("==================================================")
    
    cfg = load_voice_config()
    
    components = [
        {
            "name": "wake_kws",
            "backend": "openwakeword_onnx",
            "configured_path": cfg.wake_tflite,
            "required_companions": []
        },
        {
            "name": "stop_kws",
            "backend": "openwakeword_onnx",
            "configured_path": cfg.stop_tflite,
            "required_companions": []
        },
        {
            "name": "vad",
            "backend": "fsmn_vad",
            "configured_path": cfg.vad_dir,
            "required_companions": ["config.yaml", "configuration.json", "am.mvn"]
        },
        {
            "name": "asr",
            "backend": "funasr_onnx_offline",
            "configured_path": cfg.asr_dir,
            "required_companions": ["tokens.json", "config.yaml", "configuration.json", "am.mvn"]
        },
        {
            "name": "tts",
            "backend": "piper",
            "configured_path": cfg.piper_model,
            "required_companions": [Path(cfg.piper_model).name + ".json" if cfg.piper_model else ""]
        },
        {
            "name": "tts_config",
            "backend": "piper",
            "configured_path": cfg.piper_config,
            "required_companions": []
        }
    ]
    
    for comp in components:
        name = comp["name"]
        backend = comp["backend"]
        conf_path = comp["configured_path"]
        companions = comp["required_companions"]
        if not companions or companions == [""]:
            companions = []
            
        print(f"\n[ Component: {name} (backend={backend}) ]")
        print(f"  configured_path    : {conf_path}")
        
        if not conf_path:
            print("  status             : SKIPPED (not configured)")
            continue

        if name in {"tts", "tts_config"} and cfg.disable_tts:
            print("  status             : DISABLED_OPTIONAL (profile disables local Piper)")
            continue
            
        try:
            res_path = resolve_path(conf_path)
            exists = res_path.exists()
        except Exception as e:
            print(f"  status             : FAIL (resolution error: {e})")
            continue
            
        print(f"  resolved_path      : {res_path}")
        print(f"  exists             : {exists}")
        
        if not exists:
            print("  status             : SKIPPED_MODEL_MISSING (not found on disk)")
            continue
            
        # Check size and contents
        if res_path.is_file():
            size = res_path.stat().st_size
            print(f"  file_size          : {size} bytes ({size / (1024*1024):.2f} MB)")
        else:
            print("  directory contents summary:")
            files = list(res_path.rglob("*"))
            print(f"    Total files: {len(files)}")
            for f in files[:6]:
                if f.is_file():
                    print(f"      - {f.relative_to(res_path)} ({f.stat().st_size} bytes)")
            if len(files) > 6:
                print("      - ...")
                
        # Check companions
        missing_companions = []
        if companions:
            print("  required companion files:")
            for c in companions:
                print(f"    - {c}")
            missing_companions = check_companion_files(res_path, companions)
            if missing_companions:
                print("  missing companion files:")
                for mc in missing_companions:
                    print(f"    - {mc}")
                print("  status             : BLOCKED_MODEL_LAYOUT (companions missing)")
                continue
                
        # Inspect ONNX shapes
        onnx_file = None
        if res_path.is_file() and res_path.suffix.lower() == ".onnx":
            onnx_file = res_path
        elif res_path.is_dir():
            # Search inside directory for model_quant.onnx or model.onnx
            for name_cand in ("model_quant.onnx", "model.onnx"):
                cand = res_path / name_cand
                if cand.exists():
                    onnx_file = cand
                    break
                    
        if onnx_file:
            print(f"  ONNX analysis of: {onnx_file.name}")
            meta = inspect_onnx_metadata(onnx_file)
            if meta["status"] == "SKIPPED_ENV_DEPENDENCY":
                print(f"  status             : SKIPPED_ENV_DEPENDENCY (onnxruntime missing)")
            elif meta["status"] == "FAIL":
                print(f"  status             : FAIL (error loading session: {meta['error']})")
            else:
                print(f"    opset            : {meta['opset']}")
                print("    input tensors    :")
                for inp in meta["inputs"]:
                    print(f"      name={inp['name']} shape={inp['shape']} dtype={inp['type']}")
                print(f"    output names     : {', '.join(meta['outputs'])}")
                print(f"    available providers: {', '.join(meta['providers'])}")
                print(f"    selected provider: {meta['selected_provider']}")
                print("  status             : PASS")
        else:
            # File is a TFLite model or something else
            print("  status             : PASS (file check OK)")

if __name__ == "__main__":
    inspect_models()
