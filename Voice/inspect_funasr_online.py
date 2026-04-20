#!/usr/bin/env python3
import inspect
import importlib

mods = [
    ("funasr_onnx", "Paraformer"),
    ("funasr_onnx", "ParaformerOnline"),
    ("funasr_onnx.paraformer_online_bin", "Paraformer"),
]
for mod_name, attr in mods:
    print(f"\n=== {mod_name}.{attr} ===")
    try:
        mod = importlib.import_module(mod_name)
        obj = getattr(mod, attr)
        print("FOUND:", obj)
        try:
            print("SIGNATURE:", inspect.signature(obj))
        except Exception as e:
            print("SIGNATURE_ERR:", e)
        try:
            print("CALL_SIGNATURE:", inspect.signature(obj.__call__))
        except Exception as e:
            print("CALL_SIGNATURE_ERR:", e)
    except Exception as e:
        print("IMPORT_ERR:", e)
