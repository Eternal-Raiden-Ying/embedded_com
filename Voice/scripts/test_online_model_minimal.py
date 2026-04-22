#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import numpy as np
import soundfile as sf

from funasr_onnx.paraformer_online_bin import Paraformer


def parse_text(out):
    if out is None:
        return ""

    if isinstance(out, str):
        return out.strip()

    if isinstance(out, dict):
        if "text" in out and isinstance(out["text"], str):
            return out["text"].strip()
        if "result" in out and isinstance(out["result"], str):
            return out["result"].strip()
        if "preds" in out:
            preds = out["preds"]
            if isinstance(preds, tuple) and len(preds) >= 1:
                if isinstance(preds[0], str):
                    return preds[0].strip()
            if isinstance(preds, list):
                texts = [x.strip() for x in preds if isinstance(x, str) and x.strip()]
                return "".join(texts)
        return ""

    if isinstance(out, list):
        texts = []
        for item in out:
            if isinstance(item, str):
                t = item.strip()
                if t:
                    texts.append(t)
            elif isinstance(item, dict):
                if "text" in item and isinstance(item["text"], str):
                    t = item["text"].strip()
                    if t:
                        texts.append(t)
                elif "result" in item and isinstance(item["result"], str):
                    t = item["result"].strip()
                    if t:
                        texts.append(t)
                elif "preds" in item:
                    preds = item["preds"]
                    if isinstance(preds, tuple) and len(preds) >= 1 and isinstance(preds[0], str):
                        t = preds[0].strip()
                        if t:
                            texts.append(t)
                    elif isinstance(preds, list):
                        for x in preds:
                            if isinstance(x, str) and x.strip():
                                texts.append(x.strip())
        return "".join(texts).strip()

    return str(out).strip()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", required=True)
    ap.add_argument("--wav", required=True)
    ap.add_argument("--chunk-size", default="5,10,5")
    ap.add_argument("--quantize", action="store_true")
    ap.add_argument("--device-id", default="-1")
    ap.add_argument(
        "--call-style",
        default="auto",
        choices=[
            "auto",
            "audio_in+cache_is_final",
            "audio_in+param_dict",
            "positional+cache_is_final",
            "positional+param_dict",
            "audio_in_only",
            "positional_only",
        ],
    )
    args = ap.parse_args()

    chunk_size = [int(x) for x in args.chunk_size.split(",")]

    speech, sr = sf.read(args.wav, dtype="float32")
    if speech.ndim > 1:
        speech = speech[:, 0]

    if sr != 16000:
        raise ValueError(f"Expected 16k wav, got sr={sr}")

    speech = np.asarray(speech, dtype=np.float32).reshape(-1)
    speech = np.ascontiguousarray(speech, dtype=np.float32)

    print("sr=", sr)
    print("samples=", len(speech))
    print("dtype=", speech.dtype, "min=", float(speech.min()), "max=", float(speech.max()))

    model = Paraformer(
        model_dir=args.model_dir,
        batch_size=1,
        chunk_size=chunk_size,
        device_id=args.device_id,
        quantize=args.quantize,
    )

    step = chunk_size[1] * 960
    print("chunk_size=", chunk_size, "step=", step)

    cache = {}
    chosen_style = None
    pieces = []

    off = 0
    idx = 0
    while off < len(speech):
        end = min(off + step, len(speech))
        chunk = speech[off:end]
        is_final = end >= len(speech)

        attempts = {
            "audio_in+cache_is_final": lambda: model(audio_in=chunk, cache=cache, is_final=is_final),
            "audio_in+param_dict": lambda: model(audio_in=chunk, param_dict={"cache": cache, "is_final": is_final}),
            "positional+cache_is_final": lambda: model(chunk, cache=cache, is_final=is_final),
            "positional+param_dict": lambda: model(chunk, param_dict={"cache": cache, "is_final": is_final}),
            "audio_in_only": lambda: model(audio_in=chunk),
            "positional_only": lambda: model(chunk),
        }

        out = None

        if args.call_style == "auto":
            last_err = None
            if chosen_style is None:
                for name, fn in attempts.items():
                    try:
                        out = fn()
                        chosen_style = name
                        print(f"[choose call_style] {chosen_style}")
                        break
                    except Exception as e:
                        last_err = e
                        print(f"[call fail] {name}: {repr(e)}")
                if chosen_style is None:
                    raise RuntimeError(f"all call styles failed, last_err={repr(last_err)}")
            else:
                out = attempts[chosen_style]()
        else:
            if idx == 0:
                print(f"[force call_style] {args.call_style}")
            out = attempts[args.call_style]()

        # 某些实现会把 cache 塞回返回对象里
        if isinstance(out, dict) and isinstance(out.get("cache"), dict):
            cache = out["cache"]

        text = parse_text(out)
        pieces.append(text)

        print("=" * 60)
        print(f"chunk#{idx} off={off} end={end} final={is_final}")
        print("raw_out =", out)
        print("text    =", text)

        off = end
        idx += 1

    final_text = "".join(pieces).strip()
    print("\nFINAL_TEXT =", final_text)


if __name__ == "__main__":
    main()