#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Dict, List


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="批量跑 Offline_Edge_Test/live_capture 里的 depth/color 样本")
    parser.add_argument("--input-dir", type=Path, default=Path("test_data/live_capture"))
    parser.add_argument("--calib-json", type=Path, default=Path("calib.json"))
    parser.add_argument("--out-dir", type=Path, default=Path("test_data/offline_batch_output"))
    parser.add_argument("--python", default=sys.executable, help="用于调用 offline_depth_png_test.py 的解释器")
    return parser


def _resolve(base: Path, path: Path) -> Path:
    return (base / path).resolve() if not path.is_absolute() else path.resolve()


def _pair_samples(input_dir: Path) -> List[Dict[str, Path]]:
    depth_files = sorted(list(input_dir.glob("depth_*.png")) + list(input_dir.glob("capture_*_depth.png")))
    pairs: List[Dict[str, Path]] = []
    for depth_path in depth_files:
        if depth_path.name.startswith("depth_"):
            suffix = depth_path.name[len("depth_") :]
            color_path = input_dir / ("color_" + suffix)
        elif depth_path.name.endswith("_depth.png"):
            stem = depth_path.stem[:-len("_depth")]
            color_path = input_dir / (stem + "_color.png")
        else:
            color_path = None
        item = {"depth": depth_path}
        if color_path is not None and color_path.exists():
            item["color"] = color_path
        pairs.append(item)
    return pairs


def main() -> None:
    args = build_parser().parse_args()
    here = Path(__file__).resolve().parent
    input_dir = _resolve(here, args.input_dir)
    calib_json = _resolve(here, args.calib_json)
    out_dir = _resolve(here, args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    pairs = _pair_samples(input_dir)
    if not pairs:
        raise RuntimeError(f"no depth samples found in {input_dir}")

    offline_script = here / "offline_depth_png_test.py"
    summary = []

    for idx, pair in enumerate(pairs, start=1):
        depth_path = pair["depth"]
        color_path = pair.get("color")
        cmd = [
            str(args.python),
            str(offline_script),
            "--depth-png",
            str(depth_path),
            "--calib-json",
            str(calib_json),
            "--out-dir",
            str(out_dir),
        ]
        if color_path is not None:
            cmd.extend(["--color-png", str(color_path)])
        proc = subprocess.run(cmd, capture_output=True, text=True)
        item = {
            "index": idx,
            "depth_png": str(depth_path),
            "color_png": str(color_path) if color_path is not None else None,
            "ok": proc.returncode == 0,
            "stdout": proc.stdout.strip(),
            "stderr": proc.stderr.strip(),
        }
        try:
            if proc.returncode == 0 and proc.stdout.strip():
                payload = json.loads(proc.stdout)
                item["result"] = payload
        except Exception:
            pass
        summary.append(item)
        print("[%02d/%02d] %s -> %s" % (idx, len(pairs), depth_path.name, "OK" if proc.returncode == 0 else "FAIL"))

    summary_path = out_dir / "batch_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
        f.write("\n")

    ok_count = sum(1 for item in summary if item["ok"])
    print("summary:", summary_path)
    print("ok=%d total=%d" % (ok_count, len(summary)))


if __name__ == "__main__":
    main()
