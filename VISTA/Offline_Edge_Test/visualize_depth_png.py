#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
from pathlib import Path

import cv2


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="把 16-bit depth png 生成可读的伪彩图，便于板端离线查看")
    parser.add_argument("--depth-png", type=Path, required=True)
    parser.add_argument("--out-png", type=Path, default=None)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    here = Path(__file__).resolve().parent
    depth_png = (here / args.depth_png).resolve() if not args.depth_png.is_absolute() else args.depth_png.resolve()
    out_png = args.out_png
    if out_png is None:
        out_png = depth_png.with_name(depth_png.stem + "_vis.png")
    elif not out_png.is_absolute():
        out_png = (here / out_png).resolve()

    depth_raw = cv2.imread(str(depth_png), cv2.IMREAD_ANYDEPTH)
    if depth_raw is None:
        raise RuntimeError(f"failed to load depth png: {depth_png}")

    depth_vis = cv2.convertScaleAbs(depth_raw, alpha=0.03)
    depth_vis = cv2.applyColorMap(depth_vis, cv2.COLORMAP_JET)
    cv2.imwrite(str(out_png), depth_vis)
    print("saved:", out_png)


if __name__ == "__main__":
    main()
