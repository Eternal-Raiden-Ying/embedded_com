#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import shutil
import time
from pathlib import Path
from typing import List, Tuple


def _entry_mtime(path: Path) -> float:
    try:
        return float(path.stat().st_mtime)
    except Exception:
        return 0.0


def _iter_entries(directory: Path) -> List[Path]:
    if not directory.exists() or not directory.is_dir():
        return []
    return [p for p in directory.iterdir()]


def _remove_path(path: Path, dry_run: bool) -> bool:
    if dry_run:
        return True
    try:
        if path.is_dir():
            shutil.rmtree(path)
        else:
            path.unlink(missing_ok=True)
        return True
    except Exception:
        return False


def _cleanup_dir(directory: Path, keep_count: int, cutoff_ts: float, dry_run: bool) -> Tuple[List[Path], List[Path]]:
    removed: List[Path] = []
    failed: List[Path] = []
    entries = _iter_entries(directory)
    entries_sorted = sorted(entries, key=_entry_mtime, reverse=True)
    keep_set = set(entries_sorted[: max(0, int(keep_count))])

    for path in entries_sorted:
        mtime = _entry_mtime(path)
        remove_for_age = mtime < cutoff_ts
        remove_for_count = path not in keep_set
        if not (remove_for_age or remove_for_count):
            continue
        ok = _remove_path(path, dry_run=dry_run)
        if ok:
            removed.append(path)
        else:
            failed.append(path)
    return removed, failed


def main():
    parser = argparse.ArgumentParser(
        description="Clean VISTA runs/logs: remove entries older than N days or outside latest K entries."
    )
    parser.add_argument("--days", type=int, default=14, help="Age threshold in days. Default: 14")
    parser.add_argument("--keep", type=int, default=30, help="Keep latest entry count per directory. Default: 30")
    parser.add_argument("--project-root", default="", help="VISTA root path. Default: infer from script location")
    parser.add_argument("--dry-run", action="store_true", help="Show removal plan without deleting files")
    args = parser.parse_args()

    if args.project_root:
        vista_root = Path(args.project_root).resolve()
    else:
        vista_root = Path(__file__).resolve().parents[1]

    runs_dir = vista_root / "runs"
    logs_dir = vista_root / "logs"

    now_ts = time.time()
    cutoff_ts = now_ts - max(0, int(args.days)) * 86400

    print(f"[INFO] vista_root={vista_root}")
    print(f"[INFO] runs_dir={runs_dir}")
    print(f"[INFO] logs_dir={logs_dir}")
    print(f"[INFO] days={args.days} keep={args.keep} dry_run={bool(args.dry_run)}")

    total_removed: List[Path] = []
    total_failed: List[Path] = []

    for target in (runs_dir, logs_dir):
        removed, failed = _cleanup_dir(
            directory=target,
            keep_count=int(args.keep),
            cutoff_ts=cutoff_ts,
            dry_run=bool(args.dry_run),
        )
        total_removed.extend(removed)
        total_failed.extend(failed)
        print(f"[INFO] {target.name}: removed={len(removed)} failed={len(failed)}")
        for path in removed:
            print(f"  - remove: {path}")
        for path in failed:
            print(f"  - failed: {path}")

    print(f"[INFO] total_removed={len(total_removed)} total_failed={len(total_failed)}")
    if total_failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()

