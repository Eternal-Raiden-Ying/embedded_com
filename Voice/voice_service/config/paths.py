#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
from pathlib import Path
from typing import Union, Tuple, Optional

def find_repo_root() -> Path:
    # 1. Environment variable override
    env_root = os.environ.get("VOICE_REPO_ROOT") or os.environ.get("REPO_ROOT")
    if env_root:
        return Path(env_root).resolve()

    # 2. Structural marker check in parent hierarchy
    current = Path(__file__).resolve().parent
    for parent in [current] + list(current.parents):
        if (parent / "configs" / "system_config.yaml").exists() or (parent / "start_robot_stack.sh").exists():
            return parent

    # 3. Fallback to default relative structure (four levels up from Voice/voice_service/config/paths.py)
    return Path(__file__).resolve().parents[3]

REPO_ROOT = find_repo_root()

def resolve_path(relative_or_absolute: Union[str, Path]) -> Path:
    if not relative_or_absolute:
        raise ValueError("Cannot resolve empty path")

    p = Path(relative_or_absolute)
    if p.is_absolute():
        resolved = p.resolve()
    else:
        resolved = (REPO_ROOT / p).resolve()

    # Reject paths attempting to escape the repo root via ".."
    if ".." in p.parts or ".." in p.as_posix():
        try:
            resolved.relative_to(REPO_ROOT)
        except ValueError:
            raise ValueError(f"Path escape attempt rejected: {relative_or_absolute}")

    return resolved

def resolve_and_verify_model(name: str, config_val: str, env_var: str, backend: str) -> Tuple[Optional[Path], bool]:
    val = os.environ.get(env_var)
    source = "env" if val else "config"
    if not val:
        val = config_val

    if not val:
        return None, False

    try:
        resolved = resolve_path(val)
        exists = resolved.exists()
    except Exception as e:
        print(f"[VOICE][ERROR] path resolution failed for {name}: {e}")
        return None, False

    # Print validation message matching: configured_path; resolved_path; exists; backend
    print(f"[VOICE][MODEL] name={name} source={source} configured_path={val} resolved_path={resolved} exists={exists} backend={backend}")

    try:
        from voice_service.runtime.common import jlog
        jlog({
            "level": "info",
            "src": "model",
            "name": name,
            "source": source,
            "configured_path": str(val),
            "resolved_path": str(resolved),
            "exists": exists,
            "backend": backend
        })
    except Exception:
        pass

    return resolved, exists
