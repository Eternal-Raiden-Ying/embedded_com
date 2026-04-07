#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from common.runtime_logging import (
    RunLogger,
    configure_stream_logger,
    ensure_dir,
    logging_level_for_mode,
    monotonic_ts,
    safe_dump,
)


def configure_logging(mode: str = "concise"):
    return configure_stream_logger("orch", mode=mode, enabled=True)


__all__ = [
    "RunLogger",
    "configure_logging",
    "ensure_dir",
    "logging_level_for_mode",
    "monotonic_ts",
    "safe_dump",
]
