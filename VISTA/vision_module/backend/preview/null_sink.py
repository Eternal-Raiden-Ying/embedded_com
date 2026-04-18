#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from typing import Any, Dict

from .base import PreviewFrame, PreviewSink


class NullPreviewSink(PreviewSink):
    """No-op preview sink used when preview is disabled."""

    sink_name = "null"

    def __init__(self):
        self._opened = False

    def open(self) -> None:
        """Mark the null sink as active without allocating real resources."""
        self._opened = True

    def render(self, frame: PreviewFrame) -> bool:
        """Accept preview frames and intentionally do nothing with them."""
        _ = frame
        return True

    def close(self) -> None:
        """Mark the null sink as closed."""
        self._opened = False

    def snapshot(self) -> Dict[str, Any]:
        """Expose sink state for diagnostics."""
        snap = super().snapshot()
        snap.update({"opened": self._opened})
        return snap
