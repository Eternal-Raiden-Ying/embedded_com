#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class PreviewOverlay:
    """Render metadata passed from stage/mode logic to a preview sink."""

    title: str = ""
    lines: List[str] = field(default_factory=list)
    annotations: List[Dict[str, Any]] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class PreviewFrame:
    """One preview-ready frame bundle emitted by the preview manager."""

    ts: float
    image: Any = None
    stage: str = "IDLE"
    mode: str = "IDLE"
    overlay: PreviewOverlay = field(default_factory=PreviewOverlay)


class PreviewSink(ABC):
    """Abstract preview output sink.

    Concrete sinks may draw to an OpenCV window, a websocket stream, or a
    debug-only null sink.
    """

    sink_name: str = "base"

    @abstractmethod
    def open(self) -> None:
        """Allocate sink resources before preview rendering begins."""

    @abstractmethod
    def render(self, frame: PreviewFrame) -> bool:
        """Render one preview frame and return False if the sink requests exit."""

    @abstractmethod
    def close(self) -> None:
        """Release sink resources during shutdown."""

    def snapshot(self) -> Dict[str, Any]:
        """Return a lightweight sink diagnostic snapshot."""
        return {"sink_name": self.sink_name}


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
