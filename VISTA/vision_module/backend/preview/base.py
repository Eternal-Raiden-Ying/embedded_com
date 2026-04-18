#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class PreviewOverlay:
    title: str = ""
    lines: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class PreviewFrame:
    ts: float
    image: Any
    stage: str = "IDLE"
    mode: str = "IDLE"
    overlay: Optional[PreviewOverlay] = None


class PreviewSink:
    sink_name = "base"

    def open(self) -> None:
        return None

    def close(self) -> None:
        return None

    def render(self, frame: PreviewFrame) -> bool:
        _ = frame
        return True

    def snapshot(self) -> Dict[str, Any]:
        return {"sink_name": self.sink_name}


class NullPreviewSink(PreviewSink):
    sink_name = "null"


class OpenCVPreviewSink(PreviewSink):
    sink_name = "opencv"

    def __init__(self, window_name: str = "VISTA Preview"):
        self.window_name = str(window_name or "VISTA Preview")
        self._cv2 = None
        self._opened = False
        self._available = False

    def _ensure_cv2(self) -> bool:
        if self._cv2 is not None:
            return bool(self._available)
        try:
            import cv2  # type: ignore
        except Exception:
            self._cv2 = None
            self._available = False
            return False
        self._cv2 = cv2
        self._available = True
        return True

    def open(self) -> None:
        if not self._ensure_cv2() or self._opened:
            return
        try:
            self._cv2.namedWindow(self.window_name)  # type: ignore[union-attr]
            self._opened = True
        except Exception:
            self._opened = False
            self._available = False

    def close(self) -> None:
        if self._cv2 is None or not self._opened:
            return
        try:
            self._cv2.destroyWindow(self.window_name)  # type: ignore[union-attr]
        except Exception:
            pass
        self._opened = False

    def render(self, frame: PreviewFrame) -> bool:
        if not self._ensure_cv2():
            return True
        if not self._opened:
            self.open()
        if not self._opened:
            return True
        canvas = frame.image
        try:
            canvas = frame.image.copy()
        except Exception:
            pass
        overlay = frame.overlay
        if overlay is not None:
            try:
                title = str(overlay.title or "").strip()
                if title:
                    self._cv2.putText(canvas, title, (16, 28), self._cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                for idx, line in enumerate(overlay.lines or []):
                    self._cv2.putText(
                        canvas,
                        str(line),
                        (16, 58 + idx * 24),
                        self._cv2.FONT_HERSHEY_SIMPLEX,
                        0.55,
                        (255, 255, 255),
                        1,
                    )
            except Exception:
                pass
        try:
            self._cv2.imshow(self.window_name, canvas)
            key = self._cv2.waitKey(1) & 0xFF
        except Exception:
            return True
        return key != 27

    def snapshot(self) -> Dict[str, Any]:
        return {
            "sink_name": self.sink_name,
            "window_name": self.window_name,
            "opened": bool(self._opened),
            "available": bool(self._available),
        }
