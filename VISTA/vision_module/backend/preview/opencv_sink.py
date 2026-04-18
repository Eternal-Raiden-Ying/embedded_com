#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from typing import Any, Dict

import numpy as np

try:
    import aidcv as cv2
except ImportError:
    import cv2

from .base import PreviewFrame, PreviewSink


class OpenCVPreviewSink(PreviewSink):
    """Preview sink interface for a future OpenCV-backed dashboard.

    Rendering details are intentionally deferred. This class documents the
    expected lifecycle and return contract for the preview capability.
    """

    sink_name = "opencv"

    def __init__(self, window_name: str = "VISTA App Dashboard"):
        self.window_name = window_name
        self._opened = False
        self._last_frame_ts = 0.0

    def open(self) -> None:
        """Prepare the dashboard window and sink-local resources."""
        cv2.namedWindow(self.window_name)
        self._opened = True

    def render(self, frame: PreviewFrame) -> bool:
        """Render one frame bundle and return False when the user asks to exit."""
        if not self._opened:
            self.open()
        image = frame.image
        if image is None:
            canvas = np.zeros((480, 640, 3), dtype=np.uint8)
        else:
            canvas = image.copy()
            if len(getattr(canvas, "shape", ())) == 2:
                canvas = cv2.cvtColor(canvas, cv2.COLOR_GRAY2BGR)
        title = str((frame.overlay.title or "").strip() or "VISTA Preview")
        cv2.putText(canvas, title, (16, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (50, 255, 50), 2)
        y = 54
        for line in list(frame.overlay.lines or [])[:8]:
            cv2.putText(canvas, str(line), (16, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (220, 220, 220), 2)
            y += 24
        cv2.imshow(self.window_name, canvas)
        self._last_frame_ts = float(frame.ts or 0.0)
        if cv2.waitKey(1) & 0xFF == 27:
            return False
        return True

    def close(self) -> None:
        """Destroy sink-local resources and close the dashboard window."""
        if self._opened:
            try:
                cv2.destroyWindow(self.window_name)
            except Exception:
                pass
        self._opened = False

    def snapshot(self) -> Dict[str, Any]:
        """Expose sink configuration and last-render bookkeeping."""
        snap = super().snapshot()
        snap.update(
            {
                "window_name": self.window_name,
                "opened": self._opened,
                "last_frame_ts": self._last_frame_ts,
            }
        )
        return snap
