from .base import NullPreviewSink, PreviewFrame, PreviewOverlay, PreviewSink

try:
    from .opencv_sink import OpenCVPreviewSink
except Exception:
    class OpenCVPreviewSink(NullPreviewSink):  # type: ignore
        sink_name = "opencv"

        def __init__(self, window_name: str = "VISTA App Dashboard"):
            super().__init__()
            self.window_name = window_name

__all__ = [
    "PreviewFrame",
    "PreviewOverlay",
    "PreviewSink",
    "NullPreviewSink",
    "OpenCVPreviewSink",
]
