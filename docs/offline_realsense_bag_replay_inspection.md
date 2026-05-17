# Offline RealSense Bag Replay Inspection

Date: 2026-05-17

## Summary

The repository already has RealSense `.bag` playback support, but it lives outside the main `VISTA/vision_module` runtime path:

- `VISTA/Online_Edge_Detect/stream_source.py` can open a `.bag` via `rs.config.enable_device_from_file(...)`, disables real-time playback with `playback.set_real_time(False)`, aligns depth to color when configured, and returns synchronized `depth` and `color` numpy arrays.
- `VISTA/Online_Edge_Detect/app.py` runs the online table-edge detector from that stream source. It can therefore test table-edge detection from a bag through the online edge service path.
- `VISTA/Offline_Edge_Test/read_realsense_bag.py` can inspect a `.bag`, export depth/color frames, preview them, and optionally run the older `Offline_Edge_Test/TableEdgeDetector.py`.
- `VISTA/Offline_Edge_Test/REALSENSE_BAG_USAGE.md` documents the existing bag inspection flow.

## Existing Commands

Online edge detector using a bag:

```bash
EDGE_BAG_PATH=VISTA/20260516_161436.bag EDGE_PREVIEW=1 python3 VISTA/Online_Edge_Detect/app.py
```

Offline bag inspection/export:

```bash
python3 VISTA/Offline_Edge_Test/read_realsense_bag.py --bag VISTA/20260516_161436.bag --max-frames 30 --preview --run-detector --calib-json VISTA/Offline_Edge_Test/calib.json
```

## Gaps

There is no `VISTA/vision_module` example that directly replays a `.bag` while reusing the current `vision_module` ROI preset helpers and preview overlay style. The existing `Online_Edge_Detect` app has environment-variable configuration, but it does not provide script-level controls for:

- `--stride`
- `--start-frame`
- `--max-frames`
- `--roi-preset`
- `--save-dir`

## Minimal Integration Point

The least invasive path is a standalone script under `VISTA/vision_module/examples/` that:

- reuses `VISTA/Online_Edge_Detect.stream_source.RealSenseStreamSource` or the same `pyrealsense2` playback pattern for RGB+Depth frame input;
- reuses `VISTA/Online_Edge_Detect.detector.OnlineTableEdgeDetector` for table-edge detection;
- reuses `VISTA/vision_module/backend/table_edge_roi.py` for ROI presets;
- reuses `VISTA/vision_module/backend/preview/opencv_sink.py` by rendering a `PreviewFrame` with `table_edge_obs` metadata.
