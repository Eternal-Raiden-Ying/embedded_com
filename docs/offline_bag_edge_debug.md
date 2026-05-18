# Offline Bag Edge Debug

`VISTA/vision_module/examples/offline_bag_edge_debug.py` 用 RealSense `.bag` 离线回放 RGB + Depth，并把抽帧送入当前在线桌边检测逻辑。它用于不接真实相机时调试 table edge、ROI preset 和 preview overlay。

## Dependencies

需要当前 Python 环境可导入：

- `pyrealsense2`
- `numpy`
- `cv2` 或板端 `aidcv`

本机可先检查：

```bash
/usr/bin/python3 -c "import pyrealsense2, numpy, cv2; print('ok')"
```

如果缺 `pyrealsense2`，优先在板端或 RealSense SDK 环境中安装/验证，例如使用已带 librealsense 的系统 Python。普通 `pip install pyrealsense2` 是否可用取决于平台架构和 Python 版本。

## Run

显示预览窗口：

```bash
/usr/bin/python3 VISTA/vision_module/examples/offline_bag_edge_debug.py \
  --bag VISTA/20260516_161436.bag \
  --stride 10 \
  --roi-preset center_lower \
  --show
```

不显示窗口，只保存抽帧预览：

```bash
/usr/bin/python3 VISTA/vision_module/examples/offline_bag_edge_debug.py \
  --bag VISTA/20260516_161436.bag \
  --stride 10 \
  --max-frames 500 \
  --roi-preset center_lower \
  --save-dir /tmp/offline_bag_edge_debug
```

从第 100 帧开始，每 20 帧处理一次：

```bash
/usr/bin/python3 VISTA/vision_module/examples/offline_bag_edge_debug.py \
  --bag VISTA/20260516_161436.bag \
  --start-frame 100 \
  --stride 20 \
  --max-frames 800 \
  --roi-preset full_width_lower \
  --show
```

预览窗口按 `q` 退出，`Esc` 也会退出。

## ROI Presets

ROI preset 来自 `VISTA/vision_module/backend/table_edge_roi.py`，离线脚本没有单独硬编码 ROI。当前可选：

- `full_frame`
- `center_mid`
- `center_lower`
- `full_width_lower`
- `right_lower`

切换方式：

```bash
--roi-preset full_frame
--roi-preset center_mid
--roi-preset center_lower
--roi-preset full_width_lower
--roi-preset right_lower
```

`full_frame` 会把整个 depth 图作为 edge detector 的输入 ROI，适合先做全局检测对照实验。

## YOLO Box Preview

默认离线脚本不跑 YOLO。需要观察 RGB 中桌子检测框时，加 `--yolo`：

```bash
/usr/bin/python3 VISTA/vision_module/examples/offline_bag_edge_debug.py \
  --bag VISTA/20260516_161436.bag \
  --stride 10 \
  --roi-preset full_frame \
  --yolo \
  --show
```

保存带 YOLO 检测框的抽帧预览：

```bash
/usr/bin/python3 VISTA/vision_module/examples/offline_bag_edge_debug.py \
  --bag VISTA/20260516_161436.bag \
  --stride 10 \
  --roi-preset full_frame \
  --yolo \
  --save-dir /tmp/offline_bag_edge_yolo
```

`--yolo` 只用于 RGB 预览框渲染，不会根据 YOLO bbox 动态调整 ROI。ROI 仍然完全由 `--roi-preset` 决定。

## Saved Preview

提供 `--save-dir` 后，脚本会保存每个抽帧结果：

```text
bag_edge_000120_center_lower.png
```

图片使用 `OpenCVPreviewSink` 的同类面板：RGB、Depth colormap、edge debug、状态/俯视图。可用它对比不同 ROI preset 下 ROI 框、边缘线和检测状态。

## Reading Output

控制台会低频打印抽帧结果：

```text
[BAG_EDGE] frame=120 valid=1 dist=-0.012 yaw=0.035 roi=[160, 240, 480, 408] preset=center_lower age_ms=24.1
```

判断 ROI 是否合理时重点看：

- `valid=1`：当前 ROI 内检测到了有效 edge。
- `dist`：相对目标距离的误差，绝对值越小越接近期望桌边距离。
- `yaw`：边缘线角度误差，绝对值越小越接近正对桌边。
- `roi=[x0, y0, x1, y1]`：确认 ROI 框覆盖桌边区域，而不是大面积背景或无效深度。
- `age_ms`：当前抽帧处理耗时，用于观察 ROI 变大后是否明显变慢。

调 ROI 时建议先跑 `center_lower`，如果桌边偏右再试 `right_lower`，如果不确定桌边纵向位置先试 `full_width_lower`，需要全局对照时试 `full_frame`。

## Differences From Live Camera

- `.bag` 回放不依赖实时相机，但 librealsense 仍可能需要 udev/device 访问权限。
- 脚本默认 `playback.set_real_time(False)`，处理速度由 CPU 和 `--stride` 决定，不代表实时帧率。
- bag 内录制的分辨率、fps、depth scale 和真实在线相机配置可能不同；脚本使用 `VISTA/Offline_Edge_Test/calib.json` 作为默认标定。
- 离线脚本重点复用 table-edge detector、ROI preset 和 preview overlay，不启动完整 `vision_module` app/scheduler，因此不包含在线模式切换和 IPC 输出；`--yolo` 只做抽帧 RGB 检测框预览。
