# VISTA

**Vision Intelligent Search & Tracking Assistant**

视觉检测与目标追踪服务。基于 Qualcomm QNN NPU 加速的 YOLO 实例分割模型，通过 GStreamer 摄像头采集图像，向 Orchestrator 提供目标观测结果。

## 架构

```
Camera (GStreamer/aarch64) → VisionEngine → QNN Predictor (QCS6490 NPU)
                                          ↓
Orchestrator ──vision_req──▶ [app.py]  ──vision_obs──▶ Orchestrator
```

## 目录结构

```
VISTA/
├── vision_module/
│   ├── app/app.py              # 服务入口
│   ├── backend/
│   │   ├── vision_engine.py    # 主推理引擎
│   │   ├── new_engine.py       # 新版引擎
│   │   ├── camera/
│   │   │   ├── HardwareCamera.py
│   │   │   ├── fast_cam.*.so   # GStreamer C++ 扩展（aarch64 预编译）
│   │   │   └── cxx/            # C++ 源码（cam_gst.cpp）
│   │   └── predictor/
│   │       └── QNNPredictor.py # Qualcomm QNN 推理封装
│   ├── config/board_config.py  # 主配置文件
│   ├── ipc/                    # TCP 传输层
│   ├── model/                  # YOLO 模型（QCS6490 编译版）
│   └── test/                   # 调试工具
├── grasp_module/               # 抓取模块
├── tools/
└── logs/ / runs/ / pids/
```

## 模型

| 模型目录 | 用途 |
|----------|------|
| `model/yolo26s-seg/` | 目标检测与分割（主模型） |
| `model/yolo26s-seg-grasp/` | 抓取点检测 |
| `model/yolov8s-seg/` | 备用分割模型 |

> 模型为 QCS6490 QNN 2.36 编译版（`.amf` / `.ctx.bin`），仅可在目标硬件上运行。

## 运行

```bash
cd /home/aidlux/2026/VISTA
/usr/bin/python3 -m vision_module.app.app
```

## IPC 协议

| 方向 | 消息类型 | 地址 |
|------|----------|------|
| 接收 | `vision_req` | `127.0.0.1:9003` |
| 发送 | `vision_obs` | `127.0.0.1:9002` |

`vision_req` 关键字段：`mode`（`SEARCH` / `APPROACH` / `IDLE`）、`session_id`、`req_id`、`target`

`vision_obs` 关键字段：`session_id`、`req_id`、`target_obs`、`home_tag_obs`

## 硬件依赖

- SoC：Qualcomm QCS6490（NPU 推理必须）
- 摄像头：通过 GStreamer pipeline 采集，依赖 `fast_cam` C++ 扩展（aarch64 预编译 `.so`）
- 深度相机：RealSense（可选，`RealSenseDepthCamera.py`）

## 本地开发说明

`fast_cam.cpython-38-aarch64-linux-gnu.so` 为 ARM 预编译产物，Windows 本地无法直接运行视觉模块。
如需重新编译，在 AidLux 上执行：

```bash
cd VISTA/vision_module/backend/camera/cxx
mkdir -p build && cd build
cmake .. && make
```
