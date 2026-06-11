# VISTA 摄像头后端 (VISTA Camera Backend)

此目录包含 VISTA 运行时管理器所使用的摄像头后端。

它不是一个独立的 Demo 包，而是当前 VISTA stage/mode 架构的一部分。

## 当前角色

摄像头后端提供由 `CameraManager` 选择的底层摄像头实现。

当前导出的运行时类：

- `ColorCamera`: 彩色相机
- `IRCamera`: 红外（IR）相机
- `HardwareCamera`: 硬件加速相机
- `RealSenseDepthCamera`: RealSense 深度相机

导入选择器（import selector）位于 `__init__.py` 中，当前支持：

- `VISTA_BACKEND=mock`
- `VISTA_BACKEND=real`
- `VISTA_BACKEND=auto`

## 当前目录内容

- `ColorCamera.py`: 彩色摄像头实现
- `IRCamera.py`: 红外摄像头实现
- `HardwareCamera.py`: 硬件加速摄像头路径
- `RealSenseDepthCamera.py`: 深度摄像头实现
- `base.py`: 摄像头抽象基类
- `mock.py`: Mock 后端
- `_fast_gst_camera.py`: 快速摄像头路径的 Python 桥接
- `cxx/`: 用于摄像头扩展开发的原生源码/构建目录
- `camera_info.md`: 当前硬件的原始参数与能力笔记

预构建的目标设备二进制文件 `fast_cam.cpython-38-aarch64-linux-gnu.so`
存储在此 Python 源码目录之外的 `VISTA/vision_module/libs/aarch64/` 路径下。

## 构建路径 (Build Path)

如果需要在目标设备上重新构建原生摄像头扩展，请使用当前仓库路径：

```bash
cd VISTA/vision_module/backend/camera/cxx
mkdir -p build
cd build
cmake ..
make
```

将生成的 `fast_cam.cpython-38-aarch64-linux-gnu.so` 复制到
`VISTA/vision_module/libs/aarch64/` 中。对于本仓库，旧的 `aidlux_cam/csrc` 路径已废弃。

## 当前架构说明 (Current Architectural Notes)

- 摄像头生命周期由 `vision_module/backend/camera_manager.py` 拥有，而不是 App 层。
- 摄像头实例根据运行时模式计划（mode plans）进行选择和重配置。
- 当前板端默认值仍保留在 `vision_module/config/board_config.py` 中。
- `GRASP_REMOTE` 现在使用显式的 `ModeProfile.camera_overrides`，而不是隐式地重用本地跟踪默认配置。
- 板端配置仍提供源默认值，但运行时所有权现归模式/配置文件（mode/profile）数据所有。
- 默认彩色摄像头基线现为 `BGR`，且模式配置文件拥有每个模式最终的 RGB 摄像头格式 / 裁剪 / FPS 约定。

## 当前限制 (Current Limitations)

- 真实的运行目标是 AidLux / QCS6490，而不是 Windows。
- `fast_cam` 原生扩展是一个 Python 3.8 aarch64 Linux 二进制文件。Windows
  主机环境无法导入它；除非 `platform.machine() == "aarch64"` 且 `fast_cam` 可导入，否则需要此模块的主机端测试必须跳过。
- 在 Windows 或不支持的环境中，后端可能会根据运行时设置解析为 `mock`。
- `camera_info.md` 是硬件说明，而不是权威的架构约定。

## 相关文档 (Related Docs)

- `VISTA/ReadMe.md`
- `VISTA/ARCHITECTURE.md`
- `VISTA/PRODUCT_REQUIREMENTS.md`
