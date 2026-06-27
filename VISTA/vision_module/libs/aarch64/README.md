# VISTA aarch64 原生库 (VISTA aarch64 Native Libraries)

此目录存储 VISTA 使用的仅适用于板端的原生二进制文件。

## fast_cam

- 文件：`fast_cam.cpython-38-aarch64-linux-gnu.so`
- 来源：从 `VISTA/vision_module/backend/camera/cxx` 下的摄像头扩展源码编译构建
- Python ABI：CPython 3.8
- 平台：aarch64 Linux 目标板
- 主机行为：Windows 主机无法导入此模块
- 测试行为：除非在 aarch64 上运行且模块可导入，否则需要 `fast_cam` 的测试必须跳过

请将原生目标二进制文件保留在此处，而不是混入 Python 摄像头源码目录中。
