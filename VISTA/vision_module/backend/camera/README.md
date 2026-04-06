# AidLux Fast Camera 🚀

专为 AidLux / 高通边缘计算平台打造的**硬件加速、零拷贝 (Zero-Copy)** 摄像头采集模块。
完美解决 USB 摄像头高分辨率 YUYV 裸流导致 CPU 占用率爆满、推流卡顿的问题。

## 🧠 核心实现原理



本模块绕过了 Python OpenCV 传统的软解码路径，采用 C++ 与 GStreamer 底层 API 重构了图像采集流水线：
1. **DMA-BUF 零拷贝**: 图像数据从 USB (`v4l2src`) 进入内存后，全程驻留在硬件隔离区（DMA-BUF），不经过任何用户态与内核态的内存拷贝（Memcpy）。
2. **高通硬件级前处理 (`qtivtransform`)**: 色彩空间转换（YUYV -> RGB）、缩放（Resize）、裁剪（Crop）、翻转等极其消耗算力的操作，全部由高通芯片底层的硬件处理单元（GPU/VPE）完成，CPU 参与度降至 0%。
3. **Pybind11 显存直通**: C++ 层捕获硬件内存指针后，直接将其伪装成 NumPy 数组递给 Python。AI 模型（如 QNN HTP）可以直接读取这块物理显存进行极速推理。
4. **异步 GIL 释放**: 在 C++ 阻塞等待画面时主动释放 Python 全局解释器锁（GIL），彻底解决 AidLux CVS 网页端推流的心跳死锁问题。

## 🛠️ 编译与安装

由于涉及底层硬件加速驱动，本模块需要使用 CMake 编译。

1. 进入 C++ 源码目录：
   ```bash
   cd aidlux_cam/csrc
   mkdir build && cd build
   cmake ..
   make  
  
2. 将编译完成得到的.so文件移到此目录下

## 帮助

python API 请参考__init__.py
