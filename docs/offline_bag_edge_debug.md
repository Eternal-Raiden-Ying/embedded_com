# 离线数据包与桌边感知调试指南 (Offline Bag & Table Edge Debugging)

本文档将 RealSense 离线 `.bag` 数据包的回放检测、参数验证、ROI 设定以及桌面场景的深度图获取与录制逻辑进行合并整理，作为板端离线感知调试的统一指导。

---

## 1. 概述与核心工具 (Overview & Core Tools)

在不连接真实相机硬件时，我们可以利用录制好的 RealSense `.bag` 二进制包回放 RGB + Depth 图像流，输入给在线的桌边检测算法以调试 table edge、ROI preset 和 preview overlay。

主要有以下几个调试工具入口：
*   **`VISTA/vision_module/examples/offline_bag_edge_debug.py`**：支持加载指定 `.bag` 包，并提供丰富的命令行调试参数（如 `--stride`、`--roi-preset`、`--save-dir` 等），可将处理后的调试状态图输出保存。
*   **`VISTA/Online_Edge_Detect/stream_source.py`**：内置 `RealSenseStreamSource`，支持读取 `.bag` 文件，通过 `playback.set_real_time(False)` 能够实现非实时抽帧对齐。
*   **`VISTA/Offline_Edge_Test/read_realsense_bag.py`**：用于快速诊断 `.bag` 文件内含的通道流列表，导出指定帧数的 `depth` 和 `color` 图像，并执行旧版的 `TableEdgeDetector.py`。
*   **`VISTA/Offline_Edge_Test/record_realsense_bag.py`**：用于在连接有真实 RealSense D435 相机的设备上，通过 Python 脚本直接控制采集并录制指定时间的 `.bag` 文件。

---

## 2. 离线回放与调试命令 (Run & Debug Commands)

在 Python 运行环境中，确保已正确安装并导入 `pyrealsense2`、`numpy`、`cv2`。

### 2.1 基础回放预览
```bash
/usr/bin/python3 VISTA/vision_module/examples/offline_bag_edge_debug.py \
  --bag VISTA/20260516_161436.bag \
  --stride 10 \
  --roi-preset center_lower \
  --show
```
*   `--bag`：指定 `.bag` 数据包文件路径。
*   `--stride`：抽帧步长。对于高频录制的包，可通过增大步长来加速本地回放与分析。
*   `--roi-preset`：指定 ROI 预设类型。

### 2.2 不显示 GUI 窗口，直接保存调试分析图片
```bash
/usr/bin/python3 VISTA/vision_module/examples/offline_bag_edge_debug.py \
  --bag VISTA/20260516_161436.bag \
  --stride 10 \
  --max-frames 500 \
  --roi-preset center_lower \
  --save-dir /tmp/offline_bag_edge_debug
```

### 2.3 加载 YOLO 目标检测框进行对比预览
默认回放仅跑桌边几何检测，若想同步预览 YOLO 模型对桌面的检测框，可以附加 `--yolo` 标志：
```bash
/usr/bin/python3 VISTA/vision_module/examples/offline_bag_edge_debug.py \
  --bag VISTA/20260516_161436.bag \
  --stride 10 \
  --roi-preset full_frame \
  --yolo \
  --show
```
*注：`--yolo` 仅用于预览渲染，不影响几何桌边检测的 ROI 边界控制。*

---

## 3. ROI 预设选项与控制台输出解析 (ROI Presets & Outputs)

### 3.1 ROI 预设列表
预设定义在 `VISTA/vision_module/backend/table_edge_roi.py` 中：
*   `full_frame`：将整张深度图作为输入，通常用于进行全局检测对照。
*   `center_mid`：中部居中。
*   `center_lower`：中下部居中（桌边停靠的最常用预设）。
*   `full_width_lower`：底部全宽度。
*   `right_lower`：右下部。

### 3.2 控制台关键日志解读
运行期间控制台会低频输出检测摘要，示例：
```text
[BAG_EDGE] frame=120 valid=1 dist=-0.012 yaw=0.035 roi=[160, 240, 480, 408] preset=center_lower age_ms=24.1
```
*   `valid=1`：说明当前 ROI 区域内成功拟合出了桌边线。
*   `dist`：相对期望停车距离的偏差（米），绝对值越小说明越符合停稳要求。
*   `yaw`：小车与桌边的夹角偏差（弧度）。
*   `roi`：当前计算出的生效矩形裁剪框。
*   `age_ms`：单帧图像的处理时延。如果 ROI 范围过大，时延会明显增加。

---

## 4. 真实数据包录制与常见问题诊断 (Bag Recording & Verification)

### 4.1 离线回放时 TableEdgeDetector 报错或提示跳过深度图？
*   **原因**：有些 `.bag` 数据在录制时仅开启了彩色摄像头流（`stream.color`），没有将深度流（`stream.depth`）真正写入文件，而几何桌边检测必须依赖 16-bit 深度流。
*   **诊断工具**：
    ```bash
    python3 VISTA/Offline_Edge_Test/read_realsense_bag.py --bag 你的文件.bag --max-frames 3
    ```
    观察控制台输出的流通道列表，确认是否同时存在 `stream.depth` 和 `stream.color`。

### 4.2 使用 RealSense Viewer 录制完整数据包的最佳步骤
1.  启动 `realsense-viewer`。
2.  展开并启用左侧的 **`Stereo Module -> Depth`** 和 **`RGB Camera -> Color`** 开关。
3.  **关键步骤**：必须等主界面预览里同时出现了实时刷新的深度图（彩虹色深度图）与彩色图之后，再点击顶部的 **`Record to File`** 开始录制。
4.  录制结束保存为 `.bag`。

### 4.3 使用 Python 脚本进行自动化录制
如果在 Linux 开发板上没有图形显示环境，可直接调用以下命令行开启双流录制：
```bash
python3 VISTA/Offline_Edge_Test/record_realsense_bag.py --output new_rgb_depth.bag --duration 10 --preview
```
这会使用默认配置（Depth: `640x480@30fps`，Color: `1280x720@30fps`）自动录制 10 秒并存盘。

### 4.4 依赖项配置说明
几何桌边拟合算法依赖于聚类与直线拟合，若在新的虚拟环境中运行检测，必须确保安装了 `scikit-learn`：
```bash
pip install scikit-learn
```
