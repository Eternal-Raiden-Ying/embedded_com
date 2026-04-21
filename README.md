# embedded_com

本仓库当前维护的主项目是 `grasp_module`。

根目录下其他文件夹大多是为了参考、依赖对照或源码比对而保留的外部仓库，不作为本项目主线代码维护对象。后续开发、测试和文档说明，默认都以 `grasp_module` 为中心。

## 项目定位

该项目用于基于 RealSense RGB-D 输入做目标抓取检测，并将抓取推理能力封装为服务接口，供机器人端或边缘端调用。

当前主链路已经收敛为仅支持 `class_id` 输入的模式：

1. 服务端接收 RGB、Depth 和目标 `class_id`
2. 服务内部使用 YOLO 做目标分割
3. 将目标区域和扩展后的 bbox 区域投影为点云
4. 使用基于 MinkowskiEngine 的 GraspNet 网络生成抓取候选
5. 按配置执行碰撞过滤、NMS 和排序
6. 返回抓取位姿、夹爪宽度、评分等结果

## 目录说明

- `grasp_module/`
  - 业务主代码
  - `app/`：FastAPI 服务入口、服务日志、warmup 输入
  - `backend/`：推理引擎、点云预处理、YOLO 分割、碰撞检测、模型定义
  - `config/`：预测配置、日志配置、RealSense 相机内参
  - `weights/`：抓取模型权重、YOLO 分割权重
  - `test/`：业务测试脚本、样例 RGB/Depth 数据、调试输出
- `env/`
  - 项目专用 Python 环境
  - 已包含 `MinkowskiEngine`、`torch`、`open3d`、`pyrealsense2`、`ultralytics`、`pointnet2`
  - 系统默认 Python 环境不能直接运行本项目
- `output_dataset/`
  - 离线测试数据
  - 包含从 RealSense `.bag` 导出的彩色图、16bit 深度图、伪彩深度图和部分点云
- `test/`
  - 数据准备脚本
  - 主要用于从 `.bag` 导出图像、深度和相机内参

## 运行环境

本项目依赖自带环境运行，不应使用系统默认 Python。

推荐解释器：

```powershell
E:\Documents_E\vscode\embedded_com\env\python.exe
```

已确认该环境可导入以下关键依赖：

- `MinkowskiEngine`
- `torch`
- `open3d`
- `pyrealsense2`
- `ultralytics`
- `pointnet2`

## 服务说明

服务入口文件：

- `grasp_module/app/server_app.py`

核心推理引擎：

- `grasp_module/backend/engine.py`

当前提供的接口：

- `POST /api/v1/init`
  - 加载抓取模型到 GPU
  - 初始化全局 predictor
  - 执行一次 warmup 推理
- `POST /api/v1/predict`
  - 输入：`rgb_file`、`depth_file`、`class_id`、`metadata`
  - 输出：抓取结果列表
- `POST /api/v1/release`
  - 释放 predictor
  - 清理显存和缓存

`metadata` 当前主要用于日志和调用方标识，通常包含：

```json
{
  "robot_id": "edge-sim",
  "cmd": "predict",
  "class_id": 46
}
```

## 当前推理链路

1. 接收 RGB、Depth 和目标 `class_id`
2. 使用 YOLO 分割出目标掩码
3. 将目标区域和扩展 bbox 区域投影为点云
4. 对目标点云采样并构造 MinkowskiEngine 输入
5. 运行 GraspNet 生成抓取候选
6. 可选执行碰撞过滤
7. 执行 NMS、排序并返回结果

## 测试与调试脚本

业务测试脚本位于 `grasp_module/test/`：

- `test_engine.py`
  - 直接调用 `RealSenseGraspPredictor`
  - 适合做单机推理验证
- `simulate_client_request.py`
  - 通过 HTTP 模拟边缘端请求服务
  - 用于验证 `/init`、`/predict`、`/release`
- `get_mask_from_img.py`
  - 用 YOLO 从图片生成二值分割掩码
  - 当前不属于主链路必需脚本

数据准备脚本位于 `test/`：

- `export_pic.py`
  - 从 RealSense `.bag` 导出颜色图、深度图、伪彩深度图和点云
- `export_camera_intrinsics.py`
  - 导出相机内参到 JSON

## 运行示例

启动服务：

```powershell
E:\Documents_E\vscode\embedded_com\env\python.exe -m grasp_module.app.server_app
```

直调引擎测试：

```powershell
E:\Documents_E\vscode\embedded_com\env\python.exe grasp_module\test\test_engine.py --yolo_class_id 46
```

模拟端侧请求：

```powershell
E:\Documents_E\vscode\embedded_com\env\python.exe grasp_module\test\simulate_client_request.py --class_id 46
```

## 配置与数据

默认配置定义在：

- `grasp_module/config/predictor_config.py`
- `grasp_module/config/global_config.py`

默认相机内参文件：

- `grasp_module/config/realsense_metadata.json`

默认权重文件：

- `grasp_module/weights/minkuresunet_realsense.tar`
- `grasp_module/weights/yolo26m-seg.pt`

测试样例数据：

- `grasp_module/test/data/color/color_00000.png`
- `grasp_module/test/data/depth/depth_raw_00000.png`

服务日志文件：

- `grasp_module/log/server.log`

查看最近日志建议只取末尾若干行，例如：

```powershell
Get-Content grasp_module\log\server.log -Tail 50
```

## 当前代码状态

当前主链路已经完成以下收敛：

- `/predict` 仅支持 `class_id`
- 引擎内部统一走 YOLO 分割，不再兼容外部 `seg_file`
- 测试脚本已同步到 `class_id` 模式
- 根目录说明文档已与当前实现对齐

当前服务端链路在本地验证中可以正常完成：

- `/init`
- 多次 `/predict`
- `/release`

典型预测耗时约为 `0.5s` 量级，warmup 后时延稳定。
