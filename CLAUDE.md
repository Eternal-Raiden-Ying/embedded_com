# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## 项目概览

本仓库的主项目是 `grasp_module` — 一个基于 RealSense RGB-D 输入的机器人抓取检测系统。核心链路：RGB+Depth 输入 → YOLO 目标分割 → 3D 点云投影 → GraspNet (MinkowskiEngine) 抓取候选生成 → 碰撞过滤 → NMS → robot 坐标系下的抓取位姿输出。通过 FastAPI 服务对外暴露 `/api/v1/init`、`/api/v1/predict`、`/api/v1/release` 接口。

根目录下 `anygrasp_sdk/`、`graspness_unofficial/`、`graspnetAPI/`、`MinkowskiEngineCuda13/`、`yolo-source/` 都是本地引用/对照用，不作为本项目主线代码维护。

## 语言

始终使用中文与用户交流。

## 运行环境

本项目**必须**使用项目自带 Python 环境，不能使用系统默认 Python：

```powershell
E:\Documents_E\vscode\embedded_com\env\python.exe
```

关键依赖（已安装到 `env/`）：`torch`、`MinkowskiEngine`（本地修改版，适配 Windows 编译）、`open3d`、`pyrealsense2`、`ultralytics`、`pointnet2`（vendored 到 `third_party/pointnet2/`）、`graspnetAPI`。

具体依赖说明见 `ENVIRONMENT.md`。

## 常用命令

启动 FastAPI 服务：
```powershell
E:\Documents_E\vscode\embedded_com\env\python.exe -m grasp_module.app.server_app
```

YOLO 预检（不加载 GraspNet，启动快，先看 class_id 是否可能命中）：
```powershell
E:\Documents_E\vscode\embedded_com\env\python.exe grasp_module\test\test_yolo.py --rgb_path <color.png> --yolo_class_id 47
```

Grasp 主调试入口（先做 YOLO 预检，通过后再加载模型推理）：
```powershell
E:\Documents_E\vscode\embedded_com\env\python.exe grasp_module\test\test_engine.py --rgb_path <color.png> --depth_path <depth.png> --yolo_class_id 47
```

从 .bag 回放 debug（不跑 YOLO 不跑 grasp，纯看点云/深度质量）：
```powershell
E:\Documents_E\vscode\embedded_com\env\python.exe grasp_module\test\handeye_from_bag.py --bag_file <xxx.bag>
```

HTTP 接口联调：
```powershell
E:\Documents_E\vscode\embedded_com\env\python.exe grasp_module\test\simulate_client_request.py --class_id 47
```

查看服务最近日志：
```powershell
Get-Content grasp_module\log\server.log -Tail 50
```

编译检查（无完整 CI，用 py_compile 验证语法）：
```powershell
E:\Documents_E\vscode\embedded_com\env\python.exe -m py_compile <file.py>
```

## 代码架构

### `grasp_module/app/`
FastAPI 服务层。`server_app.py` 管理全局 `RealSenseGraspPredictor` 单例，提供 init/predict/release 三个端点。predict 接口接收 `rgb_file` (UploadFile)、`depth_file` (UploadFile)、`class_id` (int)、`metadata` (JSON string)。

### `grasp_module/backend/`
核心推理引擎和相关模型：
- `engine.py` — `RealSenseGraspPredictor` 类，核心推理入口。`infer()` 方法串联完整链路：YOLO 分割 → 点云生成 → 采样/预处理 → MinkowskiEngine 前向 → 碰撞检测 → 后处理。`build_protocol_targets()` 将 grasp 结果转换到 robot 坐标系并过滤不可行角度。`save_debug_visualizations()` 导出 masked_cloud / scene_cloud / grasp mesh 到 PLY。
- `models/graspnet.py` — GraspNet 网络定义（MinkUNet14D backbone → ApproachNet → CloudCrop → SWADNet）+ `pred_decode()` 将网络输出解码为 GraspGroup。
- `models/label_generation.py` — 训练时的标签生成，包含 `M_POINT=1024` 等与模型结构耦合的常量。
- `models/knn/` — 自维护的 KNN 扩展（从 graspness_unofficial 修复 bug 后迁移至此），不再依赖外部 graspness_unofficial。
- `utils/data_utils.py` — `CameraInfo`、RGB-D 到点云投影（`create_colored_point_cloud_from_rgbd`）、PLY 读写。
- `utils/yolo_utils.py` — YOLO 分割封装：`predict_target_masks()` 返回 seg_mask、bbox_mask（扩张后的 bbox）、overlay_img、检测信息。
- `utils/collision_detector.py` — `ModelFreeCollisionDetector`：基于体素占用的无模型碰撞检测，参数（finger_width/length/height_override）由 predictor config 控制。
- `utils/frames.py` — `FrameTransformer`：camera → robot 坐标系变换（旋转+平移），从 config 的 CSV 参数初始化。
- `utils/gripper_mesh.py` — 本地构建 debug 夹爪 mesh，不依赖 graspnetAPI 运行时签名。

### `grasp_module/config/`
- `predictor_config.py` — `PredictorConfig` dataclass 定义所有可调参数及其默认值，提供 `add_predictor_args()` 用于 argparse。
- `global_config.py` — 继承 `PredictorConfig` 的 `AppConfig`，服务启动时通过 `parse_known_args()` 加载。
- `logging_config.py` — 统一 logger name `vision.grasp`。

### `grasp_module/test/`
调试脚本和工具：
- 建议使用顺序：`test_yolo.py` → `test_engine.py` → `handeye_from_bag.py` → `simulate_client_request.py`
- `utils/bag_io.py` — 统一的 .bag 读取/抽帧/深度后处理/点云导出，所有 bag 相关脚本共享。内含深度中值滤波 + 零值孔洞填充，以及 RealSense SDK 官方 filter chain 的封装。
- `utils/yolo_probe.py` — YOLO 轻量预检和 fallback 路由（不导入 GraspNet）。
- `utils/io_utils.py` — 通用 IO：`ensure_dir`、`save_json`、`parse_int_list_csv`。

### `third_party/pointnet2/`
Vendored pointnet2 源码，供 GraspNet 的 `furthest_point_sample` / `gather_operation` 使用。已在 `env/` 中编译安装。

### `.gitignore`
忽略 `env/`、`output_dataset/`、`grasp_module/weights/`、`grasp_module/log/`、debug 输出目录、`.bag`/`.ply`/`.npy`、第三方仓库镜像、编译产物 (knn build/、pointnet2 build/)。如有历史已跟踪文件需要移除，用 `git rm --cached`。

## 当前开发状态与优先级

详见 `TODO.md`（最后更新 2026-04-26）：

- **P0 基础重构**：坐标系转换模块化 (done)、中间结果可观测性 (doing，需补碰撞剔除 grasp 单独可视化)
- **P1 上游感知稳定**：YOLO 偶发漏检排查 (doing，瓶颈在 apple=47 置信度偏低与类别混淆，已有 fallback 47→32→55)、3D 投影畸变 (todo)、深度后处理验证 (todo)、边界缺失问题 (todo)
- **P2 机器人约束**：夹爪/机械臂执行语义重定义 (todo)、机械臂约束过滤优化 (doing，已有 pitch/roll/feasible_angle，待补 pitch 硬过滤和 IK)
- **P3 碰撞/评分优化**：碰撞剔除 grasp 可视化 (todo)、场景点云选取优化 (todo)、评分逻辑优化 (todo)、seed/M_POINT 调整 (blocked，等上游稳定)
- **P4 自动化与工具**：手眼标定自动化 (todo)、测试脚本体系整理 (doing)

## 编写注意事项

- 所有测试脚本通过 argparse 控制参数，不走硬编码。参数设置与 `PredictorConfig` 的 default 保持一致。
- `engine.py` 中 YOLO 的 conf/iou 在 `predict()` 调用时传入，分别是 `yolo_conf`（默认 0.25）和 `yolo_iou`（默认 0.7）。YOLO 预检/fallback 阶段使用 `fallback_probe_conf`（默认 0.10）。
- 坐标系转换统一走 `FrameTransformer`：camera 向量/点通过配置的 R,t 转到 robot 坐标系。debug 点云和 grasp mesh 保留在原始 camera 坐标系。
- 碰撞检测的几何参数（`collision_finger_width_m`、`collision_finger_length_m`、`collision_height_override_m`）与 debug 可视化夹爪 mesh 参数（`gripper_height_m`、`gripper_finger_width_m` 等）是**两组独立参数**，分别控制碰撞判据和可视化外观。
- 修改 `predictor_config.py` 中的 default 值时，确保 test 脚本的 argparse default 保持一致（它们共享 `add_predictor_args`）。
- .md 文件若出现乱码，用 UTF-8 编码重新打开。
