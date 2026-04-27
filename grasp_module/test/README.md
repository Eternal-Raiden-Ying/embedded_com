# Test Scripts

本目录主要用于离线调试 `YOLO -> grasp -> robot` 链路、RealSense `.bag` 回放、点云导出和接口联调。

统一运行环境：

```powershell
E:\Documents_E\vscode\embedded_com\env\python.exe
```

## 脚本概览

### `test_yolo.py`

用途：
- 只做 YOLO 预检
- 支持单图和 `.bag`
- 不导入 GraspNet，启动更快
- 适合检查：
  - `class_id`
  - fallback 是否生效
  - 是否单目标 / 多目标
  - overlay 结果

常用启动：

```powershell
E:\Documents_E\vscode\embedded_com\env\python.exe E:\Documents_E\vscode\embedded_com\grasp_module\test\test_yolo.py --rgb_path <color.png>
```

```powershell
E:\Documents_E\vscode\embedded_com\env\python.exe E:\Documents_E\vscode\embedded_com\grasp_module\test\test_yolo.py --bag_file <xxx.bag>
```

主要参数：

- `--rgb_path`
  - 单图输入 RGB 路径
  - 默认：`grasp_module/test/data/color/color_00000.png`
- `--depth_path`
  - 单图模式下仅用于保存本地调试输入
  - 默认：`grasp_module/test/data/depth/depth_raw_00000.png`
- `--yolo_model`
  - YOLO 权重名或路径
  - 默认：`yolo26m-seg.pt`
- `--yolo_weights_dir`
  - YOLO 权重目录
  - 默认：`grasp_module/weights`
- `--yolo_class_id`
  - 主检测类别
  - 默认：`47`
- `--yolo_conf`
  - 主配置里的 YOLO 阈值，当前主要用于打印和参考
  - 默认：`0.25`
- `--yolo_iou`
  - YOLO NMS IoU 阈值
  - 默认：`0.7`
- `--bbox_expand_scale`
  - bbox 扩张倍数
  - 默认：`2.0`
- `--fallback_class_ids_csv`
  - 检测 fallback 顺序
  - 默认：`32,55`
- `--fallback_probe_conf`
  - fallback 探测时的置信度阈值
  - 默认：`0.10`
- `--dump_dir`
  - 单图输出目录
  - 默认：`grasp_module/test/yolo_debug`
- `--bag_file`
  - `.bag` 输入路径
  - 默认：空
- `--bag_output_dir`
  - bag 输出目录
  - 默认：`grasp_module/test/yolo_debug/<bag名>`
- `--bag_top_k`
  - 从候选帧中最多检查前几帧
  - 默认：`1`
- `--bag_stride`
  - 每隔多少帧采样一次
  - 默认：`1`
- `--bag_max_frames`
  - 最多检查多少个采样帧
  - 默认：`60`
- `--bag_min_valid_ratio`
  - depth 非零比例低于此值的帧会被丢弃
  - 默认：`0.0`

输出内容：

- 单图：
  - `summary.json`
  - `yolo_overlay.jpg`
- bag：
  - `summary.json`
  - 每个候选帧一个目录，包含：
    - `color.png`
    - `depth_raw.png`
    - `summary.json`
    - `yolo_overlay.jpg`

bag 处理逻辑：

1. 用 `pyrealsense2` 读取 `.bag`
2. `align(color -> depth 输出坐标)`
3. 每隔 `bag_stride` 取样
4. 统计每帧：
   - `zero_count`
   - `valid_ratio`
5. 排序规则：
   - `zero_count` 少优先
   - `valid_ratio` 高优先
   - `frame_index` 小优先
6. 默认只检查排序后的前 `1` 帧
7. 如果 `bag_top_k > 1`，则会在前 `K` 帧中优先选“单目标检出”的第一帧，否则退到“多目标中最高置信度”的帧

### `test_engine.py`

用途：
- Grasp 调试主入口
- 先做 YOLO 预检
- 只有预检通过时才导入并构造 `RealSenseGraspPredictor`
- 支持单图和 `.bag`
- 适合检查：
  - fallback 后最终是否能进入 grasp
  - 输出的 `robot` 坐标
  - debug 点云和夹爪 mesh

常用启动：

```powershell
E:\Documents_E\vscode\embedded_com\env\python.exe E:\Documents_E\vscode\embedded_com\grasp_module\test\test_engine.py --rgb_path <color.png> --depth_path <depth.png>
```

```powershell
E:\Documents_E\vscode\embedded_com\env\python.exe E:\Documents_E\vscode\embedded_com\grasp_module\test\test_engine.py --bag_file <xxx.bag>
```

主要参数：

- `--checkpoint_path`
  - GraspNet 权重路径
  - 默认：`grasp_module/weights/minkuresunet_realsense.tar`
- `--dump_dir`
  - 单图 debug 输出目录
  - 默认：`grasp_module/test/debug_res`
- `--seed_feat_dim`
  - GraspNet seed feature 维度
  - 默认：`512`
- `--num_point`
  - 点云采样点数
  - 默认：`15000`
- `--voxel_size`
  - 稀疏卷积输入体素尺寸
  - 默认：`0.005`
- `--collision_thresh`
  - 碰撞阈值，`<=0` 表示关闭碰撞检测
  - 默认：`-1.0`
- `--voxel_size_cd`
  - 碰撞检测体素尺寸
  - 默认：`0.01`
- `--random_seed`
  - 随机种子
  - 默认：`0`
- `--scene_max_depth`
  - 调试场景点云的最大深度
  - 默认：`3.0`
- `--debug_grasp_count`
  - debug 输出的 grasp 数量
  - 默认：`15`
- `--rgb_path`
  - 单图 RGB 路径
  - 默认：`grasp_module/test/data/color/color_00000.png`
- `--depth_path`
  - 单图 depth 路径
  - 默认：`grasp_module/test/data/depth/depth_raw_00000.png`
- `--camera_metadata`
  - 相机内参 JSON
  - 默认：`grasp_module/config/realsense_metadata.json`
- `--yolo_model`
  - YOLO 权重名或路径
  - 默认：`yolo26m-seg.pt`
- `--yolo_weights_dir`
  - YOLO 权重目录
  - 默认：`grasp_module/weights`
- `--yolo_class_id`
  - 主检测类别
  - 默认：`47`
- `--yolo_conf`
  - YOLO 正式推理阈值
  - 默认：`0.25`
- `--yolo_iou`
  - YOLO NMS IoU 阈值
  - 默认：`0.7`
- `--bbox_expand_scale`
  - bbox 扩张倍数
  - 默认：`2.0`
- `--collision_depth_margin`
  - bbox 场景点云的深度扩张 margin
  - 默认：`0.15`
- `--protocol_depth_base`
  - 输出末端点相对 grasp 原点的偏移
  - 默认：`-0.085`
- `--protocol_feasible_angle_deg`
  - 允许的机械臂约束角度
  - 默认：`5.0`
- `--protocol_min_score`
  - 协议输出的最小置信度阈值
  - 默认：`0.3`
- `--response_max_targets`
  - 最多输出多少个 target
  - 默认：`5`
- `--robot_cam_rotation_csv`
  - 标定右手系到 robot 的旋转矩阵
  - 默认：`0.7701,-0.0703,0.6340,0.1184,0.9924,-0.0338,-0.6268,0.1011,0.7726`
- `--robot_calibration_translation_cm_csv`
  - 标定右手系到 robot 的平移向量，单位 `cm`
  - 默认：`-8.86370095,2.28825035,34.46293759`
- `--debug`
  - 是否保存 debug 结果
  - 默认：开启
- `--bag_file`
  - `.bag` 输入路径
  - 默认：空
- `--bag_output_dir`
  - bag 输出目录
  - 默认：`grasp_module/test/bag_debug/<bag名>`
- `--bag_top_k`
  - 检查前多少个候选帧
  - 默认：`1`
- `--bag_stride`
  - 每隔多少帧采样一次
  - 默认：`1`
- `--bag_max_frames`
  - 最多检查多少个采样帧
  - 默认：`60`
- `--bag_min_valid_ratio`
  - depth 非零比例低于此值的帧会被丢弃
  - 默认：`0.0`
- `--fallback_class_ids_csv`
  - 测试层 fallback 顺序
  - 默认：`32,55`
- `--fallback_probe_conf`
  - fallback 探测阈值
  - 默认：`0.10`

输出内容：

- 单图：
  - `summary.json`
  - `yolo_overlay.jpg`
  - `ply/masked_cloud.ply`
  - `ply/scene_cloud.ply`
  - `ply/grasps_top15_heatmap.ply`
  - `ply/best_protocol_grasp.ply`
- bag：
  - bag 根目录 `summary.json`
  - 只对最终选中的 1 帧做 grasp 推理
  - 所有 inspected 候选帧都会有自己的 `summary.json` 和 `yolo_overlay.jpg`

bag 处理逻辑：

1. 从 `.bag` 抽帧并按 `zero_count / valid_ratio / frame_index` 排序
2. 默认只检查前 `1` 帧
3. 先做 YOLO 预检
4. 优先选“单目标检出”的帧
5. 如果没有单目标，但有多目标，则选“多目标中置信度最高”的帧
6. 只对最终选中的这 `1` 帧做 grasp 推理

### `handeye_from_bag.py`

用途：
- 从 `.bag` 导出对齐后的 RGB、Depth 和点云
- 不跑 YOLO
- 不跑 grasp
- 适合手眼标定和点云质量检查

常用启动：

```powershell
E:\Documents_E\vscode\embedded_com\env\python.exe E:\Documents_E\vscode\embedded_com\grasp_module\test\handeye_from_bag.py --bag_file <xxx.bag>
```

主要参数：

- `--bag_file`
  - `.bag` 输入路径
  - 默认：必填，无默认值
- `--output_dir`
  - 输出目录
  - 默认：`grasp_module/test/handeye_debug/<bag名>`
- `--bag_top_k`
  - 导出前多少个候选帧
  - 默认：`1`
- `--bag_stride`
  - 每隔多少帧采样一次
  - 默认：`5`
- `--bag_max_frames`
  - 最多检查多少个采样帧
  - 默认：`80`
- `--bag_min_valid_ratio`
  - depth 非零比例低于此值的帧会被丢弃
  - 默认：`0.0`
- `--z_min`
  - 导出点云最小深度，单位 `m`
  - 默认：`0.15`
- `--z_max`
  - 导出点云最大深度，单位 `m`
  - 默认：`2.0`
- `--browse_voxel_size`
  - 浏览版点云的体素降采样尺寸，单位 `m`
  - 默认：`0.01`

输出内容：

- `camera_metadata.json`
- 每个候选帧一个目录，包含：
  - `color.png`
  - `depth_raw.png`
  - `summary.json`
  - `ply/scene_cloud_raw.ply`
  - `ply/scene_cloud_filtered.ply`
  - `ply/scene_cloud_browse.ply`

### `simulate_client_request.py`

用途：
- 模拟 HTTP 客户端
- 联调 `/api/v1/init -> /api/v1/predict -> /api/v1/release`
- 不用于排查 YOLO 细节和点云质量

常用启动：

```powershell
E:\Documents_E\vscode\embedded_com\env\python.exe E:\Documents_E\vscode\embedded_com\grasp_module\test\simulate_client_request.py --class_id 47
```

主要参数：

- `--server_url`
  - 服务地址
  - 默认：`http://127.0.0.1:6006`
- `--robot_id`
  - metadata 里的 robot id
  - 默认：`edge-sim`
- `--cmd`
  - metadata 里的命令字段
  - 默认：`predict`
- `--class_id`
  - 发送给服务端的类 id
  - 默认：`46`
- `--skip_init`
  - 跳过 `/init`
  - 默认：关闭
- `--skip_release`
  - 跳过 `/release`
  - 默认：关闭
- `--timeout`
  - HTTP 超时时间，单位 `s`
  - 默认：`120.0`
- `--predict_repeats`
  - 重复请求次数
  - 默认：`3`

### `get_mask_from_img.py`

用途：
- 对单张图片跑 YOLO 分割
- 导出 `seg_*.npy` 和 overlay 图
- 适合快速验证某个类在单图上能不能检出

常用启动：

```powershell
E:\Documents_E\vscode\embedded_com\env\python.exe E:\Documents_E\vscode\embedded_com\grasp_module\test\get_mask_from_img.py --img_path <color.png> --class_id 47
```

主要参数：

- `--img_path`
  - 输入图片路径
  - 默认：必填，无默认值
- `--weights_dir`
  - 权重目录
  - 默认：`../weights`
- `--model`
  - 模型名或路径
  - 默认：`yolo26m-seg.pt`
- `--class_id`
  - 指定类别 id
  - 默认：`46`
- `--output_dir`
  - 输出目录
  - 默认：`./data/seg`

## `test/utils` 模块说明

### `utils/bag_io.py`

作用：
- 统一 `.bag` 读取、对齐、抽帧、排序、metadata 导出、点云导出

主要能力：
- `collect_bag_frames`
- `collect_bag_candidates`
- `save_bag_frame_inputs`
- `save_point_cloud_frame_outputs`

### `utils/yolo_probe.py`

作用：
- 统一 YOLO 轻量预检和 fallback 路由
- 不导入 grasp 模型

主要能力：
- `load_probe_model`
- `probe_single_class`
- `resolve_detection_route`
- `choose_detection_frame`

### `utils/io_utils.py`

作用：
- 通用 IO 和参数打印

主要能力：
- `ensure_dir`
- `save_json`
- `parse_int_list_csv`
- `log_kv_block`

### `utils/reporting.py`

作用：
- 统一构造 grasp 调试输出 summary

主要能力：
- `build_downstream_response`
- `summarize_top_raw_grasp`

## 建议使用顺序

1. 先用 `test_yolo.py` 看当前类 id 和 fallback 是否可用
2. 再用 `test_engine.py` 跑 grasp
3. 如果主要看点云质量或手眼标定，使用 `handeye_from_bag.py`
4. 如果主要看服务端接口联调，使用 `simulate_client_request.py`
