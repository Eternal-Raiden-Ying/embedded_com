# Table Plane Detector Modes Analysis

本文分析当前 `vision_module` 中桌边/桌前平面检测的两条核心路径：

- `full` detector mode
- `fast_plane_only` detector mode

重点说明各阶段算法细节、滤波和拟合逻辑、两者共同复用的模块，以及 offline bag 测试和 online runtime 测试时的调用路径差异。

## 结论概览

当前真正被 `vision_module` 统一入口调用的是：

- `VISTA/vision_module/backend/table_edge_manager.py`
  - `TableEdgeManager.process_camera_frame(...)`
  - `TableEdgeManager._process_depth(...)`
  - `TableEdgeManager._process_depth_fast_plane_only(...)`

两种 detector mode 的分流点在：

```python
TableEdgeManager._process_depth(...)
```

逻辑是：

- `detector_mode == "fast_plane_only"`：走 `TableEdgeManager._process_depth_fast_plane_only(...)`
- 否则默认 `full`：走 `OnlineTableEdgeDetector.process_depth(...)`

`full` mode 的核心算法仍在旧目录：

```text
VISTA/Online_Edge_Detect/detector.py
```

`fast_plane_only` 的核心算法在当前主模块：

```text
VISTA/vision_module/backend/table_edge_manager.py
```

## 共同入口和共同复用模块

### 统一入口

无论 offline bag 还是 online runtime，只要走当前 `vision_module` 的 table edge path，最终都会调用：

```python
TableEdgeManager.process_camera_frame(...)
```

该函数负责：

1. 接收 `frames` 字典，读取 `frames["depth"]`
2. 检查 depth 是否存在、是否是 2D ndarray
3. 设置 frame seq、时间戳、source mode、本地感知上下文
4. 调用 `_process_depth(depth, seq)`
5. 包装 freshness / age / process_ms 等运行时字段
6. 返回统一的 `table_edge_obs`

### 共同复用内容

两种模式共同复用：

- `TableEdgeManager`
- `VisionServiceConfig`
- `table_edge.detector_mode`
- `choose_depth_roi(...)`
- ROI metadata 和 `roi_payload`
- YOLO gate / local perception gate 逻辑
- 标定加载结果中的 camera intrinsics
- `target_dist_m`
- 输出字段兼容层：`table_edge_obs`
- preview 渲染入口
- offline bag eval 的 JSON/CSV/preview 导出框架
- online runtime 的 scheduler publish 路径

共同 ROI 选择逻辑在：

```python
TableEdgeManager._select_roi(...)
```

其来源优先级大致是：

1. locked ROI
2. local perception / YOLO table bbox
3. debug roi preset，例如 `center_lower`
4. static fallback

共同 YOLO gate 在：

```python
TableEdgeManager._yolo_table_confirmation(...)
```

注意：当前用户运行 `fast_plane_only + roi-preset center_lower` 时，ROI 是通过 `TableEdgeManager._select_roi(...)` 进入 fast path 的。fast path 本身没有 YOLO/RGB 检测逻辑，但仍会经过同一个 gate/preset 体系。

## Full Detector Mode

### 调用路径

`full` mode 的调用链：

```text
TableEdgeManager.process_camera_frame(...)
  -> TableEdgeManager._process_depth(...)
    -> TableEdgeManager._select_roi(...)
    -> TableEdgeManager._yolo_table_confirmation(...)
    -> OnlineTableEdgeDetector.process_depth(...)
```

核心实现文件：

```text
VISTA/Online_Edge_Detect/detector.py
```

`TableEdgeManager` 在初始化时通过 `_load_detector()` 动态加载：

```python
VISTA.Online_Edge_Detect.board_config.CONFIG
VISTA.Online_Edge_Detect.detector.OnlineTableEdgeDetector
VISTA.Online_Edge_Detect.detector.load_calib
```

所以 `full` mode 现在仍然依赖 `VISTA/Online_Edge_Detect`。

### Stage 1: ROI 和 depth preprocessing

入口：

```python
OnlineTableEdgeDetector.process_depth(depth_image_16bit, roi_override=None)
```

先调用：

```python
_preprocess_depth(...)
```

处理内容：

1. 解析 ROI：
   - 如果 `TableEdgeManager` 传入 `roi_override`，优先使用 override
   - 否则使用 `Online_Edge_Detect` 自己的 config ROI
2. ROI clipping：
   - 防止 ROI 超出 depth image
3. depth median blur：
   - 使用 `cfg.depth_median_ksize`
   - kernel 合法且大于 1 时做 `cv2.medianBlur`
4. depth scale：
   - 16-bit depth 转米
   - `depth_m = raw * calib.depth_scale`
5. depth range filter：
   - `z_min < depth_m < z_max`

输出：

- `valid_mask`
- `depth_meters`
- `roi_box`

### Stage 2: Camera XYZ 点云构建

调用：

```python
_depth_to_3d(depth_meters, valid_mask, roi_box)
```

用 pinhole camera model 投影：

```text
X_cam = (u - cx) * Z / fx
Y_cam = (v - cy) * Z / fy
Z_cam = depth
```

输出 `pc_cam = [X_cam, Y_cam, Z_cam]`。

如果点数少于 `cfg.min_all_points`，直接返回 `roi_empty`。

### Stage 3: Legacy table point mask

调用：

```python
_find_table_plane(pc_cam)
```

逻辑很简单：

```python
table_y_min < Y_cam < table_y_max
```

这个结果主要用于统计 `table_point_count` 和 debug。当前 full mode 的主姿态估计更多依赖后面的 front plane / crease line。

### Stage 4: Front Plane Estimation

调用：

```python
_estimate_front_plane(depth_meters, valid_mask, roi_box)
```

这是 full mode 的主要平面检测逻辑之一。

#### 4.1 XYZ map

先构建 ROI 内每个像素的：

- `x_map`
- `y_map`
- `z_map`

#### 4.2 局部法向量候选筛选

对 ROI 内每个非边界像素，用左右相邻点形成 `vx`，上下相邻点形成 `vy`：

```text
vx = P(u+1,v) - P(u-1,v)
vy = P(u,v+1) - P(u,v-1)
normal = cross(vx, vy)
```

然后过滤：

- 中心点和四邻域 depth 都必须有效
- normal norm 必须有效
- `abs(normal_y) <= plane_max_abs_normal_y`
- `abs(normal_z) >= plane_min_abs_normal_z`

这些候选被视为“可能的桌前竖直平面像素”。

#### 4.3 RANSAC 平面拟合

对候选点做 RANSAC：

1. 如果候选点超过 5000，随机采样 5000 个做拟合
2. 每轮随机选 3 个点
3. 通过 3 点构造平面：

```text
normal = cross(p2 - p1, p3 - p1)
d = -dot(normal, p1)
plane: normal dot P + d = 0
```

4. 过滤不符合竖直前平面方向的 normal：
   - `abs(normal_y) <= plane_max_abs_normal_y`
   - `abs(normal_z) >= plane_min_abs_normal_z`
5. 计算所有 fit points 到平面的 residual
6. residual 小于 `plane_max_residual_m` 的点为 inlier
7. 选择 inlier count 最多、mean residual 最小的平面

#### 4.4 平面姿态转 yaw/dist

平面拟合得到：

```text
normal = [nx, ny, nz]
d
```

转换成 bird-view 线：

```text
k = -nx / nz
b = -d / nz
yaw = atan(k)
dist = b - target_dist_m
```

这里的 full mode 仍主要使用 camera X/Z 几何关系，而不是 fast mode 里的 robot XYZ ground-frame height filter。

#### 4.5 平面 confidence

front plane confidence 来自：

- residual score
- x span score
- area ratio score
- inlier count score

大致形式：

```text
conf = 0.30 * residual_score
     + 0.25 * span_score
     + 0.25 * area_score
     + 0.20 * inlier_score
```

#### 4.6 Front plane reject reason

`_plane_reject_reason(...)` 会检查：

- no reliable plane
- front face area 太小
- residual 太高
- x span 太短
- front plane score 太低

### Stage 5: Crease Line Estimation

如果 `plane_only_mode=False` 且 `enable_crease_line=True`，full mode 还会跑 crease line。

调用：

```python
_estimate_crease_line(depth_meters, valid_mask, roi_box, plane)
```

#### 5.1 沿列寻找 depth trend 突变

对 ROI 内每隔 `trend_col_step_px` 的列做扫描：

1. 对每个候选 row，用上下窗口分别拟合 `row -> depth`
2. 计算上下窗口 slope 差：

```text
score = abs(slope_below - slope_above)
```

3. score 大于 `trend_min_slope_delta` 的位置被视为可能的边缘/折线点
4. 每列保留 top-k 候选

#### 5.2 分 upper crease 和 lower contact

根据候选点在 ROI 内的 normalized y 位置分成：

- upper crease
- lower contact

分别调用：

```python
_fit_line_hypothesis(...)
```

#### 5.3 RANSAC XZ 线拟合

line hypothesis 内部调用：

```python
_fit_ransac_xz(points, threshold_m=line_select_max_residual_m)
```

逻辑：

1. 取点的 `X_cam` 和 `Z_cam`
2. RANSAC 每轮随机选 2 点构造线：

```text
Z = k * X + b
```

3. residual：

```text
abs(Z - (kX + b))
```

4. residual 小于阈值的是 inlier
5. 用最佳 inliers 再 `np.polyfit` 精拟合
6. 输出：
   - yaw = `atan(k)`
   - dist = `b - target_dist_m`
   - confidence = `inlier_count / candidate_count`
   - x span
   - residual mean

#### 5.4 line 额外滤波

每条 line hypothesis 还会检查：

- candidate count
- residual
- x span
- confidence
- ROI boundary touch ratio
- 与 front plane yaw 一致性
- 与 front plane 上/下边界距离一致性
- object-like score

object-like score 会惩罚：

- 线太短
- 点太稀疏
- inlier ratio 低
- 离 plane boundary 太远
- 触碰 ROI 边界过多

最终可能的 reject reason：

- `too_few_candidates`
- `ransac_failed`
- `line_plane_boundary_mismatch`
- `object_like_line`
- `line_too_short`
- `line_residual_high`
- `line_confidence_low`

### Stage 6: Front Plane 和 Crease Line 融合

调用：

```python
_fuse_front_pose(plane, line)
```

融合逻辑：

- 如果 plane-only 或 crease line disabled：只用 front plane
- 如果 plane 和 line 都可靠：
  - yaw 差小于 `fusion_yaw_consistency_rad` 时做加权融合
  - 权重来自各自 confidence
  - boundary consistency 差时偏向 plane
- 如果二者 yaw 冲突：
  - 选择 confidence 高的一个
  - pose_source 标记为 `conflict`
  - reject reason `plane_line_yaw_conflict`
- 如果只有 plane 可靠：用 front plane
- 如果只有 line 可靠：用 crease line
- 都不可靠：pose not found

### Stage 7: Geometry Score 和 Control Gate

调用：

```python
_score_table_geometry(...)
_validate_pose_for_control(...)
```

geometry score 综合：

- front plane score
- line score
- plane-line consistency
- ROI boundary score
- temporal score

control gate 检查：

- confidence
- x span
- residual
- yaw range
- temporal jump
- stable frame count

full mode 的 control level 命名是：

- `approach`
- `alignment`
- `stop`
- `none`

注意这和 fast mode 当前输出的：

- `approach_slow`
- `align`
- `stop_ready`
- `rotate_only`
- `none`

命名不完全一致。`TableEdgeManager` 通过兼容字段把它们都包装进 `table_edge_obs`。

## Fast Plane Only Mode

### 调用路径

`fast_plane_only` 的调用链：

```text
TableEdgeManager.process_camera_frame(...)
  -> TableEdgeManager._process_depth(...)
    -> TableEdgeManager._process_depth_fast_plane_only(...)
```

核心实现文件：

```text
VISTA/vision_module/backend/table_edge_manager.py
```

该路径不调用 `OnlineTableEdgeDetector.process_depth(...)` 做检测，但仍依赖 `_load_detector()` 加载到的 calibration、target distance 和部分 detector config。

### Stage 1: ROI 和 sparse depth sampling

fast path 先调用：

```python
_select_roi(depth_frame)
_yolo_table_confirmation()
```

ROI 选择和 full mode 共用。

然后：

```python
depth_roi = depth_frame[y0:y1:stride, x0:x1:stride]
```

其中 `stride = table_edge.fast_plane_stride`。

这意味着 fast path 不处理全分辨率 ROI，而是直接 sparse sampling。

### Stage 2: Depth valid filter

把 depth 转米：

```text
depth_m = raw_depth * calib.depth_scale
```

然后用 full detector config 里的 depth range：

```text
z_min < depth_m < z_max
```

如果有效点数小于按 stride 缩放后的 `min_all_points`，返回 `not_enough_points`。

### Stage 3: Camera XYZ projection

对 sparse valid points 投影：

```text
X_cam = (u - cx) * Z / fx
Y_cam = (v - cy) * Z / fy
Z_cam = depth
```

这里和 full mode 的投影公式一致。

### Stage 4: Camera XYZ -> Robot XYZ

fast mode 和 full mode 最大差别之一在这里。

fast mode 使用固定相机外参近似：

- `camera_pitch_deg`
- `camera_height_m`

转换到 robot/world-like frame：

```text
X_robot = X_cam
Y_robot = Z_cam * cos(theta) - Y_cam_down * sin(theta)
Z_robot = camera_height - (Z_cam * sin(theta) + Y_cam_down * cos(theta))
```

当前 corrected 参数：

```text
camera_pitch_deg = 15.0
camera_height_m = 0.70
table_height_m = 0.40
```

### Stage 5: Height Candidate Filter

fast path 不先找法向量，也不跑 full RANSAC 平面。

它先用 robot Z 过滤桌前竖直面候选：

```text
front_face_z_min_m < Z_robot < front_face_z_max_m
```

这一步生成：

- `fast_candidate_point_count`
- `fast_candidate_pixels`
- `fast_candidate_x_span_m`

语义是：有效深度点中，高度落在“桌前立面候选高度范围”的点。

这批点不一定已经是桌前面，也可能包含背景板、地面附近点、桌面边缘或其它物体。

### Stage 6: Vertical Support / Representative Selection

调用：

```python
_select_fast_front_face_representatives(...)
```

输入是 height candidates 的：

- `X_robot`
- `Y_robot`
- `Z_robot`
- pixel coordinates

处理逻辑：

1. 按 `X_robot / x_bin_width_m` 分 bin
2. 每个 X bin 内再按 `Y_robot / y_cluster_bin_m` 分 cluster
3. 对每个 local Y cluster 检查：
   - support point 数量 >= `min_vertical_support_points`
   - Z span >= `min_vertical_z_span_m`
   - Y spread 不过大
4. 对每个 X bin 选择 score 最高的 cluster
5. 每个被选 cluster 输出一个 representative：
   - median X
   - median Y
   - median Z
   - median pixel
   - support count
   - z span
   - support pixels

这一步的目标是找“沿竖直方向有高度跨度、并且在局部 Y 上连续”的桌前立面支撑，而不是把所有 height candidates 都拿去拟合。

输出：

- `support_pixels`
- `reps`
- `support_count`
- `z_span`

失败原因：

- `vertical_support_low`
- `front_face_columns_low`

### Stage 7: Bird-view Representative Fitting

fast mode 的核心拟合不是平面 RANSAC，而是 representative 的 bird-view 线拟合。

拟合变量：

```text
x_t = representative X_robot
y_t = representative Y_robot
```

拟合形式：

```text
Y_robot = k * X_robot + b
```

helper：

```python
_weighted_line_fit(x_t, y_t, weights)
```

权重来源：

- `sqrt(support_count)`
- `z_span / min_vertical_z_span`
- 最终 clip 到保守范围，避免单个大 cluster 主导

当前逻辑：

1. 对所有 reps 做 weighted line fit
2. 计算每个 rep residual：

```text
abs(Y - (kX + b))
```

3. 做 neighbor continuity check：
   - isolated representative 会被标记为不连续
4. residual 小于阈值且连续的 reps 为 inlier
5. 如果 inlier 数足够，用 inlier reps 重新 weighted fit
6. 输出：
   - `fast_rep_inlier_count`
   - `fast_rep_outlier_count`
   - `fast_fit_inlier_x_span_m`
   - `fast_residual_mean`
   - `fast_residual_p90`

姿态转换：

```text
yaw_err = atan(k)
dist_err = b - target_dist_m
```

### Stage 8: Fast Confidence

fast confidence 综合：

- representative inlier evidence
- support inlier ratio
- residual score
- fit span / coverage
- yaw geometry score
- vertical support geometry score
- temporal score

关键区别：

- candidate span 不再直接当 fit span
- support span、rep span、fit inlier span 分开统计
- confidence 中加入 support geometry score，但没有恢复 full-resolution mask

### Stage 9: Fast Control Gate

fast path gate 顺序大致是：

1. representative inlier 数不足：`vertical_support_low`
2. support inlier 数不足：`vertical_support_low`
3. rep columns 不足：`front_face_columns_low`
4. residual 太大：`residual_too_large`
5. yaw 超限：`yaw_out_of_range`
6. preview width 太小：`width_too_small`
7. fit x span 太小：`front_face_x_span_low`
8. temporal jump：`temporal_jump`
9. confidence 不足：`confidence_too_low`

control level：

- far/middle/near 根据 `dist_err` 分段
- yaw 大时可能 `rotate_only`
- near 且 dist/yaw/confidence 满足时 `stop_ready`
- 中等情况 `align`
- 其它可用但弱时 `approach_slow`

额外弱支撑 downgrade：

- `rep_inlier_count <= 3` 或 `fit_inlier_x_span < 0.15m`
  - 不允许 `stop_ready`
  - `align/stop_ready` 会降到 `approach_slow`
- `rep_inlier_count = 4..7` 或 `fit_span = 0.15..0.30m`
  - 不允许 `stop_ready`
  - 最多 `align`
- `yaw_out_of_range` 仍然是硬 gate

## Full 和 Fast 的关键差异

| 项目 | full mode | fast_plane_only |
|---|---|---|
| 核心文件 | `Online_Edge_Detect/detector.py` | `vision_module/backend/table_edge_manager.py` |
| 点输入 | ROI 内较完整 depth map | ROI sparse sampled depth |
| 坐标系 | camera XYZ / XZ bird-view | camera XYZ 后转 robot XYZ |
| 候选过滤 | depth valid + local normal / trend | depth valid + robot Z height range |
| 平面检测 | local normal candidates + RANSAC plane | 不做 plane RANSAC |
| 线检测 | crease trend + RANSAC XZ line | support cluster reps + weighted XY line |
| 拟合对象 | front plane pixels / crease candidates | representative points |
| residual 空间 | plane residual 或 XZ line residual | robot XY bird-view residual |
| 支撑定义 | inlier pixels / area ratio | support pixels behind representatives |
| outlier 处理 | RANSAC inlier mask | representative residual + neighbor continuity |
| control level 命名 | `approach/alignment/stop` | `approach_slow/align/stop_ready/rotate_only` |
| 速度 | 更重 | 更轻 |
| 主要风险 | 背景竖直平面或 crease line 被误选 | height candidate 很多但 vertical support/reps 不成立 |

## Offline 和 Online 调用路径

### Offline bag eval

用户当前常用命令：

```bash
/usr/bin/python3 -m VISTA.vision_module.examples.bag_table_plane \
  --bag VISTA/20260516_161436.bag \
  --mode eval \
  --detector-mode fast_plane_only \
  --roi-preset center_lower \
  --output runs/...
```

调用路径：

```text
bag_table_plane.py
  -> iter_bag_frames(...)
    -> pyrealsense2 playback from .bag
    -> align depth to color
  -> TableEdgeManager(cfg=...)
  -> process_camera_frame(
       frames={"rgb": ..., "depth": ...},
       source_mode="OFFLINE_BAG_EVAL",
       local_perception={...},
       runtime_status={...}
     )
  -> _process_depth(...)
```

offline bag eval 的特点：

- 不通过 scheduler 取帧
- 不启动 TableEdgeManager worker thread
- 每帧显式调用 `process_camera_frame`
- 可以通过 CLI 覆盖：
  - detector mode
  - ROI preset
  - config path
- 输出：
  - `table_edge_obs.jsonl`
  - `metrics_summary.json`
  - preview frames
  - debug CSV

### Full-vs-fast paired offline compare

调用路径：

```text
full_fast_stride30_compare.py
  -> iter_bag_frames(...)
  -> full_processor = TableEdgeManager(detector_mode="full")
  -> fast_processor = TableEdgeManager(detector_mode="fast_plane_only")
  -> process_camera_frame(...) twice per selected frame
  -> OpenCVPreviewSink render paired images
```

特点：

- 同一 bag frame 分别喂给 full 和 fast
- 两个 `TableEdgeManager` 实例各自维护 temporal state
- 用于 visual compare，不是在线控制路径

### Online runtime

在线 runtime 调用路径：

```text
Vision app / scheduler
  -> camera manager publishes "camera_frames"
  -> TableEdgeManager.start_runtime()
  -> TableEdgeManager._worker_loop()
    -> scheduler.read_slot("camera_frames")
    -> process_camera_frame(...)
    -> _process_depth(...)
    -> scheduler.publish_result("table_edge_obs", payload)
```

online 的特点：

- 通过 scheduler 读最新 camera frame
- worker 按 `table_edge.update_hz` 或 `track_local_update_hz` 调度
- 会统计 dropped frames、latest frame lag、publish delay
- 使用真实 runtime status 和 local perception
- 输出直接进入 runtime 的 `table_edge_obs` topic

## Offline 和 Online 的共同路径

只要 offline 使用的是当前 `VISTA/vision_module/examples/bag_table_plane.py`，它和 online runtime 的共同路径是：

```text
TableEdgeManager.process_camera_frame(...)
  -> TableEdgeManager._process_depth(...)
    -> full or fast branch
  -> TableEdgeManager._with_freshness(...)
```

共同使用：

- 当前 `vision_params.yaml`
- `TableEdgeManager`
- ROI selection
- YOLO/local perception gate
- detector mode selection
- calibration loaded through `Online_Edge_Detect`
- target distance
- output schema `table_edge_obs`
- preview sink rendering逻辑

主要差异只在“帧来源”和“发布/保存方式”：

- offline：bag replay -> direct function call -> files
- online：camera scheduler -> worker loop -> scheduler publish

## 旧 Offline_Edge_Test 和 Online_Edge_Detect 的关系

仓库里还有旧路径：

```text
VISTA/Offline_Edge_Test
VISTA/Online_Edge_Detect
VISTA/Vertical_Plane_Pose_Estimator
```

当前关系：

- `VISTA/Online_Edge_Detect` 仍被 `TableEdgeManager._load_detector()` 使用，不能直接删除。
- `VISTA/Offline_Edge_Test/calib.json` 是 `Online_Edge_Detect` 默认 calibration 来源之一，不能随意删除。
- `VISTA/Offline_Edge_Test/Online` 更像旧副本，当前主 `vision_module` 不直接调用。
- `VISTA/Vertical_Plane_Pose_Estimator` 是独立原型，当前主路径未引用。

## 当前风险点

### Full mode 风险

- front plane RANSAC 可能拟合到背景竖直板。
- crease line 可能被物体边缘或 ROI 边界影响。
- plane 和 line 的冲突会进入 conflict 或 fallback，控制 gate 可能拒绝。
- full mode 依赖旧 `Online_Edge_Detect` 配置，配置迁移前不宜删除该目录。

### Fast mode 风险

- height candidates 多，不代表 front face support 成立。
- camera pitch / camera height 错误会直接影响 robot Z height filtering。
- center_lower ROI 如果没有包含有效桌前立面，后续 support/reps 一定失败。
- 后期帧仍可能选择背景板或背景竖直结构作为 front-face-like support。
- representative outlier / continuity 策略偏严格时，可能导致拟合失败。

## 建议的后续整理顺序

1. 先不要删除 `VISTA/Online_Edge_Detect`。
2. 若要清理目录，先把 calibration、DetectorConfig、load_calib 和 full detector 依赖迁移进 `vision_module`。
3. `Vertical_Plane_Pose_Estimator` 可以作为 archive 候选，因为当前主路径未引用。
4. `Offline_Edge_Test/Online` 可以作为 legacy duplicate 候选，但要先确认旧 PNG 测试脚本是否还要保留。
5. 真正要改检测质量时，优先处理 ROI 动态选择；否则 fast path 很多失败帧不是拟合算法问题，而是 ROI 内没有足够有效 front-face 信息。
