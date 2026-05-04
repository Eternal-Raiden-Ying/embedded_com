# 板端通信协议

协议版本：`1.2`

## 通信概览

| 项目 | 说明 |
|------|------|
| 传输协议 | HTTP/1.1 |
| 数据格式 | 请求 `multipart/form-data`，响应 `application/json` |
| 服务地址 | `http://{host}:6006` |
| 编码 | UTF-8 |

## 接口一览

| 方法 | 路径 | 用途 |
|------|------|------|
| POST | `/api/v1/init` | 加载模型到 GPU 并 warmup |
| POST | `/api/v1/predict` | 提交 RGB-D 帧，获取抓取目标 |
| POST | `/api/v1/release` | 释放模型，回收 GPU 显存 |

## 推荐调用顺序

```
POST /api/v1/init          → 等待 200
POST /api/v1/predict × N   → 逐帧推理
POST /api/v1/release       → 释放资源
```

注意：
- `predict` 必须在 `init` 成功后调用，否则返回 400。
- `init` 已加载时重复调用返回 `{"status": "already_loaded"}`，不会重新加载。
- `release` 后再次调用 `predict` 需要先 `init`。

---

## 接口详情

### POST /api/v1/init

加载 GraspNet + YOLO 模型到 GPU，执行一次 warmup 推理。耗时约 5–15 秒（取决于硬件）。

**请求**：无 body。

**响应**（200）：

```json
{
  "status": "success",
  "message": "Predictor loaded and warmed up successfully."
}
```

可能的其他状态：

```json
{"status": "already_loaded", "message": "Predictor is already running."}
```

---

### POST /api/v1/predict

提交一帧 RGB-D (1280*720)图像，返回抓取目标列表。

**请求**（multipart/form-data）：

| 字段 | 类型 | 必须 | 说明 |
|------|------|------|------|
| `rgb_file` | file (PNG/JPG) | 是 | RGB 彩色图像，建议 1280×720 |
| `depth_file` | file (PNG, 16-bit) | 是 | 深度图像，单位 mm，与 RGB 对齐 |
| `class_id` | int | 是 | 目标 YOLO 类别 ID，≥ 0 |
| `metadata` | string (JSON) | 是 | 调用方标识信息 |

`metadata` 格式：

```json
{
  "robot_id": "edge-sim",
  "cmd": "predict"
}
```

**响应**（200）：

```json
{
  "format_version": "1.1",
  "status": "success | reposition_required | failure",
  "reason": "no_detection | no_grasp_detected | no_feasible_grasp | score_below_threshold | null",
  "message": "人类可读的调试信息",
  "detection": { ... },
  "grasp_count": 1024,
  "feasible_count": 14,
  "output_count": 3,
  "targets": [ ... ]
}
```

#### status 三分法

| status | 含义 | 建议动作 |
|--------|------|---------|
| `success` | 至少一个 target 通过全部过滤 | 取 targets[0] 执行 |
| `reposition_required` | 目标可见，生成了抓取，但角度/分数不满足约束 | 微调机械臂位姿后重试 |
| `failure` | 硬失败，调整位姿无法解决 | 检查目标是否在视野内、光照、class_id |

#### reason 枚举

| reason | 归属 status | 含义 |
|--------|-------------|------|
| `null` | success | 正常 |
| `no_detection` | failure | YOLO 未检测到任何匹配类别的目标 |
| `no_grasp_detected` | failure | YOLO 找到了目标，但 GraspNet 未生成抓取 |
| `no_feasible_grasp` | reposition_required | 所有抓取的可执行角度超过阈值 |
| `score_below_threshold` | reposition_required | 所有可行抓取的置信度低于最低分阈值 |

#### reposition_proposal 对象（可选）

当 `reason = "no_feasible_grasp"` 且 `build_reposition_proposal()` 成功生成时出现：

```json
{
  "dx_cm": 0.0,
  "dy_cm": -20.0,
  "reference_line_new_xy_cm": [0.0, -20.0],
  "distance_lg_cm": 55.9,
  "capped": false,
  "reference_grasp": {
    "score": 0.58,
    "x_cm": 50.0,
    "y_cm": 5.0,
    "z_cm": 7.0,
    "feasible_distance_cm": 17.9
  }
}
```

| 字段 | 类型 | 说明 |
|------|------|------|
| `dx_cm` | float | 参考 Z 线建议移动的 X 距离 (cm) |
| `dy_cm` | float | 参考 Z 线建议移动的 Y 距离 (cm) |
| `reference_line_new_xy_cm` | [float, float] | 建议的参考 Z 线新 XY 坐标 |
| `distance_lg_cm` | float | 新参考线到 grasp 点的 XY 距离 (cm) |
| `capped` | bool | 是否被 `reposition_max_distance_cm` 截断 |
| `reference_grasp.score` | float | 参考 grasp 的 GraspNet 置信度 |
| `reference_grasp.x_cm` / `y_cm` / `z_cm` | float | 参考 grasp 的 robot 坐标 (cm) |
| `reference_grasp.feasible_distance_cm` | float | 参考 grasp 当前的方向约束距离 (cm) |

#### detection 对象

```json
{
  "requested_class_id": 47,
  "resolved_class_id": 47,
  "found": true,
  "confidence": 0.1338,
  "detection_count": 1,
  "multiple_detections": false,
  "similar_detection_result": false,
  "bbox": [554, 228, 778, 436]
}
```

| 字段 | 类型 | 说明 |
|------|------|------|
| `requested_class_id` | int | 调用方请求的类别 ID |
| `resolved_class_id` | int | 实际检测到的类别 ID |
| `found` | bool | 是否检测到目标 |
| `confidence` | float \| null | YOLO 置信度，未检测到时为 null |
| `detection_count` | int | 检测框数量 |
| `multiple_detections` | bool | 是否多目标 |
| `similar_detection_result` | bool | `resolved_class_id ≠ requested_class_id` 时为 true，表示触发了回退检测 |
| `bbox` | [int, int, int, int] \| null | 最佳检测框 `[x1, y1, x2, y2]` |

#### targets 数组元素

```json
{
  "x_cm": 27.54,
  "y_cm": -1.43,
  "z_cm": 7.01,
  "pitch_deg": -29.22,
  "roll_deg": -6.18,
  "gripper_width_cm": 8.66,
  "approach_depth_cm": 4.00,
  "confidence": 0.5488,
  "feasible_distance_cm": 1.95,
  "position_frame": "robot",
  "angle_frame": "robot"
}
```

| 字段 | 类型 | 单位 | 说明 |
|------|------|------|------|
| `x_cm` | float | cm | 夹爪后端中心 X（robot 基坐标，X 前） |
| `y_cm` | float | cm | 夹爪后端中心 Y（Y 左） |
| `z_cm` | float | cm | 夹爪后端中心 Z（Z 上） |
| `pitch_deg` | float | 度 | 近似后 approach（P 面投影）相对于水平面的仰角 |
| `roll_deg` | float | 度 | 绕近似后 approach 轴的旋转角（垂直 v_proj 平面内），±180° |
| `gripper_width_cm` | float | cm | 所需夹爪开口宽度 |
| `approach_depth_cm` | float | cm | 夹爪沿接近方向的插入深度 |
| `confidence` | float | — | GraspNet 置信度，[0, 1] |
| `feasible_distance_cm` | float | cm | approach 直线到参考 Z 线的空间距离（cm），越小越好 |
| `position_frame` | string | — | 固定 `"robot"` |
| `angle_frame` | string | — | 固定 `"robot"` |

**坐标系约定**：robot 基坐标系，右手系。X 前 / Y 左 / Z 上。所有 target 位置已含 `camera → robot` 的旋转和平移变换。

---

### POST /api/v1/release

释放 GPU 资源。

**请求**：无 body。

**响应**（200）：

```json
{"status": "success", "message": "GPU memory freed."}
```

```json
{"status": "already_released", "message": "Predictor is not running."}
```

---

## 调用示例

### 例 1：success（正常抓取）

请求：

```bash
curl -X POST http://127.0.0.1:6006/api/v1/predict \
  -F "rgb_file=@color.png" \
  -F "depth_file=@depth_raw.png" \
  -F "class_id=47" \
  -F 'metadata={"robot_id":"edge-01","cmd":"predict"}'
```

响应（节选）：

```json
{
  "format_version": "1.1",
  "status": "success",
  "reason": null,
  "message": "YOLO detected class_id=47, 11 feasible grasps, 1 passed score filter",
  "detection": {
    "requested_class_id": 47,
    "resolved_class_id": 47,
    "found": true,
    "confidence": 0.1338,
    "detection_count": 1,
    "multiple_detections": false,
    "similar_detection_result": false,
    "bbox": [554, 228, 778, 436]
  },
  "grasp_count": 1024,
  "feasible_count": 11,
  "output_count": 1,
  "targets": [
    {
      "x_cm": 27.54,
      "y_cm": -1.43,
      "z_cm": 7.01,
      "pitch_deg": -29.22,
      "roll_deg": -6.18,
      "gripper_width_cm": 8.66,
      "approach_depth_cm": 4.00,
      "confidence": 0.5488,
      "feasible_distance_cm": 1.95,
      "position_frame": "robot",
      "angle_frame": "robot"
    }
  ]
}
```

### 例 2：failure / no_detection（YOLO 未检测到目标）

```json
{
  "format_version": "1.1",
  "status": "failure",
  "reason": "no_detection",
  "message": "YOLO did not detect class_id=47",
  "detection": {
    "requested_class_id": 47,
    "resolved_class_id": 47,
    "found": false,
    "confidence": null,
    "detection_count": 0,
    "multiple_detections": false,
    "similar_detection_result": false,
    "bbox": null
  },
  "grasp_count": 0,
  "feasible_count": 0,
  "output_count": 0,
  "targets": []
}
```

### 例 3：failure / no_grasp_detected（YOLO 找到但无抓取）

```json
{
  "format_version": "1.1",
  "status": "failure",
  "reason": "no_grasp_detected",
  "message": "YOLO detected class_id=47 (conf=0.1338, 1 instance(s)) but GraspNet produced no grasps",
  "detection": {
    "requested_class_id": 47,
    "resolved_class_id": 47,
    "found": true,
    "confidence": 0.1338,
    "detection_count": 1,
    "multiple_detections": false,
    "similar_detection_result": false,
    "bbox": [554, 228, 778, 436]
  },
  "grasp_count": 0,
  "feasible_count": 0,
  "output_count": 0,
  "targets": []
}
```

### 例 4：reposition_required / score_below_threshold（分数不足）

```json
{
  "format_version": "1.1",
  "status": "reposition_required",
  "reason": "score_below_threshold",
  "message": "YOLO detected class_id=47, 14 feasible grasps, but all below the minimum score threshold of 0.3 (best: 0.2100)",
  "detection": {
    "requested_class_id": 47,
    "resolved_class_id": 47,
    "found": true,
    "confidence": 0.1338,
    "detection_count": 1,
    "multiple_detections": false,
    "similar_detection_result": false,
    "bbox": [554, 228, 778, 436]
  },
  "grasp_count": 1024,
  "feasible_count": 14,
  "output_count": 0,
  "targets": []
}
```

### 例 5：fallback 检测（similar_detection_result = true）

请求 `class_id=47`，YOLO 未检出，回退到 `class_id=55` 检出：

```json
{
  "format_version": "1.1",
  "status": "success",
  "reason": null,
  "message": "YOLO detected class_id=55, 5 feasible grasps, 4 passed score filter",
  "detection": {
    "requested_class_id": 47,
    "resolved_class_id": 55,
    "found": true,
    "confidence": 0.4730,
    "detection_count": 3,
    "multiple_detections": true,
    "similar_detection_result": true,
    "bbox": [513, 248, 729, 463]
  },
  "grasp_count": 1024,
  "feasible_count": 5,
  "output_count": 4,
  "targets": [
    {
      "x_cm": 24.86,
      "y_cm": 2.88,
      "z_cm": 6.81,
      "pitch_deg": -29.24,
      "roll_deg": -1.49,
      "gripper_width_cm": 8.39,
      "approach_depth_cm": 4.00,
      "confidence": 0.7023,
      "feasible_distance_cm": 2.00,
      "position_frame": "robot",
      "angle_frame": "robot"
    }
  ]
}
```

---

## 错误码

| HTTP 状态 | 场景 |
|-----------|------|
| 200 | 正常（业务状态在 response body 的 `status` 字段中） |
| 400 | `class_id` 为负数、`metadata` 非法 JSON、解码失败、未 `init` 即调用 `predict` |
| 500 | 模型加载失败 |

---


## 版本历史

| 版本 | 日期 | 变更 |
|------|------|------|
| 1.2 | 2026-05-04 | 方向约束重构：`feasible_angle_deg` → `feasible_distance_cm`；`pitch_deg`/`roll_deg` 语义更新；新增 `reposition_proposal` |
| 1.1 | 2026-05-02 | 新增 `failure` status、`detection` 对象、`similar_detection_result` |
| 1.0 | 2026-05-02 | 初始冻结，`success` / `reposition_required` 两种 status |
