# 控制语义字段说明（统一版）

| 字段名 | 中文含义 |
|---|---|
| `table_bbox_current_found` | 当前帧是否真实检测到 table bbox。 |
| `table_bbox_control_valid` | 控制层是否允许使用 table bbox；当前检测或短时 hold 均可为 True。 |
| `table_bbox_hold_active` | table bbox 是否来自历史保持。 |
| `table_bbox_hold_age_frames` | table bbox 历史保持帧数。 |
| `table_bbox_xyxy` | 控制层看到的 table bbox，RGB 输出图坐标。 |
| `table_bbox_conf_raw` | YOLO 原始置信度，仅记录。 |
| `table_bbox_conf_used_for_gate` | 当前是否用 confidence 做 table bbox 门控；固定为 False。 |
| `edge_detected` | 是否检测到 edge 几何候选。 |
| `edge_geometry_valid` | 单帧 edge 几何是否有效；不是控制授权。 |
| `edge_stable` | edge 是否连续稳定。 |
| `edge_trusted` | edge 是否被授权参与姿态控制。 |
| `edge_quality` | 控制层接收到的 edge 质量字典。 |
| `edge_conf_raw` | edge 原始置信度。 |
| `edge_residual_raw` | edge 残差。 |
| `edge_support_count` | edge 支撑点数量。 |
| `edge_inlier_count` | edge 内点数量。 |
| `edge_x_span_m` | edge 横向物理跨度。 |
| `edge_background_penalty` | edge 背景惩罚。 |
| `edge_trust_reason` | edge 被判定 trusted 的原因。 |
| `edge_reject_for_control_reason` | edge 不能参与控制的原因。 |
| `control_source` | 当前控制源，保留：`yolo_forward`、`edge_adjust`、`local_rotate_search`、`final_lock`、`stop` 等。 |
| `control_intent` | 当前控制意图，例如 `forward`、`posture_adjust`、`search`、`hold`、`stop`。 |
| `allow_forward` | 本帧是否允许前进。 |
| `allow_rotate` | 本帧是否允许旋转。 |
| `forward_block_reason` | 前进被阻止的原因。 |
| `rotate_block_reason` | 旋转被阻止的原因。 |
| `edge_control_allowed` | edge 是否允许进入控制，等价于 `edge_trusted`。 |
| `docking_enabled_by_table_bbox` | table bbox 是否允许 docking/edge 被考虑。 |
| `stale_level` | 当前观测新鲜度级别。 |
| `stale_source` | stale 来源，例如 edge / table / vision。 |
| `zero_cmd_reason` | 最终输出 0/STOP 的原因。 |

## 控制约定

- `yolo_forward` 固定为 `vx>0,wz=0`。
- `edge_adjust` 只能在 `edge_trusted=True` 时产生。
- 没有 `table_bbox_control_valid` 时，不允许 docking/edge 控制，只能本地搜索或停止。
- 0.40 bbox area gate 已废弃，bbox 面积只作为诊断字段。
