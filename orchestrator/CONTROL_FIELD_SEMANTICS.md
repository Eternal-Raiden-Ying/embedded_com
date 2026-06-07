# 控制层语义字段表

本表记录本轮控制层重构后建议保留和使用的核心字段。后续日志、summary、preview、offline replay 尽量以这些字段为准。

| 字段名 | 中文含义 |
|---|---|
| `table_bbox_current_found` | 当前帧是否真实检测到 table bbox；只表示当前帧视觉结果，不包含历史保持。 |
| `table_bbox_control_valid` | 控制层是否允许使用 table bbox；当前实现基本等价于检测到 bbox，后续可加入短时 hold。 |
| `table_bbox_hold_active` | 当前 table bbox 是否来自短时保持/记忆，而不是当前帧检测。 |
| `table_bbox_hold_age_frames` | table bbox 保持已经持续的帧数。 |
| `table_bbox_xyxy` | table bbox 的坐标，格式为 `[x1, y1, x2, y2]`。 |
| `table_bbox_area_ratio` | table bbox 面积占图像面积比例；仅用于诊断，不参与控制分支。 |
| `table_bbox_conf_raw` | YOLO/table bbox 的原始置信度，仅记录，不作为 table 有效门控。 |
| `table_bbox_conf_used_for_gate` | table bbox 的置信度是否参与控制门控；当前固定为 `False`。 |
| `edge_detected` | ROI 内是否检测到 edge/docking 几何候选结构。 |
| `edge_geometry_valid` | 单帧 edge 几何检测是否通过基础有效性检查；属于感知层结果，不等于控制可信。 |
| `edge_stable` | edge 几何结果是否连续稳定达到配置帧数。 |
| `edge_trusted` | edge 是否被允许进入姿态控制；必须建立在 table bbox 有效、edge 几何有效和稳定基础上。 |
| `edge_stable_count` | 当前 edge 连续稳定帧数。 |
| `edge_conf_raw` | edge/docking 几何检测置信度原始值。 |
| `edge_residual_raw` | edge 拟合残差原始值。 |
| `edge_support_count` | edge 支撑点数量。 |
| `edge_inlier_count` | edge 拟合内点数量。 |
| `edge_trust_reason` | edge 被判定为 trusted 的原因。 |
| `edge_reject_for_control_reason` | edge 未被允许进入控制的原因。 |
| `docking_enabled_by_table_bbox` | table bbox 有效时，docking/edge 才有资格进入控制候选。 |
| `edge_control_allowed` | edge/docking 是否被允许参与控制；语义上应等价于 `edge_trusted`。 |
| `edge_control_block_reason` | edge/docking 被禁止参与控制的原因。 |
| `control_source` | 当前帧最终控制源。 |
| `control_intent` | 当前帧控制意图，例如 forward、posture_adjust、search、final_lock。 |
| `allow_forward` | 本帧是否允许前进。 |
| `allow_rotate` | 本帧是否允许旋转。 |
| `forward_block_reason` | 前进被禁止的原因。 |
| `rotate_block_reason` | 旋转被禁止的原因。 |
| `vx_norm` | 控制器输出的归一化前进速度。 |
| `vy_norm` | 控制器输出的归一化横移速度。 |
| `wz_norm` | 控制器输出的归一化角速度。 |
| `stale_level` | 当前观测新鲜度等级，如 fresh、soft_stale、hard_stale、dead。 |
| `stale_source` | stale 来源，如 edge、table、vision。 |
| `zero_cmd_reason` | 输出零速度或 STOP 的原因。 |

## 控制源枚举

| control_source | 中文含义 |
|---|---|
| `yolo_forward` | table bbox 有效，默认直行；语义要求 `vx>0, wz=0`。 |
| `edge_adjust` | edge 已 trusted，允许 docking/edge 参与姿态修正。 |
| `local_rotate_search` | table bbox 当前不可用，本地旋转搜索 table bbox。 |
| `final_lock` | 最终锁定/停车阶段控制。 |
| `search_failed_stop` | 本地搜索超时后的失败停止。 |
| `explicit_stop` | 外部 STOP / 急停 / 明确停止。 |
| `stop` | 普通停止或安全停止。 |

## 已废弃语义

以下旧语义在本轮不再作为控制依据：

| 旧字段/旧控制源 | 处理方式 |
|---|---|
| `yolo_assist` | 废弃，统一映射到 `yolo_forward` 的直行语义。 |
| `yolo_edge_blend` | 废弃，edge 可信时直接使用 `edge_adjust`，不再做模糊 blend。 |
| `yolo_table_area_gate_for_docking` | 废弃，bbox area 仅用于诊断，不再决定 docking 是否可用。 |
| `docking_allowed_by_yolo_area` | 废弃，不再输出为核心字段。 |
| `docking_blocked_by_yolo_area` | 废弃，不再输出为核心字段。 |
