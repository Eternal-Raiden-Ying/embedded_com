# Vision Field Semantics 字段表

| 字段名 | 中文注释 |
|---|---|
| `table_bbox_current_found` | 当前帧 YOLO 是否检测到 table bbox；只表示当前帧视觉检测事实。 |
| `table_bbox_control_valid` | 当前帧 table bbox 是否可作为控制层的桌子存在性信号；当前规则为检测到 bbox 即有效，不使用 confidence 门控。 |
| `table_bbox_xyxy` | table bbox 的像素坐标，格式为 `[x1, y1, x2, y2]`。 |
| `table_bbox_source` | table bbox 来源，例如 `yolo_table_bbox`、`mock_table_bbox`、`none`。 |
| `table_bbox_conf_raw` | YOLO 输出的原始 table bbox 置信度，仅记录，不参与 table 有效性门控。 |
| `table_bbox_conf_used_for_gate` | table bbox 是否使用 confidence 作为门控；当前固定为 `False`。 |
| `table_bbox_area_ratio` | table bbox 面积占 RGB 图像面积的比例；仅诊断，不参与控制权切换。 |
| `table_bbox_center` | table bbox 中心点像素坐标 `[cx, cy]`。 |
| `table_bbox_center_norm` | table bbox 中心点归一化坐标 `[cx_norm, cy_norm]`。 |
| `table_bbox_found` | 兼容旧字段，等价于 `table_bbox_current_found`。 |
| `table_bbox_detected` | 兼容旧字段，等价于 `table_bbox_current_found`。 |
| `yolo_table_control_valid` | 兼容旧字段，等价于 `table_bbox_control_valid`。 |
| `table_confirmed_by_yolo` | 兼容旧字段，表示 table bbox 来自 YOLO 检测。 |
| `docking_enabled_by_yolo` | YOLO table bbox 是否允许 docking/edge 进入候选流程；没有 bbox 时必须为 `False`。 |
| `edge_control_allowed` | edge 是否已经达到控制使用资格；当前跟随 `edge_trusted`。 |
| `edge_control_block_reason` | edge 没有控制权限的原因。 |
| `edge_detected` | fast/full edge 算法是否在 ROI 内检测到几何边缘候选。 |
| `edge_geometry_valid` | 单帧几何结果是否有效；只表示感知层几何有效，不表示可控制。 |
| `edge_valid` | 兼容旧字段，当前等价于 `edge_geometry_valid`。 |
| `edge_stable` | edge 几何结果是否连续稳定达到配置帧数。 |
| `edge_stable_count` | 当前连续稳定 edge 帧数。 |
| `edge_stable_required_frames` | 判定 edge stable 需要的最小帧数。 |
| `edge_trusted` | edge 是否可信到可以给控制层用于 docking 姿态调整。 |
| `valid_for_control` | 兼容旧字段，当前等价于 `edge_trusted`，比 `edge_valid` 更严格。 |
| `edge_trust_reason` | edge 被判定为 trusted 的原因。 |
| `edge_reject_for_control_reason` | edge 未被允许进入控制的原因。 |
| `edge_quality` | edge 质量特征字典，包含 conf、residual、support、inlier、span、frontness 等信息。 |
| `edge_quality.edge_conf` | edge 置信度。 |
| `edge_quality.residual_mean` | edge/line 拟合平均残差。 |
| `edge_quality.residual_p90` | edge/line 拟合 P90 残差。 |
| `edge_quality.residual_max` | edge/line 拟合最大残差。 |
| `edge_quality.candidate_count` | ROI 内候选点数量。 |
| `edge_quality.support_count` | 支撑当前 edge 的点数。 |
| `edge_quality.inlier_count` | 拟合内点数量。 |
| `edge_quality.x_span_m` | edge 在 X 方向覆盖的物理宽度。 |
| `edge_quality.line_score` | fast edge 线质量评分。 |
| `edge_quality.frontness_score` | 前边缘/前平面特征评分。 |
| `edge_quality.edge_consistency_score` | edge 一致性评分。 |
| `edge_quality.background_penalty` | 背景干扰惩罚。 |
| `edge_quality.reject_reason` | edge reject 原因。 |
| `roi_source` | 当前 depth ROI 来源，例如 YOLO bbox 映射、静态 fallback、无 bbox 禁用。 |
| `roi_phase` | 当前 ROI 阶段语义，例如 `disabled_no_table_bbox`、`yolo_guided` 等。 |
| `depth_edge_roi` | depth 图中的 ROI，格式 `[x1, y1, x2, y2]`。 |
| `table_edge_roi` | 兼容旧字段，等价于当前 table edge depth ROI。 |
| `mapped_depth_center` | RGB table bbox 映射到 depth 图后的中心点。 |
| `table_bbox_rgb_xyxy` | RGB 图中的 table bbox 坐标。 |
| `table_bbox_rgb_center` | RGB table bbox 中心点。 |
| `table_bbox_rgb_center_norm` | RGB table bbox 归一化中心点。 |
| `yolo_table_roi_valid` | YOLO table bbox 是否成功生成了可用 ROI。 |
| `yolo_valid_reason` | table bbox 被认为有效的原因。 |
| `yolo_invalid_reason` | table bbox 不可用的原因。 |
