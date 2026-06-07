# 视觉语义字段说明（统一版）

| 字段名 | 中文含义 |
|---|---|
| `table_bbox_current_found` | 当前帧 YOLO/table 检测是否真实检测到桌子 bbox；不使用 confidence 门控。 |
| `table_bbox_control_valid` | 控制层是否允许使用 table bbox；当前帧检测到或显式 hold 时为 True。 |
| `table_bbox_hold_active` | 当前 table bbox 是否来自历史保持，而不是当前帧真实检测。 |
| `table_bbox_hold_age_frames` | table bbox 历史保持已经持续的帧数。 |
| `table_bbox_xyxy` | 当前控制/感知使用的桌子 bbox，格式为 `[x1,y1,x2,y2]`，坐标属于 RGB 输出图。 |
| `table_bbox_source` | table bbox 来源，例如 `yolo_table_bbox`、`mock_table_bbox`、`table_bbox_hold`、`none`。 |
| `table_bbox_invalid_reason` | table bbox 不可用时的原因。 |
| `table_bbox_conf_raw` | YOLO 输出的原始置信度，仅记录，不作为 table 是否存在的门控。 |
| `table_bbox_conf_used_for_gate` | table bbox 是否使用置信度作为有效性门控；当前固定为 False。 |
| `table_bbox_area_ratio` | table bbox 面积占 RGB 图像面积比例，仅诊断，不参与控制。 |
| `table_bbox_center` | table bbox 在 RGB 输出图中的中心像素坐标。 |
| `table_bbox_center_norm` | table bbox 中心归一化坐标。 |
| `table_bbox_found` | 兼容旧字段，等价于 `table_bbox_current_found`。新逻辑不要优先使用它。 |
| `yolo_table_control_valid` | 兼容旧字段，等价于 `table_bbox_control_valid`。新逻辑不要优先使用它。 |
| `edge_detected` | fast edge / docking 感知是否检测到几何边缘候选。 |
| `edge_geometry_valid` | 当前单帧 edge 几何结果是否有效；它不是控制许可。 |
| `edge_stable` | edge 几何结果是否连续稳定达到配置帧数。 |
| `edge_trusted` | edge 是否满足稳定性和质量门控，可以交给控制层参与姿态调整。 |
| `edge_quality` | 统一的 edge 几何质量字典，包含 conf、residual、support、inlier、span 等。 |
| `edge_trust_reason` | edge 被判定 trusted 的原因。 |
| `edge_reject_for_control_reason` | edge 不能用于控制的原因。 |
| `edge_valid` | 兼容旧字段，当前等价于 `edge_geometry_valid`。 |
| `valid_for_control` | 兼容旧字段，当前等价于 `edge_trusted`。 |
| `edge_control_allowed` | edge 是否允许进入控制层，当前等价于 `edge_trusted`。 |
| `docking_enabled_by_yolo` | table bbox 是否允许 docking/edge 被考虑；不是最终控制许可。 |
| `edge_conf` | edge 置信度，来自 `confidence` 或 fast edge 质量字段。 |
| `fast_residual_mean` | fast edge 拟合残差均值。 |
| `fast_residual_p90` | fast edge 拟合残差 p90。 |
| `fast_support_point_count` | 支撑 edge 的候选点数量。 |
| `fast_rep_inlier_count` | 拟合内点数量。 |
| `fast_fit_inlier_x_span_m` | edge 内点在横向 X 上覆盖的物理跨度。 |
| `fast_background_penalty` | 背景干扰惩罚，越高说明越可能包含背景误检。 |
| `roi_source` | ROI 来源，例如 `yolo_table_bbox_mapped`、`disabled_no_table_bbox`、`static_no_yolo_fallback`。 |
| `roi_reason` | ROI 选择原因。 |
| `yolo_table_roi_mode` | YOLO table ROI 策略，当前推荐 `bbox_expand`。 |
| `mapped_depth_bbox_xyxy` | RGB table bbox 映射到 depth 图后的 bbox。 |
| `table_edge_roi` | 最终用于 depth edge 检测的 ROI。 |
| `roi_mapping_mode` | RGB bbox 到 depth ROI 的映射模式。 |
| `roi_clamped` | ROI 是否被 depth 图边界裁剪。 |

## 统一约定

- `table_bbox_current_found` 表示当前帧真实检测结果；`table_bbox_control_valid` 表示控制层可用性。
- `edge_geometry_valid` 是感知结果；`edge_trusted` 才是控制授权。
- `bbox_area_ratio`、confidence 等字段只做诊断，不作为当前桌子是否存在的主门控。
- ROI 默认采用 bbox-size driven：由 YOLO bbox 映射到 depth 后生成，不再只是固定窗口跟随中心点。
