# 控制-视觉语义统一修改说明

本版基于控制层 refactor v2 hotfix1 和视觉语义 refactor v1 继续统一：

1. 控制层和视觉层统一使用 `table_bbox_current_found / table_bbox_control_valid / table_bbox_hold_active`。
2. 控制层和视觉层统一使用 `edge_geometry_valid / edge_stable / edge_trusted / edge_quality`。
3. 控制层的 edge_trusted 判断能够读取视觉层输出的 `edge_quality` 字典。
4. 视觉层默认将动态 ROI 改为 bbox-size driven 的 `bbox_expand`。
5. 旧的模糊字段保留为兼容日志，不再作为新控制逻辑主输入。
