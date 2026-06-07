# 视觉语义统一与 bbox-size driven ROI 修改说明

本版在视觉层完成三类修改：

1. 统一 table bbox 语义：当前帧检测、控制可用、历史保持分开记录。
2. 统一 edge 语义：`edge_detected -> edge_geometry_valid -> edge_stable -> edge_trusted` 分层明确。
3. 将动态 ROI 默认改为 `bbox_expand`：RGB table bbox 映射到 depth bbox 后按 margin 扩展，得到最终 `table_edge_roi`。

核心目标是让控制层只消费清晰字段：

```text
table_bbox_control_valid
edge_geometry_valid
edge_stable
edge_trusted
edge_quality
table_edge_roi
```

保留的旧字段仅用于兼容日志和旧代码，不应作为新控制逻辑的主输入。
