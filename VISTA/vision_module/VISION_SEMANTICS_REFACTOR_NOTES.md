# Vision Semantics Refactor V1

本轮修改目标是把视觉链路输出的语义字段标准化，方便控制层稳定消费。

## 修改重点

1. 新增 `backend/vision_semantics.py`，集中生成 table bbox 与 edge 语义字段。
2. `predictor_manager.py` 中 table bbox 的存在性不再受 `enable_yolo_table_search` 影响；该开关只应影响 RGB search ROI，不应隐藏 table bbox 语义。
3. `table_edge_manager.py` 标准化输出：
   - `table_bbox_current_found`
   - `table_bbox_control_valid`
   - `edge_detected`
   - `edge_geometry_valid`
   - `edge_stable`
   - `edge_trusted`
   - `edge_quality`
4. `edge_valid` 作为兼容字段保留，但语义调整为几何有效，不再等同于控制有效。
5. `valid_for_control` 作为兼容字段保留，但现在跟随更严格的 `edge_trusted`。
6. 新增 `VISION_FIELD_SEMANTICS.md`，列出所有核心字段中文说明。

## 控制层使用建议

控制层优先消费：

- `table_bbox_control_valid`
- `edge_geometry_valid`
- `edge_stable`
- `edge_trusted`
- `edge_quality`

不要再直接把 `edge_valid=True` 当作 docking 控制权限。
