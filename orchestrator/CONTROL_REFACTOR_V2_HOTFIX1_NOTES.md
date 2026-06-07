# CONTROL_REFACTOR_V2_HOTFIX1_NOTES

## 修复内容

本热修复解决离线 bag 回放中的语义字段兼容问题：

```text
AttributeError: TablePerceptionSemantics object has no attribute table_bbox_found
```

原因是控制层第二轮已经将主语义字段改为：

- `table_bbox_current_found`
- `table_bbox_control_valid`
- `edge_geometry_valid`

但 `controller._rotate_gate()` 中仍残留了旧字段访问：

- `sem.table_bbox_found`
- `sem.edge_valid`

## 修改点

1. `runtime/perception_semantics.py`
   - 给 `TablePerceptionSemantics` 增加只读兼容 alias：
     - `table_bbox_found` -> `table_bbox_current_found`
     - `yolo_table_control_valid` -> `table_bbox_control_valid`
     - `edge_valid` -> `edge_geometry_valid`
   - `to_dict()` 显式导出兼容字段，避免旧日志/离线脚本断裂。

2. `runtime/controller.py`
   - `_rotate_gate()` 改为使用新字段：
     - `table_bbox_control_valid`
     - `edge_geometry_valid`

## 语义说明

- `table_bbox_current_found`：当前帧是否检测到 table bbox。
- `table_bbox_control_valid`：控制层是否允许使用 table bbox，后续可包含 hold/memory。
- `edge_geometry_valid`：感知层单帧几何有效，不等于控制可信。
- `edge_trusted`：允许 edge/docking 参与姿态控制的条件。

## 验证

已执行：

```bash
python3 -m compileall -q orchestrator_service
```

语法检查通过。
