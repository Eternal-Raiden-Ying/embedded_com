# CONTROL_REFACTOR_V2_NOTES

本版基于 `orchestrator_control_refactor_v1.zip` 继续修改，重点是控制层第二轮语义清理。

## 本轮核心目标

1. 清理控制相关语义字段，保留清晰、必要、可解释的字段。
2. 移除旧的模糊控制源：`yolo_assist`、`yolo_edge_blend`、0.40 bbox area gate。
3. 控制源统一为：
   - `yolo_forward`
   - `edge_adjust`
   - `local_rotate_search`
   - `final_lock`
   - `search_failed_stop`
   - `explicit_stop`
   - `stop`
4. 明确语义：
   - `table_bbox_current_found`：当前帧 YOLO/table bbox 是否存在。
   - `table_bbox_control_valid`：控制层是否允许使用 table bbox。
   - `edge_geometry_valid`：感知层检测到几何边缘。
   - `edge_trusted`：允许 edge/docking 进入姿态控制。
5. `yolo_forward` 保持为 `vx>0, wz=0`。
6. edge/docking 只有在 `edge_trusted=True` 后才能进入 `edge_adjust`。
7. 没有 table bbox 时，不启用 edge/docking 控制，只进入 `local_rotate_search`。

## 修改文件

- `orchestrator_service/runtime/perception_semantics.py`
- `orchestrator_service/runtime/control_authority.py`
- `orchestrator_service/runtime/controller.py`
- `orchestrator_service/runtime/state_machine.py`
- `orchestrator_service/runtime/service.py`
- `orchestrator_service/config/schema.py`
- `orchestrator_service/config/board_config.py`
- `configs/stage_params.yaml`
- `CONTROL_FIELD_SEMANTICS.md`

## 基础验证

已执行：

```bash
python3 -m compileall -q orchestrator_service
```

返回码为 0。运行时出现的 `artifact_tool` spreadsheet warmup stderr 是环境启动噪声，不是 orchestrator 语法错误。
