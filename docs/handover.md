# Handover — 状态映射表待处置事项

审计日期：2026-05-05

---

## ORCHESTRATOR_STATE_MAP 幻影状态

**文件**: `orchestrator/orchestrator_service/mobile_gateway/runtime/service.py`

`State` enum (`runtime/context.py:13-34`) 定义了 22 个状态。但 `ORCHESTRATOR_STATE_MAP` (lines 37-64) 和 `_status_message_for_state` (lines ~1140-1174) 中包含 3 个在 enum 中**不存在**的状态名：

| 幻影键 | map 行号 | message 行号 | 全仓库引用 |
|--------|---------|-------------|-----------|
| `SEARCH_OBJECT` | 40 | 1160 | 仅此两处 |
| `EDGE_FOLLOW_OBJECT_SEARCH` | 44 | 1160 | 仅此两处 |
| `DETECTOR_UNAVAILABLE` | 45 | 1162-1163 | 仅此两处 |

**推测**: 这些是旧版状态机中的名称，在重构为当前 `State` enum 时被重命名（`SEARCH_OBJECT` → `SEARCH_TARGET_INIT`，`EDGE_FOLLOW_OBJECT_SEARCH` → `EDGE_SLIDE_SEARCH`），但 map 表未同步更新。没有设计文档引用这些名称作为未来规划。

**当前影响**: 无运行时错误——`_handle_state_block` 使用 `.get(raw_state, ("unknown", 0))` 防御性查询，未命中时返回 `"unknown"` 状态。这些条目永远不会被匹配到（没有代码 emit 这些状态名），属于死数据。

**另外**: `STOP` / `STOPPED` (lines 61-62, 1168) 也不在 State enum 中。Orchestrator 处理 STOP intent 时 emit 的是 `IDLE`。这两个属于防御性条目。

**建议**: 确认后可安全删除 map 中的 3 个幻影键和 message handler 中的对应分支。删除时需同步改 `_status_message_for_state` 的 set 和 if-block。

---

## GRASP / GRASPING 命名修复（已完成）

**问题**: map 中写的是 `"GRASPING"`，State enum 值是 `"GRASP"`。字符串不匹配 → `.get()` 返回默认 `("unknown", 0)`。

**修复**: map line 59 和 message handler line 1166 均已改为 `"GRASP"`。移动端现在能正确显示抓取状态。

---

## 目录迁移（已完成）

- `VISTA/vision_module_v2/` → `old/VISTA/vision_module_v2/` — 启动崩溃，零引用
- `VISTA/Offline_Edge_Test/Online/` → `old/VISTA/Offline_Edge_Test/Online/` — `Online_Edge_Detect/` 的旧版，零引用

**注意**: `Online_Edge_Detect/` 不能删 — `table_edge_manager.py:75` 运行时依赖。

---

## 已完成的 GRASP 串联工作（来自 next_todo.md）

### VISTA grasp.py 适配 v1.1 协议
- **文件**: `VISTA/vision_module/app/stages/grasp.py:425-485`
- **改动**: 三分法 status 检查 (`success→RESULT_READY`, `failure→FAILED`, `reposition_required→RUNNING+reposition_hint`)
- **提交**: `4153fd4`

### target → class_id 映射
- **文件**: `orchestrator/orchestrator_service/utils/target_utils.py`
- **函数**: `target_to_class_id()` — apple→47, banana→46, bottle→39, cup→41

### Orchestrator GRASP 状态 + 通信基础设施 (8 文件)
- **新建**: `bridge/arm_protocol.py`, `utils/grasp_utils.py`, `examples/test_grasp_dryrun.py`
- **修改**: `ipc/protocol.py` (ArmCommand/ArmResponse/make_grasp_req), `runtime/context.py` (State.GRASP+8 字段), `runtime/controller.py` (MotionDecision.arm_cmd), `bridge/uart_bridge.py` (send_arm_command), `runtime/state_machine.py` (_tick_grasp 三子状态), `runtime/service.py` (grasp_obs+arm_cmd dispatch)
- **验证**: `test_grasp_dryrun.py` 全流程 FREEZE_BASE→GRASP→POSE→OK POSE→DONE→IDLE 通过

### grasp server v1.2 协议适配
- VISTA: `reposition_hint: True` → `reposition_proposal` dict 透传
- Orchestrator: 新增 `REPOSITIONING` 子状态，`dx_cm`/`dy_cm` → 底盘定时开环微调
- 验证: dry-run 含 reposition 步骤，底盘命令方向正确

### 过渡策略（占位，后续替换）
| 项 | 当前 | 后续 |
|----|------|------|
| operating_time | 500ms 默认 | 外部传入或 server 侧提供 |
| claw 查表 | `int(width_cm * 10)` | STM32 真实张合角度映射表 |
| reposition 微调 | v1.2 reposition_proposal 已集成 | done |

---

## 代码清理（2026-05-05 审计后完成）

- `src/hal/` → `old/src/hal/` (HAL 墓碑 README)
- `_fast_gst_camera.py` → `old/VISTA/.../camera/_fast_gst_camera.py` (从未被注册/导出)
- 删除所有 `.pyc` / `__pycache__/` (30+ 文件, 18 目录)
- 删除 `TableEdgeDetector.zip`
- `.gitignore` 新增 `*.zip` 规则
