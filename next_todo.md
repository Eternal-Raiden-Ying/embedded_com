# next_todo.md — 已完成事项

更新时间：2026-05-04

---

## 已完成 — VISTA + Orchestrator GRASP 串联

### 1. VISTA grasp.py 适配 v1.1 协议
- **文件**: `VISTA/vision_module/app/stages/grasp.py:425-485`
- **改动**: 三分法 status 检查 (`success→RESULT_READY`, `failure→FAILED`, `reposition_required→RUNNING+reposition_hint`)，提取 `targets[0]→result.grasp`，`detection→result.detection`
- **验证**: `py_compile` 通过
- **提交**: `4153fd4` VISTA: adapt GraspStagePlan to grasp server v1.1 three-way status

### 2. target → class_id 映射
- **文件**: `orchestrator/orchestrator_service/utils/target_utils.py`
- **函数**: `target_to_class_id(target: str) -> int`，COCO80 查表，独立包装便于替换
- **映射**: apple→47, banana→46, bottle→39, cup→41

### 3. Orchestrator GRASP 状态 + 通信基础设施 (8 文件)
- **新建**:
  - `bridge/arm_protocol.py` — `encode_pose()`, `parse_arm_response()`
  - `utils/grasp_utils.py` — `grasp_to_pose_params()`, `normalize_roll()`, `width_to_claw_angle()`(占位)
  - `examples/test_grasp_dryrun.py` — dry-run 测试脚本
- **修改**:
  - `ipc/protocol.py` — `ArmCommand`, `ArmResponse`, `make_grasp_req()`
  - `runtime/context.py` — `State.GRASP` + 8 上下文字段
  - `runtime/controller.py` — `MotionDecision.arm_cmd` 可选字段
  - `bridge/uart_bridge.py` — `send_arm_command()` 直写（不走 latest-override）
  - `runtime/state_machine.py` — `_tick_grasp()` 三子状态 + vision binding + FREEZE_BASE→GRASP 条件过渡
  - `runtime/service.py` — `grasp_obs` 提取 + `arm_cmd` 分发 + arm response UART 解析
- **验证**: `py_compile` 全部 8 文件通过；`test_grasp_dryrun.py` 全流程通过
- **测试结论**: FREEZE_BASE→GRASP→AWAITING_RESPOND→AWAITING_RESULT→AWAITING_ARM→DONE→IDLE，POSE 命令正确生成 (`POSE 15 0 12 0 0 85 500`)

### 过渡策略（当前占位，后续替换）

| 项 | 当前 | 后续 |
|----|------|------|
| operating_time | 500ms 默认 | 外部传入或 server 侧提供 |
| claw 查表 | `int(width_cm * 10)` | STM32 真实张合角度映射表 |
| reposition 微调 | 占位 0/空值 | grasp server proposal 字段 |

---

## 待完成 — P0 阻塞项

- **端到端测试**: 依赖板端测试 skill 就位（真车联调：mobile_cmd → search → lock → grasp → DONE）
- **VISTA 端到端验证**: grasp.py 修改已被 Orchestrator 消费的验证（需 VISTA 运行环境）

## 待完成 — P1

- effects 正式化、req_type 语义清理、状态双写消除、TRACK_TABLE 合入评估、文档同步、release_cooldown_s 决策
- Mobile Gateway 生产化（MQTT broker、ASR 管线、盲人反馈、生产加固）
