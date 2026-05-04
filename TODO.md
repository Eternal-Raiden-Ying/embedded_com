# TODO.md — 项目总后续工作跟踪

更新时间：2026-05-05

状态约定：
- `todo`：未开始
- `doing`：正在推进
- `done`：已完成
- `blocked`：存在外部依赖或前置条件未满足

---

## 项目当前状态概览

| 模块 | 状态 | 关键未完成项 |
|------|------|-------------|
| mobile_gateway | MVP 可用，小程序已稳定 | MQTT broker 生产化、ASR 管线（P1） |
| Orchestrator | **GRASP 状态 + v1.2 适配完成**（reposition proposal 集成，dry-run 全通过） | 端到端真车联调 |
| VISTA | **v1.1 适配完成**（三分法 + result 规范化） | 端到端联调 |
| grasp server | **协议 v1.1 已冻结** | `operating_time`/proposal 后续版本 |
| Voice/ASR | 已从板端归档 | ASR→NLU→mobile_cmd 管线（P1） |

---

## P0 — Grasp 串联进入主链路

参考：
- [branch_discovery.md](docs/branch_discovery.md) — VISTA IPC 架构发现（Section 1 必须修改项）
- [grasp_protocol_analysis.md](docs/grasp_protocol_analysis.md) — 全链路协议分析
- [api_protocol.md](E:/Documents_E/vscode/embedded_com/grasp_module/docs/api_protocol.md) — grasp server v1.1

### 1. Grasp Server 输出协议 v1.1
- 状态：`done`
- 已冻结：
  - `status` 三分法：`success` / `reposition_required` / `failure`
  - `reason` 5 值：`null` / `no_detection` / `no_grasp_detected` / `no_feasible_grasp` / `score_below_threshold`
  - 新增 `detection` 对象（`requested/resolved_class_id`, `found`, `similar_detection_result` 等）
  - 新增 `format_version`, `message`
  - `targets[]` 11 字段 + 坐标系约定（robot 系，X前/Y左/Z上）
- `operating_time` 预留后续版本

### 2. VISTA 适配 v1.1 协议
- 状态：`done`
- 文件：`VISTA/vision_module/app/stages/grasp.py:425-485`
- 实现：三分法 status (`success→RESULT_READY`, `failure→FAILED`, `reposition_required→RUNNING+reposition_hint`)
- 验证：`py_compile` 通过

### 3. vision_obs.result 结构规范化
- 状态：`done`
- 与 Item 2 同一改动，合并实现：`targets[0]→result.grasp`, `detection→result.detection`，删除调试字段透传

### 4. target → class_id 映射函数
- 状态：`done`
- 文件：`orchestrator/orchestrator_service/utils/target_utils.py`
- 映射：apple→47, banana→46, bottle→39, cup→41

### 5. Orchestrator 侧 GRASP 状态 + 通信基础设施
- 状态：`done`
- 8 文件改动：新建 `arm_protocol.py`, `grasp_utils.py`, `test_grasp_dryrun.py`；修改 `protocol.py`, `context.py`, `controller.py`, `uart_bridge.py`, `state_machine.py`, `service.py`
- 三子状态：`AWAITING_RESPOND → AWAITING_RESULT → AWAITING_ARM → DONE`
- 验证：`py_compile` 全部通过，`test_grasp_dryrun.py` 全流程通过（FREEZE_BASE→GRASP→POSE→OK POSE→DONE→IDLE）

**过渡策略**：
- `operating_time` → 默认 500ms
- `width_to_claw_angle()` → 占位 `int(width_cm * 10)`
- reposition 微调 → `done`（v1.2 `reposition_proposal` 已集成，见 Item 5a）

### 5a. Grasp Server v1.2 协议适配
- 状态：`done`
- 变更：`feasible_angle_deg→feasible_distance_cm`，新增 `reposition_proposal` 对象（dx_cm/dy_cm 等）
- 文件：`VISTA/grasp.py` + `context.py` + `state_machine.py` + `test_grasp_dryrun.py` + `docs/grasp_protocol_analysis.md`
- 实现：reposition 不再使用占位 boolean，改为透传完整 proposal dict；新增 `REPOSITIONING` 子状态实现底盘微调移动（dx/dy → vx/vy 定时开环移动）
- 验证：`py_compile` 全部通过，`test_grasp_dryrun.py` 全流程通过含 reposition 步骤，底盘命令方向正确

### 6. 端到端测试
- 状态：`blocked`（依赖 Item 2+3+5）

---

## P1 — VISTA 代码优化（来自 branch_discovery.md）

### 7. effects 通道正式化
- 状态：`todo`
- 内容：
  - `BaseStagePlan` 新增 `emit_event(route, payload)` 方法
  - 替代当前 `grasp.py` 中手写的 `_remote_effect()` dict
  - `_publish_effects()` 已就绪，只需生产侧规范化
- 参考：`branch_discovery.md` Section 2.3, Section 6

### 8. req_type / op 语义清理
- 状态：`todo`
- 内容：
  - `StageController._request_type()` 将 RESPOND 归为 `target_update`，语义不准
  - 建议新增 `respond` 类型或独立归类
- 参考：`branch_discovery.md` Section 2.1

### 9. VistaApp / StageContext 状态双写消除
- 状态：`todo`
- 内容：
  - VistaApp 通过 `self.stage_controller.context()` 直接读取
  - 去除 `_sync_runtime_from_stage_context()` 同步函数和自有副本
- 参考：`branch_discovery.md` Section 2.2

### 10. vision_develop TRACK_TABLE 合入评估
- 状态：`todo`
- 内容：vision_develop 的 TRACK_TABLE 工作是否合入 up2date

### 11. VISTA ReadMe.md 文档同步
- 状态：`todo`
- 内容：DEPTH_PERCEPTION/TABLE_EDGE_PERCEPTION 标注修正 + 日志文件名更新

### 12. release_cooldown_s 决策
- 状态：`todo`
- 内容：实现字段行为 或 从 profile schema 移除

### 13. preview/table_edge 设计一致性审查
- 状态：`todo`
- 注意：开发前需讨论框架设计

---

## P1 — Mobile Gateway 生产化（优先级下调）

小程序框架稳定，不做架构调整。

### 14. MQTT Broker 选型与部署
- 状态：`todo`
### 15. 小程序端 ASR → 结构化命令管线
- 状态：`todo`
### 16. 视障用户音频反馈
- 状态：`todo`
### 17. 生产加固
- 状态：`todo`

---

## P2 — 远期

### 18. Vision Module v2 深度桌边提取
- 状态：`blocked`（Offline_Edge_Test 验证中）
### 19. vision_module vs vision_module_v2 关系清理
- 状态：`todo`
### 20. 多入口 Control Plane
- 状态：`todo`
