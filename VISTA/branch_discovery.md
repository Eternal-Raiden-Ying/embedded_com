# Branch Discovery — Code-VS-Spec Audit

本文件由 Code-VS-Spec agent 于 2026-05-05 创建，记录文档与代码对照审查中发现的待决策事项。

---

## 1. TableEdgeManager 缺失于架构文档

- **来源**: Code-VS-Spec audit, 2026-05-05
- **发现**: `TableEdgeManager` 是拥有独立 worker 线程的完整运行时管理器（发布 `table_edge_obs`，消费 `camera_frames`/`local_perception`/`runtime_status`），但未在 ARCHITECTURE.md 的架构图、manager 列表和数据流图中体现。
- **评估**: 已修复 — 已将 `TableEdgeManager` 加入 ARCHITECTURE.md 的拓扑图、mode 列表、manager 层描述、Scheduler 路由表和数据流图。
- **决策**: 已解决。

---

## 2. `release_cooldown_s` 声明但未使用

- **来源**: Code-VS-Spec audit, 2026-05-05（ARCHITECTURE.md 自承此问题）
- **发现**: `ModeProfile.release_cooldown_s` 在 `backend/mode_profiles.py:48` 声明，6 个 mode profile 均设置了非零值（2.0s–5.0s，见 `config/mode_defaults.py`）。但 `_compile_plan()` 从不将该字段编入 plan dict，整个 `vision_module/` 目录下零处运行时读取。运行时始终立即 stop/release。
- **评估**: 字段是装饰性的，配置承诺了 cooldown 语义但运行时未兑现。
- **决策**: 二选一 — (a) 在 `RuntimeSupervisor` 的 shutdown/disable 路径中实现真实的 cooldown 等待；或 (b) 从 `ModeProfile` 和所有 profile 配置中删除该字段以避免虚假抽象。

---

## 3. 错误处理模式分层不一致

- **来源**: Code-VS-Spec audit, 2026-05-05
- **发现**: 错误处理在三个层级使用三种不同模式：
  - Scheduler 层：返回 `bool`（`publish_result`/`publish_event`）
  - Manager 层：将错误信息嵌入 payload 字段（`contract_ok`/`contract_error`/`contract_warnings`）
  - Stage 层：使用状态枚举（`RUNNING`/`FAILED` 在 `StageOutput` 中）
  - RuntimeSupervisor：`reconcile()` 返回 `bool`，内部 catch 并 log 每个 manager 的异常
- **评估**: 这可能是故意的分层设计——每层有不同的错误语义（基础设施 vs 领域 vs 协议），但文档从未明确说明这种混合策略。当前不影响功能正确性。
- **决策**: 建议在 ARCHITECTURE.md 中显式描述此分层错误处理策略，确认其为设计意图而非疏漏。或者评估是否需要统一到单一模式。

---

## 4. PreviewManager 对未注册路由 `target_obs` 的无效读取

- **来源**: Code-VS-Spec audit, 2026-05-05
- **发现**: `PreviewManager` 在 `preview/manager.py:347` 调用 `scheduler.read_result("target_obs", default={})`，但 `target_obs` 未在 `ModeController._compile_plan()` 中注册为路由。该读取始终返回默认值 `{}`，是无效操作。Preview 实际上从 `local_perception` payload 中提取目标数据。
- **评估**: 无害的死代码，但可能误导读者认为存在 `target_obs` 路由。
- **决策**: 移除 `PreviewManager` 中的死读取，或注册 `target_obs` 为正式路由（如果有未来用例）。

---

## 5. IDLE stage 无独立 StagePlan 类

- **来源**: Code-VS-Spec audit, 2026-05-05
- **发现**: IDLE 在文档中与 SEARCH/GRASP/RETURN 并列，但没有 `stages/idle.py` 文件。IDLE 由 `StageController._transition_to()`（`stage_controller.py:342-349`）隐式处理：创建空 `StageContext` 并返回 `None`。其他三个 stage 各有独立的 `BaseStagePlan` 子类。
- **评估**: IDLE 确实是 no-op 状态，不需要复杂逻辑。但不对称性在文档中未解释，新读者可能困惑为何只有三个 StagePlan 文件。
- **决策**: 在 ARCHITECTURE.md 中说明 IDLE 以空 StageContext 隐式实现，无需独立 plan 文件。或为一致性创建一个最小化的 `IdleStagePlan`。

---

## 6. `capability_placeholder` 残留于配置

- **来源**: Code-VS-Spec audit, 2026-05-05
- **发现**: `capability_placeholder` 仍存在于 `config/schema.py:33` 作为配置字段，从环境变量加载（`config/board_config.py:52`），并包含在诊断转储中（`app/app.py:130`）。但它不参与任何运行时路径选择——`VISTA_BACKEND` 已完全接替。
- **评估**: 死配置字段，增加维护负担。
- **决策**: 从 schema、board_config 加载和诊断输出中移除 `capability_placeholder`，或明确标记为 deprecated 并附加删除时间线。

---

## 7. `_fast_gst_camera.py` 依赖未清理

- **来源**: Code-VS-Spec audit, 2026-05-05
- **发现**: `_fast_gst_camera.py` 曾被移至 `old/` 目录，但 `ColorCamera.py:4` 和 `IRCamera.py:4` 仍有 `from ._fast_gst_camera import FastGstCameraBase` 导入。`HardwareCamera.py` 通过继承 `ColorCamera` 存在传递依赖。
- **评估**: 文件已还原至源路径（`old/` 保留了原始层级备份）。待后续评估是否可安全移除——需确认 `FastGstCameraBase` 的逻辑是否可合并到现有 camera 基类中。
- **决策**: 暂保留，待重构时处理。

---

## 8. `class_names_source` 字段冗余

- **来源**: Code-VS-Spec audit, 2026-05-05
- **发现**: `class_names_source` 在 `predictor_manager.py` 中有三种值（`"profile"` / `"fallback_coco80"` / `"missing"`），通过 `local_perception` 发布。但下游消费者（search.py、preview/manager.py、opencv_sink.py、grasp.py、return_home.py）均只读取 `class_names`，**无一下游检查 `class_names_source`**。该字段的唯一实际用途是在 `predictor_manager.py` 内部生成 `contract_warnings`。
- **评估**: 字段不影响业务逻辑，属于诊断/追溯用途。如果合约警告已足够覆盖降级场景，收敛时可以考虑删去此字段以精简 `local_perception` payload。
- **决策**: 暂保留。设计文档（INTERFACES.md）不需要修改——该字段属于内部实现细节而非外部 IPC contract。
