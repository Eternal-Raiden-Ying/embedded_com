# VISTA Event Description

## 范围

本文档描述当前代码会写入 `VISTA/runs/<stack_run_id>/event.jsonl` 的事件。

不包含：

- `ipc.jsonl` 中的 transport 事件
- `heartbeat.jsonl` 中的低频健康快照

## 写入路径

`event.jsonl` 当前有 3 条写入路径：

1. App 事件
   - `VistaApp._record_event(...) -> RunLogger.write_event_record(...)`
2. Stage 事件
   - `StageController._emit_event(...) -> VistaApp._record_stage_event(...) -> RunLogger.write_event_record(...)`
3. Backend 事件
   - `ModeController -> VistaApp._record_backend_event(...)`
   - `RuntimeSupervisor` / 各 manager -> `VisionEngine._emit_event(...) -> VistaApp._record_backend_event(...)`
   - 最终统一进入 `VistaApp._record_event(...) -> RunLogger.write_event_record(...)`

补充：

- `VistaApp._record_backend_event(...)` 会对 backend 事件补齐当前 `stage/mode/session_id/req_id/epoch/interaction_id`
- 如果未来重新启用 `BACKEND_DIAGNOSTIC`，它只会在 `VISION_LOG_MODE=full` 或 `VISION_DEBUG=1` 时落盘；当前代码中没有实际 emitter

## 当前事件清单

| 事件名 | 类别 | 触发位置 | 触发条件 | 说明 |
| --- | --- | --- | --- | --- |
| `SERVICE_STARTING` | 服务生命周期 | `VistaApp.start()` | 启动流程开始 | 已写入 `meta.json`，准备启动 IPC 和 runtime |
| `SERVICE_READY` | 服务生命周期 | `VistaApp.start()` | request server 与 runtime 就绪 | 服务进入主循环前写入 |
| `SERVICE_STOPPING` | 服务生命周期 | `VistaApp.stop()` | 停机流程开始 | 开始关闭 IPC 和 runtime |
| `SERVICE_STOPPED` | 服务生命周期 | `VistaApp.stop()` | 停机流程结束 | 资源释放完成 |
| `FATAL` | 服务生命周期 | `VistaApp.run()` | 主循环抛出未处理异常 | `level=error` |
| `VISION_REQ` | 请求接收 | `VistaApp._handle_request_payload()` | 一个 `vision_req` 或兼容请求被解析并交给 stage controller 处理后 | 此时 `stage/mode` 已同步到当前上下文 |
| `VISION_STOP` | 请求接收 | `VistaApp._handle_stop_request()` | stop 流程被接受 | 记录 stop 进入流程 |
| `ENTER_IDLE` | 运行态切换 | `VistaApp._enter_cold_idle()` | stop 后进入冷 idle | 与 `IDLE` mode 相关 |
| `ENTER_HOT_STANDBY` | 运行态切换 | `VistaApp._enter_hot_standby()` | stop 后进入热待机 | 与 `IDLE_HOT` mode 相关 |
| `STAGE_TRANSITION` | Stage 工作流 | `StageController._emit_transition()` | enter / restart / stop 时 | 描述 `from_stage`/`to_stage` 和 `from_mode`/`to_mode` |
| `INTERACTION_RESPONSE_HANDLED` | 交互 | `StageController.handle_request()` | `RESPOND` 请求被当前 stage 消费 | 主要用于 `GRASP` 交互回合 |
| `INTERACTION_STATE_CHANGED` | 交互 | `StageController._emit_output_events()` | 输出状态进入 `WAITING_RESPONSE` / `RESULT_READY` / `FAILED` 且状态键发生变化 | 不是每个 tick 都重复写 |
| `MODE_APPLY_FAILED` | Stage / Mode | `StageController._mode_apply_failed_output()` | mode 切换申请失败 | 常见于 enter/update/respond/tick 后 runtime apply 失败 |
| `BACKEND_LIFECYCLE_CHANGED` | Backend 生命周期 | `VisionEngine._emit_backend_lifecycle()` | backend 初始化、启动、停止 | `action` 常见为 `initialized` / `started` / `stopped` |
| `BACKEND_MODE_CHANGED` | Backend mode | `ModeController.commit_switch()` | mode plan 已经成功生效并提交 | 这是 backend mode 真值事件 |
| `BACKEND_RUNTIME_RECONCILED` | Backend runtime | `RuntimeSupervisor._apply_plan()` | 一次 runtime plan reconcile 结束 | 记录 `mode`、`generation`、`ok` |
| `CAPABILITY_CHANGED` | Capability | `VisionEngine._emit_capability_change()` | camera / predictor / remote / preview 状态变化 | 由 engine 汇总 manager 回调后统一写事件 |
| `BACKEND_FAILURE` | Backend 故障 | `ModeController` / `VisionEngine` | mode 不存在、mode plan 非法、mode apply 不完整等失败 | `level=error` |

## 事件顺序说明

### 1. 请求与 stage 切换

同一个请求里，当前实现通常是：

1. `StageController.handle_request(...)` 先完成 stage/mode 更新
2. 如果发生 stage enter/restart/stop，会先写 `STAGE_TRANSITION`
3. `VistaApp` 再记录 `VISION_REQ`

因此，`STAGE_TRANSITION` 可能早于同一次请求对应的 `VISION_REQ`。

### 2. mode 真值

mode 相关有两类事件：

- `STAGE_TRANSITION`
  - 来自控制面
  - 反映 stage 逻辑中的目标变化
- `BACKEND_MODE_CHANGED`
  - 来自 `ModeController.commit_switch()`
  - 反映 runtime plan 已成功应用后的 backend mode 真值

如果要追踪资源真实生效情况，以 `BACKEND_MODE_CHANGED` 为准。

### 3. mode 失败

mode 切换失败时常见事件链：

- `BACKEND_FAILURE`
  - 来自 `ModeController.prepare_switch()` 或 `VisionEngine.apply_mode_plan()`
- `MODE_APPLY_FAILED`
  - 来自 `StageController`
  - 表示当前 request/tick 对应的业务流程未能完成 mode 应用

## `CAPABILITY_CHANGED` 说明

当前常见 `capability`：

- `camera`
- `predictor_model`
- `remote`
- `preview`

当前常见 `action`：

- `enabled`
- `disabled`
- `loaded`
- `released`
- `sink_changed`
- `reconfigured`
- `enable_failed`
- `load_failed`

实际 action 集合由 runtime 路径决定，不要求固定枚举完全闭合。

## 交互事件说明

当前交互主路径主要在 `GRASP` stage。

- `INTERACTION_STATE_CHANGED`
  - VISTA 进入一个新的交互状态
  - 当前代码只对 `WAITING_RESPONSE` / `RESULT_READY` / `FAILED` 发出该事件
- `INTERACTION_RESPONSE_HANDLED`
  - 上游发送了一次 `RESPOND`
  - 当前 stage 已消费该响应

常见 `decision`：

- `ACCEPT`
- `REJECT`

## 字段约定

`event.jsonl` 顶层字段顺序由 `common/runtime_logging.py` 中的 `EVENT_FIELD_ORDER` 固定为：

1. `ts`
2. `level`
3. `module`
4. `stack_run_id`
5. `event`
6. `stage`
7. `mode`
8. `trigger`
9. `session_id`
10. `req_id`
11. `epoch`
12. `interaction_id`
13. `data`

未知附加字段会被折叠进 `data`，而不是持续扩展顶层列。
