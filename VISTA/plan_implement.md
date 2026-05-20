# VISTA 架构优化 — 实现需求

日期：2026-05-20（更新：ABCD 四个步骤已全部实现）
来源：arch_optimize.md 讨论后的设计决策

---

## 一、已完成的框架工作

### 1.1 架构税务清理

| 项 | 内容 |
|----|------|
| App 状态去重 | VistaApp 不再维护 StageContext 副本，统一走 `_ctx()` |
| Mode 切换收拢 | 死代码删除、TABLE_EDGE→DEPTH 回退移入 SearchStagePlan、Grasp tick 改为 allowlist |
| Tick 自主订阅 | BaseStagePlan 加 `common_routes`/`optional_routes`，Scheduler 按 filter 采集 |
| Edge 标准化 | `Online_Edge_Detect/` → `edge_detect/`，standalone 文件移至 `old/`，动态 import 改为直接 import |

### 1.2 Step A-D（已提交）

| Step | Commit | 内容 |
|------|--------|------|
| A | `f3cc925` | RemoteProfile 扩展 `kind: "loop"|"task"` / `action` / `max_retries`，ModeProfile 扩展 `table_edge_path` / `table_edge_update_hz`，`_compile_plan()` 透传，环境变量弃用警告 |
| B | `31d7a3c` | RemoteManager task worker（`_run_task`，执行完自动退出线程），孤儿线程 kill（join 1s + daemon），TableEdgeManager 去 `_active_mode()`，`configure(path, update_hz)` 从 plan 接收配置 |
| C | `5a39631` | GRASP_REMOTE_INIT ModeProfile（第 8 个 mode，`kind=task, action=init, max_retries=3`），StageContext 加 `server_status: str`，StageController.tick() 从 remote_result 同步，GraspStagePlan 完整 mode 链：`GRASP_REMOTE_INIT → GRASP_REMOTE`，on_respond BUSY 处理，MICRO_ADJUST allowlist 扩展 |
| D | `acc6b1d` | 删除 `_remote_effect()` 及其两个调用点（INIT/PREDICT effects）。`remote_cmd` route、`_publish_effects()`、RemoteManager loop consumer 保留等待确认无其他使用者 |

---

## 二、待实现

### 2.1 VisionEngine 删除

VisionEngine 当前是 Scheduler + RuntimeSupervisor + 5 Managers 的薄 facade。删除后的职责分流：

| 原 VisionEngine 功能 | 移往 | 说明 |
|----------------------|------|------|
| 构造函数（组装 Manager） | `VistaApp.__init__` | App 直接创建 Scheduler + Managers + RuntimeSupervisor |
| `apply_mode_plan` | `ModeController.switch_mode()` | 内部直接 `scheduler.configure()` + `supervisor.reconcile()` |
| `start` / `stop` | `ModeController` | runtime 生命周期跟随 mode controller |
| `collect_tick_input` | `VistaApp._tick_stage` → `scheduler.collect_tick_input` | 直接调，不需要中间层 |
| `push_stage_signals` | `StageController` → `scheduler.push_stage_signals` | 直接调 |
| `publish_result("runtime_status")` | `StageController` → `scheduler.publish_result` | generation 从 `mode_controller.generation` 获取 |
| `publish_event` | 删除（随 effects 移除） | 已无调用方 |
| `runtime_snapshot` | `ModeController` | Manager 快照聚合，供心跳使用 |
| `_active_runtime_generation` | 删除 | 冗余副本，直接用 `ModeController._generation` |
| `_active_runtime_plan` | 删除 | 冗余副本 |
| capability 事件回调 | **删除** | 日志/事件记录后面再优化，当前阶段不保留 |

### 2.2 删除后的层级关系

```
VistaApp
  ├── Scheduler
  ├── ModeController(scheduler, supervisor, managers)
  │     ├── switch_mode() 内含 scheduler.configure → supervisor.reconcile
  │     ├── runtime_snapshot()
  │     └── start() / stop()
  ├── StageController(scheduler, mode_controller)
  │     ├── push_stage_signals → scheduler
  │     ├── publish_result → scheduler
  │     └── _apply_context_mode → mode_controller.switch_mode
  └── RuntimeSupervisor(scheduler, managers, backend_event_sink)
```

### 2.3 INIT stage

全局 INIT stage 挂 `INIT` mode（与 GRASP_REMOTE_INIT 不同），在 `VistaApp.start()` 阶段触发：

- 负责全局初始化：camera、predictor、remote 健康检查
- 进入 INIT stage → 执行完毕后自动转入 IDLE，等待 Orchestrator 发请求
- INIT mode 的 plan 中包含所有启动时需要验证的 capability（camera check、predictor load、remote best-effort `/init`）

### 2.4 effects 残留清理（原 Step D 收尾）

- `stage_controller.py` 中 `_publish_effects()` 确认无其他使用者后删除
- `Scheduler` 中 `remote_cmd` route 确认无其他使用者后删除
- `remote_ack` route 确认无消费者后一并删除

---

## 三、后续工作（不在本次范围）

| 项 | 说明 |
|----|------|
| edge_detect 配置迁移 | `backend/edge_detect/board_config.py` → 主 `config/schema.py` |
| Backend 目录分层 | 框架/管理器/驱动/能力包 |
| 日志/事件记录优化 | 数据面 route 记录、manager 状态快照、capability 事件恢复 |
| Mode 切换优化 | 避免同 worker 反复启停 |
| ARCHITECTURE.md 同步 | VisionEngine 删除后更新分层图和调用链 |

---

## 四、实施顺序

| Step | 内容 | 依赖 | 状态 |
|------|------|------|------|
| A | ModeProfile + RemoteProfile 扩展，TableEdgeManager 去 mode 感知，环境变量弃用 | 无 | done |
| B | RemoteManager task worker + 孤儿线程 kill + TableEdgeManager path config | A | done |
| C | GRASP_REMOTE_INIT mode + server_status + GraspStagePlan mode 链 | B | done |
| D | effects producer 删除（`_remote_effect`） | C | done |
| E | VisionEngine 删除 + capability 回调删除 + 层级重连 | D | **待做** |
| F | INIT stage + INIT mode（全局初始化） | E | **待做** |
| G | effects 残留清理（`_publish_effects`、`remote_cmd`、`remote_ack`） | D | **待做** |
| H | ARCHITECTURE.md 同步 | E+F+G | **待做** |
