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

### 2.4 Silent mode — 所有 stage 的 default

当前各 StagePlan 的 `default_mode` 直接指向业务 mode（`SEARCH→TRACK_LOCAL`、`GRASP→MICRO_ADJUST`、`RETURN→TRACK_LOCAL`），导致 default mode 携带能力实现。改为每个 stage 有一个 **silent mode** 作为兜底：

- 无任何 capability 实现（无 camera、无 predictor、无 remote、无 table_edge）
- 仅响应 `req.mode_hint` 进行 mode 切换，或由 tick 逻辑主动设置 mode
- 作为"非预期情况"的安全兜底

**以 GRASP stage 为例**：

| Mode | 触发 | 能力 |
|------|------|------|
| `IDLE`（silent default） | 进入 stage 时的初始状态，或 mode_hint 未匹配 | 无 |
| `MICRO_ADJUST` | req.mode_hint 驱动，或 tick 中 allowlist 回退 | camera + predictor |
| `GRASP_REMOTE_INIT` | ACCEPT + server_status != ready | remote task init |
| `GRASP_REMOTE` | ACCEPT + server_status == ready，或 init 完成后自动切换 | camera + depth + remote task predict |

进入 GRASP stage 时 Orchestrator 必带 `mode_hint`，所以不会停留在 silent mode。

### 2.5 remote_init_status 路由 — 拆分 init/predict 结果

当前 `remote_result` 混用：INIT 任务写 `service_init_state`/`service_init_confirmed`，PREDICT 任务写 `result`。拆分为两个路由：

| 路由 | 生产者 | 消费者 | 内容 |
|------|--------|--------|------|
| `remote_init_status` | RemoteManager INIT task | StageController（RESPOND 路径）+ GraspStagePlan tick（GRASP_REMOTE_INIT 分支） | `service_init_state`、`service_init_confirmed`、retry 计数 |
| `remote_result` | RemoteManager PREDICT task | GraspStagePlan tick（GRASP_REMOTE 分支） | `last_action`、`last_ok`、`has_result`、`result` |

RemoteManager `_run_task` 根据 `action` 写入对应路由。Silent mode 订阅 `remote_init_status`，GRASP_REMOTE 订阅 `remote_result`。StageController.handle_request() RESPOND 路径从 `remote_init_status` 读取 server_status，不再依赖 MICRO_ADJUST 的订阅。

### 2.6 ABCD 审计发现

#### HIGH（已修复）

| # | 问题 | 修复 |
|---|------|------|
| A1 | `RemoteProfile.command` 与 `action` 不匹配 | `command` 是 HTTP 线格式字段，task worker 用 `action`。暂保留 `command`，待 loop worker 全清后删除 |
| C1 | RESPOND 路径 server_status 恒为 "unknown"，直达 GRASP_REMOTE 分支死代码 | 由 silent mode + `remote_init_status` 路由解决 |
| D1 | PREDICT task 在 camera frames 就绪前执行 → `missing_camera_frames` | 已修复：`_run_task("predict")` 自旋等待帧 + generation 匹配 |

#### MEDIUM

| # | 问题 | 评估 |
|---|------|------|
| C2 | BUSY 守卫在 server_status 检查前触发——init 刚好完成时 RESPOND 仍返回 FAILED/busy | 将 BUSY 守卫移到 server_status 检查之后。如果 server_status 已是 "ready" 且 mode 仍为 GRASP_REMOTE_INIT（task 刚完成），跳过 BUSY 直接接受 |
| C3 | GRASP_REMOTE_INIT→GRASP_REMOTE 切 mode 强制重新分配 camera | GRASP_REMOTE_INIT 目前 `enabled_cameras=()`，切到 GRASP_REMOTE 需要 rgb+depth。改为让 GRASP_REMOTE_INIT 也启用 camera（但不做 predict），减少切换开销 |
| C4 | GRASP_REMOTE_INIT 和 GRASP_REMOTE 各自检查不同的 init 确认字段 | 拆分 `remote_init_status` 后统一用 `service_init_confirmed` |
| A2 | GRASP_REMOTE 启用 depth camera → 触发 table_edge 启发式 → 意外启用 table_edge 10Hz | `_compile_plan()` 的 table_edge 启发式需增加 escape hatch：`table_edge_path="none"` 时跳过 |
| A3 | `VISTA_TABLE_EDGE_HZ` 未被列入弃用警告 | 加入 `_warn_deprecated_env()` |

### 2.7 effects 残留清理

**已清理**（`a21ce7f`）：
- `RemoteManager._worker_loop()` loop 路径 + `_handle_command()` + `_publish_event()` 删除

**待清理**（VisionEngine 删除后）：
- `stage_controller.py` 中 `_publish_effects()` 确认无其他使用者后删除
- `Scheduler` 中 `publish_event()` / `consume_event()` 确认无其他 route 使用后删除
- `mode_controller.py` 中 `remote_cmd` / `remote_ack` route 注册 + `_verify_plan` 校验删除
- `grasp.py` 中 `remote_ack` 订阅删除
- 测试中 `remote_cmd` / `remote_ack` 引用更新

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
| E | VisionEngine 删除 + capability 回调删除 + 层级重连 | D | done |
| F1 | INIT stage + INIT mode（全局初始化） | E | done |
| F2 | Silent mode — 所有 stage 的 default mode 改为无 capability 的兜底 | F1 | **待做** |
| F3 | remote_init_status 路由 — 拆分 init/predict 结果到独立 route | F2 | **待做** |
| F4 | Audit findings 修复（C2 C3 C4 A2 A3） | F3 | **待做** |
| G | effects 残留清理（`remote_cmd`、`remote_ack` route 删除） | E, F3 | **待做** |
| H | ARCHITECTURE.md 同步 | all done | **待做** |
