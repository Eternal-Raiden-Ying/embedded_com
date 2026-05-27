# VISTA 当前架构

本文档描述当前代码中的实际结构，以及当前仍未收口的架构缺口。

它不是理想化蓝图，而是当前实现的近端基线。

## 目标

VISTA 当前的主线设计已经收敛为：

- `VistaApp` 负责服务生命周期、IPC、主循环、日志、心跳，直接创建 Scheduler + Managers + ModeController
- `StageController` 负责业务阶段控制和 stage state，持有 Scheduler 引用
- `ModeController` 负责 mode profile、switch state、runtime plan 编译，持有 Scheduler + RuntimeSupervisor
- `RuntimeSupervisor` 负责 capability reconcile
- `Scheduler` 负责 manager 与 stage 之间的摘要数据交换
- 各 `Manager` 自己维护 worker loop，不把高频工作塞回主线程

## 总体拓扑

```text
Orchestrator --vision_req--> VistaApp
                              |
                              v
                        StageController ──┬── scheduler.read_result("remote_init_status")
                         |           \    │    (RESPOND path, GRASP stage only)
                         v            v   │
                    StagePlan      ModeController(scheduler + supervisor)
                         |             |
                         |             | switch_mode() 内含 scheduler.configure + supervisor.reconcile
                         |             |
                         +-------> RuntimeSupervisor
                         |          /   |   |    \        \
                         |         v    v   v     v        v
                         |    Camera Predictor Remote Preview TableEdge
                         |       \      |       |      /     /
                         |        +-----+------+------+-----+
                         v                     |
                      Scheduler <─────────────┘
                         |
                         v
              scheduler.collect_tick_input() / StageController.tick()
                         |
                         v
               VistaApp --vision_obs--> Orchestrator
```

## 活跃基线

### 当前 stage

- `INIT`：服务初始化，启动时自动执行 remote `/init`
- `SEARCH`
- `GRASP`
- `RETURN`
- `IDLE`

### 当前默认 mode

所有 StagePlan 的 `default_mode` 统一为 `SILENT`（零能力兜底）。业务 mode 由 Orchestrator 的 `mode_hint` 或 StageController 的 START 兼容逻辑触发。

| Mode | 类型 | 能力 | 说明 |
|------|------|------|------|
| `SILENT` | 兜底 | 无 | 所有 stage 的默认 mode，无任何 capability |
| `INIT` | 全局 | remote task init | 服务启动时执行一次 remote `/init` |
| `TRACK_LOCAL` | SEARCH/RETURN | rgb+depth+Predictor+TableEdge | 本地目标追踪 |
| `DEPTH_PERCEPTION` | SEARCH | depth(+可选 rgb)+TableEdge | 纯深度桌边感知 |
| `TABLE_EDGE_PERCEPTION` | SEARCH | rgb+depth+Predictor+TableEdge | 完整桌边+目标观测 |
| `MICRO_ADJUST` | GRASP | 无 | 等待 Orchestrator 调整决策 |
| `GRASP_REMOTE_INIT` | GRASP | rgb+depth+remote task init | 远程抓取初始化（至多 3 次重试） |
| `GRASP_REMOTE` | GRASP | rgb+depth+remote task predict | 远程抓取预测 |
| `IDLE_HOT` | IDLE | rgb+preview | 热待机 |

GRASP stage 状态机：`GRASP_REMOTE_INIT → GRASP_REMOTE ↔ MICRO_ADJUST`。`GRASP_REMOTE_INIT` 在 server ready 后自动切 `GRASP_REMOTE`。`GRASP_REMOTE` 收到 RESPOND ACCEPT 切 `MICRO_ADJUST`（等待调整），`MICRO_ADJUST` 收到 ACCEPT 切回 `GRASP_REMOTE`（重试 predict）。

## 分层职责

### 1. App 层

入口文件：`vision_module/app/app.py`

职责：

- 直接创建 `Scheduler`、各 Manager、`RuntimeSupervisor`、`ModeController`、`StageController`
- 启动 `req_in` 和 `obs_out` IPC
- 在主循环中接收请求、驱动 stage tick、发送 `vision_obs`
- 记录 `event.jsonl`、`ipc.jsonl`、`heartbeat.jsonl`
- 管理 stop 后的 `IDLE` / `IDLE_HOT` 过渡

不负责：

- 不直接读 raw frame
- 不直接操作 camera/predictor/remote/preview manager
- 不直接做细粒度 capability 时序控制

### 2. Stage 控制层

文件：

- `vision_module/app/stage_controller.py`
- `vision_module/app/stages/base.py`
- `vision_module/app/stages/init.py`
- `vision_module/app/stages/search.py`
- `vision_module/app/stages/grasp.py`
- `vision_module/app/stages/return_home.py`

职责：

- 维护 `StageContext`（含 `server_status`）
- 处理 `START` / `UPDATE` / `RESPOND` / `STOP`
- START 兼容：缺 `mode_hint` 时自动补（GRASP→GRASP_REMOTE, SEARCH→TRACK_LOCAL, RETURN→TRACK_LOCAL）
- 进行 stage enter / restart / stop
- 调用 `ModeController.switch_mode()` 完成 mode 变更
- 从 `Scheduler` 采样 `StageTickInput`
- 每 tick 同步 `ctx.server_status` 从 `remote_init_status`
- RESPOND 路径直接从 Scheduler 读 `remote_init_status`（保证时效性）
- 产出 `StageOutput`（含 `next_stage` 用于自动 stage 切换），并通过 `vision_obs` 对外输出结果

当前 stage 消费关系：

- `INIT`：消费 `remote_init_status`，就绪后自动切 IDLE
- `SEARCH`：主要消费 `local_perception`
- `GRASP`：消费 `remote_init_status`（GRASP_REMOTE_INIT）/ `remote_result`（GRASP_REMOTE）
- `RETURN`：当前消费 `local_perception`，并已通过 detect 主线生成 outward-compatible 的 `home_tag_obs`

### 3. Mode 控制层

文件：

- `vision_module/backend/mode_controller.py`
- `vision_module/backend/mode_profiles.py`
- `vision_module/config/mode_defaults.py`

职责：

- 注册并解析 `ModeProfile`
- 把 mode 编译成 runtime plan
- 维护当前 mode、target mode、generation、last switch result
- 在 mode 成功生效后发出 `BACKEND_MODE_CHANGED`

当前边界：

- `ModeController` 持有 `Scheduler` + `RuntimeSupervisor`
- `switch_mode()` 内部直接 `scheduler.configure()` + `supervisor.reconcile()`，失败时回滚 scheduler
- 提供 `start_runtime()` / `stop_runtime()` / `runtime_snapshot()`
- 不直接拥有 manager worker

### 4. Runtime 层

文件：

- `vision_module/backend/runtime_supervisor.py`

职责：

- `RuntimeSupervisor`
  - 根据 mode plan 配置 managers
  - 把 capability plan 转成 camera / predictor / remote / preview / table_edge 的启停与配置动作
  - 发出 `BACKEND_RUNTIME_RECONCILED`

### 5. Scheduler / 数据总线层

文件：`vision_module/backend/scheduler.py`

职责：

- 保存当前 `routes`
- 管理 `result_slots`
- 管理 `event_latches`
- 管理 `pending_signals`
- 依据 `generation` 过滤旧数据
- 为 stage 生成 `StageTickInput`

当前重要 route：

- `camera_frames`：slot，backend scope
- `frame_meta`：slot，stage scope
- `local_perception`：slot，stage scope
- `table_edge_obs`：slot，stage scope — 由 `TableEdgeManager` 发布，`SearchStagePlan` 和 `PreviewManager` 消费
- `remote_init_status`：slot，stage scope — 条件注册（INIT / GRASP_REMOTE_INIT mode），RemoteManager INIT task 写入，StageController 读取 `server_status`
- `remote_result`：slot，stage scope — RemoteManager PREDICT task 写入，GRASP_REMOTE mode 消费
- `runtime_status`：slot，backend scope

设计边界：

- `Scheduler` 只做共享数据总线
- 不拥有业务 worker
- 不直接做高频采集、推理或远程请求

### 6. Manager 层

文件：

- `vision_module/backend/camera_manager.py`
- `vision_module/backend/predictor_manager.py`
- `vision_module/backend/remote/manager.py`
- `vision_module/backend/preview/manager.py`
- `vision_module/backend/table_edge_manager.py`

职责：

- `CameraManager`：相机实例生命周期和采集 worker，发布 `camera_frames`、`frame_meta`
- `PredictorManager`：模型生命周期和推理 worker，读取 `camera_frames`，发布 `local_perception`
- `RemoteManager`：远程抓取协作 worker，支持 task 和 loop 两种模式。task 模式执行一次后线程退出（INIT→`remote_init_status`，PREDICT→`remote_result`，RELEASE→`remote_result`）
- `PreviewManager`：预览 worker，读取 `camera_frames`、`runtime_status`、`local_perception`、`table_edge_obs`，渲染调试预览
- `TableEdgeManager`：桌边感知 worker，读取 `camera_frames`、`local_perception`、`runtime_status`，发布 `table_edge_obs`

约束：

- 每个 manager 自己拥有 worker 线程
- manager 之间只通过 `Scheduler` 交换数据
- manager 不直接修改 `StageController` 或 `StageContext`

### 7. Backend driver / sink 层

文件：

- `vision_module/backend/camera/*`
- `vision_module/backend/predictor/*`
- `vision_module/backend/remote/client.py`
- `vision_module/backend/preview/*`
- `vision_module/backend/edge_detect/*`

职责：

- 封装真实设备、真实模型、远程 HTTP 客户端和 preview sink
- 提供 manager 可复用的最底层能力接口

## 当前主调用链

### 请求处理链

```text
VistaApp._handle_request_payload()
  -> VisionReq.from_dict(...)
  -> StageController.handle_request(...)
  -> StagePlan.on_enter / on_update / on_respond / on_stop(...)
  -> StageController._apply_context_mode(...)
  -> ModeController.switch_mode(...)           ← 内含 scheduler.configure + supervisor.reconcile + 失败回滚
  -> VistaApp 发送 vision_obs
```

注：`ModeController.switch_mode()` 内部直接操作 Scheduler 和 RuntimeSupervisor，不再通过回调。

### 控制循环链

```text
VistaApp._tick_stage()
  -> scheduler.collect_tick_input(ts=now, route_filter=...)
  -> StageController.tick(tick_input)
  -> StagePlan.tick(...)
  -> VistaApp 发送 vision_obs
```

注：`VistaApp` 直接调 `scheduler.collect_tick_input()`，`route_filter` 由当前 StagePlan 的 `subscribed_routes(mode)` 提供。

### 数据面链路

```text
CameraManager -> Scheduler(camera_frames, frame_meta)
PredictorManager -> Scheduler(local_perception)
TableEdgeManager -> Scheduler(table_edge_obs)
RemoteManager -> Scheduler(remote_init_status | remote_result)
PreviewManager <- Scheduler(camera_frames, runtime_status, local_perception, table_edge_obs)
TableEdgeManager <- Scheduler(camera_frames, local_perception, runtime_status)
StageController -> Scheduler(runtime_status)
StageController <- Scheduler(remote_init_status)  # RESPOND path only
```

## 当前架构的关键边界

- `App` 不处理 raw frame
- `StageTickInput` 只带摘要结果，不带原始图像
- `StageOutput.signals` 只做短生命周期控制面反馈
- 跨 tick 的业务状态放在 `StageContext.stage_state`
- mode 以 `generation` 作为隔离边界，旧代结果不会进入当前 stage 采样
- preview 是 backend/data-plane 的旁路能力，不再经过 `App` 处理图像

## 当前已知架构缺口

这部分是当前文档必须明确写出的 contract 基线与剩余问题，而不是隐藏在代码里的实现债务。

### 1. Detect 输出 contract 当前基线

现状：

- detect predictor 当前允许保留 predictor-local 后处理语义
- `PredictorManager` 已将 `local_perception` 收口为稳定摘要：
  - `infer_boxes`: `[[x1, y1, x2, y2, score, class_id], ...]`
  - `infer_box_format`: 当前 detect 基线为 `xyxy_score_class_id`
  - `class_names`: 来自 active model/profile 的类别表
  - NumPy 输出在 manager 边界被转换为纯 Python 结构

影响：

- detect 输出已能稳定穿过 manager -> stage 边界
- 后续若更换 predictor-specific postprocess，只能通过显式 contract 变更完成，不能再做未记录漂移

### 2. Detect 类别语义所有权当前基线

现状：

- 默认本地主线是 `coco80 detect`
- stage 侧 detect 解析现在优先消费 `local_perception.class_names`

影响：

- 默认 detect 路径不再依赖 `grasping_coco20` 这一全局表
- 旧的全局 `TARGET_CLASSES` 已从可执行代码中删除，主线真值由 active model/profile 提供

架构方向：

- 类别 vocabulary 应跟随 active model/profile，而不是由单一全局 target 表硬编码主导

### 3. Remote task execution 当前基线

现状：

- RemoteManager 支持两种 worker 模式：`kind="task"`（执行一次后线程退出）和 `kind="loop"`（持续运行，已废弃）
- INIT task：进入 INIT / GRASP_REMOTE_INIT mode 时触发，至多 3 次重试，结果写入 `remote_init_status`
- PREDICT task：进入 GRASP_REMOTE mode 时触发，自旋等待 fresh camera frames（generation 匹配），结果写入 `remote_result`
- RELEASE task：mode 停止/引擎关闭时触发 `/release`
- 孤儿线程 kill：`stop_runtime()` 设 stop event + join(1s)，超时后线程成为孤儿（daemon 线程，HTTP client 10s timeout 后自然退出）

影响：

- 不再依赖 effects 通道，mode 切换即命令派发
- INIT/PREDICT 时序由 mode 链保证：`GRASP_REMOTE_INIT → GRASP_REMOTE`（server ready 后自动切换）
- fresh-frame gate 由 RemoteManager `_run_task("predict")` 内部处理（轮询 camera_frames 等 generation 匹配）

### 5. Camera 与上传参数所有权当前基线

现状：

- `ModeProfile.camera_overrides` 现在承载每个 mode 的显式 capture contract
- `GRASP_REMOTE` 默认 profile 会下发自己的 `rgb/depth` camera overrides
- remote upload encoding 与压缩参数现在属于 `RemoteProfile`

影响：

- remote 的相机与上传设置不再只是 manager 内部硬编码
- 后续要改 remote 抓拍质量时，应优先改 mode/profile，而不是改 worker 私有常量

### 6. `release_cooldown_s` 仍是声明多于行为

现状：

- mode profile 中存在 `release_cooldown_s`
- runtime 当前仍以立即 stop/release 为主

影响：

- 文档和配置承诺了 cooldown 语义，但运行时并未真正兑现

架构方向：

- 要么真正实现 cooldown
- 要么删去装饰性字段，不再制造虚假抽象

### 7. Remote `class_id` 真值来源当前基线

现状：

- `class_id` 现在只从外部请求读取，并可存入 `StageContext.stage_state`
- remote manager 不再从 `target` 或 ASR vocabulary 推导 `class_id`

影响：

- remote contract 不再有双真值来源
- 上游未提供 `class_id` 时，`GRASP` 会在进入 remote 路径前显式失败

## 当前建议的所有权划分

为减少重复定义和中间层膨胀，建议坚持以下边界：

- `StagePlan`
  - 负责业务流程
  - 决定何时等待外部交互、何时请求 remote、何时结束阶段
- `ModeProfile`
  - 负责资源需求和默认 capture contract
  - 不负责业务轮次逻辑
- `Predictor` / model profile
  - 负责输入输出与类别 vocabulary 的模型语义
- `Scheduler`
  - 负责数据交换，不负责业务含义
- `RemoteManager`
  - 负责请求执行，不负责推断业务真值来源

## 当前文档基线

当前目录中的文档应这样分工：

- `ReadMe.md`：当前总览与操作基线
- `INTERFACES.md`：外部协议基线
- `ARCHITECTURE.md`：内部结构与已知缺口基线
- `EVENT_DESCRIPTION.md`：内部事件与日志语义基线
- `docs/handover.md`：已完成工作与待处置事项移交

如果代码继续调整，应优先同步这些文件。
## 2026-04 Audit Follow-Up Baseline

- Camera and predictor runtime backend ownership now belongs to package-level backend selectors. `VISTA_BACKEND=mock|real|auto` is the control-plane truth. `capability_placeholder` is no longer allowed to choose the main runtime path.
- `PredictorManager` now validates detect output at the manager boundary before publishing `local_perception`. Stable detect payload fields now include `contract_ok`, `contract_error`, `contract_warnings`, `class_names`, `class_names_source`, `infer_box_format`, `has_infer`, `implementation`, `model_name`, `predictor_type`, `box_count`, `infer_boxes`, `infer_masks`, `rgb_shape`, `obs_ts`, `frame_seq`, `age_ms`, `table_bbox`, `table_quadrant`, `rgb_search_roi`, `table_roi_source`.
- Detect class-name fallback no longer returns to the legacy `grasping_coco20` table. Structural fallback is now normalized `coco80`, and weakened payloads are marked with `class_names_source=fallback_coco80`.
- Frame-consuming managers are now generation-aware. `PredictorManager` and `PreviewManager` gate on `(generation, seq)` rather than raw `seq`, so `Scheduler.configure()` slot reset on mode switch does not stall local inference or freeze preview.
- The default camera color baseline is now BGR. Detect follows BGR end-to-end to match the tmp benchmark path; the optional segment predictor converts BGR to RGB internally where needed.
- Mode-profile camera ownership is now explicit rather than just structural. `TRACK_LOCAL` and `GRASP_REMOTE` each publish their own RGB capture contract through `ModeProfile.camera_overrides`.
- Legacy alias cleanup is in progress: `vision_stream.py` is removed and `QNNPredictor` is no longer part of the supported predictor export surface.
- `VisionEngine` has been deleted (2026-05). Responsibilities distributed to ModeController (owns Scheduler + Supervisor), StageController (holds Scheduler directly), and VistaApp (direct component assembly).
- `remote_cmd` / `remote_ack` event routes removed. effects mechanism replaced by mode-driven task execution (RemoteManager `kind=task`).
- GRASP stage mode chain: `SILENT(default) → GRASP_REMOTE_INIT → GRASP_REMOTE ↔ MICRO_ADJUST`. All StagePlan default_mode unified to `SILENT`.
- `remote_init_status` route registered conditionally for INIT / GRASP_REMOTE_INIT modes. RemoteManager task writes init/predict payloads to separate routes.
- `ModeProfile` now has explicit `table_edge_enabled`, `table_edge_path`, `table_edge_update_hz` fields. No longer derived from camera config.
