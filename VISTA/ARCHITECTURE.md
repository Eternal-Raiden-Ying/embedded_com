# VISTA 当前架构

本文档描述当前代码中的实际结构，以及当前仍未收口的架构缺口。

它不是理想化蓝图，而是当前实现的近端基线。

## 目标

VISTA 当前的主线设计已经收敛为：

- `VistaApp` 负责服务生命周期、IPC、主循环、日志和心跳
- `StageController` 负责业务阶段控制和 stage state
- `ModeController` 负责 mode profile、switch state 和 runtime plan 编译
- `VisionEngine` 负责 backend runtime 装配与对外 facade
- `RuntimeSupervisor` 负责 capability reconcile
- `Scheduler` 负责 manager 与 stage 之间的摘要数据交换
- 各 `Manager` 自己维护 worker loop，不把高频工作塞回主线程

## 总体拓扑

```text
Orchestrator --vision_req--> VistaApp
                              |
                              v
                        StageController
                         |           \
                         |            \
                         v             v
                    StagePlan      ModeController
                         |             |
                         |             | prepare_switch / commit_switch
                         |             |
                         +-------> VisionEngine.apply_mode_plan(...)
                                       |
                                       v
                               RuntimeSupervisor
                             /    |      |      \         \
                            v     v      v       v         v
                   CameraManager Predictor Remote Preview TableEdgeManager
                            \      |      /        |         /
                             +-----+-----+---------+--------+
                                           |
                                           v
                                      Scheduler
                                           |
                                           v
                      VisionEngine.collect_tick_input() / StageController.tick()
                                           |
                                           v
                                 VistaApp --vision_obs--> Orchestrator
```

## 活跃基线

### 当前 stage

- `SEARCH`
- `GRASP`
- `RETURN`
- `IDLE`

### 当前默认 mode

- `IDLE`
- `TRACK_LOCAL`
- `DEPTH_PERCEPTION`
- `TABLE_EDGE_PERCEPTION`
- `MICRO_ADJUST`
- `GRASP_REMOTE_INIT`
- `GRASP_REMOTE`
- `IDLE_HOT`

说明：

- `DEPTH_PERCEPTION` 和 `TABLE_EDGE_PERCEPTION` 是当前已注册、已收口的默认 runtime 基线 mode
- `DEPTH_PERCEPTION` 提供纯深度桌边感知；`TABLE_EDGE_PERCEPTION` 同时保持 RGB + depth + 本地模型的桌边与目标观测
- `TableEdgeManager` 是负责 `table_edge_obs` 发布的完整运行时组件，由 `RuntimeSupervisor` 按 mode plan 中的 `table_edge.enabled` 标志启停

## 分层职责

### 1. App 层

入口文件：`vision_module/app/app.py`

职责：

- 创建 `ModeController`、`VisionEngine`、`StageController`
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
- `vision_module/app/stages/search.py`
- `vision_module/app/stages/grasp.py`
- `vision_module/app/stages/return_home.py`

职责：

- 维护 `StageContext`
- 处理 `START` / `UPDATE` / `RESPOND` / `STOP`
- 进行 stage enter / restart / stop
- 调用 `ModeController` + `VisionEngine` 完成 mode 变更
- 从 `Scheduler` 采样 `StageTickInput`
- 产出 `StageOutput`，并通过 `vision_obs` 对外输出结果

当前 stage 消费关系：

- `SEARCH`：主要消费 `local_perception`
- `GRASP`：主要消费 `remote_result`，并在微调阶段消费 `local_perception`
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

- `ModeController` 负责资源需求描述
- 不直接拥有 manager worker
- capability reconcile 真正发生在 `VisionEngine -> RuntimeSupervisor`

### 4. Runtime 层

文件：

- `vision_module/backend/vision_engine.py`
- `vision_module/backend/runtime_supervisor.py`

职责划分：

- `VisionEngine`
  - runtime 根对象
  - 持有 `Scheduler`、各 manager、`RuntimeSupervisor`
  - 暴露 `start()`、`stop()`、`apply_mode_plan()`、`collect_tick_input()`、`publish_result()`、`publish_event()`
- `RuntimeSupervisor`
  - 根据 mode plan 配置 managers
  - 把 capability plan 转成 camera / predictor / remote / preview 的启停与配置动作
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
- `remote_result`：slot，stage scope
- `table_edge_obs`：slot，stage scope — 由 `TableEdgeManager` 发布，`SearchStagePlan` 和 `PreviewManager` 消费
- `runtime_status`：slot，backend scope
- `remote_cmd`：event，backend scope
- `remote_ack`：event，backend scope，当前更接近辅助诊断 route，不应被视为主业务 contract 真值

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
- `RemoteManager`：远程抓取协作 worker，消费 `remote_cmd`，发布 `remote_result`、`remote_ack`
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
  -> ModeController.switch_mode(...)           ← 封装了 prepare_switch → apply_mode_plan → commit_switch
  -> VisionEngine.apply_mode_plan(...)
  -> RuntimeSupervisor.reconcile(...)
  -> VistaApp 发送 vision_obs
```

注：`StageController` 通过 `ModeController.switch_mode()` 单一方法触发 mode 切换，而非分别调用 `prepare_switch` / `commit_switch`。`switch_mode()` 内部依次执行 prepare → apply_mode_plan 回调 → commit，语义等价但 API 表面是一个统一入口。

### 控制循环链

```text
VistaApp._tick_stage()
  -> VisionEngine.collect_tick_input(ts=now)   ← vision_engine.py:191
  -> Scheduler.collect_tick_input(ts=ts)       ← scheduler.py:209
  -> StageController.tick(tick_input)
  -> StagePlan.tick(...)
  -> VistaApp 发送 vision_obs
```

注：`StageController` 没有 `collect_tick_input()` 方法；tick input 收集从 `VistaApp` 直接到 `VisionEngine` 再到 `Scheduler`，绕过 `StageController`。

### 数据面链路

```text
CameraManager -> Scheduler(camera_frames, frame_meta)
PredictorManager -> Scheduler(local_perception)
TableEdgeManager -> Scheduler(table_edge_obs)
RemoteManager -> Scheduler(remote_result, remote_ack)
StageController effects -> Scheduler(remote_cmd)
PreviewManager <- Scheduler(camera_frames, runtime_status, local_perception, table_edge_obs)
TableEdgeManager <- Scheduler(camera_frames, local_perception, runtime_status)
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

### 3. Remote request sequencing 当前基线

现状：

- `RemoteManager` 在 service startup 时会对可用 `base_url` 做一次 best-effort `/init`
- `GRASP` stage 在 `RESPOND ACCEPT` 后进入 `GRASP_REMOTE`
- `StagePlan` 会等待 service-level init confirmed 和 fresh frame gate，同时最多触发 3 次 init retry
- `RemoteManager` 也会在 manager 层拒绝 `init_not_confirmed` 的 `PREDICT`

影响：

- integrated path 现在与最小脚本的 session-style 主干顺序一致：
  - service startup best-effort `/init`
  - grasp-time wait init success / retry if needed
  - `/predict`
  - `/release` on shutdown / disable / explicit reset

### 4. `GRASP_REMOTE` fresh-frame barrier 当前基线

现状：

- mode 切换仍会带来 generation 变化和 scheduler state 清空
- `GRASP_REMOTE` 当前通过 stage-visible `frame_meta` gate 等待新 generation 下的 fresh frame

影响：

- `mode applied` 与 `data ready` 已被拆开，不再依赖 timing 碰运气
- 远程抓取何时发起 `PREDICT` 现在由 stage 明确控制

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
- Mode-profile camera ownership is now explicit rather than just structural. `TRACK_LOCAL`, `MICRO_ADJUST`, and `GRASP_REMOTE` each publish their own RGB capture contract through `ModeProfile.camera_overrides`.
- Legacy alias cleanup is in progress: `vision_stream.py` is removed and `QNNPredictor` is no longer part of the supported predictor export surface.
