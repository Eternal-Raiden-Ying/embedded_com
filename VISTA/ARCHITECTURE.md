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
                             /    |      |      \
                            v     v      v       v
                   CameraManager Predictor Remote Preview
                            \      |      /        |
                             +-----+-----+---------+
                                           |
                                           v
                                      Scheduler
                                           |
                                           v
                      StageController.collect_tick_input() / tick()
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
- `MICRO_ADJUST`
- `GRASP_REMOTE`
- `IDLE_HOT`

说明：

- `DEPTH_PERCEPTION` 等概念仍可作为未来扩展方向
- 但它们不是当前已注册、已收口的默认 runtime 基线

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
- `RETURN`：当前消费 `local_perception`，但真实适配路径仍弱于 `SEARCH`

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

职责：

- `CameraManager`：相机实例生命周期和采集 worker，发布 `camera_frames`、`frame_meta`
- `PredictorManager`：模型生命周期和推理 worker，读取 `camera_frames`，发布 `local_perception`
- `RemoteManager`：远程抓取协作 worker，消费 `remote_cmd`，发布 `remote_result`、`remote_ack`
- `PreviewManager`：预览 worker，读取 `camera_frames`、`runtime_status`、`local_perception`，渲染调试预览

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
  -> ModeController.prepare_switch(...)
  -> VisionEngine.apply_mode_plan(...)
  -> RuntimeSupervisor.reconcile(...)
  -> ModeController.commit_switch(...)
  -> VistaApp 发送 vision_obs
```

### 控制循环链

```text
VistaApp._tick_stage()
  -> StageController.collect_tick_input(...)
  -> Scheduler.collect_tick_input(...)
  -> StageController.tick(...)
  -> StagePlan.tick(...)
  -> VistaApp 发送 vision_obs
```

### 数据面链路

```text
CameraManager -> Scheduler(camera_frames, frame_meta)
PredictorManager -> Scheduler(local_perception)
RemoteManager -> Scheduler(remote_result, remote_ack)
StageController effects -> Scheduler(remote_cmd)
PreviewManager <- Scheduler(camera_frames, runtime_status, local_perception)
```

## 当前架构的关键边界

- `App` 不处理 raw frame
- `StageTickInput` 只带摘要结果，不带原始图像
- `StageOutput.signals` 只做短生命周期控制面反馈
- 跨 tick 的业务状态放在 `StageContext.stage_state`
- mode 以 `generation` 作为隔离边界，旧代结果不会进入当前 stage 采样
- preview 是 backend/data-plane 的旁路能力，不再经过 `App` 处理图像

## 当前已知架构缺口

这部分是当前文档必须明确写出的现实问题，而不是隐藏在代码里的实现债务。

### 1. Detect 输出 contract 仍未完全收口

现状：

- real detect predictor 会返回 NumPy 结构
- `PredictorManager` 与 stage 侧消费对真实输出的边界仍需收口

影响：

- `local_perception` 的稳定性依赖于 manager 边界是否把真实输出安全归一化

### 2. Detect 类别语义所有权仍不清晰

现状：

- 默认本地主线是 `coco80 detect`
- 但 stage 侧的 target 解析历史上依赖过 `grasping_coco20`

影响：

- 当前默认 detect 成功路径可能被错误类别映射削弱

架构方向：

- 类别 vocabulary 应跟随 active model/profile，而不是由单一全局 target 表硬编码主导

### 3. Remote `INIT -> PREDICT` gate 不够严格

现状：

- `GRASP` stage 会触发 remote command
- 但当前 integrated path 对“服务器 init 已完成”这一条件缺少清晰强制 gate

影响：

- 框架化之后，可能削弱 `simulate_client_request.py` 中的最小成功路径

### 4. `GRASP_REMOTE` 缺 fresh-frame barrier

现状：

- mode 切换会带来 generation 变化和 scheduler state 清空
- remote predict 当前仍可能在新 mode 的新 frame 就绪前被触发

影响：

- `missing_camera_frames`
- `missing_depth_frame`
- race-dependent false failures

架构方向：

- `mode applied` 不等于 `data ready`
- 该 gate 应由 stage 驱动，而不是靠 worker timing 假设

### 5. Camera 参数所有权仍偏向全局配置

现状：

- 关键相机参数仍主要在 `board_config.py`
- `GRASP_REMOTE` 尚未完整拥有自己的 capture profile contract

影响：

- remote 路径可能沿用 local tracking 的默认捕获设置

架构方向：

- remote 相关分辨率、输出格式、上传前编码策略应进入 mode/profile 所有权

### 6. `release_cooldown_s` 仍是声明多于行为

现状：

- mode profile 中存在 `release_cooldown_s`
- runtime 当前仍以立即 stop/release 为主

影响：

- 文档和配置承诺了 cooldown 语义，但运行时并未真正兑现

架构方向：

- 要么真正实现 cooldown
- 要么删去装饰性字段，不再制造虚假抽象

### 7. Remote `class_id` 真值来源仍需收口

现状：

- 架构方向已经明确：`class_id` 来自外部输入
- 当前实现仍保留从 `target` 推导 `class_id` 的回退逻辑

影响：

- remote contract 仍有双真值风险

架构方向：

- `class_id` 可以存于 stage state
- 但来源应只来自外部请求，不应继续由 VISTA 内部猜测生成

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
- `AUDIT_TODO.md`：编码交接 backlog

如果代码继续调整，应优先同步这些文件，而不是继续依赖旧的迁移计划文档。
