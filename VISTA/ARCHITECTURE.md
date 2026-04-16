# VISTA 当前架构

本文档描述当前代码中的实际结构，以 `vision_module/` 现状为准。

## 目标

VISTA 当前已经收敛为一条明确主链路：

- `VistaApp` 负责服务生命周期、IPC、主循环、日志和心跳
- `StageController` 负责业务阶段控制和 mode 触发
- `ModeController` 负责 mode profile 和 mode 切换状态
- `VisionEngine` 负责 backend runtime 装配与执行
- `Scheduler` 负责 managers 与 stage 之间的摘要数据交换
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
- 不直接做业务 stage 判定

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
- 进行 stage enter/restart/stop
- 调用 `ModeController` + `VisionEngine` 完成 mode 变更
- 从 `Scheduler` 采样 `StageTickInput`
- 产出 `StageOutput`，并通过 `vision_obs` 对外输出结果

当前 stage：

- `SEARCH`：本地搜索与跟踪，消费 `local_perception`
- `GRASP`：微调交互和远程抓取协作，消费 `remote_result`
- `RETURN`：回航标记/回航目标观测，消费 `local_perception`

核心 contract：

- `StageContext`：跨请求、跨 tick 的可变业务状态
- `StageTickInput`：当前控制周期对 `Scheduler` 的采样摘要
- `StageOutput`：当前 stage 的输出 envelope，包含 `vision_obs`、`signals`、`effects`

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

当前默认 mode：

- `IDLE`
- `TRACK_LOCAL`
- `MICRO_ADJUST`
- `GRASP_REMOTE`
- `IDLE_HOT`

当前实现中，`ModeController` 不直接管理各 manager 的启停；真正的 capability reconcile 在 `VisionEngine -> RuntimeSupervisor` 中完成。

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
  - 把 capability plan 转成 camera/predictor/remote/preview 的启停与配置动作
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
- `remote_ack`：event，backend scope

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
  -> StagePlan.on_enter/on_update/on_respond/on_stop(...)
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

## 当前架构的关键约束

- `App` 不处理 raw frame
- `StageTickInput` 只带摘要结果，不带原始图像
- `StageOutput.signals` 只做短生命周期控制面反馈
- 跨 tick 的业务状态放在 `StageContext.stage_state`
- mode 以 `generation` 作为隔离边界，旧代结果不会进入当前 stage 采样
- preview 是 backend/data-plane 的旁路能力，不再经过 `App` 处理图像

## 当前文档基线

本目录下的旧计划文档已经不再作为主基线。后续如果代码继续调整，以当前 `app/`、`backend/`、`config/` 实现和本文件为准。
