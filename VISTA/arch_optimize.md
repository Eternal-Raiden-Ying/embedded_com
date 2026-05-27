# VISTA 架构优化分析

日期：2026-05-05
来源：Code-VS-Spec audit 后续 — 基于当前代码实际结构的架构问题梳理

本文档列出当前架构中的设计张力，作为后续架构设计讨论的基础。不包含具体修改方案。

---

## 一、App 层职责过重与状态双写

### 1.1 现状

`VistaApp`（`app/app.py`）当前承担至少 **九种输出机制** 和 **七种独立职责域**：

| 职责域 | 具体内容 | 位置 |
|--------|---------|------|
| 主循环/生命周期 | `run()` 固定频率循环、`start()`/`stop()`、异常兜底 | app.py:897-936 |
| IPC 收发 | `_handle_request_payload()` 请求路由、`_send_obs()` 发送入队、IPC 事件日志 | app.py:292-365, 741-813 |
| 状态同步 | 自有 stage/mode/session/req/epoch/interaction 副本 + `_sync_runtime_from_stage_context()` | app.py:85-95, 637-675 |
| 日志/遥测 | `_record_event()`、`_record_ipc()`、`_record_stage_event()`、`_record_backend_event()` | app.py:158-235 |
| 心跳 | `_emit_heartbeat_if_needed()` — 汇聚 4+ 子系统数据 | app.py:577-622 |
| 速率统计 | `_emit_rate_summary_if_needed()` — 9 个 deque、百分位计算、per-mode 门控 | app.py:466-514 |
| 请求追踪 | `_record_request_trace()` — 逐请求 trace 写入 `vision_request_trace.jsonl` | app.py:516-549 |
| 边缘分析 | `_record_rate_sample()` — 逐帧 edge_profile 写入 `edge_profile.jsonl` | app.py:414-464 |
| IDLE 过渡 | `_enter_cold_idle()` / `_enter_hot_standby()` / `_expire_hot_standby()` | app.py:551-575, 829-841 |
| 操作员控制台 | 请求行、stage/mode 变更行、IPC 事件过滤、心跳行、速率摘要行 | 散布于各处 |

### 1.2 核心问题：状态双写

VistaApp 维护了 **StageContext 中每个字段的自有副本**：

```
VistaApp.current_stage        ⇄  StageContext.current_stage
VistaApp.current_mode         ⇄  StageContext.current_mode
VistaApp.target_name          ⇄  StageContext.target_name
VistaApp.current_session_id   ⇄  StageContext.session_id
VistaApp.current_req_id       ⇄  StageContext.req_id
VistaApp.current_epoch        ⇄  StageContext.epoch
VistaApp.active_interaction_id ⇄  StageContext.interaction_id
```

每次 `handle_request()` 和 `tick()` 后必须调用 `_sync_runtime_from_stage_context()` 同步。`_enter_hot_standby()` 和 `_enter_cold_idle()` 返回 7 元组，需在 **三个调用点** 以完全相同的顺序解包。任何遗漏导致静默状态发散。

### 1.3 根因

App 层同时做了三件事：**编排**（服务生命周期、主循环节奏）、**数据采集**（tick input 收集、速率统计）、**运维输出**（日志、心跳、控制台）。这三件事混在一个类里，导致 App 需要通过自有状态副本来弥合不同职责间的数据需求。

---

## 二、Stage/Mode 双控制路径

### 2.1 现状

VISTA 有两条并行的控制路径，桥接点分散：

```
Stage 路径（业务流程）          Mode 路径（资源配置）
─────────────────────          ─────────────────────
StageController                ModeController
  ├─ handle_request()            ├─ switch_mode()
  ├─ tick()                      ├─ _compile_plan()
  ├─ _transition_to()            ├─ prepare_switch()
  └─ StagePlan                    ├─ commit_switch()
       ├─ on_enter()              └─ VisionEngine
       ├─ on_update()                  ├─ apply_mode_plan()
       ├─ on_respond()                 ├─ Scheduler.configure()
       └─ tick()                       └─ RuntimeSupervisor.reconcile()
```

两者通过 `StageController._apply_context_mode()` 桥接，但 **映射逻辑分散在三处**：

| 位置 | 作用 | 文件:行 |
|------|------|---------|
| StagePlan.default_mode | 各 stage 的类属性默认 mode | search.py:340, grasp.py:190 |
| StagePlan._resolve_mode() | 根据请求动态选定 mode | search.py:346-352 |
| StageController._apply_context_mode() | TABLE_EDGE_PERCEPTION → DEPTH_PERCEPTION 硬编码回退 | stage_controller.py:186-194 |

### 2.2 具体不一致

- **`build_default_stage_entry_modes()` 是死代码**：`mode_defaults.py:17` 定义了 stage→mode 映射表但从未被调用。实际生效的是各 StagePlan 的 `default_mode` 类属性。
- **RETURN stage 配置存在但无实现**：映射表中有 `RETURN: TRACK_LOCAL`，但没有 ReturnStagePlan 类注册。
- **TABLE_EDGE 回退逻辑耦合 stage 知识**：`_apply_context_mode()` 硬编码了 `ctx.current_stage == "SEARCH"` 和 `search_kind == "TABLE_EDGE"` 的判断。通用 mode 切换桥接器不应知道具体 stage 的语义。
- **GraspStagePlan.tick() 无条件覆写 mode**：`grasp.py:548` 在 mode 非 `GRASP_REMOTE` 时强制设为 `MICRO_ADJUST`。如果 GRASP 阶段未来需要使用其他 mode（如 TRACK_LOCAL 做重捕获），tick 会静默覆盖。

---

## 三、tick input 收集绕过 StageController

### 3.1 现状

ARCHITECTURE.md 描述 StageController "从 Scheduler 采样 StageTickInput"，但实际调用链：

```
VistaApp._tick_stage()
  → self.runtime.collect_tick_input(ts=now)    ← VisionEngine
  → Scheduler.collect_tick_input(ts=ts)
  → StageController.tick(tick_input)           ← 被动接收
```

`StageController` 没有 `collect_tick_input()` 方法。数据采集由 VisionEngine（runtime 层）暴露接口，由 VistaApp（app 层）直接调用。StageController 仅作为 `tick()` 的被动接收方。

### 3.2 问题

- **层级跳跃**：tick input 收集本应是 stage 层关心的事（需要哪些数据、以什么频率采样），但实际接口在 runtime 层，调用在 app 层，stage 层被架空。
- **扩展性**：如果未来不同 stage 需要不同的 tick 数据聚合策略（如 SEARCH 需要高频、GRASP 需要等待特定事件），当前路径无从差异化。
- **循环依赖**：VistaApp 在 `_tick_stage()` 中向 `tick_input.snapshot` 注入 app 层快照（`app.py:823`），形成 app → tick → stage → output → app 的环。

---

## 四、effects 机制未成形

### 4.1 现状

effects 是 StagePlan 向 Manager 发送控制指令的唯一通道（如 GraspStagePlan 触发 RemoteManager 的 INIT/PREDICT）。但当前实现是半成品：

- **BaseStagePlan 无 helper**：没有 `emit_event(route, payload)` 方法。Stage 通过手写 dict 构造 effect。
- **唯一使用者**：`grasp.py` 的模块级函数 `_remote_effect()`，硬编码 `"type": "PUBLISH_EVENT"` 和 `"route": "remote_cmd"`。
- **魔法字符串**：`"PUBLISH_EVENT"` 是裸字符串常量，生产方（grasp.py）和消费方（stage_controller.py:297）各写各的，没有共享枚举或常量。
- **路由策略三重校验**：`_compile_plan()` 声明、`_verify_plan()` 检查、`Scheduler.publish_event()` 运行时再检查。意图是防御性的但造成重复。

### 4.2 与其他控制机制的关系

effects 和 signals 是并行的控制反馈通道：

| 通道 | 方向 | 生命周期 | 当前用途 |
|------|------|---------|---------|
| signals | StagePlan → StageController | 短（下个 tick 消费） | response/status 反馈 |
| effects | StagePlan → Scheduler → Manager | 异步（worker loop 消费） | remote_cmd |
| runtime_status | StageController → Scheduler | slot（持续可读） | stage/mode 状态广播 |

三者各自独立运作，但 effects 的 API 成熟度远低于另外两者。

---

## 五、TableEdgeManager + Online_Edge_Detect 的双重身份

### 5.1 现状

`Online_Edge_Detect` 是一个 **可独立运行的子应用**，拥有：

- 独立的 `app.py` — 主循环（10Hz）、RunLogger、RealSense 直连
- 独立的 `board_config.py` — `OnlineEdgeConfig` 配置体系（`EDGE_*` 环境变量前缀）
- 独立的 `protocol.py` — `TableEdgeObsMsg` dataclass
- 独立的 `stream_source.py` — `RealSenseStreamSource` 封装
- 独立的 `schema.py` — `DetectorConfig`/`RealSenseConfig` 等配置 dataclass

它同时被 `TableEdgeManager` 通过 **动态 `__import__`** 加载：

```
TableEdgeManager._load_detector()
  → __import__("vision_module.backend.Online_Edge_Detect.board_config")
  → __import__("vision_module.backend.Online_Edge_Detect.detector")
  → OnlineTableEdgeDetector(calib, CONFIG.detector, target_dist)
```

### 5.2 两层人格的冲突

| 维度 | 独立模式（app.py main） | 托管模式（TableEdgeManager） |
|------|------------------------|---------------------------|
| 数据源 | RealSense SDK 直连 | Scheduler `camera_frames` slot |
| 输出 | `JsonlClientSender` TCP/UDS | Scheduler `table_edge_obs` slot |
| 日志 | 自有 `RunLogger("online_edge", ...)` | 无（依赖 VISTA 主 RunLogger） |
| 配置 | 完整 `OnlineEdgeConfig`（含相机参数） | 仅提取 `detector.*` 和 `calib_json` |
| 主循环 | 自有 `run()` @ 10Hz | TableEdgeManager `_worker_loop` @ 可变频率 |

在托管模式下，`Online_Edge_Detect` 的 `app.py`、`protocol.py`、`stream_source.py`、`board_config.py` 的 RuntimeConfig/OutputConfig/RealSenseConfig 部分 **完全未被使用**。它们是死代码路径，但仍保留在 `backend/` 中。

### 5.3 与其他 Manager 的对比

| 方面 | PredictorManager | TableEdgeManager |
|------|-----------------|------------------|
| 导入方式 | 直接 `from .predictor import ...` | 动态 `__import__` 带回退 |
| 子应用 | 无 | Online_Edge_Detect（完整独立 app） |
| 后端连接 | 构造函数实例化 predictor | `__import__` 动态加载 detector |
| 轻量回退 | 无 | `_process_depth_lightweight()`（绕过 RANSAC） |

其他 manager 向下委托给 driver，TableEdgeManager 则 **向上包装了一个独立应用**。这在 manager 层中是一个抽象层次异常。

---

## 六、Edge Detect 数据路由与框架不一致

### 6.1 数据流对比

**正常 Manager 模式（PredictorManager）：**

```
CameraManager → Scheduler(camera_frames)
PredictorManager ← Scheduler(camera_frames)    # 消费
PredictorManager → Scheduler(local_perception)  # 生产
StagePlan ← Scheduler(local_perception)         # stage 消费
```

**Edge 路径（TableEdgeManager）：**

```
CameraManager → Scheduler(camera_frames)
TableEdgeManager ← Scheduler(camera_frames)      # 消费 depth
TableEdgeManager ← Scheduler(local_perception)   # 消费（用于 ROI 选择）
TableEdgeManager ← Scheduler(runtime_status)     # 消费（用于 locked_roi）
TableEdgeManager → Scheduler(table_edge_obs)     # 生产
SearchStagePlan ← Scheduler(table_edge_obs)      # stage 消费
PreviewManager ← Scheduler(table_edge_obs)       # preview 消费
```

TableEdgeManager 是唯一 **同时消费和生产的 manager**（PredictorManager 只消费 camera_frames；PreviewManager 只消费）。更重要的是，它的输入包括 `local_perception` 和 `runtime_status`——分别是 PredictorManager 和 StageController 的输出。这形成了一条跨 manager 的隐式依赖链：

```
PredictorManager → local_perception → TableEdgeManager → table_edge_obs → StagePlan
```

### 6.2 问题

- **Manager 间隐式依赖**：TableEdgeManager 的 ROI 选择逻辑依赖 PredictorManager 产出的 `table_bbox`。如果 PredictorManager 未启用或产出空结果，TableEdgeManager 的行为会退变，但此依赖未在任何 contract 中显式声明。
- **Mode 感知泄漏**：`TableEdgeManager._active_mode()` 直接读 `scheduler.snapshot().get("active_mode")` 来决定使用轻量级还是完整路径。Manager 不应需要知道自己运行在哪个 mode 下——mode 是 Stage 层的概念，应由 plan 配置控制行为。
- **独立日志路径**：Online_Edge_Detect 的 `app.py` 使用自定义 RunLogger 写入 `timeline.jsonl`，与 VISTA 主路径的 `event.jsonl` 完全分离、命名格式不同。调试时需要交叉对照两个日志源。

---

## 七、Backend 目录扁平化

### 7.1 现状

`backend/` 当前混装了四类不同性质的代码：

```
backend/
├── 框架/编排层
│   ├── vision_engine.py         # runtime facade
│   ├── runtime_supervisor.py    # capability reconcile
│   ├── mode_controller.py       # mode switch orchestration
│   ├── scheduler.py             # data bus
│   └── mode_profiles.py         # mode profile definitions
├── 管理器层
│   ├── camera_manager.py
│   ├── predictor_manager.py
│   ├── table_edge_manager.py
│   └── preview/manager.py + remote/manager.py
├── 驱动/能力包
│   ├── camera/                  # 5 个相机类 + C++ 桥接
│   ├── predictor/               # 2 个 QNN predictor + mock + utilities
│   ├── preview/                 # opencv_sink + null_sink
│   ├── remote/                  # client + protocol
│   └── Online_Edge_Detect/      # 完整子应用（~10 个文件）
└── 工具
    └── table_edge_roi.py
```

### 7.2 问题

- 没有明确的"框架 vs 能力"分界。新增一个 backend component 时，没有指引判断它应该放在平级文件还是子包。
- `mode_controller.py` 和 `mode_profiles.py` 是 mode 体系的定义者，与 `vision_engine.py`/`runtime_supervisor.py` 的 runtime 编排职责不同，但平铺在同一层。
- `Online_Edge_Detect/` 作为子包包含 `board_config.py` 和 `app.py`，按体积和独立性应该是一个独立的顶层模块，但现在嵌在 backend 内部。

---

## 八、日志/JSONL 记录策略

以下不涉及当前已写入的具体内容，而是从**架构层面的调试便利性**出发，讨论理想情况下应在哪些节点进行结构化记录。

### 8.1 现有记录的覆盖盲区

基于当前调用链分析，以下关键节点缺少结构化日志：

| 节点 | 当前状态 | 调试价值 |
|------|---------|---------|
| Mode 切换前后 manager 状态快照 | 无记录 | 排查"切 mode 后为什么某 manager 没启动" |
| Scheduler route 读写操作 | 无记录（仅 transport 层有 IPC 日志） | 排查数据流中断——哪个 manager 写了、哪个 stage 读了 |
| effects 投递与消费 | 无记录 | 排查 remote_cmd 是否被 RemoteManager 成功消费 |
| StagePlan.tick() 决策路径 | 仅 vision_obs 透出 | 排查"为什么这次 tick 没发 PREDICT" |
| `_apply_context_mode()` 的 no-op 快速路径 | 无记录 | mode 切换被静默跳过时无法追踪原因 |

### 8.2 记录分层建议

从调试视角，VISTA 的数据流有三个自然观察点：

| 层 | 应记录的内容 | 对应现有机制 |
|----|-------------|-------------|
| IPC 边界 | 入站请求、出站观测、序列化异常 | `ipc.jsonl`（已有） |
| 控制面 | stage 变迁、mode 切换、请求分类决策、effects 投递 | `event.jsonl`（已有但不完整） |
| 数据面 | route 写入/读取、manager worker 启停、帧消费/跳过 | **缺失** |

数据面日志是当前最大盲区。建议新增一种轻量 route 操作记录（route_name, operation, generation, seq），为每个 Scheduler route 的操作提供可审计的读写轨迹。这不需要高频——仅记录关键变化点（首次写入、generation 切换后的首次读取、消费滞后告警）。

### 8.3 日志 namespace 统一

当前存在三套日志标识：

| 子系统 | logger name | 输出目标 |
|--------|------------|---------|
| VISTA 主路径 | `vision.*` | `event.jsonl`/`ipc.jsonl`/`heartbeat.jsonl` |
| Online_Edge_Detect | `online_edge.*`（独立 RunLogger） | `timeline.jsonl` |
| Python logging | `vision.inference` 等 | `logs/vision.log` |

如果未来需要统一日志收集，建议将所有结构化日志收敛到同一个 RunLogger 实例（或至少同一目录、同一命名约定），通过 `module` 字段区分来源。

---

## 九、低优先级问题

以下问题已识别但不在本次优化优先级内：

### 9.1 两个配置体系
- `vision_module/config/board_config.py` — VISTA 主配置
- `vision_module/backend/Online_Edge_Detect/board_config.py` — 桌边检测独立配置
- 二者无共享机制。未来统一配置管理时处理。

### 9.2 ipc/ 目录的归属
- `ipc/protocol.py` 定义 `VisionReq`/`VisionObs` dataclass 和传输工具
- 当前与 `app/` 平级，无明确依赖方向
- 如果 IPC 工具未来被其他模块复用，当前结构可接受；如果永远是 VISTA 独占，可考虑合并入 app/

### 9.3 `capability_placeholder` 残留在配置 schema/schema.py:33、board_config.py:52、app.py:130，但不再参与运行时选择。应在清理配置时一并处理。

### 9.4 `build_default_stage_entry_modes()` 死代码 — `mode_defaults.py:17` 定义了完整映射但从未被调用。

---

## 十、问题关联图

```
┌─────────────────────────────────────────────────────────────┐
│                        App 层过重                            │
│  (状态双写 + 9 种输出 + 速率统计 + IDLE 过渡混在一个类)        │
└───────────┬─────────────────────────┬───────────────────────┘
            │                         │
            v                         v
┌───────────────────┐     ┌──────────────────────────┐
│ Stage/Mode 双路径  │     │  Edge 集成打破层级划分      │
│ 映射分散三处       │     │  Online_Edge_Detect =     │
│ tick 绕过 Stage   │     │  独立子应用嵌在 backend     │
│ effects 半成品     │     │  数据路由跨 manager 隐式链  │
└───────────────────┘     └──────────────────────────┘
            │                         │
            └─────────┬───────────────┘
                      v
          ┌──────────────────────┐
          │ 控制面与数据面边界模糊  │
          │ Manager 需感知 mode    │
          │ backend 目录无分层     │
          │ 日志/JSONL 覆盖不完整   │
          └──────────────────────┘
```

核心矛盾是：**Edge 能力的加入暴露了原有框架在"非标准 manager"接入时的路径缺失**——原有的 Camera→Predictor→Stage 数据流是线性的，而 Edge 需要多源输入（depth + local_perception + runtime_status），这不在最初设计的 manager 模式范围内。同时 App 层承担了太多横切关注点，导致添加新能力时无法通过清晰的委托边界来消化复杂度。
