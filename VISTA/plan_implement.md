# VISTA 架构优化 — 实现需求

日期：2026-05-13
来源：arch_optimize.md 讨论后的设计决策

---

## 一、已完成的框架工作（不在本次范围）

| 项 | 内容 |
|----|------|
| App 状态去重 | VistaApp 不再维护 StageContext 副本，统一走 `_ctx()` |
| Mode 切换收拢 | 死代码删除、TABLE_EDGE→DEPTH 回退移入 SearchStagePlan、Grasp tick 改为 allowlist |
| Tick 自主订阅 | BaseStagePlan 加 `common_routes`/`optional_routes`，Scheduler 按 filter 采集 |
| Edge 标准化 | `Online_Edge_Detect/` → `edge_detect/`，standalone 文件移至 `old/`，动态 import 改为直接 import |

---

## 二、Mode 能力静态配置扩展

### 2.1 ModeProfile 的 table_edge 字段扩展

当前 `_compile_plan()` 的 `table_edge` 只有 `enabled: bool`。扩展为：

```
capabilities.table_edge:
  enabled: bool         # 是否启用 TableEdgeManager
  path: str             # "lightweight" | "full" | "disabled"
  update_hz: float      # worker loop 频率
```

**涉及的 ModeProfile 配置变更**（`config/mode_defaults.py`）：

| Mode | table_edge.path | table_edge.update_hz |
|------|----------------|---------------------|
| TRACK_LOCAL | `"lightweight"` | 5.0 |
| DEPTH_PERCEPTION | `"full"` | 10.0 |
| TABLE_EDGE_PERCEPTION | `"full"` | 10.0 |
| MICRO_ADJUST | — | disabled |
| GRASP_REMOTE | — | disabled |
| IDLE / IDLE_HOT | — | disabled |

### 2.2 传递路径

```
ModeProfile.table_edge.{path, update_hz}
  → _compile_plan() 写入 plan["capabilities"]["table_edge"]
  → VisionEngine.apply_mode_plan()
  → RuntimeSupervisor._configure_table_edge(plan)
  → TableEdgeManager 接收配置（替代 _active_mode() 读取）
```

### 2.3 TableEdgeManager 去 mode 感知

- 删除 `_active_mode()` 方法
- 删除 `_current_interval_s()` 中的 mode 判断
- `_process_depth()` 的路由条件从 `self._active_mode() == "TRACK_LOCAL"` 改为检查传入的 `path` 配置
- 环境变量 `VISTA_TRACK_LOCAL_LIGHT_EDGE` / `VISTA_TRACK_LOCAL_EDGE_STRIDE` / `VISTA_TRACK_LOCAL_EDGE_UPDATE_HZ` 迁移为 ModeProfile 静态配置字段

---

## 三、以 Mode 能力替代 effects 通道

### 3.1 核心思路

当前 `GraspStagePlan` 通过 effects (`remote_cmd` event) 向 `RemoteManager` 发 INIT/PREDICT/RELEASE 指令。改为：**mode 的能力配置定义动作，切 mode 即触发执行**。StagePlan 不再发指令，只读结果 + 切 mode。

### 3.2 两层 INIT 设计

| Mode | 触发时机 | 职责 |
|------|---------|------|
| `INIT` | `VistaApp.start()` 阶段 | 全局初始化：camera、predictor、remote 健康检查（best-effort `/init`） |
| `GRASP_REMOTE_INIT` | GRASP stage 收到 `RESPOND ACCEPT`，`server_status != ready` | 仅远程 `/init`，至多 3 次重试 |

`VistaApp` 或 `StageController` 维护 `server_status` 字段，来源：

```
remote capability (GRASP_REMOTE_INIT mode)
  → RemoteManager 执行 INIT，结果写入 Scheduler(remote_result)
  → Scheduler → StageTickInput → StageController → server_status
```

`StageController.handle_request()` 处理 `stage=GRASP` 请求时，`server_status` 随 payload 传入，StagePlan 据此判断进入 `GRASP_REMOTE_INIT` 还是 `GRASP_REMOTE`。

### 3.3 GRASP stage mode 转换

```
收到 RESPOND ACCEPT:
  ├─ server_status != ready → GRASP_REMOTE_INIT (至多 3 次 INIT，失败 → FAILED)
  └─ server_status == ready → GRASP_REMOTE

GRASP_REMOTE:
  ├─ 动作：开 camera、等 fresh frame、发 PREDICT
  ├─ success → RESULT_READY → Orchestrator 进入下游
  ├─ failure → FAILED
  └─ reposition_required → RUNNING + proposal → Orchestrator 再次 RESPOND

GRASP_REMOTE_INIT:
  ├─ 动作：仅发 /init（至多 3 次）
  ├─ ready → 自动切 GRASP_REMOTE
  └─ 3 次失败 → FAILED
```

### 3.4 RemoteManager 能力动作配置

`capabilities.remote` 扩展 `action` 字段：

```
capabilities.remote:
  enabled: bool
  action: str          # "init" | "predict" | "release" | "" (idle)
  max_init_retries: int  # GRASP_REMOTE_INIT 模式专用
  base_url: str
  ...
```

`RemoteManager` worker loop 读 plan 中的 `action`，执行对应操作，结果写入 `remote_result` slot。

### 3.5 effects 通道清理

- `grasp.py` 中 `_remote_effect()` 函数删除
- `BaseStagePlan` 暂不新增 `emit_event()` 方法（effects 概念消失）
- `stage_controller.py` 中 `_publish_effects()` 保留（等待确认无其他使用者后删除）
- `Scheduler` 中 `remote_cmd` route 保留（等待确认无其他使用者后删除）

---

## 四、后续工作（不在本次实现范围）

| 项 | 说明 |
|----|------|
| edge_detect 配置迁移 | `backend/edge_detect/board_config.py` → 主 `config/schema.py` 的 `EdgeDetectConfig` dataclass |
| Backend 目录分层 | `backend/` 按框架/管理器/驱动/能力包分层 |
| 日志覆盖补全 | 数据面 route 操作记录、manager 状态快照 |
| Mode 切换优化 | 避免频繁切 mode 调用同一 worker 的不必要启停开销 |
| effects 残留清理 | 确认 `remote_cmd` route 和 `_publish_effects()` 无其他使用者后删除 |

---

## 五、设计决策细节（2026-05-13 grilling 确认）

### 5.1 GRASP_REMOTE_INIT 是新 ModeProfile

独立 ModeProfile，与 GRASP_REMOTE 并列。ModeProfile 数量从 7 增到 8。GraspStagePlan 内编辑切换逻辑。

### 5.2 server_status 放 StageContext

- **写入**：`StageController.tick()` 从 `tick_input.results["remote_result"]` 提取 → 写 `ctx.server_status`
- **读取**：StagePlan（GraspStagePlan._resolve_mode）读 `ctx.server_status` 决定切哪个 mode
- **handle_request 路径**：`on_respond()` 中直接从 Scheduler 读 `remote_result` 获取 `server_status`（绕过 tick 缓存，保证时效性）。仅限 GRASP 请求，加注释说明时效性原因。

### 5.3 有限 task vs 无限 loop

ModeProfile 在 dataclass 中声明 `kind`：

```python
@dataclass
class RemoteProfile:
    enabled: bool = False
    kind: str = "loop"       # "loop" | "task"
    action: str = ""         # task only: "init" | "predict" | "release"
    max_retries: int = 1     # task only
```

- **loop**：camera/predictor/preview — 无限 worker，mode 存活期间持续运行
- **task**：INIT/PREDICT/RELEASE — mode 激活时触发一次，执行完自动退出线程，不空转

`kind`/`action` 由 RemoteManager 的 worker loop 自己读 plan 决定行为。`RuntimeSupervisor._configure_remote()` 不特殊处理，只负责启停。

### 5.4 Scheduler 不新增函数，复用现有接口

- Task 状态 = result_slot（`remote_result`），只不过从持续更新的帧摘要变成"完成时写一次"
- 不需要显式区分持续 vs 一次性语义
- task_state 在 `tick_input` 中，不持久化到 `ctx`
- handle_request 不访问底层 task 状态

### 5.5 孤儿线程 kill 方案

切 mode → `stop_runtime()` → `_worker_stop.set()` + `join(timeout=1.0)`。旧 worker 在 join 超时后成为孤儿线程：

- Scheduler 已切换 generation，旧 gen 数据被过滤
- `daemon=True`，主线程退出时强制回收
- HTTP client 已有 10s timeout，至多 10s 后线程自己结束
- 不阻塞主循环

### 5.6 on_respond 重复请求处理

- 当前 mode 已是 GRASP_REMOTE_INIT 且 task 未完成 → 返回 BUSY（status=FAILED, reason=busy），拒绝此次请求
- 要求进入其他 mode → kill 孤儿线程（通过标准 stop_runtime + join(1s)）

### 5.7 table_edge.path 替代环境变量

- `"lightweight"` → `_process_depth_lightweight()`（纯 numpy 拟合）
- `"full"` → `_process_depth()`（完整 RANSAC detector）
- 环境变量 `VISTA_TRACK_LOCAL_LIGHT_EDGE` / `VISTA_TRACK_LOCAL_EDGE_STRIDE` / `VISTA_TRACK_LOCAL_EDGE_UPDATE_HZ` 值硬编码到 ModeProfile
- App 层检测这些环境变量是否被赋值，如赋值则打印弃用警告

### 5.8 实施顺序

| Step | 内容 | 依赖 |
|------|------|------|
| A | ModeProfile 扩展（table_edge.path/update_hz + remote.kind/action/max_retries），`_compile_plan()` 透传，环境变量弃用警告 | 无 |
| B | RemoteManager task worker + Scheduler task 语义 + 孤儿线程 kill + handle_request 切 mode 清理 | A |
| C | Mode 新增 GRASP_REMOTE_INIT，GraspStagePlan mode 逻辑：server_status 读写、on_respond BUSY 返回、mode 链 GRASP_REMOTE_INIT→GRASP_REMOTE | B |
| D | effects 通道清理：删 `_remote_effect()`、`_publish_effects()`（确认无其他使用者后）、`remote_cmd` route | C |

每步独立验证，编译通过后提交。