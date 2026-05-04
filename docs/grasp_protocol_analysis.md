# Grasp 协议全链路分析

分析时间：2026-05-02

---

## 链路总览

```
mobile_gateway ──task_cmd──▶ Orchestrator ──vision_req──▶ VISTA ──HTTP──▶ grasp server
       ◀──task_ack───              ◀──vision_obs───        ◀──HTTP────
                                      │
                                      └──UART──▶ STM32 ──▶ 机械臂
```

## 消息流经 4 跳

| Hop | 方向 | 传输 | 消息类型 |
|-----|------|------|---------|
| 1 | mobile_gateway → Orchestrator | TCP 9001 | `task_cmd` |
| 2 | Orchestrator → VISTA | TCP 9003 | `vision_req` |
| 3 | VISTA → grasp server | HTTP multipart | `/api/v1/predict` |
| 4 | VISTA → Orchestrator | TCP 9002 | `vision_obs` |
| — | Orchestrator → mobile_gateway | TCP 9012 | `task_ack` |

---

## Hop 1: mobile_gateway → Orchestrator

**文件**: `orchestrator/orchestrator_service/ipc/protocol.py:222-269`

```json
{
  "type": "task_cmd",
  "intent": "FIND",
  "confidence": 1.0,
  "cmd_id": "wx_1777293209382",
  "session_id": "wx_session_001",
  "epoch": 1,
  "source": "wechat_miniprogram",
  "ts": 1777293208.6,
  "target": "apple"
}
```

**消费方**: `OrchestratorService._drain_task_cmds()` → 状态机

**关键约束**:
- `intent` 必须是 `FIND` / `RETURN` / `STOP` 之一
- `FIND` 时 `target` 必须在冻结词表内（`frozen_targets`），否则 `ProtocolError`
- 当前冻结词表由 `target_aliases.json` 定义：apple/banana/bottle/cup

**grasp 相关**: task_cmd 中没有 `class_id` 字段。class_id 需要在后续 hop 中由 Orchestrator 查表填入。

---

## Hop 2: Orchestrator → VISTA

**文件**: 
- 生产者: `orchestrator/orchestrator_service/ipc/protocol.py:622-649` (`make_vision_req()`)
- 消费者: `VISTA/vision_module/ipc/protocol.py:105-141` (`VisionReq.from_dict()`)

```json
{
  "type": "vision_req",
  "ts": 1777293208.7,
  "op": "START",
  "stage": "GRASP",
  "target": "apple",
  "mode_hint": "GRASP_REMOTE",
  "session_id": "wx_session_001",
  "req_id": "req_abc123def0",
  "epoch": 1,
  "payload": {
    "remote_grasp": true,
    "need_depth": true,
    "class_id": 47,
    "remote_timeout_s": 10.0,
    "robot_id": "arm_001",
    "remote_metadata": {},
    "target_obs": {},
    "proposal": {},
    "result": {}
  }
}
```

**VISTA 侧状态路由** (来自 `stage_controller.py`):
- `stage=GRASP` → 激活 `GraspStagePlan`
- `mode_hint=GRASP_REMOTE` → 目标 mode（若 RESPOND ACCEPT）

**`payload` 字段消费** (来自 `grasp.py:76-101` `_grasp_state_from_req()`):

| 字段 | 类型 | 必需 | 说明 |
|------|------|------|------|
| `class_id` | int | **是** | YOLO 目标类别 ID，缺失时 GRASP 拒绝 |
| `remote_grasp` | bool | 否(默认 true) | false 时跳过远程调用，返回 mock 结果 |
| `need_depth` | bool | 否(默认 true) | 是否需要 depth 帧 |
| `remote_timeout_s` | float | 否(默认 10.0) | HTTP 超时 |
| `robot_id` | str | 否(默认 "arm_001") | 传递给 grasp server |
| `remote_metadata` | dict | 否 | 合并进 grasp server metadata |
| `target_obs` | dict | 否 | mock/override 目标观测 |
| `proposal` | dict | 否 | mock/override 微调建议 |
| `result` | dict | 否 | mock/override 抓取结果 |

**⚠️ 当前缺失**: Orchestrator 侧没有 `make_grasp_req()` 辅助函数。需要用 `make_vision_req(stage="GRASP", mode_hint="GRASP_REMOTE", payload={...})` 手动构造。

**⚠️ class_id 来源未解决**: `INTERFACES.md` 规定 class_id 必须来自显式外部输入，但 task_cmd 中无此字段。Orchestrator 需要做 `target → class_id` 查表（`target_aliases.json` 目前不包含 class_id 映射）。

---

## Hop 3: VISTA → grasp server

**VISTA 侧文件**:
- `backend/remote/client.py` — `RemoteGraspClient` (HTTP client)
- `backend/remote/protocol.py` — `RemotePredictRequest`, `RemotePredictResponse`
- `backend/remote/manager.py` — `RemoteManager` (生命周期 + 编码)
- `app/stages/grasp.py` — `GraspStagePlan` (状态机)

**grasp server 侧文件**:
- `grasp_module/app/server_app.py` — FastAPI 端点
- `grasp_module/backend/engine.py` — `RealSenseGraspPredictor`

### 3a. VISTA 内部状态机 (GraspStagePlan)

文件: `app/stages/grasp.py:190-518`

```
on_enter(START, stage=GRASP)
  │
  ├─ mode = MICRO_ADJUST
  │   └─ tick() → 发送 WAITING_RESPONSE (proposal + interaction_id)
  │       └─ Orchestrator 应 RESPOND(ACCEPT)
  │
  ├─ on_respond(ACCEPT)
  │   ├─ class_id 缺失 → FAILED (missing_class_id)
  │   ├─ remote_grasp=false → RESULT_READY (mock result)
  │   └─ remote_grasp=true → mode = GRASP_REMOTE
  │
  └─ mode = GRASP_REMOTE (tick 循环)
      ├─ INIT 未确认 → 发送 INIT (最多重试 3 次)
      ├─ INIT 确认 + 帧就绪 → 发送 PREDICT
      ├─ PREDICT 成功 → RESULT_READY
      └─ PREDICT 失败 → FAILED
```

### 3b. HTTP /api/v1/init

**请求**: POST，无 body
**响应**: `{"status": "success", "message": "Predictor loaded and warmed up successfully."}`

VISTA 侧只检查 HTTP 200 + `ok=true`，不检查 `status` 字段。

### 3c. HTTP /api/v1/predict

**请求**: multipart/form-data

| 字段 | 类型 | 说明 |
|------|------|------|
| `rgb_file` | file | RGB 图像，编码由 `rgb_encoding` 决定（jpeg/png） |
| `depth_file` | file | 深度图，编码由 `depth_encoding` 决定（默认 png） |
| `class_id` | int (form) | YOLO 目标类别 |
| `metadata` | str (JSON) | `{"robot_id":"arm_001","cmd":"predict","class_id":47}` |

**响应** (成功):

```json
{
  "status": "success",
  "grasp_count": 15,
  "feasible_count": 8,
  "output_count": 3,
  "targets": [
    {
      "x_cm": 12.5,
      "y_cm": -3.2,
      "z_cm": 18.0,
      "pitch_deg": 15.3,
      "roll_deg": -2.1,
      "gripper_width_cm": 8.5,
      "approach_depth_cm": 5.0,
      "confidence": 0.87,
      "feasible_angle_deg": 4.2,
      "position_frame": "robot",
      "angle_frame": "robot"
    }
  ]
}
```

**响应** (失败/需重定位):

```json
{
  "status": "reposition_required",
  "grasp_count": 0,
  "feasible_count": 0,
  "output_count": 0,
  "targets": [],
  "reason": "no_grasp_detected",
  "message": "placeholder"
}
```

`reason` 可能值: `no_grasp_detected` | `no_feasible_grasp` | `score_below_threshold`

**响应** (`/api/v1/release`): `{"status": "success", "message": "GPU memory freed."}`

### ⚠️ 协议断层

`GraspStagePlan.tick()` 消费 grasp server 响应的方式（`grasp.py:425-441`）:

```python
if matched and last_action == "PREDICT" and bool(remote_result.get("last_ok", False)) and bool(remote_result.get("has_result", False)):
    result = deepcopy(remote_result.get("result") or {})
    result.setdefault("target", ctx.target_name)
    result["source"] = "remote_grasp_client"
    result["request_id"] = request_id
    # → vision_obs.result = result (整个 grasp server 响应 dict)
```

这里 `remote_result["result"]` 是 grasp server 的完整 HTTP 响应 JSON（包含 `status`, `targets`, `grasp_count` 等），不是一个扁平的单个 grasp pose。这意味着 **Orchestrator 收到 `vision_obs.result` 后会拿到 `{status, targets: [...], ...}` 结构**——这需要 Orchestrator 侧知道如何解析 `targets` 数组。

**需要冻结的协议点**:
1. `targets[]` 中每个元素的字段是否就是 `_build_protocol_target()` 的当前输出？
2. `targets` 排序规则（当前按 confidence 降序，top-K=5，min_score 可配）
3. 返回第一个还是全部 targets？Orchestrator 需要知道最佳抓取位姿
4. `status` 枚举值: `success` | `reposition_required`
5. `reason` 枚举值需要和 VISTA/Orchestrator 的错误处理对齐

---

## Hop 4: VISTA → Orchestrator

**文件**: 
- 生产者: `VISTA/vision_module/app/stages/grasp.py` + `VISTA/vision_module/ipc/protocol.py:178-194`
- 消费者: Orchestrator `_drain_vision_obs()` → 状态机

### 4a. 中间状态消息 (mode=GRASP_REMOTE, status=RUNNING)

```json
{
  "type": "vision_obs",
  "ts": 1777293209.0,
  "stage": "GRASP",
  "mode": "GRASP_REMOTE",
  "status": "RUNNING",
  "session_id": "wx_session_001",
  "req_id": "req_abc123def0",
  "epoch": 1,
  "perception": {
    "target_obs": {"found": true, "target": "apple", "confidence": 0.88, ...}
  },
  "result": {
    "remote_state": "awaiting_predict_result",
    "request_id": "rr_1777293209000",
    "init_confirmed": true,
    "predict_sent": true,
    "frame_ready": true,
    ...
  }
}
```

Orchestrator 在此期间等待 `status` 变为 `RESULT_READY` 或 `FAILED`。

### 4b. 成功消息 (mode=GRASP_REMOTE, status=RESULT_READY)

```json
{
  "type": "vision_obs",
  "ts": 1777293210.0,
  "stage": "GRASP",
  "mode": "GRASP_REMOTE",
  "status": "RESULT_READY",
  "session_id": "wx_session_001",
  "req_id": "req_abc123def0",
  "epoch": 1,
  "perception": {
    "target_obs": {"found": true, "target": "apple", "confidence": 0.88, ...}
  },
  "result": {
    "status": "success",
    "grasp_count": 15,
    "feasible_count": 8,
    "output_count": 3,
    "targets": [
      {
        "x_cm": 12.5,
        "y_cm": -3.2,
        "z_cm": 18.0,
        "pitch_deg": 15.3,
        "roll_deg": -2.1,
        "gripper_width_cm": 8.5,
        "approach_depth_cm": 5.0,
        "confidence": 0.87,
        "feasible_angle_deg": 4.2,
        "position_frame": "robot",
        "angle_frame": "robot"
      }
    ],
    "target": "apple",
    "source": "remote_grasp_client",
    "request_id": "rr_1777293209000"
  }
}
```

**⚠️ 这里的 result 结构是 grasp server HTTP 响应 + VISTA 附加的 `target`/`source`/`request_id`。Orchestrator 必须理解 `result.targets[0]` 才是最优抓取位姿。**

### 4c. 失败消息

```json
{
  "type": "vision_obs",
  "stage": "GRASP",
  "mode": "GRASP_REMOTE",
  "status": "FAILED",
  "result": {
    "reason": "remote_init_failed",
    "request_id": "rr_...",
    "remote_error": "connection refused",
    "init_attempts": 3
  }
}
```

`reason` 可能值: `remote_init_failed` | `remote_predict_failed` | `missing_class_id` | `interaction_id_mismatch`

---

## Orchestrator → mobile_gateway (task_ack)

**文件**: `orchestrator/orchestrator_service/ipc/protocol.py:610-619`

```json
{
  "type": "task_ack",
  "ts": 1777293210.5,
  "cmd_id": "wx_1777293209382",
  "session_id": "wx_session_001",
  "epoch": 1,
  "accepted": true,
  "state": "GRASP",
  "reason": "",
  "source": "orchestrator"
}
```

然后 mobile_gateway 将其转为 MQTT `mobile/ack` (kind=task_ack) + `mobile/status`。

---

## 当前协议缺口汇总

| # | 位置 | 问题 | 影响 |
|---|------|------|------|
| 1 | Hop 2 | **class_id 来源未定**：task_cmd 无此字段，target_aliases.json 不包含映射 | Orchestrator 无法构造 GRASP vision_req |
| 2 | Hop 2 | **Orchestrator 缺少 `make_grasp_req()`** 辅助函数 | 需手动组装 payload |
| 3 | Hop 3-4 | **grasp server 输出被透传**：`vision_obs.result` 直接嵌入 HTTP 响应 JSON，未做字段筛选/重命名 | Orchestrator 必须理解 grasp server 的 `{status, targets: [{x_cm, y_cm, ...}]}` 结构 |
| 4 | Hop 3 | **`targets[]` 结构未冻结**：当前 `_build_protocol_target()` 输出 11 个字段，但未经正式 schema 定义 | 下游解析可能因字段变更断裂 |
| 5 | Hop 3 | **status 枚举未文档化**：仅 `success` / `reposition_required`，与 VISTA 的 status 体系(RUNNING/RESULT_READY/FAILED)无映射约定 |
| 6 | Hop 3 | **`GraspStagePlan` 不检查 `status=="success"`**：只检查 `has_result`，可能误将 `reposition_required` 当作成功 | 错误处理逻辑缺陷 |
| 7 | Hop 3-4 | **多 target 返回时的选择策略未定**：server 返回 top-K，VISTA 透传全部，Orchestrator 需要知道选哪个 |

---

## 建议的冻结协议步骤

### Step 1: 冻结 grasp server → VISTA 输出 (最小可行)

```json
{
  "status": "success" | "reposition_required",
  "grasp_count": 15,
  "output_count": 3,
  "targets": [
    {
      "x_cm": 12.5,
      "y_cm": -3.2,
      "z_cm": 18.0,
      "pitch_deg": 15.3,
      "roll_deg": -2.1,
      "gripper_width_cm": 8.5,
      "approach_depth_cm": 5.0,
      "confidence": 0.87,
      "feasible_angle_deg": 4.2,
      "position_frame": "robot",
      "angle_frame": "robot"
    }
  ],
  "reason": "no_grasp_detected"
}
```

需要确认：
- `x_cm/y_cm/z_cm` 坐标系原点（当前: robot 系，rear_edge_center）
- `pitch_deg/roll_deg` 零位定义
- `gripper_width_cm` 是否是夹爪开口宽度
- `approach_depth_cm` 的含义（接近方向深度？）

### Step 2: 决定 VISTA vision_obs.result 的格式

选项 A（当前）: 透传 grasp server 完整响应 + 附加字段
选项 B: VISTA 提取 `targets[0]` 为扁平结构，放入 `result`
选项 C: VISTA 做字段映射/重命名，统一到内部 grasp result schema

### Step 3: 解决 class_id 来源

选项 A: `target_aliases.json` 扩展为 `{"apple": {"class_id": 47}, ...}`
选项 B: `mobile_cmd.payload` 新增 `class_id` 字段，小程序侧传入
选项 C: VISTA 从本地推理得到 class_id（与 INTERFACES.md 的 "explicit external input only" 冲突）

### Step 4: Orchestrator 侧补充

- 新增 `make_grasp_req()` 辅助函数
- GRASP 状态实现（Item 6 in TODO.md）
- 解析 `vision_obs.result` 中的 grasp pose，转为 UART 命令
