# VISTA 上下游接口说明

本文档只描述 VISTA 与外部模块之间的协议边界，不描述 VISTA 内部 `Scheduler` route 或 manager 间内部 contract。

## 链路总览

| 方向 | 消息类型 | 默认地址 | 说明 |
| --- | --- | --- | --- |
| 上游 -> VISTA | `vision_req` | `127.0.0.1:9003` | Orchestrator 或调试工具发给 VISTA 的控制请求 |
| VISTA -> 上游 | `vision_obs` | `127.0.0.1:9002` | VISTA 输出给 Orchestrator 的统一观测 envelope |

当前 transport 配置来源：`vision_module/config/board_config.py`

## 入站协议：`vision_req`

### 语义

`vision_req` 是上游驱动 VISTA 的唯一主请求协议。

当前代码支持的操作：

- `START`
- `UPDATE`
- `RESPOND`
- `STOP`

当前代码支持的 stage：

- `SEARCH`
- `GRASP`
- `RETURN`
- `IDLE`

### 字段

| 字段 | 类型 | 必填 | 说明 |
| --- | --- | --- | --- |
| `type` | `str` | 否 | 新协议建议固定为 `vision_req` |
| `ts` | `float` | 否 | 请求时间戳，未传则由接收端补当前时间 |
| `op` | `str` | 建议必填 | `START` / `UPDATE` / `RESPOND` / `STOP` |
| `stage` | `str` | 建议必填 | `SEARCH` / `GRASP` / `RETURN` / `IDLE` |
| `target` | `str` | 否 | 目标名称，例如 `bottle` |
| `mode_hint` | `str` | 否 | 建议优先 mode，如 `TRACK_LOCAL` / `MICRO_ADJUST` |
| `session_id` | `str` | 否 | 会话 ID |
| `req_id` | `str` | 否 | 请求 ID |
| `epoch` | `int` | 否 | 上游任务 epoch |
| `interaction_id` | `str` | 否 | `RESPOND` 时对应交互回合 |
| `response` | `dict` | 否 | `RESPOND` 的决策内容 |
| `payload` | `dict` | 否 | stage 专属请求体 |

### 当前兼容行为

`VisionReq.from_dict()` 仍保留旧消息兼容逻辑：

- `type=home_tag_req` 会被归一化为：
  - `stage=RETURN`
  - `op=START`
- 如果未显式传 `stage/op`，会基于旧字段做有限推断

建议新调用方始终显式发送 `stage` 和 `op`，不要依赖兼容推断。

### `STOP` 判定

当前代码里，以下任一条件都会被视为 stop：

- `op == "STOP"`
- `stage == "IDLE"`

### 操作语义

#### `START`

- 进入指定 stage
- 如果当前已经在同一 stage，会按 restart 语义重新进入
- 允许携带 `target`、`mode_hint`、`payload`

#### `UPDATE`

- 保持当前 stage，不重新 enter
- 只更新该 stage 关心的参数

#### `RESPOND`

- 用于回应 `WAITING_RESPONSE` 状态下的交互
- 当前主要用于 `GRASP` stage
- 典型 `response.decision`：`ACCEPT` / `REJECT`

#### `STOP`

- 结束当前任务流
- VISTA 内部会进入 `IDLE` 或 `IDLE_HOT`
- 当前实现不保证一定返回一个专门用于 stop ack 的 `vision_obs`

### 示例

#### 1. 启动搜索

```json
{
  "type": "vision_req",
  "ts": 1710000000.0,
  "session_id": "sess_001",
  "req_id": "req_001",
  "epoch": 1,
  "op": "START",
  "stage": "SEARCH",
  "target": "bottle"
}
```

#### 2. 启动抓取阶段

```json
{
  "type": "vision_req",
  "ts": 1710000010.0,
  "session_id": "sess_001",
  "req_id": "req_010",
  "epoch": 1,
  "op": "START",
  "stage": "GRASP",
  "target": "bottle",
  "payload": {
    "remote_grasp": true,
    "need_depth": true
  }
}
```

#### 3. 响应一次微调建议

```json
{
  "type": "vision_req",
  "ts": 1710000012.0,
  "session_id": "sess_001",
  "req_id": "req_011",
  "epoch": 1,
  "op": "RESPOND",
  "stage": "GRASP",
  "interaction_id": "ia_007",
  "response": {
    "decision": "ACCEPT"
  },
  "payload": {
    "executed_motion": {
      "dx_m": 0.03,
      "dy_m": -0.01,
      "dyaw_rad": 0.08
    }
  }
}
```

#### 4. 停止当前任务

```json
{
  "type": "vision_req",
  "ts": 1710000020.0,
  "session_id": "sess_001",
  "req_id": "req_020",
  "epoch": 1,
  "op": "STOP",
  "stage": "IDLE"
}
```

## 出站协议：`vision_obs`

### 语义

`vision_obs` 是 VISTA 当前唯一的对外结果 envelope。

它承载：

- 当前 stage
- 当前 active mode
- 当前状态机输出状态
- 可选感知结果
- 可选动作建议
- 可选阶段结果
- 可选交互元信息

### 字段

| 字段 | 类型 | 必填 | 说明 |
| --- | --- | --- | --- |
| `type` | `str` | 是 | 固定为 `vision_obs` |
| `ts` | `float` | 是 | VISTA 生成该消息的时间 |
| `stage` | `str` | 是 | 当前业务 stage |
| `mode` | `str` | 是 | 当前 active mode |
| `status` | `str` | 是 | 当前输出状态 |
| `session_id` | `str` | 否 | 当前会话 ID |
| `req_id` | `str` | 否 | 当前请求 ID |
| `epoch` | `int` | 否 | 当前 epoch |
| `interaction` | `dict` | 否 | 交互信息 |
| `perception` | `dict` | 否 | 感知摘要 |
| `proposal` | `dict` | 否 | 对上游的动作建议 |
| `result` | `dict` | 否 | 阶段结果或失败原因 |

### 当前实际状态值

当前代码实际会输出的 `status`：

- `RUNNING`
- `WAITING_RESPONSE`
- `RESULT_READY`
- `FAILED`

`DONE` 不是当前实现中的稳定输出状态，调用方不应依赖它。

### 当前 stage 对应的典型 payload

| Stage | Status | 典型内容 |
| --- | --- | --- |
| `SEARCH` | `RUNNING` | `perception.target_obs` |
| `RETURN` | `RUNNING` | `perception.home_tag_obs` |
| `GRASP` | `WAITING_RESPONSE` | `perception.target_obs` + `proposal` + `interaction` |
| `GRASP` | `RUNNING` | 远程抓取进行中的中间状态，`result.remote_state` 等 |
| `GRASP` | `RESULT_READY` | 抓取结果或远程结果，位于 `result` |
| `GRASP` | `FAILED` | 失败原因，位于 `result.reason` 等字段 |

### 典型 `interaction`

当前 `GRASP` 阶段的微调建议通常会输出：

```json
{
  "required": true,
  "interaction_id": "ia_007",
  "kind": "MOVE_HINT",
  "round": 1
}
```

### 示例

#### 1. 搜索观测

```json
{
  "type": "vision_obs",
  "ts": 1710000001.0,
  "session_id": "sess_001",
  "req_id": "req_001",
  "epoch": 1,
  "stage": "SEARCH",
  "mode": "TRACK_LOCAL",
  "status": "RUNNING",
  "perception": {
    "target_obs": {
      "found": true,
      "target": "bottle",
      "confidence": 0.82,
      "cx_norm": 0.47,
      "size_norm": 0.19,
      "bbox": [100, 120, 240, 300]
    }
  }
}
```

#### 2. 抓取前等待确认

```json
{
  "type": "vision_obs",
  "ts": 1710000011.0,
  "session_id": "sess_001",
  "req_id": "req_010",
  "epoch": 1,
  "stage": "GRASP",
  "mode": "MICRO_ADJUST",
  "status": "WAITING_RESPONSE",
  "interaction": {
    "required": true,
    "interaction_id": "ia_007",
    "kind": "MOVE_HINT",
    "round": 1
  },
  "perception": {
    "target_obs": {
      "found": true,
      "target": "bottle"
    }
  },
  "proposal": {
    "motion_delta": {
      "dx_m": 0.03,
      "dy_m": -0.01,
      "dyaw_rad": 0.08
    },
    "reason": "mock_micro_adjust_before_remote_grasp"
  }
}
```

#### 3. 抓取结果就绪

```json
{
  "type": "vision_obs",
  "ts": 1710000015.0,
  "session_id": "sess_001",
  "req_id": "req_012",
  "epoch": 1,
  "stage": "GRASP",
  "mode": "GRASP_REMOTE",
  "status": "RESULT_READY",
  "perception": {
    "target_obs": {
      "found": true,
      "target": "bottle"
    }
  },
  "result": {
    "target": "bottle",
    "source": "remote_grasp_client",
    "request_id": "rr_1710000012345"
  }
}
```

## 调用方约束

- 上游应把 `session_id`、`req_id`、`epoch` 当作关联键保留下来
- 上游不要依赖 raw frame 或内部 route 名称，这些不属于外部协议
- 上游不要依赖未在本文档列出的内部事件名，它们属于日志语义，不属于 IPC contract
- 上游最好显式传 `stage` 和 `op`，不要依赖兼容推断
- 上游处理 `GRASP` 时，应支持 `WAITING_RESPONSE -> RESPOND -> RESULT_READY/FAILED` 这一轮交互链路
