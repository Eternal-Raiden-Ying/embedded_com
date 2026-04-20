# VISTA 上下游接口说明

本文档描述 VISTA 对外的 IPC 协议边界。

不描述的内容：

- `Scheduler` 内部 route 细节
- manager 间内部 contract
- 具体 worker loop 时序

当代码与文档冲突时，应以当前实现和本文件为近端基线；历史兼容逻辑不应反向定义新协议。

## 链路总览

| 方向 | 消息类型 | 默认地址 | 说明 |
| --- | --- | --- | --- |
| 上游 -> VISTA | `vision_req` | `127.0.0.1:9003` | Orchestrator 或调试工具发给 VISTA 的控制请求 |
| VISTA -> 上游 | `vision_obs` | `127.0.0.1:9002` | VISTA 输出给上游的统一观测 envelope |

当前 transport 默认配置来源：`vision_module/config/board_config.py`

## 入站协议：`vision_req`

### 当前语义

`vision_req` 是上游驱动 VISTA 的唯一主请求协议。

当前支持的 `op`：

- `START`
- `UPDATE`
- `RESPOND`
- `STOP`

当前支持的 `stage`：

- `SEARCH`
- `GRASP`
- `RETURN`
- `IDLE`

### 顶层字段

| 字段 | 类型 | 必填 | 说明 |
| --- | --- | --- | --- |
| `type` | `str` | 否 | 建议固定为 `vision_req` |
| `ts` | `float` | 否 | 请求时间戳，未传则由接收端补当前时间 |
| `op` | `str` | 建议必填 | `START` / `UPDATE` / `RESPOND` / `STOP` |
| `stage` | `str` | 建议必填 | `SEARCH` / `GRASP` / `RETURN` / `IDLE` |
| `target` | `str` | 否 | 目标名称，例如 `bottle`；用于业务语义，不等于 remote `class_id` |
| `mode_hint` | `str` | 否 | 建议优先 mode |
| `session_id` | `str` | 否 | 会话 ID |
| `req_id` | `str` | 否 | 请求 ID |
| `epoch` | `int` | 否 | 上游任务 epoch |
| `interaction_id` | `str` | 否 | `RESPOND` 时对应交互回合 |
| `response` | `dict` | 否 | 对交互回合的响应内容 |
| `payload` | `dict` | 否 | stage 专属请求体 |

### 当前兼容行为

`VisionReq.from_dict()` 仍保留旧协议兼容逻辑：

- `type=home_tag_req` 会归一化为 `stage=RETURN`、`op=START`
- 如果未显式传 `stage/op`，会基于旧字段做有限推断

这只是兼容行为，不是推荐 contract。新调用方应显式发送 `stage` 和 `op`。

### `STOP` 判定

当前代码里，以下任一条件都会被视为 stop：

- `op == "STOP"`
- `stage == "IDLE"`

## Stage 专属请求体

### 1. `SEARCH`

当前稳定语义较简单：

- `target` 表示目标名称
- `payload.target_obs` / `payload.mock_target_obs` 仍可用于调试或 mock 注入

不建议把内部 detect 细节暴露到 `SEARCH` 的外部请求 contract 中。

### 2. `RETURN`

当前稳定语义：

- `payload.home_tag_obs` / `payload.mock_home_tag_obs` 仍可用于调试或 mock 注入

`RETURN` 当前对真实 local perception 的适配能力仍弱于 `SEARCH`，调用方不应假设它已经拥有和 `SEARCH` 同等成熟的 detect 驱动路径。

### 3. `GRASP`

`GRASP` 是当前最需要明确 contract 的 stage。

#### 当前稳定方向

- `remote_grasp`: 是否走 remote grasp 路径
- `need_depth`: remote grasp 是否要求 depth
- `class_id`: remote grasp 的目标类别 ID

#### `class_id` 规则

这是当前需要收口的关键点：

- remote grasp 的 `class_id` 应由外部输入显式提供
- `class_id` 可以被 VISTA 保存到 stage state 中，作为请求上下文的一部分
- `target` 只是业务目标名称，不应再被上游当作 remote `class_id` 的替代物

当前代码中仍存在从 `target` 推导 `class_id` 的兼容/回退行为，但调用方不应依赖该行为；该回退路径是待收缩对象，不是目标 contract。

#### 当前 `GRASP.payload` 中已被实现消费的字段

| 字段 | 类型 | 当前状态 | 说明 |
| --- | --- | --- | --- |
| `remote_grasp` | `bool` | 当前支持 | 是否走 remote grasp，默认当前实现偏向 `true` |
| `need_depth` | `bool` | 当前支持 | 是否要求 depth，remote 路径通常应显式传入 |
| `class_id` | `int` | 当前支持，推荐显式必传 | remote grasp 目标类别 |
| `remote_timeout_s` | `float` | 当前支持 | request 级超时覆盖项 |
| `remote_base_url` | `str` | 当前支持 | 临时 endpoint 覆盖项，更适合调试/测试 |
| `remote_metadata` | `dict` | 当前支持 | 附加 remote metadata |

#### 上游约束

- 如果请求 remote grasp，上游应显式提供 `payload.class_id`
- 不应依赖 VISTA 内部从 `target` 自动推导 `class_id`
- `remote_base_url`、`remote_metadata`、`remote_timeout_s` 当前可用，但更接近高级覆盖项，而不是最小稳定生产 contract

#### `RESPOND` 语义

`GRASP` 当前的交互主链路是：

1. VISTA 发出 `WAITING_RESPONSE`
2. 上游执行或确认动作
3. 上游发送 `RESPOND`
4. VISTA 继续进入 `MICRO_ADJUST` 或 `GRASP_REMOTE` 的后续链路

典型字段：

- `interaction_id`
- `response.decision`: `ACCEPT` / `REJECT`

## 出站协议：`vision_obs`

### 当前语义

`vision_obs` 是 VISTA 当前唯一的对外结果 envelope。

它承载：

- 当前 `stage`
- 当前 active `mode`
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

### 当前稳定 `status`

当前代码稳定输出的状态值：

- `RUNNING`
- `WAITING_RESPONSE`
- `RESULT_READY`
- `FAILED`

`DONE` 不是当前稳定 contract，调用方不应依赖它。

### 当前 stage 对应的典型 payload

| Stage | Status | 典型内容 |
| --- | --- | --- |
| `SEARCH` | `RUNNING` | `perception.target_obs` |
| `RETURN` | `RUNNING` | `perception.home_tag_obs` |
| `GRASP` | `WAITING_RESPONSE` | `perception.target_obs` + `proposal` + `interaction` |
| `GRASP` | `RUNNING` | remote 进行中的中间状态，常见于 `result.remote_state` 等 |
| `GRASP` | `RESULT_READY` | remote grasp 结果或最终阶段结果，位于 `result` |
| `GRASP` | `FAILED` | 失败原因，常见于 `result.reason` |

### 关于 `GRASP` 的结果语义

当前 `GRASP` 阶段的 `RESULT_READY` / `FAILED` 更多反映：

- remote grasp 请求是否完成
- remote 结果是否可用
- 当前阶段是否产出阶段性结果

它不等于上层运动执行已经完成。

## 示例

### 1. 启动搜索

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

### 2. 启动 remote 抓取阶段

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
    "need_depth": true,
    "class_id": 39
  }
}
```

### 3. 响应一次微调建议

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

### 4. 搜索观测输出

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

## 调用方约束

- 上游应把 `session_id`、`req_id`、`epoch` 当作关联键保留
- 上游不要依赖 raw frame 或内部 route 名称，这些不属于外部协议
- 上游不要依赖未在本文档列出的内部事件名，它们属于日志语义，不属于 IPC contract
- 上游最好显式传 `stage` 和 `op`，不要依赖兼容推断
- 如果触发 remote grasp，上游应显式传 `payload.class_id`
- 上游不应把 `target` 当作 remote `class_id` 的替代字段
- 上游处理 `GRASP` 时，应支持 `WAITING_RESPONSE -> RESPOND -> RESULT_READY/FAILED` 这一轮交互链路

## 当前未完全收口的地方

以下项目已经是架构方向，但当前实现仍在收口中：

- VISTA 内部仍残留从 `target` 推导 `class_id` 的回退逻辑
- remote request 的最小稳定字段集合仍在收缩中
- `GRASP_REMOTE` 的 fresh-frame barrier 和 init-completion gate 仍需继续落实

这些属于当前实现债务，不应被上游拿来当正式 contract 依赖。
