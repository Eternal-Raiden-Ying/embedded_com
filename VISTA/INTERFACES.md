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
| `type` | `str` | 是 | 代码中固定为 `"vision_req"`；旧协议 type 值归一化后存入 `legacy_type` |
| `ts` | `float` | 否 | 请求时间戳，未传则由接收端补当前时间 |
| `op` | `str` | 建议必填 | `START` / `UPDATE` / `RESPOND` / `STOP` |
| `stage` | `str` | 建议必填 | `SEARCH` / `GRASP` / `RETURN` / `IDLE` |
| `target` | `str` | 否 | 目标名称，例如 `bottle`；用于业务语义，不等于 remote `class_id` |
| `mode_hint` | `str` | 否 | 建议优先 mode |
| `session_id` | `str` | 否 | 会话 ID |
| `req_id` | `str` | 否 | 请求 ID |
| `req_type` | `str` | 否 | 控制面请求分类：`mode_request` / `target_update` / `keepalive`；由 StageController 判定 |
| `legacy_type` | `str` | 否 | 旧协议 `type` 原值（如 `home_tag_req`），归一化后保留用于追溯 |
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

- `vision_req.target` 是当前 detect-backed `RETURN` 的权威返回目标
- `payload.home_tag_obs` / `payload.mock_home_tag_obs` 仍可用于调试或 mock 注入

`RETURN` 现在直接消费 `local_perception.infer_boxes`、`class_names`、`rgb_shape`、`contract_ok`、`contract_error`、`contract_warnings`，并对外继续发布 `perception.home_tag_obs`。detect 路径下不会猜测 target；缺少可用 `target` 时会返回 `found=false` 且附带 `reason=missing_return_target`。

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

当前基线下，remote grasp 不再从 `target` 推导 `class_id`。
如果上游未显式提供 `payload.class_id`，remote 路径会被明确拒绝。

#### 当前 `GRASP.payload` 中已被实现消费的字段

| 字段 | 类型 | 当前状态 | 说明 |
| --- | --- | --- | --- |
| `remote_grasp` | `bool` | 当前支持 | 是否走 remote grasp，默认当前实现偏向 `true` |
| `need_depth` | `bool` | 当前支持 | 是否要求 depth，remote 路径通常应显式传入 |
| `class_id` | `int` | 当前支持，推荐显式必传 | remote grasp 目标类别 |
| `robot_id` | `str` | 当前支持 | 机器人标识，用于 remote manager 区分请求来源 |
| `remote_timeout_s` | `float` | 当前支持 | request 级超时覆盖项 |
| `remote_metadata` | `dict` | 当前支持 | 附加 remote metadata |

#### 上游约束

- 如果请求 remote grasp，上游应显式提供 `payload.class_id`
- 不应依赖 VISTA 内部从 `target` 自动推导 `class_id`
- `remote_metadata`、`remote_timeout_s` 当前可用，但更接近高级覆盖项，而不是最小稳定生产 contract

#### `RESPOND` 语义

`GRASP` 当前的交互主链路（VISTA 侧职责）：

1. VISTA 发出 `WAITING_RESPONSE`，等待上游决策
2. 上游执行或确认动作后发送 `RESPOND`（含 `response.decision` 和 `payload.class_id`）
3. VISTA 收到 `RESPOND ACCEPT` 后，若进入 remote grasp 路径：
   - `class_id` 校验：缺失 `class_id` 时显式拒绝，返回 `FAILED` + `reason=missing_class_id`
   - `INIT`：确认 remote 服务就绪（含最多 3 次 retry）
   - 等待 `GRASP_REMOTE` mode 的新 generation fresh frame
   - `PREDICT`：发送抓取预测请求
   - 返回 `RESULT_READY`（成功）/ `FAILED`（失败）/ `RUNNING`（`reposition_required`，需上游再次 RESPOND）
4. 上游 Orchestrator 负责循环逻辑：根据结果决定重试（再次 RESPOND）或进入下游执行

**重要边界**：
- **重试循环在 Orchestrator，不在 VISTA**。VISTA 仅处理单次 RESPOND → PREDICT → 返回结果
- **`RELEASE` 不是 per-grasp 步骤**。remote 服务的 `/release` 仅在 VISTA 引擎停止/禁用/显式 reset 时调用（`RuntimeSupervisor._stop_remote()`），属于资源清理而非抓取生命周期的一部分
- 状态转换：`WAITING_RESPONSE` → (收到 RESPOND) → `RUNNING` → `RESULT_READY` / `FAILED` / `RUNNING`(reposition)
- `DONE` 不是稳定状态值，调用方不应依赖

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
| `type` | `str` | 是 | 固定为 `vision_obs`。注：在 JSON 输出中 `type` 为最后一个字段（dataclass 声明顺序由 `asdict()` 保留） |
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
- `RELAXING`

`DONE` 不是当前稳定 contract，调用方不应依赖它。

### 当前 stage 对应的典型 payload

| Stage | Status | 典型内容 |
| --- | --- | --- |
| `SEARCH` | `RUNNING` | `perception.target_obs` |
| `RETURN` | `RUNNING` | `perception.home_tag_obs` |
| `GRASP` | `RELAXING` | SILENT mode idle — no capability, waiting for instruction |
| `GRASP` | `WAITING_RESPONSE` | MICRO_ADJUST mode — waiting for orchestrator adjust/accept decision |
| `GRASP` | `RUNNING` | GRASP_REMOTE_INIT/GRASP_REMOTE 进行中，常见于 `result.remote_state` |
| `GRASP` | `RESULT_READY` | remote grasp 成功结果，位于 `result.grasp` |
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
    "class_id": 4
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

## 当前收口状态

remote request 的最小稳定字段集合已收缩完成，segmentation 相关 remote surface 已删除。上游可以依赖当前文档列出的字段作为稳定 contract。

## 序列化行为

本节描述 `vision_obs` 序列化时上游需要注意的行为，这些行为在 `protocol.py` 中实现。

### `_compact()` 与 None 值处理

`vision_obs` 输出 JSON 前会经过 `_compact()` 处理（`protocol.py:49-68`）：

- **默认行为**：值为 `None` 的字段会被**省略**，不出现在 JSON 中
- **例外（`_PRESERVE_NONE_KEYS`）**：以下 key 即使值为 `None` 也会输出 `null`（`protocol.py:39-46`）：
  - `yaw_err`、`dist_err` — 姿态估计误差
  - `obs_ts`、`age_ms` — 观测时间戳与年龄
  - `frame_id`、`seq` — 帧标识与序号
- **空 dict**：值为 `{}` 的字段会被移除
- **列表**：列表中的 `None` 元素会被移除

上游解析方应注意：同一个字段可能在一条消息中存在（有值或 `null`），在下一条消息中完全不存在。

### Transport 行为

- **帧协议**：JSONL（每行一个 JSON object，以 `\n` 分隔），UTF-8 编码
- **发送队列**：容量 `maxsize=5`；满时丢弃最旧消息（`transport.py:98-103`）
- **发送频率**：主循环 8Hz，`vision_obs` 发送受速率限制（默认 `send_hz=5.0`，TRACK_LOCAL 模式 `track_local_send_hz=8.0`）。请求处理后的 `vision_obs` 不受速率限制（`force_send=True`）
- **重连**：对端断开后自动重连（间隔 1.0s），同一消息最多重试 2 次后丢失
- **入站校验**：只接受 `type=vision_req` 或 `type=home_tag_req` 的消息；其他 type 值**静默丢弃**（无日志、无错误回复）
- **JSON parse 失败**：记录 warning 日志，静默跳过该行，不回复上游

## 2026-04 Contract Notes

- `target` and remote `class_id` are different fields. Remote grasp requests must provide explicit `payload.class_id`; VISTA no longer infers `class_id` from `target`.
- When detect contract weakens or fails, stage-side `perception.target_obs` may now include `contract_error` and `contract_warnings` instead of silently disappearing.
- The stable detect manager contract is `infer_boxes = [[x1, y1, x2, y2, score, class_id], ...]` with `infer_box_format=xyxy_score_class_id`.
- If detect `class_names` are missing from the active model profile, VISTA now weakens explicitly to normalized `coco80` rather than falling back to the legacy grasp-only class table.
- External IPC still does not expose raw frame transport, but the internal service color baseline is now BGR. Debug and preview paths should assume BGR unless a specific predictor says otherwise.
- Remote `/init` is now service-scoped. `GRASP` waits for service init confirmation plus fresh frames before `PREDICT`, and `/release` is no longer a default per-grasp action.
- `remote_base_url` is no longer part of the supported request-level override surface. Endpoint ownership now belongs to mode/profile/runtime.
- Segmentation-specific remote request surface has been removed from the integrated contract. Older references to `seg_file` or segmentation upload are historical only.
