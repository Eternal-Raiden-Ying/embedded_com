# 移动端网关运行手册与协议说明 (Mobile Gateway Runbook & Protocol)

本文档将小车移动端网关（`mobile_gateway`）的设计设计、运行模式、配置变量、MQTT 北向协议以及南向 Orchestrator 的映射逻辑合并整理，作为唯一的开发运行参考规范。

---

## 1. 概述与设计意图 (Overview & Design Intent)

`mobile_gateway` 是部署在开发板端（如 SC171）的移动端网关服务，用于桥接北向移动端网络流（如微信小程序或云端 MQTT Broker）与南向状态机控制流（Orchestrator 服务）。

*   **唯一执行授权**：保持 Orchestrator 作为板端闭环控制与执行的唯一权威，网关仅作为适配层，不直接参与机器人状态控制。
*   **北向/南向协议解耦**：北向采用面向移动端/小程序的协议格式，南向采用基于 UDS/TCP 的本地 `task_cmd` 与 `task_ack`。
*   **支持无硬件 Mock 闭环**：通过配置不同的后端，支持脱离真实小车和视觉引擎进行网关层面的闭环验证。
*   **当前基本链路**：
    ```text
    微信小程序 / 云端 MQTT ──(MQTT 协议)──> mobile_gateway ──(task_cmd: 9001/UDS)──> Orchestrator ──(串口)──> STM32 / 底盘
    微信小程序 / 云端 MQTT <──(MQTT 反馈)── mobile_gateway <──(task_ack: 9012/UDS)── Orchestrator
    ```

---

## 2. 运行模式与配置 (Runtime & Environment Config)

### 2.1 网关运行级别 (Runtime Styles)
网关支持两种运行日志级别：
*   **`production` (生产模式)**：正式服务结构，低噪日志。禁止打印 raw MQTT payload 敏感内容，在公开的 MQTT 载荷中不暴露底层私有字段。
*   **`debug` (调试模式)**：支持打印完整的北向 MQTT 收发报文。在对外状态与 ACK 消息中，允许附带 `backend_state`、`raw_error` 等调试信息，方便排查故障。

### 2.2 南向后端类型 (Southbound Backends)
网关的南向（Orchestrator 连接）支持以下三种通信模式（由 `MOBILE_GATEWAY_BACKEND` 环境变量配置）：
*   **`mock`**：模拟闭环，不需要真实的 Orchestrator 或 VISTA 服务，也不涉及真实串口，在网关内部模拟指令的接收与应答。
*   **`tcp_no_ack` / `uds_no_ack`**：单向推送模式。网关仅向 Orchestrator 发送 `task_cmd`，不读取和订阅 Orchestrator 的 `task_ack`。
*   **`orchestrator_tcp` / `orchestrator_uds`**：完整双向桥接模式。网关既转发 `task_cmd`，也接收 Orchestrator 的 `task_ack`，并监听读取 `state_blocks.jsonl` 日志用于同步车辆状态。

### 2.3 关键环境变量与配置参考 (Environment Overrides)

**北向网关监听与服务设置：**
*   `MOBILE_GATEWAY_CMD_IN_HOST` / `PORT`：本地 TCP 模式下的网关指令监听地址。
*   `MOBILE_GATEWAY_STATUS_OUT_HOST` / `PORT`：本地 TCP 模式下的网关状态目的输出。
*   `MOBILE_GATEWAY_BACKEND`：`mock` / `tcp_no_ack` / `orchestrator_tcp` / `orchestrator_uds`。
*   `MOBILE_GATEWAY_STATE_BLOCKS_PATH`：指定要监听的特定 `state_blocks.jsonl` 物理路径。
*   `MOBILE_GATEWAY_ORCH_RUNS_DIR`：Orchestrator 运行目录根路径（默认 `orchestrator/runs`），用于自动扫描最新 run 文件夹中的状态日志。

**南向连接端口（Orchestrator）：**
*   默认的 `task_cmd` 输入端口为 `127.0.0.1:9001` 或 Unix Domain Socket `/tmp/robot_stack/task_cmd.sock`。
*   默认的 `task_ack` 输出端口为 `127.0.0.1:9012` 或 Unix Domain Socket `/tmp/robot_stack/task_ack.sock`。
*   可通过环境变量如 `ORCH_TASK_CMD_IN_SOCKET_PATH` 等对其进行覆盖。

**北向 MQTT 连接设置 (当 `MOBILE_GATEWAY_MQTT_ENABLED=true` 时)：**
*   `MOBILE_GATEWAY_MQTT_BROKER_HOST` / `PORT`：MQTT Broker 地址及端口。
*   `MOBILE_GATEWAY_MQTT_TRANSPORT`：`tcp` / `websocket`。
*   `MOBILE_GATEWAY_MQTT_USE_TLS`：是否启用 TLS 安全加密（生产环境下微信小程序强制要求 WSS 协议）。
*   `MOBILE_GATEWAY_MQTT_WEBSOCKET_PATH`：WebSocket 连接路径（例如 `/mqtt`）。
*   `MOBILE_GATEWAY_MQTT_USERNAME` / `PASSWORD`：认证用户名及密码。
*   `MOBILE_GATEWAY_MQTT_CLIENT_ID`：客户端 ID。

---

## 3. 固定 MQTT 北向主题与安全规范 (Fixed MQTT Topics & Security)

为了保证上下游多模块交互一致性，网关的 MQTT 主题结构固定如下，其中机器人标识 ID 固定为 `SC171`：

*   **北向命令输入**：`robot/v1/SC171/mobile/cmd` (QoS 1)
*   **北向 ACK 输出**：`robot/v1/SC171/mobile/ack` (QoS 1)
*   **北向状态流发布**：`robot/v1/SC171/mobile/status` (QoS 0，Retained)
*   **北向心跳流发布**：`robot/v1/SC171/heartbeat` (QoS 0)

### 安全建议 (Security Notes)
1.  **链路加密**：在正式发布环境中，必须使用带有效证书的 WSS/TLS 连接，不要使用无加密的 `ws://`。
2.  **凭证脱敏**：不要将真实的 MQTT 用户名密码写入 Git 代码中，统一通过外部环境变量或本地未跟踪的 `configs/mobile_gateway.mqtt.yaml` 文件进行注入。
3.  **主题权限控制**：在 Broker 侧配置细粒度的 ACL，限制客户端仅拥有 `robot/v1/SC171/` 主题命名空间的发布和订阅权限，禁止越权操作。

---

## 4. 移动端指令协议格式与南向映射 (Command Payloads & Southbound Mapping)

### 4.1 北向命令格式
微信小程序向主题 `robot/v1/SC171/mobile/cmd` 发布以下格式的 JSON：

#### 示例：取物任务 (`fetch_object`)
```json
{
  "type": "mobile_cmd",
  "robot_id": "SC171",
  "cmd_id": "wx_1777293209382",
  "session_id": "wx_session_001",
  "epoch": 1,
  "cmd": "fetch_object",
  "target": "apple",
  "text": "拿苹果",
  "source": "wechat_miniprogram",
  "ts": 1777293208.5
}
```

#### 示例：停止任务 (`stop`)
```json
{
  "type": "mobile_cmd",
  "robot_id": "SC171",
  "cmd_id": "wx_1777293209383",
  "session_id": "wx_session_001",
  "epoch": 1,
  "cmd": "stop",
  "source": "wechat_miniprogram",
  "ts": 1777293215.0
}
```

*   **系统支持的正式小程序命令**：
    *   `fetch_object`：执行桌边取物（需要传入 `target`，例如 `apple`、`banana`、`bottle`、`cup`）。
    *   `stop`：强行打断当前执行流程，使底盘和视觉回归安全待机态。
*   **兼容性调试指令**：当 `runtime.enable_legacy_command_compat=true` 时，允许接收 `type=FIND_AND_PICK` 等历史指令。此外，网关本地支持 `query_status`（主动查询）、`resume`（继续）、`retry_search`（重试搜寻）、`go_home`（返航）等指令用于工程联调。

### 4.2 南向 `task_cmd` 映射格式
网关在校验指令合法后，会将其转换并发往南向 Orchestrator 的 `task_cmd` 端点。

#### 映射为 `fetch_object`：
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

#### 映射为 `stop`：
```json
{
  "type": "task_cmd",
  "intent": "STOP",
  "confidence": 1.0,
  "cmd_id": "wx_1777293209382",
  "session_id": "wx_session_001",
  "epoch": 1,
  "source": "wechat_miniprogram",
  "ts": 1777293208.7
}
```

---

## 5. ACK、状态与心跳语义映射 (ACK, Status, Heartbeat Semantics)

### 5.1 网关应答级别 (ACK Kinds)
在主题 `robot/v1/SC171/mobile/ack` 下，网关会发出两类应答：
1.  **`gateway_ack`**：
    *   在网关接收并解析完北向消息后**立即**发出。
    *   表示网关层校验该指令格式通过，已成功接收（或由于非法字段拒绝）。
2.  **`task_ack`**：
    *   在南向 Orchestrator 完成应答后发出。
    *   表示南向控制层已经接受或拒绝了此项任务。

#### `gateway_ack` 载荷示例：
```json
{
  "type": "mobile_ack",
  "kind": "gateway_ack",
  "robot_id": "SC171",
  "cmd_id": "wx_1777293209382",
  "session_id": "wx_session_001",
  "epoch": 1,
  "cmd": "fetch_object",
  "target": "apple",
  "message": "gateway command accepted",
  "accepted": true,
  "source": "mobile_gateway",
  "ts": 1777293208.6
}
```

### 5.2 状态流映射 (Mobile Status)
主题：`robot/v1/SC171/mobile/status`
状态载荷中提供经过归一化的用户友好型字段，可以直接在小程序界面上呈现实时信息：

```json
{
  "type": "mobile_status",
  "kind": "status",
  "robot_id": "SC171",
  "session_id": "wx_session_001",
  "epoch": 1,
  "state": "searching",
  "target": "apple",
  "message": "开始桌边任务，目标 apple",
  "progress": 20,
  "command": "fetch_object",
  "source": "mobile_gateway",
  "ts": 1777293210.0
}
```

*   **北向统一状态分类 (state)**：
    *   `submitted`：命令已提交
    *   `accepted`：任务已接收
    *   `searching`：搜寻目标中
    *   `running`：运行及停靠中
    *   `idle`：空闲就绪
    *   `stopped`：命令已被停止打断
    *   `error`：网关诊断或南向报错
*   **Orchestrator 底层状态 (backend_state) 与北向状态映射关系**：
    *   `IDLE` / `DONE` ──> `idle`
    *   `SEARCH_TABLE` / `SEARCH_TARGET_INIT` / `EDGE_SLIDE_SEARCH` ──> `searching`
    *   `YOLO_ACQUIRE_ALIGN` / `YOLO_APPROACH` / `EDGE_ADJUST` / `FINAL_SLOW_STOP` / `AT_TABLE_EDGE` / `TARGET_CONFIRM` / `TARGET_LOCKED` / `FREEZE_BASE` / `GRASP` ──> `running`
    *   `ERROR_RECOVERY` ──> `error`
    *   当 `stop` 成功执行时 ──> `stopped`
*   **异常拦截提示**：
    当网关监测到南向连接断开、视觉离线（`vision_req_out connect_failed`）或链路处于 `DEGRADED` 状态时，会强制发布 `state=error`，错误码 `1007`，以提示用户 `"视觉模块未连接，任务暂时无法继续"`。

### 5.3 网关心跳 (Heartbeat)
主题：`robot/v1/SC171/heartbeat`
用于对北向宣布网关的在线状态，同时包含当前会话 ID 和运行状态概要。
```json
{
  "type": "mobile_gateway_heartbeat",
  "kind": "heartbeat",
  "robot_id": "SC171",
  "online": true,
  "backend_mode": "orchestrator_tcp",
  "state": "idle",
  "session_id": "",
  "epoch": 0,
  "ts": 1777293212.0
}
```

---

## 6. 指令去重与可靠机制 (Deduplication & Reliable Handshaking)

*   **指令去重缓存**：网关在内存中保存最近 `64` 个历史指令的 `cmd_id`。
*   **重复指令行为**：
    *   若收到的 `cmd_id` 已存在于缓存中，网关会重复下发 `gateway_ack` 以确认通信收到。
    *   不会二次向 Orchestrator 发送重复的 `task_cmd`。
    *   避免在状态流中产生重复的状态翻转垃圾信息。
    *   重复的 `stop` 指令也会同样被拦截去重，防止连续触发日志和连接风暴。
*   **指令时效性门控 (Stale Epoch Rejection)**：对于 `epoch` 较旧或时间戳延迟过大的指令，网关可做直接拒绝处理。

---

## 7. 运行调试与验证指南 (Debugging & Runbook Steps)

### 7.1 本地测试方法 (无物理硬件，Windows/Host 环境)
1.  **拉起 Mock 网关**：
    ```bash
    python3 -m orchestrator_service.mobile_gateway.runtime.service --config configs/mobile_gateway.mqtt.example.yaml
    ```
2.  **模拟小程序发送命令**：
    使用脚本 `tools/mock_mobile_sender.py` 向监听的主题推送 `fetch_object` 或 `stop` 消息。
3.  **运行单元测试集**：
    ```bash
    python3 -m unittest tests.test_command_protocol tests.test_gateway_mapping tests.test_mock_flow tests.test_real_protocol_mapping
    ```

### 7.2 板端部署与真实对接启动顺序 (SC171 环境)
1.  **启动南向 Orchestrator** (以 UART dry-run 不打开真实串口为例)：
    ```bash
    cd /home/aidlux/embedded_com/orchestrator
    export ORCH_TASK_CMD_IN_SOCKET_PATH=/tmp/robot_stack/task_cmd.sock
    export ORCH_TASK_ACK_OUT_SOCKET_PATH=/tmp/robot_stack/task_ack.sock
    export ORCH_SERIAL_DRY_RUN=1
    python3 -m orchestrator_service.app.main
    ```
2.  **启动网关** (采用双向桥接模式并开启北向 MQTT)：
    ```bash
    cd /home/aidlux/embedded_com
    PYTHONPATH=/home/aidlux/embedded_com/orchestrator \
    /usr/bin/python3 -m orchestrator_service.mobile_gateway.runtime.service \
      --config configs/mobile_gateway.mqtt.yaml
    ```
3.  **使用调试脚本进行连通性确认**：
    使用 `tools/smoke_mobile_to_orchestrator.py` 进行端到端的 TCP 注入和响应检测。

---

## 8. 后续工作与展望 (Next Steps & Future Refactors)

1.  **语音交互接入规范**：
    不要让原始的 ASR/语音识别文本直接注入 Orchestrator 主状态机。麦克风或语音输入应当在移动端小程序侧（或轻量级云端助手）进行分词和语义解析，转换为统一标准的 `fetch_object` 或 `stop` 指令发给网关。例如：
    *   `"拿苹果"` ──> `fetch_object(target="apple")`
    *   `"停下"` ──> `stop`
2.  **视障人士音频播报策略**：
    移动端小程序可以订阅 `mobile/status` 和 `mobile/ack` 主题，对下发状态做简短且一致性的语音反馈：
    *   收到 ACK 提示：`"已收到，开始寻找苹果。"`
    *   运行节点提示：`"正在搜索桌边"`、`"已锁定目标"`、`"开始返回起点"`。
    *   中止确认：`"任务已停止。"`
3.  **南北向适配器解耦**：
    若后续需要支持多个机器人实体，可将 `mobile_gateway` 独立作为控制平面容器运行，利用 MQTT 的 `SC171` Topic 部分实现动态注册。
