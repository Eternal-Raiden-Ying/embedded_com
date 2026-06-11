# 日志标准规范说明文档 (Logging Standard)

## 1. 概述 (Overview)

本文档定义了 Robot Stack 2026 软件栈的统一日志规范（Schema），旨在规范部署于 ARM/AidLux 端侧环境下的各组件调试与诊断输出。

---

## 2. 当前各组件日志审计现状 (Current Audit)

### 2.1 状态机控制服务 (Orchestrator)
*   **控制台输出 (Stdout)**：使用 Python `logging.basicConfig`，格式统一为 `\n` 分割的文本：
    `{时间} | {级别} | {Logger名} | {消息}`。
*   **结构化文件**：采用 JSONL 格式，每次启动在 `runs/` 下建立以时间戳命名的专用文件夹，分类输出 `events.log`、`timeline.jsonl`、`ipc.jsonl`、`state_blocks.jsonl` 以及 `heartbeat.jsonl`。
*   **日志过滤级别**：支持通过 `configure_logging(mode)` 在 `full` (DEBUG) 与 `concise` (INFO) 之间进行切换。
*   **Module 标识**：Logger 统一命名为 `OrchestratorService`、`OrchestratorCore` 等。

### 2.2 视觉感知服务 (VISTA)
*   **控制台输出 (Stdout)**：文本输出，格式为 `%(asctime)s | %(levelname)-5s | %(name)s | %(message)s`。
*   **结构化文件**：在 `VISTA/runs/<stack_run_id>/` 目录下生成包含 `meta.json`、`event.jsonl`、`ipc.jsonl` 以及可选的 `heartbeat.jsonl`（默认关闭，通过 `VISION_HEARTBEAT_ENABLED=1` 开启）。
*   **控制台镜面镜像**：启动脚本会自动将 Stdout 和 Stderr 镜像重定向输出到 `VISTA/logs/vision.out` 中。
*   **过滤级别**：支持通过 `VISION_LOG_MODE=concise/full` 进行调配。
*   **Module 标识**：包含 `vision.runtime`、`vision.engine`、`vision.ipc`、`vision.stage` 等子模块前缀。

### 2.3 语音控制服务 (Voice)
*   *注：当前版本已将板端语音 ASR/Voice 移出核心运行队列，历史规范仅供审计参考。*
*   **Stdout**：通过 `jlog()` 直接以一行一个 JSON 的方式向控制台输出结构化数据。
*   **Module 标识**：在 JSON 载荷中通过 `src` 字段声明子模块类别（如 `boot`、`loop`、`oww`、`seg`、`tts`、`mic` 等）。

---

## 3. 统一日志规范结构 (Unified Logging Schema)

### 3.1 结构化日志文件格式
所有落盘的结构化运行日志必须采用 JSON 或 JSONL（每行一个完整 JSON 字典）格式：
*   **`meta.json`**：单次拉起写入一次，包含服务启动参数与最终的有效配置 Dump。
*   **`event.jsonl`**：按发生时间记录服务内的关键运行事件（如状态跳转、异常抛出、模式切换）。
*   **`ipc.jsonl`**：专用于记录跨进程、北向或串口的数据包收发记录。
*   **`heartbeat.jsonl`**：极低频的组件健康状况统计（默认关闭）。

### 3.2 结构化字段定义 (Common Fields)
日志行对应的顶层通用字段结构如下：
```json
{
  "ts": 1234567890.123,
  "level": "info",
  "module": "vision",
  "stack_run_id": "run_20260413_123456_ab12cd",
  "data": { "optional": "上下文扩展数据" }
}
```
*   `ts` (float，必须)：Unix 时间戳，精确到毫秒。
*   `level` (str，建议)：`debug` / `info` / `warn` / `error` / `critical`。
*   `module` (str，必须)：生成该日志的子系统标签（如 `vision`、`orch`）。
*   `stack_run_id` (str，必须)：单次拉起由启动脚本分配的全球唯一时间戳运行 ID。
*   `data` (dict，可选)：自定义的键值对参数，避免将不稳定参数直接扩充到顶层字段中。

### 3.3 VISTA 运行事件字段顺序 (`event.jsonl` Field Order)
为了便于自动化日志脚本快速解析，`event.jsonl` 中的 Key 排序应当固定为：
1. `ts` | 2. `level` | 3. `module` | 4. `stack_run_id` | 5. `event` | 6. `stage` | 7. `mode` | 8. `trigger` | 9. `session_id` | 10. `req_id` | 11. `epoch` | 12. `interaction_id` | 13. `data`

### 3.4 跨进程通信字段顺序 (`ipc.jsonl` Field Order)
`ipc.jsonl` 中 Key 排序固定为：
1. `ts` | 2. `level` | 3. `module` | 4. `stack_run_id` | 5. `direction` | 6. `channel` | 7. `event` | 8. `msg_type` | 9. `session_id` | 10. `req_id` | 11. `epoch` | 12. `ok` | 13. `peer` | 14. `error` | 15. `data`

---

## 4. 日志分级指南 (Level Guidelines)

*   **`debug`**：高频流式跟踪日志，如每帧三维提取的数据量、完整的二进制 IPC Payload 解析、串口发送的字节信息。
*   **`info`**：状态机状态切换、高层任务的开始与打断、周期心跳监测点、网络重连成功。
*   **`warn`**：可自动恢复的系统波动，例如传感器短时丢帧使用 Hold 缓存、网络请求超时重试、待发送串口队列满后丢弃老指令。
*   **`error`**：操作发生实质性失败，例如向底盘发送速度失败、模型加载崩溃、IPC 读写异常。
*   **`critical`**：系统致命故障，导致当前服务守护线程异常退出。

---

## 5. 日志保存与轮转策略 (Rotation Policy)

*   **单文件大小限制**：单个 JSONL 文件大小达到 10MB 时自动切分。
*   **保留策略**：默认保留最近 7 天，或者最近 100 次运行产生的数据（以先满足的条件为准）。
*   **落盘路径**：板端生产环境统一存放于 `/data/runs/<stack_run_id>/` 目录下。
