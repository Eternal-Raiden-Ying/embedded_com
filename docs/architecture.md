# 系统架构 (Architecture)

本文档将运行时层映射到当前源码树中。旨在阐明后续进行代码修改时的合理位置。

---

## 1. 应用交互层 (Application Interaction Layer)

*   **职责**：接收用户（小程序）的控制指令，发布面向用户的实时状态反馈，并将北向的移动端 MQTT 流量桥接转换为南向的状态机命令（`task_cmd`）。
*   **对应代码**：
    *   `orchestrator/orchestrator_service/mobile_gateway/`
    *   `docs/mobile_gateway_runbook.md`
*   **拥有所有权的内容**：
    *   微信小程序与 MQTT 主题。
    *   `fetch_object` 与 `stop` 命令的校验和准入。
    *   移动端应答（ACK）与状态的数据格式转换。
*   **不拥有的内容**：
    *   状态机状态转移的算法逻辑。
    *   底盘基轴物理速度计算。
    *   视觉感知提取算法。

---

## 2. 任务编排层 (Task Orchestration Layer)

*   **职责**：管理任务状态、状态转移条件、同步视觉感知请求、安全性故障恢复以及导出运行时状态快照。
*   **对应代码**：
    *   `orchestrator/orchestrator_service/runtime/core.py`
    *   `orchestrator/orchestrator_service/runtime/context.py`
    *   `orchestrator/orchestrator_service/runtime/task_runtime.py`
    *   `orchestrator/orchestrator_service/runtime/transitions.py`
    *   `orchestrator/orchestrator_service/runtime/vision_sync.py`
    *   `orchestrator/orchestrator_service/runtime/export_state.py`
    *   `orchestrator/orchestrator_service/runtime/states/`
    *   `orchestrator/orchestrator_service/runtime/safety/`
    *   `orchestrator/orchestrator_service/runtime/state_machine.py` (作为兼容层导入入口)
*   **状态组**：
    *   桌边停靠状态：`runtime/states/table_docking.py`
    *   目标搜索状态：`runtime/states/target_search.py`
    *   抓取工作流状态：`runtime/states/grasp_flow.py`
    *   异常恢复机制：`runtime/states/recovery.py`
    *   底盘运动安全过滤：`runtime/safety/base_motion_safety.py`
    *   丢失帧保持保护：`runtime/safety/stale_guard.py`
    *   急停安全策略：`runtime/safety/emergency_stop_policy.py`
*   **当前桌边停靠状态序列**：
    *   `SEARCH_TABLE`（搜索桌边）
    *   `YOLO_ACQUIRE_ALIGN`（前向粗对齐）
    *   `YOLO_APPROACH`（引导接近）
    *   `EDGE_ADJUST`（几何锁边微调）
    *   `FINAL_SLOW_STOP`（终点减速停车）
    *   `AT_TABLE_EDGE`（停靠就绪）

---

## 3. 感知算法层 (Perception Algorithm Layer)

*   **职责**：管理相机和模型后端生命周期、构建感知观测数据包以及维护 VISTA 各业务阶段（Stage）的运行时生命周期。
*   **对应代码**：
    *   `VISTA/vision_module/app/service.py`
    *   `VISTA/vision_module/app/scheduler.py`
    *   `VISTA/vision_module/app/stage_controller.py`
    *   `VISTA/vision_module/app/stages/search/`
    *   `VISTA/vision_module/app/observation/`
    *   `VISTA/vision_module/backend/`
    *   `VISTA/vision_module/utils/`
*   **当前搜索模式 (Search Modes)**：
    *   `FIND_OBJECT`：本地目标观测。
    *   `FIND_EDGE`：桌边检测与拟合。
*   **兼容别名说明**：
    *   `TRACK_LOCAL` 为 `FIND_OBJECT` 的历史兼容别名。
    *   `DEPTH_PERCEPTION` 为 `FIND_EDGE` 的历史兼容别名。

---

## 4. 数据通信层 (Data Communication Layer)

*   **职责**：定义消息 Schema 规范、管理 IPC 通信传输行为，以及维护 JSONL/TCP/UDS 边界。
*   **对应代码**：
    *   `orchestrator/orchestrator_service/ipc/protocol.py`
    *   `orchestrator/orchestrator_service/ipc/transport.py`
    *   `VISTA/vision_module/ipc/protocol.py`
    *   `VISTA/vision_module/ipc/transport.py`
    *   `common/`
*   **传输策略**：
    *   Windows 主机开发阶段默认使用 `tcp` 或 `disabled`。
    *   板端（SC171 Linux）运行时，当 `socket.AF_UNIX` 可用时，默认切换为 `uds` 通信。
    *   禁止在 Windows 上伪造 `socket.AF_UNIX` 路径。
*   **观测数据等级规则**：
    *   `obs_class="control"` 允许直接驱动状态机转移与对齐。
    *   `obs_class="diagnostic"` 仅作为性能指标和调试落盘，严禁参与小车控制闭环决策。

---

## 5. 物理执行层 (Physical Execution Layer)

*   **职责**：将状态机决策的速度指令转换为底层串口物理字节，并向真实的底层串行接口设备进行收发读写。
*   **对应代码**：
    *   `orchestrator/orchestrator_service/control/motion/`
    *   `orchestrator/orchestrator_service/control/docking/`
    *   `orchestrator/orchestrator_service/bridge/uart_bridge.py`
    *   `orchestrator/orchestrator_service/bridge/simple_car_protocol.py`
    *   `orchestrator/orchestrator_service/bridge/arm_protocol.py`
*   **模块分工**：
    *   `simple_car_protocol.py`：负责编解码 `MODE`、`VEL`、`STOP`、`BRAKE` 以及底盘状态回传。
    *   `stop_policy.py`：定义急停与安全缓停的核心物理单位与时序限制。
    *   `motion_adapter.py`：将控制决策的物理速度指令转换为串口协议包。
    *   `uart_bridge.py`：管理底盘串口打开与关闭、异步发送队列、命令覆盖机制以及底层紧急停车信号抑制。
