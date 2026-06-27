# 系统部署与运行手册 (System Runbook)

本手册记录了当前系统的运行机制，主要侧重于如何在正确配置的环境中安全地启动和操作整套服务。

---

## 1. 业务主链路概述 (Scope)

当前车辆运行的默认业务主链拓扑：
```text
微信小程序 / 云端 MQTT ──> mobile_gateway ──> Orchestrator ──> VISTA ──> STM32 / 底盘
```
*注：开发板上的语音 ASR/Voice 模块默认不再作为运行组件启动。`tts_event` 语音输出作为兼容性输出保留，默认不启用。*

---

## 2. Windows 开发环境 (Windows Development)

在 Windows 主机进行开发和代码分析时，请严格遵循以下配置：
*   **Python 解释器路径**：
    ```powershell
    D:\anaconda\Anaconda\envs\embed_sc171\python.exe
    ```
*   **环境变量设置**：
    ```powershell
    $env:PYTHONPATH="D:\55495\workspace\embedded_com;D:\55495\workspace\embedded_com\orchestrator;D:\55495\workspace\embedded_com\VISTA"
    ```
*   **开发原则**：禁止直接使用基线 Anaconda 环境或 MSYS2 环境的 Python，请勿在后台运行未加限制的全局单元测试。

---

## 3. SC171 开发板端启动说明 (SC171 Board Runtime)

板端运行使用专用的端侧 Linux 运行环境和硬件路径。通常采用项目自带的底层启动脚本拉起：
```bash
cd /home/aidlux/embedded_com

# 1. 串口仿真运行方式 (不驱动底盘物理电机)
STACK_PROFILE=dryrun ./start_robot_stack.sh

# 2. 真实底盘驱动启动方式 (串口为 /dev/ttyHS1)
STACK_PROFILE=full UART_DEV=/dev/ttyHS1 ./start_robot_stack.sh

# 3. 检查当前软件栈各进程运行状态
./start_robot_stack.sh status

# 4. 停止所有服务
./start_robot_stack.sh stop
```

---

## 4. 默认端点、通信端口与主题清单 (Ports & Topics)

| 消息/端点 | 传输通道与默认地址 | 说明 |
| :--- | :--- | :--- |
| **MQTT 北向命令** | `robot/v1/SC171/mobile/cmd` | 微信小程序发送的取物与停止指令 |
| **MQTT 北向 ACK** | `robot/v1/SC171/mobile/ack` | 网关对应答的实时推送 |
| **MQTT 状态流** | `robot/v1/SC171/mobile/status` | 小车实时业务状态和用户提示语 |
| **MQTT 心跳包** | `robot/v1/SC171/heartbeat` | 网关在线状态与诊断数据 |
| **南向 `task_cmd`** | `127.0.0.1:9001` 或 `/tmp/robot_stack/task_cmd.sock` | 网关发往 Orchestrator 的状态命令 |
| **南向 `task_ack`** | `127.0.0.1:9012` 或 `/tmp/robot_stack/task_ack.sock` | Orchestrator 应答网关的控制反馈 |
| **南向 `vision_req`** | `127.0.0.1:9003` 或 `/tmp/robot_stack/vision_req.sock` | Orchestrator 调配 VISTA 的感知请求 |
| **南向 `vision_obs`** | `127.0.0.1:9002` 或 `/tmp/robot_stack/vision_obs.sock` | VISTA 给 Orchestrator 的感知数据反馈 |
| **底盘串口** | 板载串口 `/dev/ttyHS1 @ 115200` | SC171 与 STM32 的物理串行链路 |

*注：Windows 仿真开发环境下 IPC 强制回退为 TCP 或 Disable 模式；板端运行环境在 UDS 套接字文件支持时默认优先使用 `AF_UNIX` 以降低系统延迟。*

---

## 5. Orchestrator 状态机状态语义 (Orchestrator States)

### 5.1 桌边停靠主链路状态

| 状态 | 中文说明与角色 |
| :--- | :--- |
| `SEARCH_TABLE` | 原地旋转搜索，调配 VISTA 进入 `FIND_EDGE` 搜寻桌子特征。 |
| `YOLO_ACQUIRE_ALIGN` | 检测到桌子 BBox，在接近前进行粗略的偏角与横向误差对齐。 |
| `YOLO_APPROACH` | 在 YOLO BBox 观测引导下平稳靠近桌子。 |
| `EDGE_ADJUST` | 深度图检测到可信桌边，基于高精度三维几何算法微调车辆姿态。 |
| `FINAL_SLOW_STOP` | 终点减速刹车，并进行多帧稳定判定。 |
| `AT_TABLE_EDGE` | 车辆平稳停靠在桌边，准备执行目标搜寻与抓取。 |

### 5.2 目标定位与抓取主链路状态

| 状态 | 中文说明与角色 |
| :--- | :--- |
| `SEARCH_TARGET_INIT` | 准备沿桌边移动寻找目标物体。 |
| `EDGE_SLIDE_SEARCH` | 车辆沿边横移滑动，调配 VISTA 进入 `FIND_OBJECT` 寻找目标。 |
| `TARGET_CONFIRM` | 视野内发现候选目标，进行连续多帧置信度验证。 |
| `TARGET_LOCKED` | 目标已被锁定。 |
| `FREEZE_BASE` | 锁定底盘，完全限制车辆移动。 |
| `GRASP` | 调配机械臂协作执行抓取。 |
| `DONE` | 任务执行完毕，重置会话并回归 IDLE。 |

*注：其余状态如 `LEAVE_EDGE`（离开桌边）、`RELOCATE_TO_EDGE`（寻找下一条边）、`REACQUIRE_TABLE`（重新捕获桌边）、`AVOID_OBSTACLE`（避障暂停）等为异常恢复和安全性保障状态。*

---

## 6. VISTA 感知模式定义 (VISTA Modes)

*   **`FIND_OBJECT`**：
    *   **输出**：目标位置包 `target_obs`。
    *   **历史兼容**：`TRACK_LOCAL` 会自动映射至此。
*   **`FIND_EDGE`**：
    *   **输出**：桌边误差拟合量 `table_edge_obs`。
    *   **历史兼容**：`DEPTH_PERCEPTION` 和 `TABLE_EDGE_PERCEPTION` 会自动映射至此。

---

## 7. 日志与诊断系统说明 (Logs)

运行时产生的临时日志和 runs 目录均已被 `.gitignore` 排除，避免污染代码库。

*   **运行 Stdout 输出控制镜面**：
    *   `logs/mobile_gateway.out`：移动网关输出。
    *   `orchestrator/logs/orchestrator.out`：Orchestrator 输出。
    *   `VISTA/logs/vision.out`：视觉服务输出。
*   **主要结构化日志文件**：
    *   `timeline.jsonl`：记录状态机转移、报错和里程碑事件。
    *   `ipc.jsonl`：记录北向和南向通信协议消息。
    *   `state_blocks.jsonl`：用于对外部进行状态同步的周期性快照。
    *   `cmd_vel.jsonl`：记录实际下发给底盘物理速度的历史指令。

---

## 8. 端到端本地最小验证流程 (Host Verification)

在推送任何代码修改前，请确保在本地开发机执行了以下校验：
```powershell
# 1. 语法检查
D:\anaconda\Anaconda\envs\embed_sc171\python.exe -m compileall -q common orchestrator\orchestrator_service VISTA\vision_module

# 2. 本地测试集跑通
D:\anaconda\Anaconda\envs\embed_sc171\python.exe -m pytest tests\orchestrator tests\common VISTA\vision_module\test\test_observation_router.py VISTA\vision_module\test\test_stage_contract.py -q --tb=short --disable-warnings
```
确保控制层没有引入未处理的语法错误，且核心通信包没有破坏。
