# 配置系统指南 (Configuration Guide)

本文档说明了机器人软件栈（Orchestrator 与 VISTA）的配置层级模型、关键配置文件以及运行时配置覆盖方式。

---

## 1. 配置层级 (Configuration Layers)

配置系统采用多层叠加模型，以兼顾保守默认值、特定运行环境（主机开发/板端部署）及运行时调参的需求：

1.  **Dataclass 基础默认值 (Schema Defaults)**
    *   代码位于 [common/config/schema.py](file:///d:/55495/workspace/embedded_com/common/config/schema.py)。
    *   定义了配置的数据结构与安全底线值。这部分默认值非常保守，不应直接作为实际运行参数。
2.  **项目主配置 (Project Config)**
    *   主配置文件：[configs/system_config.yaml](file:///d:/55495/workspace/embedded_com/configs/system_config.yaml)。
    *   指定了当前激活的运行 Profile，以及各底层运行参数文件的路径。
3.  **运行 Profile (Profiles)**
    *   相关文件：
        *   `configs/profiles/windows_dev.yaml`：用于 Windows 主机仿真调试。
        *   `configs/profiles/sc171_board.yaml`：用于 SC171 真实板端运行。
        *   `configs/profiles/dry_run.yaml`：用于串口 Dry-run 机制的本地测试。
    *   用于对 Windows 开发机、真实板端及模拟运行进行环境变量与网络端点的分流。
4.  **可调运行时参数文件 (Tunable Runtime Files)**
    *   相关文件：
        *   `orchestrator/configs/stage_params.yaml`：状态机运行参数。
        *   `orchestrator/configs/car_cmd_params.yaml`：底盘运动参数。
    *   **开发原则**：所有需要现场调试的参数（如阈值、速度上限、控制频率等）必须放在这些 YAML 文件中，禁止在 Python 代码里硬编码。
5.  **运行时最后一步覆盖 (Runtime Overrides)**
    *   支持通过系统环境变量进行最高优先级覆盖，在程序启动打印中会体现最终的有效生效值。

---

## 2. 核心配置文件与参数修改位置 (Key Files & Configuration Tuning)

### 2.1 状态机运行时参数 (`orchestrator/configs/stage_params.yaml`)
修改以下内容时请调整此文件：
*   桌边停靠的距离、yaw 偏角以及横向偏差的阈值和稳定判定时间。
*   沿桌边滑动寻找目标的超时限制、确认目标稳定的帧数。
*   抓取流程中的微调超时与动作执行参数。
*   丢帧保护时间（stale observation guards）及异常恢复重试次数。

### 2.2 底盘命令参数 (`orchestrator/configs/car_cmd_params.yaml`)
修改以下内容时请调整此文件：
*   串口发送周期（默认 `ORCH_CAR_SEND_PERIOD_MS = 50`）。
*   速度命令的保持时间（keepalive duration）。
*   小车前进/后退/横移/旋转的绝对速度限幅。
*   在进入新状态时的强制停车/刹车控制策略。

### 2.3 系统运行 Profile (`configs/system_config.yaml`)
修改以下内容时请调整此文件：
*   切换当前激活的 profile（如 `windows_dev` / `sc171_board`）。
*   修改 Windows 和板端的 IPC 通信端点默认配置（如 UDS 或 TCP 接口路径）。
*   重定向运行时参数文件的实际存放位置。

---

## 3. 启动打印与有效配置信息 (Effective Dump)

为了防止调试过程中由于覆盖关系导致错误参数静默生效，系统在启动时必须显式向 Stdout 打印出所有关键配置的最终真值（Effective Config Dump），内容应包含：

*   当前激活的运行 profile。
*   实际加载的配置文件路径列表。
*   串口设备名称及是否启用 Dry-run。
*   状态机心跳频率（`tick_hz`）。
*   桌边停靠接近的距离界限（`near_stop_depth_m`）。
*   沿边搜索横向速度与纵向速度上限（`edge_slide_vy_mps`、`edge_slide_max_vx_mps`）。
*   终点锁定阈值（yaw / dist / lateral 门限）。
*   急停与减速刹车控制策略（STOP/SSTOP policy）。
*   视觉观测反馈间隔。

**标准的启动日志格式示例：**
```text
loaded stage_params  : orchestrator/configs/stage_params.yaml
loaded car_cmd_params: orchestrator/configs/car_cmd_params.yaml
edge_slide_vy_mps    : 0.010
```

---

## 4. 安全机制与危险默认值校验 (Dangerous Defaults Guard)

校验层必须在小车物理移动前拦截不安全或模糊的配置选项：
1.  **横移速度校验**：检查 `edge_slide_vy_mps` 不能静默使用历史残留的危险高速度 `0.14 m/s`。
2.  **停靠策略区分**：必须严格区分 `STOP`（急停/清空命令）与 `SSTOP`（安全缓停）在底盘协议中的编解码实现。
3.  **多平台传输限制**：在 Windows 主机下禁止静默使用 UNIX Domain Socket (UDS) 传输，必须自动 fallback 到 TCP 通信，除非显式指明了其他模拟策略。
4.  **物理串口警告**：当 `ORCH_SERIAL_DRY_RUN=1` 时，必须以明显的高亮日志告知操作员当前处于串口仿真模式下。

---

## 5. VISTA 视觉配置要点 (VISTA Config Notes)

VISTA 的感知模式参数：
*   `FIND_OBJECT`：用于目标位置搜索，输出 `target_obs`。
*   `FIND_EDGE`：用于桌边感知与对齐，输出 `table_edge_obs`。
*   *兼容别名说明*：`TRACK_LOCAL` 会自动规格化映射至 `FIND_OBJECT`；`DEPTH_PERCEPTION` 会自动规格化映射至 `FIND_EDGE`。
*   **路由控制隔离**：视觉数据总线（`Scheduler`）必须确保控制相关的观测消息（`obs_class="control"`）具有独立优先级，不能被诊断或指标类消息（`obs_class="diagnostic"`）阻塞。
