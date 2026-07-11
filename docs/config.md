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
    *   指定当前激活的运行 Profile，并承载 Orchestrator/VISTA 的 canonical 运行参数。
3.  **运行 Profile (Profiles)**
    *   相关文件：
        *   `configs/profiles/windows_dev.yaml`：用于 Windows 主机仿真调试。
        *   `configs/profiles/sc171_board.yaml`：用于 SC171 真实板端运行。
        *   `configs/profiles/dry_run.yaml`：用于串口 Dry-run 机制的本地测试。
    *   用于对 Windows 开发机、真实板端及模拟运行进行环境变量与网络端点的分流。
4.  **运行时最后一步覆盖 (Runtime Overrides)**
    *   `runtime_overrides` 用于在主配置内固定当前 baseline 的最终值。
    *   `stage_params.yaml` 与 `car_cmd_params.yaml` 已删除，不再参与默认加载链。

---

## 2. 核心配置文件与参数修改位置 (Key Files & Configuration Tuning)

### 2.1 状态机与停靠控制参数 (`configs/system_config.yaml`)
修改以下内容时请调整 `orchestrator.control.*`：
*   桌边停靠的距离、yaw 偏角以及横向偏差的阈值和稳定判定时间。
*   沿桌边滑动寻找目标的超时限制、确认目标稳定的帧数。
*   抓取流程中的微调超时与动作执行参数。
*   丢帧保护时间（stale observation guards）及异常恢复重试次数。

### 2.2 底盘命令参数 (`configs/system_config.yaml`)
修改以下内容时请调整 `orchestrator.car.*`：
*   串口发送周期。
*   速度命令的保持时间（keepalive duration）。
*   小车前进/后退/横移/旋转的绝对速度限幅。
*   在进入新状态时的强制停车/刹车控制策略。

### 2.3 系统运行 Profile (`configs/system_config.yaml`)
修改以下内容时请调整此文件：
*   切换当前激活的 profile（如 `windows_dev` / `sc171_board`）。
*   修改 Windows 和板端的 IPC 通信端点默认配置（如 UDS 或 TCP 接口路径）。
*   修改 Orchestrator、VISTA 与 Gateway 的运行路径和通信端点。

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
loaded config files  : configs/system_config.yaml, configs/profiles/<profile>.yaml
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
