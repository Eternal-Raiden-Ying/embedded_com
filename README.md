# 嵌入式机器人软件栈 (Embedded Robot Stack)

本仓库包含嵌入式机器人软件栈的 Windows 开发代码和 SC171 板端运行代码。

## 运行架构布局 (Runtime Layout)

```text
移动端 App / MQTT
        |
        v
orchestrator_service.mobile_gateway
        |
        v
Orchestrator  <---- vision_obs ----  VISTA
        | ---- vision_req ---->       |
        v
STM32 底盘 + 机械臂串口协议
```

主要目录：

- `common/`: 共享的配置加载器、Schema、校验、日志以及协议助手。
- `configs/`: 项目配置与运行时 Profile。
- `orchestrator/`: 任务编排、状态运行时、底盘控制、UART 桥接以及移动端网关。
- `VISTA/`: 视觉服务、Stage 控制器、观测路由器、摄像头/模型后端。
- `tests/`: 仅限必要的主机测试。
- `scripts/manual/`: 手动、历史或交互式脚本。
- `docs/`: 架构、配置、测试和运行手册（Runbook）文档。

## Python 环境 (Python Environments)

Windows 开发必须使用：

```powershell
D:\anaconda\Anaconda\envs\embed_sc171\python.exe
```

SC171 板端运行时使用设备上安装的板端 Python 环境。请勿将 Windows 开发命令指向板端路径。

请勿使用 Base Anaconda、MSYS2 Python 或 `PATH` 中找到的任意 `python`。

## Windows PYTHONPATH

在运行主机检查前设置此环境变量：

```powershell
$env:PYTHONPATH="D:\55495\workspace\embedded_com;D:\55495\workspace\embedded_com\orchestrator;D:\55495\workspace\embedded_com\VISTA"
```

## 最小主机验证 (Minimal Host Verification)

使用针对性的测试。请勿在后台运行完整的 pytest。

```powershell
D:\anaconda\Anaconda\envs\embed_sc171\python.exe -m compileall -q common orchestrator\orchestrator_service VISTA\vision_module

D:\anaconda\Anaconda\envs\embed_sc171\python.exe -m pytest tests\orchestrator tests\common VISTA\vision_module\test\test_observation_router.py VISTA\vision_module\test\test_stage_contract.py -q --tb=short --disable-warnings
```

## 当前状态与模式名称 (Current State And Mode Names)

当前的 Orchestrator 停靠路径：

```text
SEARCH_TABLE
YOLO_ACQUIRE_ALIGN
YOLO_APPROACH
EDGE_ADJUST
FINAL_SLOW_STOP
AT_TABLE_EDGE
```

当前的 VISTA 搜索模式（search modes）：

- `FIND_OBJECT`: 目标搜索与目标观测路径。
- `FIND_EDGE`: 桌边观测路径。

仅为了兼容性而接受的遗留别名：

- `TRACK_LOCAL` -> `FIND_OBJECT`
- `DEPTH_PERCEPTION` -> `FIND_EDGE`

## 修改代码的位置 (Where To Change Things)

- 配置默认值与校验: `common/config/`
- 项目 Profile 与运行时配置: `configs/`
- Orchestrator 状态/运行时流程: `orchestrator/orchestrator_service/runtime/`
- 基础运动、停止策略（STOP policy）、速度限制: `orchestrator/orchestrator_service/control/`
- UART 与协议编码: `orchestrator/orchestrator_service/bridge/`
- VISTA 生命周期与阶段（stages）: `VISTA/vision_module/app/`
- VISTA 观测路由: `VISTA/vision_module/app/observation/`
- VISTA 搜索阶段（search stage）: `VISTA/vision_module/app/stages/search/`
- 手动脚本: `scripts/manual/`

另请参阅：

- `docs/architecture.md`
- `docs/config.md`
- `docs/testing.md`
- `docs/system_runbook.md`
