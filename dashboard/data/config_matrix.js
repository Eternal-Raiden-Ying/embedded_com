window.ROBOT_CONFIG_MATRIX = {
  "layers": [
    {
      "level": 1,
      "name": "Schema Defaults (Dataclass 默认值)",
      "file": "common/config/schema.py",
      "description": "系统定义的最底层安全兜底默认配置。参数非常保守，主要用于防止缺少配置键时代码直接崩溃。不推荐在真实运行时直接使用。"
    },
    {
      "level": 2,
      "name": "Project Config (项目主配置)",
      "file": "configs/system_config.yaml",
      "description": "项目的引导配置文件。它定义了当前激活的系统运行 Profile (例如 sc171_board 或 windows_dev) 以及其他各子配置文件所在的相对路径。"
    },
    {
      "level": 3,
      "name": "Profiles (环境配置文件)",
      "file": "configs/profiles/{windows_dev, sc171_board, dry_run}.yaml",
      "description": "区分运行环境的核心配置。决定是连接真实串口还是使用模拟串口，决定 IPC 传输使用 TCP 还是 UNIX Domain Sockets (UDS)。"
    },
    {
      "level": 4,
      "name": "Tunable Runtime Files (可调运行时参数)",
      "file": "orchestrator/configs/stage_params.yaml & car_cmd_params.yaml",
      "description": "现场调参文件。所有容易变化的控制参数（距离对齐门限、最大速度限制、超时时序、串口发送频率等）均须在此声明，严禁在代码中写死。"
    },
    {
      "level": 5,
      "name": "Environment Overrides (系统环境变量覆盖)",
      "file": "$env:ORCH_SERIAL_DRY_RUN, $env:SYSTEM_CONFIG_FILE 等",
      "description": "启动时的最后一步强制覆盖。具有最高优先级，主要用于调试及多实例测试。"
    },
    {
      "level": 6,
      "name": "Runtime Effective Dump (运行时生效快照)",
      "file": "stdout (运行日志头部 dump)",
      "description": "程序加载完毕后在控制台打印的最终参数清单，用于确认覆盖生效后的真值，是排除配置疑难问题的关键线索。"
    }
  ],
  "parameters": [
    {
      "name": "edge_slide_vy_mps",
      "description": "小车沿桌边滑动寻找物体的横移绝对速度上限。",
      "unit": "m/s (米/秒)",
      "board_val": "0.010",
      "dev_val": "0.010",
      "dangerous_threshold": ">= 0.140",
      "danger_reason": "历史残留高速度 0.140m/s 在狭窄桌边物理运行时极易因惯性滑出或发生碰撞，严禁在板端静默使用。"
    },
    {
      "name": "serial.dry_run",
      "description": "是否开启串口 Dry-run 模拟。若为 1 则不向真实底盘下发速度数据，只在标准输出打印速度字节。",
      "unit": "boolean (0 / 1)",
      "board_val": "0 (物理连接)",
      "dev_val": "1 (模拟运行)",
      "dangerous_threshold": "1 (当在板端实地运行时)",
      "danger_reason": "在真实物理跑车阶段，如果 dry_run 依然为 1，车辆将无法接收任何实际控制指令，处于静止状态。"
    },
    {
      "name": "vision_req transport",
      "description": "VISTA 接收控制编排请求的 IPC 传输协议类型。",
      "unit": "enum (uds / tcp / disabled)",
      "board_val": "uds",
      "dev_val": "tcp 或 disabled",
      "dangerous_threshold": "uds (当在 Windows 开发主机上运行时)",
      "danger_reason": "Windows 平台不支持标准的 Unix Domain Socket 句柄连接，若强行指定为 uds 会导致服务无法启动。Windows 端应自动 fallback 为 tcp 通信。"
    },
    {
      "name": "task_cmd transport",
      "description": "Mobile Gateway 发送小程序指令至任务状态机的 IPC 传输协议类型。",
      "unit": "enum (uds / tcp)",
      "board_val": "uds",
      "dev_val": "tcp",
      "dangerous_threshold": "N/A",
      "danger_reason": "需确保 UDS Socket 文件的拥有者及读写权限配置正确，否则以普通用户权限运行的 Gateway 无法连接以 root 权限运行 of Orchestrator。"
    },
    {
      "name": "STOP/SSTOP policy",
      "description": "刹车停靠策略。STOP 为急停（物理串口强制编码 0x02），SSTOP 为安全缓停（物理串口编码 0x03，配合小车平滑减速）。",
      "unit": "policy definition",
      "board_val": "STOP 优先抢占",
      "dev_val": "STOP 优先抢占",
      "dangerous_threshold": "SSTOP 丢失帧保护超时时间设得过长 (> 5.0s)",
      "danger_reason": "缓停策略如果在观测通道出现大面积延时或丢帧时没有及时触发（比如 keepalive stale 周期过长），会导致小车在传感器致盲情况下继续前行而撞击目标。"
    },
    {
      "name": "control_obs interval",
      "description": "视觉观测消息路由至状态机决策闭环的周期上限。",
      "unit": "ms (毫秒)",
      "board_val": "< 50",
      "dev_val": "N/A",
      "dangerous_threshold": ">= 100",
      "danger_reason": "如果控制观测包的发送间隔过大，状态机对桌边对齐偏差的反应速度会严重滞后，引发小车运动振荡或超调。"
    }
  ]
};
