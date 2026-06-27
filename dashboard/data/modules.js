window.ROBOT_MODULES = {
  "layers": [
    {
      "id": "app_interaction",
      "name": "应用交互层 (Application Interaction Layer)",
      "responsibility": "接收小程序控制指令，发布用户反馈，将北向 MQTT 流量桥接转换为南向状态机命令 (task_cmd)。",
      "directories": [
        "orchestrator/orchestrator_service/mobile_gateway/",
        "docs/mobile_gateway_runbook.md"
      ],
      "inputs": "微信小程序指令 / MQTT 主题",
      "outputs": "task_cmd.sock (UDS / TCP)",
      "risks": "网关掉线、消息丢失、MQTT 与 UDS 协议转换延时、输入校验绕过。",
      "tests": [
        "test_observation_router.py"
      ]
    },
    {
      "id": "task_orchestration",
      "name": "任务编排层 (Task Orchestration Layer)",
      "responsibility": "管理状态机、状态转移、同步视觉感知请求、安全性故障恢复 (safety gate) 与状态导出。",
      "directories": [
        "orchestrator/orchestrator_service/runtime/core.py",
        "orchestrator/orchestrator_service/runtime/states/",
        "orchestrator/orchestrator_service/runtime/safety/"
      ],
      "inputs": "task_cmd, vision_obs, chassis status feedback",
      "outputs": "vision_req (VISTA mode), physical car velocity cmd",
      "risks": "状态转移死锁、丢失帧导致失控、丢帧保护超时机制不合理、急停信号被覆盖。",
      "tests": [
        "test_safety_gating.py",
        "test_emergency_stop.py",
        "test_grasp_reposition.py"
      ]
    },
    {
      "id": "perception_algorithm",
      "name": "感知算法层 (Perception Algorithm Layer)",
      "responsibility": "管理相机与模型后端生命周期，计算目标/桌边观测数据包，控制视觉运行阶段。",
      "directories": [
        "VISTA/vision_module/app/service.py",
        "VISTA/vision_module/app/stages/",
        "VISTA/vision_module/backend/"
      ],
      "inputs": "vision_req, Realsense RGB-D 图像帧",
      "outputs": "vision_obs (target_obs / table_edge_obs)",
      "risks": "推理帧率过低、无物理设备时报错崩溃、YOLO桌边拟合漂移、内存泄露导致服务崩溃。",
      "tests": [
        "test_vision_state_sync.py",
        "test_stage_contract.py"
      ]
    },
    {
      "id": "data_communication",
      "name": "数据通信层 (Data Communication Layer)",
      "responsibility": "定义消息 Schema 规范，管理 IPC 通信协议边界 (UDS / TCP / JSONL)。",
      "directories": [
        "orchestrator/orchestrator_service/ipc/",
        "VISTA/vision_module/ipc/",
        "common/"
      ],
      "inputs": "原始 Python 结构或 Dict",
      "outputs": "Msgpack / JSONL 字节流",
      "risks": "Windows 端不支持 UDS 强行开启导致崩溃、Msgpack 序列化异常、控制观测被诊断消息阻塞。",
      "tests": [
        "test_observation_router.py"
      ]
    },
    {
      "id": "physical_execution",
      "name": "物理执行层 (Physical Execution Layer)",
      "responsibility": "将底盘速度指令转换为底层串口物理字节，发送到 STM32 并执行。处理急停与安全缓停时序。",
      "directories": [
        "orchestrator/orchestrator_service/control/",
        "orchestrator/orchestrator_service/bridge/uart_bridge.py",
        "orchestrator/orchestrator_service/bridge/simple_car_protocol.py"
      ],
      "inputs": "Chassis target speed cmd",
      "outputs": "Chassis UART serial package (MODE, VEL, STOP, BRAKE)",
      "risks": "串口权限缺失、串口被抢占、物理刹车响应慢、未发送 keepalive 导致底盘失联停机。",
      "tests": [
        "test_simple_car_protocol.py"
      ]
    }
  ],
  "domains": [
    {
      "name": "用户交互端",
      "responsibility": "小程序/网页端发起语音或点击取物请求，向用户展示实时车辆运行状态和抓取结果。",
      "components": "微信小程序, Web App, Mobile Gateway"
    },
    {
      "name": "SC171 边缘智能端",
      "responsibility": "运行 VISTA 算法服务进行实时视觉感知对齐，运行 Orchestrator 核心任务编排与状态决策。",
      "components": "VISTA APP, Yolov7 Detector, Orchestrator Service, UART Bridge"
    },
    {
      "name": "STM32 执行控制端",
      "responsibility": "驱动底盘电机转速，进行闭环轮速控制，执行安全刹车，检测碰撞传感器状态并反馈串口数据。",
      "components": "STM32 底盘固件, 串口通信协议解析, 电机驱动"
    },
    {
      "name": "云端 3D 抓取端",
      "responsibility": "配合处理云端复杂场景 3D 点云处理与高维抓取规划（Cloud Grasping）。",
      "components": "Cloud Grasp Server, GR-ConvNet / AnyGrasp Inference"
    }
  ]
};
