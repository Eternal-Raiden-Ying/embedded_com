# VISTA

**Vision Intelligent Search & Tracking Assistant**

部署在 QCS6490 端侧设备上的视觉能力服务。VISTA 负责接收上层任务请求，按当前业务目标切换视觉运行模式，完成目标搜索、抓取前观测、回航标记识别等工作，并通过 `vision_obs` 对外持续输出观测结果、动作建议和交互回合状态。

## 架构

VISTA 的核心分层不是“单纯推理循环”，而是：

- `stage`：业务目标层，例如 `SEARCH` / `GRASP` / `RETURN` / `IDLE`
- `mode`：当前实际运行的资源模式，通常同一时刻只有一个 active mode
- backend capability：camera / predictor / depth / network 等底层能力

典型关系：

- `SEARCH` stage 下可以先运行 `TRACK_LOCAL`，后续也可切到 `DEPTH_PERCEPTION`
- `GRASP` stage 下可以在 `GRASP_REMOTE` 和 `MICRO_ADJUST` 间往返
- `RETURN` stage 可复用 `TRACK_LOCAL`，但识别目标与输出语义不同

当前推荐的主线架构：

```
Orchestrator ──vision_req──▶ VistaApp / Stage Controller
                                  │
                                  ▼
                           VisionEngine / Mode Controller
                                  │
                    ┌─────────────┼─────────────┐
                    ▼             ▼             ▼
              CameraManager  PredictorManager  Other Capabilities
                    │             │             │
                    └──── backend/camera / predictor / depth / network

VistaApp ──vision_obs──▶ Orchestrator
```

## Stage 与 Mode

外部协议不直接命令底层 camera 或 model，而是围绕任务会话和交互回合工作。

- `stage` 表达当前业务目标
- `mode` 表达 VISTA 当前选择的运行模式
- `mode` 也承担“当前执行步”的职责，因此当前设计不再单独拆 `phase`
- 同一 `stage` 内允许多次 mode 切换
- mode 切换后可进入短暂冷却期，延迟释放旧资源，避免反复加载带来的开销

当前规划中的 mode 示例：

| Mode | 用途 |
|------|------|
| `TRACK_LOCAL` | 本地 RGB + 本地 NPU 检测/分割，低延迟追踪 |
| `GRASP_REMOTE` | RGB + Depth + 远程抓取推理 |
| `MICRO_ADJUST` | 根据视觉反馈给出位置/角度微调建议 |
| `DEPTH_PERCEPTION` | 深度感知、避障、3D 辅助观测 |

## 目录结构

```
VISTA/
├── vision_module/
│   ├── app/app.py              # 服务入口 / stage 协调层
│   ├── backend/
│   │   ├── vision_engine.py    # 主线 mode 编排引擎
│   │   ├── new_engine.py       # 实验性引擎 / 抓取流程验证
│   │   ├── camera/
│   │   │   ├── HardwareCamera.py
│   │   │   ├── fast_cam.*.so   # GStreamer C++ 扩展（aarch64 预编译）
│   │   │   └── cxx/            # C++ 源码（cam_gst.cpp）
│   │   └── predictor/
│   │       └── QNN_YOLO_Segment_Predictor.py # Qualcomm QNN 推理封装
│   ├── config/board_config.py  # 主配置文件
│   ├── ipc/                    # 协议与传输层
│   ├── model/                  # YOLO 模型（QCS6490 编译版）
│   └── test/                   # 调试工具 / 协议联调脚本
├── grasp_module/               # 抓取模块（预留）
├── tools/
└── logs/ / runs/ / pids/
```

## 模型

| 模型目录 | 用途 |
|----------|------|
| `model/yolo26s-seg/` | 本地检测与分割主模型 |
| `model/yolo26s-seg-grasp/` | 抓取链路相关模型 |
| `model/yolov8s-seg/` | 备用分割模型 |

> 模型为 QCS6490 QNN 2.36 编译版（`.amf` / `.ctx.bin`），仅可在目标硬件上运行。

## 运行

```bash
cd /home/aidlux/2026/VISTA
/usr/bin/python3 -m vision_module.app.app
```

## IPC 协议

| 方向 | 消息类型 | 地址 |
|------|----------|------|
| 接收 | `vision_req` | `127.0.0.1:9003` |
| 发送 | `vision_obs` | `127.0.0.1:9002` |

### `vision_req`

`vision_req` 保留原消息名，但升级为统一请求协议。核心字段建议为：

- `type`: 固定为 `vision_req`
- `ts`
- `session_id`
- `req_id`
- `epoch`
- `op`: `START` / `UPDATE` / `RESPOND` / `STOP`
- `stage`: `SEARCH` / `GRASP` / `RETURN` / `IDLE`
- `target`: 可选，目标类名
- `mode_hint`: 可选，建议优先模式
- `interaction_id`: 可选，响应某个交互回合时使用
- `response`: 可选，对上一个 `vision_obs` 中动作建议的确认/拒绝/反馈
- `payload`: 可选，stage 专属请求体

例 1：启动搜索

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

例 2：启动抓取阶段

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
    "need_depth": true
  }
}
```

例 3：对一次微调建议进行确认并反馈执行结果

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

### `vision_obs`

`vision_obs` 保留原消息名，并统一作为 VISTA 的唯一对外输出 envelope。核心字段建议为：

- `type`: 固定为 `vision_obs`
- `ts`
- `session_id`
- `req_id`
- `epoch`
- `stage`
- `mode`: 当前 active mode
- `status`: `RUNNING` / `WAITING_RESPONSE` / `RESULT_READY` / `DONE` / `FAILED`
- `interaction`: 可选，当前是否需要上层确认或执行动作
- `perception`: 可选，当前感知结果
- `proposal`: 可选，给外部的动作建议、微调量、移动方向、距离等
- `result`: 可选，抓取姿态、空间坐标、阶段性结果等

例 1：搜索观测

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

例 2：抓取前输出微调建议，等待上层确认

```json
{
  "type": "vision_obs",
  "ts": 1710000011.0,
  "session_id": "sess_001",
  "req_id": "req_010",
  "epoch": 1,
  "stage": "GRASP",
  "mode": "MICRO_ADJUST",
  "status": "WAITING_RESPONSE",
  "interaction": {
    "required": true,
    "interaction_id": "ia_007",
    "kind": "MOVE_HINT"
  },
  "proposal": {
    "motion_delta": {
      "dx_m": 0.03,
      "dy_m": -0.01,
      "dyaw_rad": 0.08
    },
    "reason": "target_offset_before_remote_grasp"
  }
}
```

例 3：远程抓取结果返回

```json
{
  "type": "vision_obs",
  "ts": 1710000015.0,
  "session_id": "sess_001",
  "req_id": "req_012",
  "epoch": 1,
  "stage": "GRASP",
  "mode": "GRASP_REMOTE",
  "status": "RESULT_READY",
  "result": {
    "grasp_pose": {
      "x_m": 0.41,
      "y_m": -0.06,
      "z_m": 0.18,
      "yaw_rad": 1.57
    },
    "confidence": 0.87
  }
}
```

## 协议迁移说明

当前仓库里的主服务和 Orchestrator 仍主要使用旧式字段：

- `vision_req(mode=FIND/IDLE)`
- `home_tag_req(mode=RETURN)`
- `target_obs` / `home_tag_obs`

后续主线将迁移到统一的 `vision_req` / `vision_obs` 协议。

迁移策略建议：

- VISTA 内部以新协议为主
- `test/debug_send_req.py` 与 `test/debug_recv_obj.py` 优先支持新协议联调
- legacy 输入适配仅作为过渡能力，不再限制新协议的表达能力

## 硬件依赖

- SoC：Qualcomm QCS6490（NPU 推理必须）
- 摄像头：通过 GStreamer pipeline 采集，依赖 `fast_cam` C++ 扩展（aarch64 预编译 `.so`）
- 深度相机：RealSense（可选，`RealSenseDepthCamera.py`）

## 本地开发说明

`fast_cam.cpython-38-aarch64-linux-gnu.so` 为 ARM 预编译产物，Windows 本地无法直接运行真实视觉模块。

当前建议的本地开发方式：

- mock backend smoke test
- 新协议 sender/receiver 联调
- 在 `new_engine.py` / `test_grasp_only.py` 上验证抓取实验流程

如需重新编译相机扩展，在 AidLux 上执行：

```bash
cd VISTA/vision_module/backend/camera/cxx
mkdir -p build && cd build
cmake .. && make
```
