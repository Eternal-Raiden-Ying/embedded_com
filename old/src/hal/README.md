# HAL（硬件抽象层）说明

本目录原为 HAL 的 Python 实现位置（commit 13487ee）。
经重构，HAL 已迁移至各模块的 `backend/` 目录内，与原有代码结构对齐，本目录仅保留此说明文档。

---

## 一、两次重构工作总结

### commit 13487ee（第一版 HAL）

**新建文件：**
- `src/hal/base.py` — `ICamera` / `IPredictor` 抽象基类
- `src/hal/factory.py` — `is_mock()` 工厂函数
- `src/hal/aidlux/camera.py` — `AidluxCamera`（懒加载 `HardwareCamera`）
- `src/hal/aidlux/predictor.py` — `AidluxPredictor`（懒加载 `QNN_YOLO_Segment_Predictor`）
- `src/hal/mock/camera.py` — `MockCamera`（全零帧）
- `src/hal/mock/predictor.py` — `MockPredictor`（空检测）

**修改文件：**
- `VISTA/vision_module/backend/vision_engine.py` — 移除顶层硬件 import，改为运行时按 `_IS_MOCK` 懒加载
- `orchestrator/orchestrator_service/bridge/uart_bridge.py` — `dry_run` 联动 `ENV=mock`

**问题：** `vision_engine.py` 内嵌了大量 `if _IS_MOCK:` 分支，破坏了原有 import 风格，且 HAL 与业务逻辑耦合。

### 本次重构（第二版 HAL）

**目标：** 将 HAL 内聚到 `backend/camera/` 和 `backend/predictor/` 包内，`vision_engine.py` 恢复原有 import 风格，对 mock/prod 切换完全无感知。

**新建文件：**
- `VISTA/vision_module/backend/camera/base.py` — `ICamera` ABC
- `VISTA/vision_module/backend/camera/mock.py` — `MockCamera`
- `VISTA/vision_module/backend/predictor/base.py` — `IPredictor` ABC
- `VISTA/vision_module/backend/predictor/mock.py` — `MockPredictor`

**修改文件：**
- `VISTA/vision_module/backend/camera/__init__.py` — 改为 HAL 工厂，按 `ENV` 透明导出实现
- `VISTA/vision_module/backend/predictor/__init__.py` — 改为 HAL 工厂，按 `ENV` 透明导出实现
- `VISTA/vision_module/backend/vision_engine.py` — 恢复原有 import 风格，无任何 mock 感知代码

**删除文件：**
- `src/hal/` 下所有 Python 文件（本目录仅保留此 README）

---

## 二、Windows 模拟原理

触发方式：在 Windows 上启动时设置环境变量 `ENV=mock`。

```bash
# Windows CMD
set ENV=mock && python -m vision_module.app.app

# Windows PowerShell
$env:ENV="mock"; python -m vision_module.app.app
```

工作机制：

```
ENV=mock
  └─ camera/__init__.py 读取 ENV
       ├─ _IS_MOCK = True
       └─ from .mock import MockCamera as HardwareCamera
                                       as RealSenseDepthCamera

  └─ predictor/__init__.py 读取 ENV
       ├─ _IS_MOCK = True
       └─ from .mock import MockPredictor as QNN_YOLO_Segment_Predictor
```

`vision_engine.py` 的 import 语句完全不变：

```python
from .camera import HardwareCamera, RealSenseDepthCamera
from .predictor import QNN_YOLO_Segment_Predictor
```

在 `ENV=mock` 时，这三个名字实际指向 `MockCamera` 和 `MockPredictor`，对调用方完全透明。

---

## 三、模拟程度

### 当前：流程级别（Flow-level）

| 组件 | mock 行为 | 实际效果 |
|------|-----------|----------|
| `MockCamera.read_frame()` | 返回全零 640×640×3 uint8 数组 | 视觉流水线正常运转，帧内容为空白 |
| `MockPredictor.predict_frame()` | 返回 `([], [])` | 检测结果永远为空，状态机在 SEARCH 超时后切换到 RETURN |
| `UartBridge`（dry-run） | 指令只打印日志，不发 UART | 串口协议逻辑可验证，不驱动底盘 |
| IPC（TCP 9001/9002/9003） | 完全正常 | 可用 `examples/` 下的 mock sender 注入假数据 |

**可验证的内容：**
- 状态机完整流程（IDLE → SEARCH → APPROACH → RETURN → IDLE）
- IPC 消息收发、ACK 机制、超时处理
- 日志结构（timeline / ipc / state_blocks / jsonl）

**无法验证的内容：**
- 真实目标检测（YOLO 推理结果）
- 底盘运动响应
- 相机帧质量和延迟

### 可升级：资源级别（Resource-level）

只需替换 mock 实现，无需改动任何接口或调用方代码：

**相机升级** — 修改 `backend/camera/mock.py` 读取本地视频或 Windows 摄像头：

```python
import cv2

class MockCamera(ICamera):
    def __init__(self, out_w=640, out_h=640, **kwargs):
        self._cap = cv2.VideoCapture(0)          # Windows 摄像头
        # 或 cv2.VideoCapture("test_video.mp4")  # 本地视频文件

    def read_frame(self):
        ret, frame = self._cap.read()
        return frame if ret else np.zeros((self._h, self._w, 3), dtype=np.uint8)
```

**推理器升级** — 修改 `backend/predictor/mock.py` 使用 ONNX Runtime（CPU）跑 YOLOv8：

```python
import onnxruntime as ort  # pip install onnxruntime，无需 QNN SDK

class MockPredictor(IPredictor):
    def __init__(self, args=None, **kwargs):
        self._sess = ort.InferenceSession("yolov8s.onnx")

    def predict_frame(self, frame):
        # 标准 ONNX 推理，Windows CPU 可运行
        ...
```

---

## 四、对 AidLux 端侧的影响

**零影响。**

- `ENV` 未设置时，`os.environ.get("ENV", "prod")` 返回 `"prod"`，`_IS_MOCK = False`
- `camera/__init__.py` 和 `predictor/__init__.py` 走 `else` 分支，import 路径与重构前完全一致
- `vision_engine.py` 的代码与重构前逐字相同（import 风格、实例化方式均未变）
- `uart_bridge.py` 的 `dry_run` 逻辑：`serial is None` 在 AidLux 上为 `False`，`ENV` 未设置，行为不变

---

## 五、抽象接口

### `ICamera`（`backend/camera/base.py`）

```python
class ICamera(ABC):
    def read_frame(self) -> Optional[np.ndarray]: ...
    # 返回 HxWxC uint8 numpy 数组，失败返回 None 或 size==0 的数组

    def release(self) -> None: ...
    # 释放底层硬件资源

    # 支持 with 语句（__enter__ / __exit__）
```

| 实现 | 平台 | 依赖 |
|------|------|------|
| `HardwareCamera` | AidLux | GStreamer + fast_cam C++ 扩展（aarch64） |
| `RealSenseDepthCamera` | AidLux | Intel RealSense SDK |
| `MockCamera` | Windows | 仅 numpy |

### `IPredictor`（`backend/predictor/base.py`）

```python
class IPredictor(ABC):
    def predict_frame(self, frame: np.ndarray) -> Tuple[list, list]: ...
    # 返回 (out_boxes, masks)，格式与 QNN_YOLO_Segment_Predictor 一致

    def is_ready(self) -> bool: ...
    # 推理器是否就绪（模型已加载、资源未释放）

    def release(self) -> None: ...
    # 释放 NPU/DSP 资源
```

| 实现 | 平台 | 依赖 |
|------|------|------|
| `QNN_YOLO_Segment_Predictor` | AidLux | Qualcomm QNN DSP + aidlite SDK |
| `MockPredictor` | Windows | 无 |

---

## 六、代码插入方式

HAL 通过 `__init__.py` 工厂模式插入，对调用方完全透明：

```
backend/
├── camera/
│   ├── __init__.py            ← 工厂：按 ENV 导出 HardwareCamera / MockCamera
│   ├── base.py                ← ICamera ABC（新增）
│   ├── mock.py                ← MockCamera（新增）
│   ├── HardwareCamera.py      ← 原有，未改动
│   └── RealSenseDepthCamera.py ← 原有，未改动
└── predictor/
    ├── __init__.py            ← 工厂：按 ENV 导出 QNN_YOLO_Segment_Predictor / MockPredictor
    ├── base.py                ← IPredictor ABC（新增）
    ├── mock.py                ← MockPredictor（新增）
    └── QNNPredictor.py        ← 原有，未改动
```

`vision_engine.py` 的 import 语句与重构前完全相同：

```python
from .camera import HardwareCamera, RealSenseDepthCamera   # 透明切换
from .predictor import QNN_YOLO_Segment_Predictor           # 透明切换
```

`ENV=mock` 时三个名字指向 mock 实现；`ENV=prod`（默认）时指向真实硬件实现。调用方代码零修改。
