# VISTA 整体框架与层级设计

## 1. 设计目标

VISTA 的整体架构目标是：

- 将硬件能力、算法能力、网络能力与业务控制解耦
- 支持按模式组合资源，而不是靠大量布尔开关拼接
- 让调试能力成为旁路，而不是污染正式运行路径
- 为后续抓取、避障、导航、微调等扩展预留稳定接口

## 2. 总体层级

建议按以下层级组织：

### 2.1 Hardware Backend Layer

职责：

- 提供最底层的硬件访问封装
- 管理真实设备初始化、读帧、释放资源

当前包含：

- `ColorCamera`
- `IRCamera`
- `RealSenseDepthCamera`
- `QNN_YOLO_Segment_Predictor`

特点：

- 面向能力，不感知上层业务模式
- 提供统一接口，例如 `read_frame()`、`predict_frame()`、`release()`

### 2.2 Capability Manager Layer

职责：

- 在 backend 之上做资源生命周期管理
- 对外暴露“相机能力”“推理能力”“深度能力”“网络能力”

建议拆分：

- `CameraManager`
- `PredictorManager`
- `DepthCapability` 或 `DepthProcessor`
- `NetworkGraspClient`
- `PreviewSink`

特点：

- 管理资源创建、复用、释放
- 避免上层直接操作底层 backend 类

### 2.3 Mode Orchestration Layer

职责：

- 根据任务目标决定当前需要打开哪些能力
- 负责模式切换与差量更新

建议核心对象：

- `ModeProfile`
- `ModeController`

`ModeProfile` 负责描述：

- 哪些 camera 开启
- 使用哪套相机 profile
- predictor 是否开启
- 使用哪个模型
- network 是否开启
- preview 是否开启
- 运行频率配置

`ModeController` 负责：

- 当前 mode 的保存
- 新 mode 的应用
- 对比当前资源状态和目标状态
- 只调整变化部分

### 2.4 Application / Task Layer

职责：

- 处理任务状态机与业务流程
- 决定何时切 mode，何时请求抓取，何时做微调

示例：

- 搜索目标
- 接近目标
- 抓取拍照
- 远程抓取预测
- 微调位姿
- 失败重试

特点：

- 这里处理“为什么切换”
- 不直接决定底层相机如何初始化

## 3. 建议的资源管理结构

### 3.1 CameraManager

职责：

- 保存当前已打开的 camera 实例
- 支持 `ensure_camera(name, profile)`
- 支持 `disable_camera(name)`
- 支持 camera profile 比较
- 在 profile 未变化时复用相机实例

为什么要独立：

- 相机初始化和重建代价高
- 相机在多个模式中会被复用

### 3.2 PredictorManager

职责：

- 管理 `QNN_YOLO_Segment_Predictor`
- 支持 `ensure_model(name)`
- 支持 `disable_model()`
- 避免同模型重复加载

为什么要独立：

- NPU 模型加载有明显成本
- 不是所有模式都需要 predictor

### 3.3 PreviewSink

职责：

- 仅在 debug 时订阅最新帧并显示
- 不参与 mode 核心逻辑

为什么要独立：

- 预览只是调试能力
- 关闭 preview 时应零侵入主流程

## 4. 模式设计

### 4.1 TRACK_LOCAL

资源组合：

- `rgb`
- local predictor
- preview 可选

典型参数：

- RGB 640x640
- 本地高频循环
- 低延迟

适用场景：

- 目标搜索
- 目标跟踪
- 循迹

### 4.2 GRASP_REMOTE

资源组合：

- `rgb`
- `depth`
- network grasp client
- local predictor 默认关闭

典型参数：

- RGB 1280x720
- 同步 depth
- 按需低频触发网络请求

适用场景：

- 抓取前观测
- 远程抓取姿态预测

### 4.3 待扩展模式

#### DEPTH_PERCEPTION

- 面向碰撞箱、点云、避障、导航

#### MICRO_ADJUST

- 面向抓取后微调
- 根据视觉反馈做位置纠正

## 5. 模式切换原则

模式切换必须遵循差量更新：

- 不变化的相机不重建
- 不变化的模型不重载
- 只关闭当前 mode 不再需要的能力
- 只开启目标 mode 新增的能力

错误示例：

- 每次切 mode 都全关全开

推荐示例：

- `TRACK_LOCAL -> GRASP_REMOTE`
  - 保留 `rgb`
  - 补开 `depth`
  - 关闭 local predictor
  - 打开 network

## 6. 当前代码落点建议

### 6.1 `vision_engine.py`

建议定位：

- 主线正式 engine
- 优先作为模式化引擎重构目标

先做：

- `ModeProfile`
- `ModeController`
- `CameraManager`
- `PredictorManager`
- 两个模式：`TRACK_LOCAL`、`GRASP_REMOTE`

### 6.2 `new_engine.py`

建议定位：

- 试验性 / develop 路径
- 可以继续做新模式和云抓取流程验证
- 但长期不应和主线引擎分叉过大

## 7. 建议的开发顺序

1. 统一 backend 接口与命名
2. 定义 `ModeProfile`
3. 实现 `CameraManager`
4. 实现 `PredictorManager`
5. 在 `vision_engine.py` 中加入 `ModeController`
6. 落地 `TRACK_LOCAL`
7. 落地 `GRASP_REMOTE`
8. 将预览改为旁路能力
9. 再接上更高层 app / IPC / 任务状态机

## 8. 扩展性建议

未来新增模式时，不应继续增加大量：

- `enable_xxx`
- `use_depth`
- `use_network`
- `use_local_model`

而应新增：

- 一个新的 `ModeProfile`
- 必要时新增一个新的 capability manager

这样：

- 产品需求变化时更容易适配
- 代码结构更稳定
- 端侧资源控制更清晰
