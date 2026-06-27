# VISTA 架构设计与外部接口规范说明 (VISTA Architecture, Interfaces & Requirements)

本文档将 VISTA 的系统架构拓扑、上下游接口协议、产品功能要求、桌前平面检测算法（Full / Fast 模式对比）以及动态 ROI 映射逻辑合并整理，作为唯一的开发与架构参考规范。

---

## 1. 总体架构拓扑 (Overall Architecture Topology)

VISTA 是端侧运行的视觉感知模块，主要负责目标检索定位、桌边感知对齐和远程抓取图像采样。其内部运行拓扑如下：

```text
Orchestrator ──(vision_req)──> VistaApp
                                │
                                v
                          StageController ──┬── scheduler.read_result("remote_init_status")
                           |           \    │    (RESPOND 路径，仅限 GRASP 阶段)
                           v            v   │
                      StagePlan      ModeController(scheduler + supervisor)
                           |             |
                           |             | switch_mode() (包含配置和运行时应用)
                           |             |
                           +-------> RuntimeSupervisor
                           |          /   |   |    \        \
                           |         v    v   v     v        v
                           |    Camera Predictor Remote Preview TableEdge (各 Manager 独立线程运行)
                           |       \      |       |      /     /
                           |        +-----+------+------+-----+
                           v                     |
                        Scheduler <─────────────┘ (数据总线摘要交换，不处理原始帧)
                           |
                           v
                 scheduler.collect_tick_input() / StageController.tick()
                           |
                           v
                 VistaApp ──(vision_obs)──> Orchestrator
```

### 1.1 业务阶段 (Stages)
*   `INIT`：服务初始化，向远端服务注册 `/init`。
*   `SEARCH`：本地目标搜寻。
*   `GRASP`：远程抓取协作，状态机为 `GRASP_REMOTE_INIT → GRASP_REMOTE ↔ MICRO_ADJUST`。
*   `RETURN`：返航标志物（Home Tag）检测。
*   `IDLE`：冷待机（无能力消耗）。

### 1.2 活跃感知模式 (Modes)
*   `SILENT`：零资源消耗兜底。
*   `FIND_OBJECT` (旧称 `TRACK_LOCAL`)：RGB + Depth + YOLO 目标检测，输出目标框。
*   `FIND_EDGE` (旧称 `DEPTH_PERCEPTION` / `TABLE_EDGE_PERCEPTION`)：利用 Depth 面板进行桌边拟合，输出边缘误差。
*   `MICRO_ADJUST`：等待控制端微调。
*   `GRASP_REMOTE`：采集 RGB+Depth 并向远程服务端发送 `/predict`。
*   `IDLE_HOT`：彩色相机持续预览热待机。

---

## 2. 上下游 IPC 接口协议 (External IPC Interfaces)

VISTA 与外部（Orchestrator）基于 Unix Domain Socket (UDS) + MessagePack（以大端 4 字节表示长度前缀解决分包粘包）进行二进制高性能通信。

### 2.1 输入请求：`vision_req`
通过端口 `9003`（或套接字 `vision_req.sock`）接收。
*   **核心字段**：
    *   `op`：`START` / `UPDATE` / `RESPOND` / `STOP`。
    *   `stage`：`SEARCH` / `GRASP` / `RETURN` / `IDLE`。
    *   `target`：业务目标名称（如 `bottle`），**不可用于替代 `class_id`**。
    *   `payload.class_id`：远程抓取所用的模型类别整数 ID。
*   **重要约束**：
    *   启动 `GRASP_REMOTE` 时，上游必须显式传入 `payload.class_id`，VISTA 拒绝从 `target` 进行隐式推导。
    *   重试循环归 Orchestrator 所有，VISTA 只负责单次的 `RESPOND -> PREDICT -> 返回结果`。
    *   `/release` 不是单次抓取的必要步骤，仅在服务退出或引擎销毁时由 `RuntimeSupervisor` 触发。

### 2.2 输出观测：`vision_obs`
通过端口 `9002`（或套接字 `vision_obs.sock`）以 5Hz/8Hz 频率对外发布。
*   **核心字段**：
    *   `stage` / `mode`：当前运行的阶段与模式。
    *   `status`：`RUNNING` / `WAITING_RESPONSE` / `RESULT_READY` / `FAILED` / `RELAXING`。
    *   `perception`：感知结果，包含目标框 `target_obs`、桌边线拟合量 `table_edge_obs` 等。
*   **None 值压缩约定**：为节省带宽，输出 JSON 中值为 `None` 的字段默认会被**省略**；但 `yaw_err`、`dist_err`、`obs_ts`、`age_ms`、`frame_id`、`seq` 等关键控制/指标参数即使为 `None` 也会强制保留输出为 `null`。

---

## 3. 桌边拟合检测算法分析 (Table Edge Detector Modes)

VISTA 支持两种深度面拟合模式，分流点位于 `TableEdgeManager._process_depth()` 中：

### 3.1 完整面检测模式 (`full` mode)
*   **核心算法位置**：`VISTA/Online_Edge_Detect/detector.py`。
*   **原理**：利用 ROI 内的完整 Depth 投影生成相机坐标系点云，通过局部像素的相邻梯度计算法向量，过滤出前向立面的候选像素点，最后使用 RANSAC 算法进行三维竖直前平面拟合。平面方程的法向量和距离转为 Bird-view 二维直线，再根据 Crease line（折角趋势线）的斜率突变进行联合加权融合。
*   **特点**：拟合精度高、抗噪强，但 CPU 运算较重。

### 3.2 轻量快速面检测模式 (`fast_plane_only` mode)
*   **核心算法位置**：`VISTA/vision_module/backend/table_edge_manager.py`。
*   **原理**：先在 ROI 内进行稀疏采样（Stride 间隔），直接根据相机的俯仰角和物理安装高度，将点云转到机器人（Robot）坐标系下，依据物理高度区间直接生成高度候选点。接着按 X 轴分 Bin，在局部 Y 轴聚类出“竖直高度有跨度且连续”的支撑柱代表点（Representatives），最终在 Bird-view 二维平面上对这批代表点做加权直线拟合。
*   **特点**：极度轻量，响应迅速，但对干扰物（如地面背景等）需要更精准的 ROI 隔离。

---

## 4. ROI 动态映射补丁规范 (Simplified ROI Mapping)

为了给 `fast_plane_only` 算法提供干净的深度点云输入，引入了简化后的 bbox-size 动态映射机制：

1.  **映射策略 (`centered_scale`)**：
    将 RGB 图中的 YOLO 桌子 BBox 根据相机的视场角（FOV）比值转换到深度图对应的坐标：
    $$\text{depth\_x\_norm} = 0.5 + \text{offset\_x} + (\text{rgb\_x\_norm} - 0.5) \times \text{scale\_x}$$
    如果 RGB 视野是 Depth 视野中心的 75%，则 `scale_x=0.75`。
2.  **尺寸缩放**：
    最终深度图上的 ROI 大小为映射后的 BBox 尺寸乘以 `yolo_table_roi_scale_x/y`（默认 `0.50`，做适当收缩，聚焦于桌面前立面）。
3.  **YOLO 丢失保持 (Hold)**：
    当 YOLO 短暂丢帧时，保持最后一次有效的 BBox 以继续提供 ROI 裁剪。当丢帧数超过 `yolo_table_bbox_hold_frames`（默认 `8` 帧）时，直接置 `roi_source=disabled_no_table_bbox`，拦截无效的桌边对齐，促使控制层回归原地旋转搜索。
