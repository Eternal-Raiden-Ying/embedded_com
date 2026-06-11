# 视觉感知模块源码结构说明 (VISTA Structure)

本轮结构清理工作在保持当前运行时布局（Runtime Layout）完全不被破坏的前提下，为调试脚本、ROI 辅助逻辑和未来手动控制工具划分了更清晰的文件归宿。

---

## 1. 当前各目录职责分工 (Current Responsibilities)

*   **`app/`**：服务主入口，包含 `VistaApp` 核心主循环、`StageController` 业务层控制以及各业务阶段计划（`StagePlan`）。
*   **`backend/`**：运行时各组件管理器（Managers）以及硬件/算法适配驱动。
*   **`backend/camera/`**：相机后端，包括彩色 RGB 相机、深度相机、仿真 Mock 相机以及适配板端 GStreamer 加速的硬件通道。
*   **`backend/predictor/`**：QNN 神经网络推理后端、Mock 推理器以及 YOLO 模型推理的后处理逻辑。
*   **`backend/preview/`**：用于界面调试的 OpenCV 预览窗口以及在后台 Concises 模式下静默丢弃画面的 Null 接收端。
*   **`backend/remote/`**：远程云端抓取协作的 HTTP 交互客户端。
*   **`backend/edge_detect/`**：在线几何桌边检测算法核心驱动（包含 `OnlineTableEdgeDetector`、标定加载、默认参数）。
*   **`config/`**：板端默认硬件配置参数（`board_config`）、数据 Schema 定义以及各个感知模式的默认配置。
*   **`ipc/`**：负责 UDS/TCP 的 MessagePack 通信传输帧解析及 `vision_req`/`vision_obs` 封包定义。
*   **`diagnostics/`**：控制台调试频率限制器与系统诊断快照辅助。
*   **`model/`**：存储 QNN 模型二进制文件及转换辅助资源。
*   **`test/`**：自动化单元/集成测试集，以及历史遗留的手动调试脚本。
*   **`tools/`**：未来的手动调试脚本、单例物理验证工具的集中存放地。
*   **`utils/`**：通用的跨组件工具函数。

---

## 2. 结构调整兼容性底线规则 (Compatibility Rules)

*   **严禁大范围一次性迁移核心运行时代码**：必须确保原有业务导包路径不发生大规模破坏。
*   **平滑迁移**：在将特定的测试脚本移动到 `tools/` 时，必须在 `test/` 目录下保留旧导入路径的兼容垫片（Shim）。
*   **协议持久稳定**：结构调整过程中，决不能对 `vision_req` 和 `vision_obs` 暴露给 Orchestrator 的任何协议字段进行删改。
*   **功能解耦**：切勿在目录清理时修改底层的状态机、Scheduler 或 Stage 转义逻辑。

---

## 3. 长期演进架构设计建议 (Long-Term Direction)

*推荐采用以下目录架构组织形式，但本次清理暂不做强制迁移：*
*   `runtime/`：收拢 `scheduler`（数据总线）、`mode_controller`（模式控制）以及 `runtime_supervisor`（能力调配）。
*   `perception/`：整合 `predictor`（本地推理）、`table-edge`（几何感知）与 `remote`（网络感知）。
*   `io/`：整合 `camera`（图像源）、`IPC`（外部协议）与 `preview`（画面输出）。
*   `diagnostics/`：整合系统控制台、诊断快照和轨迹 Dump。
*   `tools/`：汇集操作员调试脚本。
*   `tests/`：汇集自动化与回归测试集。
