# 机器人桌边停靠（Docking）感知与控制逻辑重构说明

为了提升机器人小车靠近桌子（Docking）时的安全性、平滑性及控制鲁棒性，我们对感知与状态机控制逻辑进行了深度重构。本次重构共分为 4 个阶段，且所有更改已全部提交到 Git 仓库。

---

## 核心修改内容说明

### 阶段 1：纯深度安全停靠与减速拦截 (P4 Pure Depth Safety Stop)
- **视觉感知模块**：
  - 在 `table_edge_manager.py` 的深度图处理中，裁剪出底部安全 ROI（底部 40% 高度，中间 80% 宽度）。
  - 过滤无效像素（深度值应大于 0.01m），计算 `depth_p10`（第 10 百分位数深度）和 `close_depth_ratio`（小于 0.40m 的像素比例）。
  - 将这些指标传递至 `TableEdgeObservation` 和通信协议 payload。
- **状态机控制模块**：
  - 在 `state_machine.py` 中实现了 `_apply_depth_safety_logic()` 拦截逻辑，并在主循环 `tick()` 返回决策前对其进行拦截过滤。
  - **紧急停靠**：若 `depth_p10` 小于 `near_stop_depth_m` (默认 0.25m)，强制设定 `vx = 0.0, wz = 0.0`。
  - **主动减速**：若 `depth_p10` 小于 `near_slow_depth_m` (默认 0.40m)，限制最大前进线速度为 `0.010 m/s`，最大角速度为 `0.04 rad/s`。
  - **避让防锁死保护**：以上安全限制仅在机器人向前运动 (`vx_mps > 0`) 或原地旋转时生效，确保在后退逃逸（如 `DOCK_RETRY` 或 `LEAVE_EDGE`）时机器人不会因距离过近而被锁死在原地。

### 阶段 2：最终锁边稳定迟滞与速度分段控制 (P5 Final Stop Stabilization & Speed Segmentation)
- **最终停靠稳定迟滞 (Hysteresis)**：
  - 在 `FINAL_LOCK` 状态中，引入了最少停留时间与丢失缓冲门限。
  - **最少停留**：进入 `FINAL_LOCK` 后，至少保持该状态 `final_lock_min_hold_ms` (800ms)，期间不响应任何异常回退或丢失切换。
  - **丢失缓冲**：若在 `FINAL_LOCK` 中丢失桌边视野，需持续丢失超过 `final_lock_lost_timeout_ms` (1000ms) 才会触发 fallback 退出，有效抵抗视觉噪声引起的瞬时抖动。
- **线速度分段式策略 (Speed Segmentation)**：
  - 取代原有的连续比例速度控制，根据与桌边的相对距离对前进线速度 `vx` 进行分段定速：
    - **远距离 (Far - YOLO搜索/无精确距离信息)**：线速度固定为 `0.015 m/s`。
    - **中距离 (Medium - 正常接近阶段 `dist_err > 0.15m`)**：线速度固定为 `0.020 m/s`。
    - **近距离 (Close - 停靠微调阶段 `dist_err <= 0.15m`)**：线速度固定为 `0.008 m/s`。

### 阶段 3：YOLO 对齐与接近阶段 (P3 YOLO Acquire Align Stage)
- **新引入状态**：在状态机中加入了 `YOLO_ACQUIRE_ALIGN`（YOLO 图像对齐）与 `YOLO_APPROACH`（YOLO 默认接近）状态。
- **对齐流程重构**：
  - 在 `SEARCH_TABLE` 状态中，一旦检测到桌子 bbox，通过辅助方法计算其归一化中心位置 `bbox_center_x_norm`。
  - **若未对齐**：若中心位置在 `[0.35, 0.65]` 范围之外，状态转移至 `YOLO_ACQUIRE_ALIGN`。在此状态下小车原地旋转对齐（`vx = 0`，利用 YOLO 偏差输出 `wz`），直到中心进入 `[0.35, 0.65]`。
  - **若已对齐**：中心在 `[0.35, 0.65]` 内时，状态转移至 `YOLO_APPROACH`，执行直行默认接近。
  - 机器人保持 `YOLO_APPROACH` 接近，直至获取到可靠的几何桌边（`edge_trusted` 为 True），才转移至下一阶段的 `COARSE_ALIGN`。
  - 状态数据导出的 status 块中暴露了 `yolo_acquire_align_active` 状态指标。

### 阶段 4：无进展超时机制 (P6 No-progress Timeout)
- **超时机制重构**：
  - 移除了所有在接近阶段硬编码的绝对计时超时 (`approach_timeout_s`)。
  - 在 `RuntimeContext` 引入了 `min_dist_seen`（记录曾到达的最近桌边距离）和 `dist_progress_last_refreshed_mono`（最后一次进度刷新的单调时间戳）并在状态切换和清理计数器时自动初始化与重置。
  - 引入了 `_check_approach_progress(obs)` 检测函数：如果在 `progress_window_ms` (5000ms) 内，机器人与桌子的距离没有减小至少 `2mm`（`0.002m`），则判定小车打滑或受阻，主动转移至 `DOCK_RETRY` 状态退回并尝试重新停靠。

---

## 修改的文件列表

1. **[schema.py](file:///d:/55495/workspace/embedded_com/orchestrator/orchestrator_service/config/schema.py)**：添加了深度安全停靠阈值、锁边迟滞时间及无进展检测窗口大小配置。
2. **[vision_semantics.py](file:///d:/55495/workspace/embedded_com/VISTA/vision_module/backend/vision_semantics.py)**：在 `TableEdgeObservation` 实体中扩展了安全 ROI 计算结果字段。
3. **[table_edge_manager.py](file:///d:/55495/workspace/embedded_com/VISTA/vision_module/backend/table_edge_manager.py)**：实现了安全 ROI 裁剪、深度缩放及 `depth_p10`/`close_depth_ratio` 的实时统计。
4. **[protocol.py](file:///d:/55495/workspace/embedded_com/orchestrator/orchestrator_service/ipc/protocol.py)**：更新了 `TableEdgeObs` 协议字段及其字典反序列化解析。
5. **[context.py](file:///d:/55495/workspace/embedded_com/orchestrator/orchestrator_service/runtime/context.py)**：在 `State` 枚举中加入了 `YOLO_ACQUIRE_ALIGN` 与 `YOLO_APPROACH`，在 `RuntimeContext` 中引入并管理无进展检测的内部状态。
6. **[motion_controller.py](file:///d:/55495/workspace/embedded_com/orchestrator/orchestrator_service/control/motion_controller.py)**：重构了 YOLO 引导和激光/视觉接近的分段限速逻辑。
7. **[state_machine.py](file:///d:/55495/workspace/embedded_com/orchestrator/orchestrator_service/runtime/state_machine.py)**：重构了状态转移字典、对齐/接近 tick 逻辑、锁边迟滞拦截、进度无变化检测以及纯深度紧急拦截机制。

---

## 部署说明与后续建议
由于硬件环境隔离，代码已在 Windows 开发区通过静态编译语法验证（`py_compile` 成功）。最终的功能验证应在 `S171` 嵌入式板子端配合实际深度相机和串口设备运行。
