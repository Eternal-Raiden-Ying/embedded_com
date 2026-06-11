# VISTA 视觉语义字段与重构说明文档 (Vision Semantics & Refactor Notes)

本文档说明了 VISTA 视觉感知链路输出的语义字段标准化规范、视觉-控制语义统一以及 BBox-Size Driven ROI（基于边界框尺寸驱动的 ROI）的实现说明。

---

## 1. 核心目标 (Core Goals)

1.  **标准化视觉输出字段**：对视觉链路产生的检测量进行分层标准化，确保 Orchestrator 控制层能够安全稳定地消费。
2.  **统一桌子边界框（Table BBox）语义**：明确区分“当前帧检测到”、“控制层可用”及“历史帧数据保持”。
3.  **细化桌边（Edge）感知状态**：建立 `edge_detected` ──> `edge_geometry_valid` ──> `edge_stable` ──> `edge_trusted` 的分层递进判断机制。
4.  **BBox-Size Driven ROI（动态 ROI 扩展）**：将传统固定窗口的 ROI 调整默认改为 `bbox_expand`。先将 RGB 图像下的桌子 BBox 映射到 Depth 深度图上，再根据配置的 margin 范围扩展，生成最终用于桌边拟合的 `table_edge_roi`，大幅减少背景干扰。

---

## 2. 统一字段规范定义 (Unified Field Dictionary)

| 字段名 | 类型 | 说明 |
| :--- | :--- | :--- |
| `table_bbox_current_found` | `bool` | 当前帧 YOLO/table 是否真实检测到桌子 BBox（不使用置信度门控）。 |
| `table_bbox_control_valid` | `bool` | 控制层是否被允许使用桌子 BBox（当前帧检测到或处于 Hold 状态）。 |
| `table_bbox_hold_active` | `bool` | 当前 BBox 是否来自历史帧的记忆保持，而不是当前帧实时检测。 |
| `table_bbox_hold_age_frames` | `int` | 桌子 BBox 历史保持已经持续的帧数。 |
| `table_bbox_xyxy` | `list` | 生效的桌子 BBox 坐标 `[x1, y1, x2, y2]`，对应 RGB 裁剪图。 |
| `table_bbox_source` | `str` | 来源：`yolo_table_bbox`、`mock_table_bbox`、`table_bbox_hold`、`none`。 |
| `table_bbox_conf_raw` | `float` | YOLO 输出的原始置信度，仅用于记录和诊断，固定不作为门控。 |
| `table_bbox_area_ratio` | `float` | 桌子 BBox 面积占 RGB 画面积比例，仅作诊断分析。 |
| `edge_detected` | `bool` | 感知算法是否捕获到了桌边线几何候选。 |
| `edge_geometry_valid` | `bool` | 单帧 edge 拟合结果是否在数学上有效，表示几何拟合成功，不等于控制授权。 |
| `edge_stable` | `bool` | 拟合出的几何边缘线在时序上连续稳定的帧数是否达标。 |
| `edge_trusted` | `bool` | 边缘线完全满足稳定性和质量阈值校验，可以正式移交底盘参与控制。 |
| `edge_quality` | `dict` | 包含残差、支撑点、内点数和物理跨度的几何质量摘要字典。 |
| `edge_trust_reason` | `str` | 判定 edge 可信的具体条件或原因。 |
| `edge_reject_for_control_reason`| `str`| 边缘线被拒绝参与姿态控制的失效原因。 |
| `table_edge_roi` | `list` | 最终在深度图上进行几何提取的裁剪 ROI 矩形。 |

---

## 3. 历史兼容字段映射 (Backward Compatibility Aliases)

为了兼容历史数据重放并防止老版本底盘控制逻辑崩溃，网关和数据接口依然导出以下兼容别名，但在新编写的控制与状态转移中**强烈建议不要优先使用它们**：

*   `table_bbox_found` ──> 等价于新字段 `table_bbox_current_found`
*   `yolo_table_control_valid` ──> 等价于新字段 `table_bbox_control_valid`
*   `edge_valid` ──> 等价于新字段 `edge_geometry_valid`
*   `valid_for_control` / `edge_control_allowed` ──> 等价于新字段 `edge_trusted`
