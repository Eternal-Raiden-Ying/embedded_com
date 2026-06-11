# 控制层语义与调速重构说明文档 (Control Layer Semantics & Speed Unification Refactor)

本文档整理了小车控制层（Orchestrator）在进行语义清理、视觉-控制语义统一以及物理速度单位归一化时的核心重构说明。

---

## 1. 核心目标 (Core Goals)

1.  **清理控制字段语义**：保留清晰、必要且可解释的字段，剔除历史重构中遗留的模糊或冗余的控制机制。
2.  **移除历史模糊逻辑**：完全移除如 `yolo_assist`、`yolo_edge_blend` 以及旧版 `0.40 bbox area gate`（YOLO 桌面检测框面积占比门槛限制）等过渡逻辑。桌面检测框的面积比例等字段仅作为诊断数据，不再作为控制状态转移的限制阈值。
3.  **统一控制源 (Control Source)**：所有的底盘控制命令来源被限制并收敛为以下几种：
    *   `yolo_forward`：当前帧检测到 YOLO 桌子 BBox，但几何桌边未被授权信任，执行安全前进（`vx > 0, wz = 0`）。
    *   `edge_adjust`：当 `edge_trusted=True` 时，进入高精度的几何桌边对齐与姿态调整。
    *   `local_rotate_search`：未检测到 YOLO 桌子 BBox 或桌边丢失时，进行本地原地旋转搜索。
    *   `final_lock`：终点减速刹车与防抖确认。
    *   `search_failed_stop`：搜索超时或异常触发安全停车。
    *   `explicit_stop` / `stop`：上游发送的强行停止或状态机进入 IDLE。
4.  **速度单位物理化 (Speed Physical Unification)**：
    *   **彻底废弃归一化速度表示**：`CmdVel`、`SimpleCarCommand`、`SimpleCarMapper`、`Stm32MotionAdapter` 统一采用真实物理单位速度表示，不再使用 `norm` 速度做二次缩放。
    *   **单位规范**：前后纵向速度 `vx_mps`、横移速度 `vy_mps` 单位统一为**米/秒 (m/s)**；旋转角速度 `wz_radps` 单位统一为**弧度/秒 (rad/s)**。
    *   底层 `Stm32MotionAdapter` 仅执行物理限幅校验与串口编码，不执行任何二次缩放。所有的调速策略均上移至控制算法层，且在 `car_cmd_params.yaml` 和 `stage_params.yaml` 中使用物理单位配置。

---

## 2. 控制语义字段说明 (Unified Control Semantics Fields)

下表为控制层与视觉感知层完全对齐的统一语义字段，不再保留任何双真值语义或重叠定义：

| 字段名 | 类型 | 说明 |
| :--- | :--- | :--- |
| `table_bbox_current_found` | `bool` | 当前帧图像是否真实检测到 table bbox。 |
| `table_bbox_control_valid` | `bool` | 控制层是否允许使用 table bbox；包含当前帧直接检测或短时保持（hold）状态。 |
| `table_bbox_hold_active` | `bool` | table bbox 当前是否来自历史帧的数据保持。 |
| `table_bbox_hold_age_frames` | `int` | table bbox 历史保持帧数。 |
| `table_bbox_xyxy` | `list` | 传入控制层的 table bbox 实际像素坐标（基于 RGB 裁剪图像）。 |
| `table_bbox_conf_raw` | `float` | YOLO 检测的原始置信度评分（仅作诊断记录，固定不参与门控）。 |
| `edge_detected` | `bool` | 感知层是否检测到边缘几何候选。 |
| `edge_geometry_valid` | `bool` | 感知层判定单帧边缘几何的有效性，表示数学上拟合成功，不等于控制授权。 |
| `edge_stable` | `bool` | 边缘线在时间连续帧上是否稳定。 |
| `edge_trusted` | `bool` | 边缘是否通过质量判定，被正式授权参与姿态控制（等价于 `edge_control_allowed`）。 |
| `edge_quality` | `dict` | 包含残差（`edge_residual_raw`）、支撑点数（`edge_support_count`）、内点数（`edge_inlier_count`）以及横向跨度（`edge_x_span_m`）的质量评估字典。 |
| `control_source` | `str` | 当前命令的物理控制源（如 `yolo_forward`、`edge_adjust` 等）。 |
| `control_intent` | `str` | 控制意图类别（如 `forward`、`posture_adjust`、`search`、`hold`、`stop`）。 |
| `allow_forward` / `allow_rotate`| `bool` | 本帧是否允许底盘前进或旋转。 |
| `forward_block_reason` / `rotate_block_reason`| `str` | 动作被安全限制拦截的具体原因（用于离线重放和定位）。 |

---

## 3. 热修复与兼容层说明 (Hotfixes & Backward Compatibility)

在之前的版本迭代中，为了保证离线 bag 数据回放脚本（`offline_debug_harness.py` 等）在字段大范围重构后不崩溃，在 `perception_semantics.py` 中引入了别名兼容代理（Alias Map）：
*   `table_bbox_found` ──> 指向新主字段 `table_bbox_current_found`
*   `yolo_table_control_valid` ──> 指向新主字段 `table_bbox_control_valid`
*   `edge_valid` ──> 指向新主字段 `edge_geometry_valid`

此外，`to_dict()` 方法中显式保留了上述兼容字段的序列化，保证旧版本日志重放的解析完整性。控制逻辑主线（如 `controller._rotate_gate()`）均已无缝变更为使用新主字段。

---

## 4. 物理速度调参参考 (Tunable Velocity Parameters)

现场调速时，请在 `stage_params.yaml` 中配置真实物理速度，避开代码直接修改。以下为关键物理单位调参项：

*   **`yolo_table.yolo_forward_vx_mps`**：当只存在桌面目标检测框、无可信桌边时，小车自动前行寻找桌边的默认物理速度。
*   **`yolo_table.rotate_search_wz_radps`**：本地原地旋转搜索桌子时的角速度。
*   **`table_docking_motion.controlled_approach` 下的 `vx_mps / vx_min_mps / vx_max_mps`**：桌边可信进入接近对齐阶段时的纵向逼近物理速度区间。
*   **`table_docking_motion.coarse_align.wz_min_radps / wz_max_radps`**：小车调整姿态与桌边平行时的对齐角速度区间。
*   **`edge_slide_search.slide_vy_mps`**：停靠桌边后，沿边缘滑动平移搜索目标的横向速度（正值表示左移，负值表示右移）。
