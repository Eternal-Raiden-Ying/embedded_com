# Speed Unification Refactor Notes

本包把 orchestrator 的速度控制统一为真实物理速度：

- `vx_mps` / `vy_mps`：米/秒。
- `wz_radps`：弧度/秒。
- `CmdVel`、`SimpleCarCommand`、`SimpleCarMapper`、`Stm32MotionAdapter` 均不再使用 normalized speed。
- `Stm32MotionAdapter` 不再执行 `norm -> m/s` 的二次缩放，只做安全限幅和串口发送。
- `car_cmd_params.yaml` 只保留通信、keepalive、stale 和绝对速度上限。
- `stage_params.yaml` 中所有运动速度均为真实速度。

## 主要结构调整

- 新增 `orchestrator_service/control/motion_controller.py`，实际 `MotionController` 实现移到 control 层。
- `orchestrator_service/runtime/controller.py` 变成兼容 shim，外部旧 import 仍然可用。

## 关键调速字段

- `yolo_table.yolo_forward_vx_mps`：YOLO 默认前进速度。
- `yolo_table.rotate_search_wz_radps`：无桌子 bbox 时本地旋转搜索角速度。
- `table_docking_motion.controlled_approach.vx_mps / vx_min_mps / vx_max_mps`：edge/docking 可信后的接近速度。
- `table_docking_motion.coarse_align.wz_min_radps / wz_max_radps`：粗对齐角速度。
- `edge_slide_search.slide_vy_mps`：沿边搜索横向速度。

## 已删除/弃用的速度语义

- `*_norm` 速度字段。
- `vx_mps_per_norm / vy_mps_per_norm / wz_radps_per_norm` 二次缩放。
- `car_cmd.max_*_norm`。

坐标归一化字段例如 `table_cx_norm`、`view_err_norm`、`size_norm` 仍然保留，因为它们是图像/几何归一化，不是速度单位。
