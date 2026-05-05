# System Upgrade Delivery

## 1. 新增及修改的文件清单

### Orchestrator 主链升级

- 修改 `orchestrator/orchestrator_service/config/schema.py`
- 修改 `orchestrator/orchestrator_service/config/board_config.py`
- 修改 `orchestrator/orchestrator_service/ipc/protocol.py`
- 修改 `orchestrator/orchestrator_service/runtime/context.py`
- 修改 `orchestrator/orchestrator_service/runtime/controller.py`
- 修改 `orchestrator/orchestrator_service/runtime/state_machine.py`
- 修改 `orchestrator/orchestrator_service/runtime/service.py`
- 修改 `orchestrator/orchestrator_service/bridge/simple_car_protocol.py`
- 修改 `orchestrator/orchestrator_service/bridge/uart_bridge.py`

### 新增控制模块

- 新增 `orchestrator/orchestrator_service/control/__init__.py`
- 新增 `orchestrator/orchestrator_service/control/types.py`
- 新增 `orchestrator/orchestrator_service/control/pid.py`
- 新增 `orchestrator/orchestrator_service/control/docking_controller.py`

### 调试与测试脚本

- 修改 `orchestrator/orchestrator_service/examples/debug_motion_sequence.py`
- 修改 `orchestrator/orchestrator_service/examples/mock_vision_obs_sender.py`
- 修改 `orchestrator/orchestrator_service/examples/uart_protocol_smoke_test.py`
- 新增 `orchestrator/orchestrator_service/examples/control_module_smoke_test.py`
- 新增 `orchestrator/orchestrator_service/examples/state_machine_regression_test.py`
- 新增 `orchestrator/orchestrator_service/examples/offline_debug_harness.py`

### 视觉备选版本

- 新增 `VISTA/vision_module_v2/__init__.py`
- 新增 `VISTA/vision_module_v2/protocol.py`
- 新增 `VISTA/vision_module_v2/app.py`
- 新增 `VISTA/vision_module_v2/README.md`

## 2. 当前测试后能够实现的功能

### 已完成并验证

- `orchestrator` 状态机已从原先的简单 `STOP/AUTOEXPLORE/AUTOSEARCH/SEARCH/RETURN` 升级为桌面取物主链：
  - `IDLE`
  - `SEARCH_TABLE`
  - `COARSE_ALIGN`
  - `CONTROLLED_APPROACH`
  - `FINAL_LOCK`
  - `DOCK_RETRY`
  - `AT_TABLE_EDGE`
  - `SEARCH_TARGET_INIT`
  - `EDGE_SLIDE_SEARCH`
  - `TARGET_CONFIRM`
  - `TARGET_LOCKED`
  - `FREEZE_BASE`
  - `LEAVE_EDGE`
  - `RELOCATE_TO_EDGE`
  - `REACQUIRE_EDGE`
  - `NEXT_TABLE`
  - `AVOID_OBSTACLE`
  - `RETURN_HOME`
  - `ERROR_RECOVERY`
  - `DONE`

- 已支持底盘三维控制量输出：
  - `vx`
  - `vy`
  - `wz`

- 已支持新的底盘 TXT 串口协议发送格式：
  - `MODE <STATE_NAME>`
  - `VEL <vx> <vy> <wz> <hold_ms>`
  - `STOP`
  - `BRAKE`

- 已支持新的底盘反馈解析格式：
  - `STATE <status> <vx> <vy> <wz> <fault_code>`
  - `ESTOP <0|1>`

- 已支持 `table_edge_obs / target_obs / home_tag_obs` 三类视觉输入。

- 已兼容一部分旧视觉嵌套消息：
  - `vision_obs.perception.table_edge_obs`
  - `vision_obs.perception.target_obs`
  - `vision_obs.perception.home_tag_obs`

- 已加入更稳的防卡死措施：
  - 视觉请求连续失败保护
  - 底盘超时/故障/急停保护
  - 避障中断与恢复
  - 丢帧保持窗
  - 状态局部超时而不是整任务时间串用
  - UART 异步发送线程，最新命令覆盖旧命令，降低串口阻塞风险

### 已跑过的本地验证

- `py -3 orchestrator/orchestrator_service/examples/control_module_smoke_test.py`
- `py -3 orchestrator/orchestrator_service/examples/state_machine_regression_test.py`
- dry-run 启动 `orchestrator` 后，通过 `debug_motion_sequence.py` 用真实 TCP 端口喂了一整段：
  - FIND
  - 桌边搜索
  - 粗对齐
  - 受控接近
  - 锁边
  - 沿边搜索
  - 目标确认

### 当前刻意保留的边界

- 机械臂与 grasp 执行链路仍未接入状态机动作闭环。
- `VISTA/vision_module_v2` 已实现 mask 进入 application 层，但 `table_edge_obs` 仍是备选版本中的协议占位与代理输出。
  说明:
  深度桌边提取仍等待 `Offline_Edge_Test` 结果确认后再并入。

## 3. 上线前应进行的测试项

### 状态机与 IPC

- 用真实语音发送 `FIND / RETURN / STOP`，确认 `task_cmd -> task_ack -> 状态切换` 全链一致。
- 用真实视觉确认 `vision_req_out` 的 `mode/stage/current_edge_id` 与视觉侧解析完全一致。
- 验证 `STOP` 在任意状态都能打断，并且视觉侧能收到 `IDLE` 切换。

### 底盘与串口

- 真实 STM32 上验证 `VEL vx vy wz hold_ms` 的字段顺序、单位和正负号约定。
- 分别验证 `vx` 正方向、`vy` 正方向、`wz` 正方向与车体实际运动方向是否一致。
- 验证底盘 `STATE` 回传是否稳定、是否会粘包、是否会出现无换行导致主控阻塞。
- 验证 `STOP` 与 `BRAKE` 在 STM32 侧优先级是否符合预期。

### 桌边停靠

- 单桌单边场景反复测试：
  - 搜桌是否稳定进入 `COARSE_ALIGN`
  - 是否能从 `COARSE_ALIGN -> CONTROLLED_APPROACH -> FINAL_LOCK -> AT_TABLE_EDGE`
  - 桌边短时丢失时是否只短暂停车，不会马上退回
  - 长时丢失时是否能正确退回搜索或重试

### 沿边搜索

- 验证 `EDGE_SLIDE_SEARCH` 的 `vy` 横移方向是否符合物理预期。
- 验证目标在工作带内进入画面时，是否能稳定经过 `TARGET_CONFIRM -> TARGET_LOCKED -> FREEZE_BASE`。
- 验证当前边超时后是否会触发 `LEAVE_EDGE -> RELOCATE_TO_EDGE -> REACQUIRE_EDGE`。

### 避障与异常恢复

- 用 mock 或真实传感触发障碍标志，确认 `AVOID_OBSTACLE` 能进入、恢复或超时退出。
- 拔掉视觉、断开串口、模拟底盘 fault/timeout，确认状态机会进入 `ERROR_RECOVERY` 而不是卡死。

### 视觉备选版本

- 在 `VISTA/vision_module_v2` 中确认 `mask_ready / mask_shape / mask_area_ratio / mask_bbox` 随目标输出正常。
- 用你后续验证过的 `Offline_Edge_Test` 结果替换代理桌边输出后，再决定是否将 `vision_module_v2` 进入主联调。

## 4. 若测试出现问题，对应的排查与修改方案

### 问题 1：状态机不切换或切换异常

- 先看 `runs/.../state_blocks.jsonl`、`timeline.jsonl`、`ipc.jsonl`。
- 重点确认：
  - `task_cmd` 是否真的收到了
  - `session_id / epoch` 是否一致
  - `table_edge_obs / target_obs` 是否因为时间戳过旧被丢弃
  - `vision_req_fail_streak` 是否把状态机打进了恢复态
- 修改方向：
  - 检查视觉侧是否透传了正确的 `session_id / req_id / epoch`
  - 调整 `*_obs_max_age_s`
  - 调整 `*_frames_*` 和 `*_hold_s`

### 问题 2：桌边停靠抖动或过冲

- 先看 `cmd_vel.jsonl` 和 `car_cmd.jsonl` 中 `vx/vy/wz` 输出是否波动过大。
- 修改方向：
  - 调整 `docking` PID 参数
  - 调整 `vx_slew_per_s / vy_slew_per_s / wz_slew_per_s`
  - 调整 `final_lock_*_tol`
  - 若视觉桌边误差本身抖动，优先先稳视觉，不要先把 PID 调激进

### 问题 3：沿边搜索方向反了或物理运动不对

- 先用 `uart_protocol_smoke_test.py` 和 dry-run 控制台确认发送的 `vx/vy/wz`。
- 再到 STM32 侧核对逆运动学和坐标系定义。
- 修改方向：
  - 优先统一 `vx / vy / wz` 三轴正负号
  - 不要在主控和底盘两边同时“各改一半”

### 问题 4：串口卡死或发包堵塞

- 当前主控侧已经改成 UART 异步发送线程，先确认是否仍然是 STM32 侧读写阻塞。
- 检查：
  - STM32 是否按行读取并等待 `CR/LF`
  - 回传是否过快或带大量调试信息
  - `STATE` 是否缺换行
- 修改方向：
  - 保持一行一包
  - 若 STM32 回传日志过多，先降频
  - 若还不稳，再推进二进制帧升级，但高层语义保持不变

### 问题 5：真实联调通过，离线脚本却和实机表现不一致

- 优先检查脚本是否仍然走真实端口与真实消息类型。
- 当前建议使用：
  - `offline_debug_harness.py`
  - `debug_motion_sequence.py`
  - `mock_vision_obs_sender.py`
- 修改方向：
  - 不要自造另一套“内部测试协议”
  - 所有 mock 都必须复用正式 `task_cmd_in / vision_obs_in / vision_req_out / task_ack_out` 链路

### 问题 6：vision_module_v2 有 mask，但上层仍拿不到

- 先确认当前运行的是 `VISTA/vision_module_v2/app.py`，不是旧版 `vision_module/app/app.py`
- 再确认 `target_obs` 中是否带出：
  - `mask_ready`
  - `mask_shape`
  - `mask_area_ratio`
  - `mask_bbox`
- 修改方向：
  - 若模型输出顺序变化，优先检查 box 和 mask 索引是否还一一对应
  - 若 mask 非空但 bbox 为空，检查二值化和 `nonzero` 提取逻辑

## 5. 当前建议的联调顺序

1. 先跑 `orchestrator` dry-run + `offline_debug_harness.py` / `debug_motion_sequence.py`
2. 接真语音，其余保持 mock
3. 接真底盘，先确认 `vx/vy/wz` 方向和串口稳定性
4. 再接真视觉桌边链路
5. 最后再评估是否切换到 `vision_module_v2`

## 6. 备注

- 这次交付优先把“找桌 - 靠桌 - 锁边 - 沿边搜索”的主链打稳。
- 机械臂和 grasp 相关状态机动作本次没有继续深入实现，保持为后续阶段。
