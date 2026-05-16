# SC171 Overnight Codex 汇总

日期：2026-05-16
分支：`sc171-table-docking-test-prep`
基线：`a22b551 Add STM32 motion protocol probe`

## 提交列表

- `814ba62 docs: record table docking test context`
- `4b7d849 test: verify STM32 motion probe`
- `71c2611 bridge: support STM32 motion protocol encoding and feedback`
- `84ba7c6 vision: add table edge ROI debug presets`
- `3774939 vision: add table detection debug and coarse direction hint`
- `6560840 orchestrator: add table edge only test mode`
- `6607ae7 orchestrator: add grasp verification gate`
- `be15f02 docs: add table docking field test plan`

## 修改范围

- `orchestrator/stm32_motion_probe.py`：确认上一轮 STM32 协议探针的 dry-run sequence 路径。
- `orchestrator/orchestrator_service/bridge/simple_car_protocol.py`：补充 STM32 兼容编码别名和反馈解析兼容。
- `VISTA/vision_module/utils/table_roi.py`：为桌边测试增加显式 ROI preset。
- `VISTA/vision_module/backend/table_edge_roi.py`：接入 ROI debug 和 preset 支持。
- `VISTA/vision_module/backend/table_edge_manager.py`：接入 edge debug 输出。
- `VISTA/vision_module/backend/predictor_manager.py`：增加 table detection debug 和粗方向提示。
- `orchestrator/orchestrator_service/runtime/state_machine.py`：增加 table edge-only 停车结束路径和抓取验证 gate。
- `orchestrator/orchestrator_service/config/schema.py`：增加 SC171 桌边停靠默认值和测试开关。
- `orchestrator/orchestrator_service/config/board_config.py`：接入环境变量配置。
- `tests/test_simple_car_protocol.py`、`tests/test_orch_operator_console.py`、`VISTA/vision_module/test/test_table_roi.py`：补充聚焦回归测试。
- `docs/tomorrow_test_context.md`、`docs/tomorrow_field_test_plan.md`：记录测试上下文和可复制现场命令。

## 新增配置

- `ORCH_TABLE_EDGE_ONLY_TEST=0`：默认关闭。打开后，最终桌边锁定完成即停车结束，不进入目标搜索/抓取。
- `ORCH_TABLE_TARGET_DIST_CM=30`：默认桌边目标测距。
- `ORCH_TABLE_DIST_TOL_CM=5`：默认距离容差。
- `ORCH_TABLE_STOP_MARGIN_CM=5`：默认最终 settle 前的停车裕量。
- `ORCH_TABLE_SETTLE_MS=500`：默认 STOP 后 settle 时间。
- `ORCH_TABLE_STABLE_FRAMES=5`：默认 dock done 前需要的稳定帧数。
- `ORCH_TABLE_YAW_TOL_DEG=8`：默认 yaw 容差。
- `ORCH_TABLE_MAX_MICRO_ADJUST=3`：默认有限微调次数。
- `ORCH_ASSUME_GRASP_SUCCESS_FOR_TEST=0`：默认关闭。打开后，`GRASP_VERIFY` 可在没有真实验证源时进入 `RETURN_HOME`。
- `VISTA_TABLE_EDGE_ROI_PRESET`：显式 ROI preset，预期值包括 `center_mid`、`center_lower`、`full_width_lower`。
- `VISTA_EDGE_DBG=1`：输出低频 edge debug。
- `ORCH_TABLE_DET_ENABLED=1`：打开 table detection debug。
- `ORCH_TABLE_DET_MIN_CONF`、`ORCH_TABLE_DET_CENTER_TOL`：table detection debug 阈值。

## 离线验证

已通过：

```bash
python3 -m unittest tests.test_simple_car_protocol tests.test_orch_operator_console
```

结果：57 个测试通过。

已通过：

```bash
python3 orchestrator/stm32_motion_probe.py --dry-run --cmd sequence
```

观察到的 dry-run TX 行：

```text
[PROBE][TX] STOP 1
[PROBE][TX] STATUS
[PROBE][TX] JOG 30 30 30 30 100 2
[PROBE][TX] STOP 3
[PROBE][TX] STATUS
```

已通过：

```bash
/usr/bin/python3 -m unittest VISTA.vision_module.test.test_table_roi VISTA.vision_module.test.test_preview_table_bbox
```

结果：19 个测试通过。

已通过：

```bash
python3 -m py_compile orchestrator/stm32_motion_probe.py orchestrator/orchestrator_service/bridge/simple_car_protocol.py orchestrator/orchestrator_service/config/schema.py orchestrator/orchestrator_service/config/board_config.py orchestrator/orchestrator_service/runtime/context.py orchestrator/orchestrator_service/runtime/state_machine.py VISTA/vision_module/utils/table_roi.py VISTA/vision_module/backend/preview/opencv_sink.py VISTA/vision_module/backend/table_edge_manager.py
```

环境说明：

- 在当前 conda Python 下运行 `python3 -m unittest VISTA.vision_module.test.test_table_roi VISTA.vision_module.test.test_preview_table_bbox` 会失败，因为该环境没有安装 `numpy`。
- 同一组 VISTA 测试在 `/usr/bin/python3` 下通过，该环境具备所需依赖。

## 明天优先步骤

1. 确认 `/dev/ttyHS1` 或实际 STM32 串口设备。
2. 先跑 STM32 probe dry-run，再在机器人架起或轮子安全的条件下跑真实串口 sequence。
3. 确认前进、后退、转向、停止的轮向映射。
4. 选择能明显覆盖真实桌边的 ROI preset。
5. 运行 `ORCH_TABLE_DET_ENABLED=1`，确认模型是否包含 COCO table 类 `60`。
6. 使用 SC171 默认参数跑 table edge-only dry-run。
7. 只有在 dry-run STOP 行为稳定后，才跑 table edge-only 真实串口。
8. 实测物理停车裕量，并在启用目标搜索或抓取前调整 `ORCH_TABLE_STOP_MARGIN_CM`。
9. 除非明确测试返航骨架，否则保持 `ORCH_ASSUME_GRASP_SUCCESS_FOR_TEST=0`。

## 人工确认点

- 实际串口设备路径和波特率。
- 四轮方向和速度比例映射。
- STM32 对 ACK、BUSY、STATUS、TIMEOUT、JOG 事件的反馈文本。
- 相机安装角度和 table-edge ROI preset。
- table detection debug 输出真实 table 类检测，还是 `[TABLE_DET][NO_TABLE_CLASS]`。
- 最终安全的 `ORCH_TABLE_TARGET_DIST_CM` 和 `ORCH_TABLE_STOP_MARGIN_CM`。

## 工作区说明

- `ROBOT_MOTION_CONTRACT.md` 仍是未跟踪文件，本轮按计划没有提交。
