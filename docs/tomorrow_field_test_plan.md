# SC171 桌边停靠现场测试计划

日期：2026-05-16
分支：`sc171-table-docking-test-prep`

## 安全默认值

- 在确认轮向、串口和停车行为前，保持 `ORCH_SERIAL_DRY_RUN=1`。
- `ORCH_TABLE_EDGE_ONLY_TEST` 默认是 `0`；只在桌边停靠测试准备时打开。
- `ORCH_ASSUME_GRASP_SUCCESS_FOR_TEST` 默认是 `0`；只有在没有真实抓取验证源、且明确要测试返航骨架时才打开。
- 在真实底盘上确认 table edge-only 的 STOP 行为前，不要启用抓取。

## 1. STM32 运动探针

先做 dry-run 命令编码验证：

```bash
python3 orchestrator/stm32_motion_probe.py --dry-run --cmd sequence
```

确认串口后再做真实串口冒烟测试：

```bash
python3 orchestrator/stm32_motion_probe.py --port /dev/ttyHS1 --baudrate 115200 --cmd sequence
```

通过标准：

- `VEL`、`STOP`、`JOG_*`、`STATUS` 命令编码无异常。
- 真实串口测试能看到预期轮向，以及 STM32 ACK/status 回传。
- 如果任何轮向不符合预期，先修正 mapping/config，再继续下一步。

## 2. ROI 调试

只使用显式 preset；默认 ROI 行为保持不变。先改 `VISTA/configs/vision_params.yaml`：

```yaml
table_edge:
  roi_preset: center_mid
debug:
  edge_debug_enabled: true
```

然后启动：

```bash
python3 -m VISTA.vision_module.app.app
```

如果桌边不在预览叠加框内，再尝试 `center_lower` 和 `full_width_lower`：

```yaml
table_edge:
  roi_preset: center_lower
```

或：

```yaml
table_edge:
  roi_preset: full_width_lower
```

通过标准：

- 预览画面里的 ROI 叠加框覆盖真实桌边。
- 低频日志出现 `[EDGE_DBG] valid=... dist=... yaw=... age_ms=... roi=...`。
- 运动过程中 `age_ms` 保持新鲜，不持续变旧。

## 3. YOLO 桌子检测调试

打开检测调试，但不伪造 table 检测。先改 `VISTA/configs/vision_params.yaml`：

```yaml
debug:
  table_det_enabled: true
  table_det_min_conf: 0.30
  table_det_center_tol: 0.12
```

然后启动：

```bash
python3 -m VISTA.vision_module.app.app
```

通过标准：

- 如果模型包含 COCO table 类 `60`，日志输出 `[TABLE_DET]`，包含置信度、bbox 中心和 `left/center/right` 粗方向提示。
- 如果模型没有 table 类，日志输出 `[TABLE_DET][NO_TABLE_CLASS]`。
- 不把非 table 类当作桌子证据。

## 4. Table Edge-Only Dry Run

使用为 SC171 准备的停靠目标参数：

```bash
ORCH_SERIAL_DRY_RUN=1 \
ORCH_TABLE_EDGE_ONLY_TEST=1 \
ORCH_TABLE_TARGET_DIST_CM=30 \
ORCH_TABLE_DIST_TOL_CM=5 \
ORCH_TABLE_STOP_MARGIN_CM=5 \
ORCH_TABLE_SETTLE_MS=500 \
ORCH_TABLE_STABLE_FRAMES=5 \
ORCH_TABLE_YAW_TOL_DEG=8 \
ORCH_TABLE_MAX_MICRO_ADJUST=3 \
python3 -m orchestrator_service.app.main
```

通过标准：

- 状态进入 `FINAL_LOCK`，在 stop window 内发送 STOP，完成 settle，并输出 `[TABLE_DOCK][DONE]`。
- 打开 edge-only 后，输出 `[TABLE_EDGE_ONLY][DONE]`，并在目标搜索或抓取前停止。
- 桌边观测 stale 时进入 STOP/hold，不盲目前进。

## 5. Table Edge-Only 真实串口

只在 dry-run 行为稳定后执行：

```bash
ORCH_SERIAL_DRY_RUN=0 \
ORCH_SERIAL_PORT=/dev/ttyHS1 \
ORCH_SERIAL_BAUDRATE=115200 \
ORCH_TABLE_EDGE_ONLY_TEST=1 \
ORCH_TABLE_TARGET_DIST_CM=30 \
ORCH_TABLE_DIST_TOL_CM=5 \
ORCH_TABLE_STOP_MARGIN_CM=5 \
ORCH_TABLE_SETTLE_MS=500 \
ORCH_TABLE_STABLE_FRAMES=5 \
ORCH_TABLE_YAW_TOL_DEG=8 \
ORCH_TABLE_MAX_MICRO_ADJUST=3 \
python3 -m orchestrator_service.app.main
```

通过标准：

- 底盘低速接近，并停在无碰撞风险的距离外。
- `JOG_FORWARD`、`JOG_BACKWARD` 或 `JOG_TURN` 只在有限次微调中出现。
- `[TABLE_EDGE_ONLY][DONE]` 之后机器人保持停止。

## 6. 抓取验证入口

机械臂 OK 后的默认行为是保守的：

```bash
ORCH_ASSUME_GRASP_SUCCESS_FOR_TEST=0 python3 -m orchestrator_service.app.main
```

没有真实验证源时，默认预期日志：

```text
[GRASP][VERIFY_UNAVAILABLE] no real grasp verification source; not assuming success
```

仅测试返航骨架时才打开：

```bash
ORCH_ASSUME_GRASP_SUCCESS_FOR_TEST=1 python3 -m orchestrator_service.app.main
```

通过标准：

- 没有验证源、且未打开 assume-success 时，系统不会标记抓取成功。
- 打开 assume-success 后，`GRASP_VERIFY` 转入 `RETURN_HOME`。
- 验证失败走现有抓取 retry/error recovery。

## 7. 现场成功标准

- STM32 probe 命令有 ACK，且轮向正确。
- ROI 叠加框能跟上明天实际相机安装角度下的桌边。
- table detection debug 要么输出真实 table 类检测，要么明确输出无 table 类。
- edge-only 模式到达 `[TABLE_EDGE_ONLY][DONE]` 后保持停车。
- 启用目标搜索或抓取前，先实测 stop margin。

## 人工确认项

- 实际串口设备路径。
- 四轮方向映射。
- STM32 对 ACK/BUSY/STATUS/TIMEOUT 的回传行格式。
- 相机安装角度和 table-edge ROI preset。
- 实际安全的 `ORCH_TABLE_STOP_MARGIN_CM`。
- YOLO 模型是否包含 COCO table 类 `60`。
