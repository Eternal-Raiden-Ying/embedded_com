# 离线控制分析与 replay 工具说明

这些工具的目的不是替代真实小车联调，而是把真实小车联调从“调代码”变成“采日志 + 验证候选参数”。大部分状态切换、滤波、速度平滑、抖动风险判断，都可以先在电脑上离线完成。

## 目录文件

| 文件 | 作用 |
|---|---|
| `common_offline.py` | JSONL 读写、字段归一化、信号统计、抖动风险评分等公共函数 |
| `sim_docking_controller.py` | 构造虚拟 yaw/dist/dropout/jump 输入，直接测试 `DockingController` |
| `replay_table_edge_log.py` | 用真实 `table_edge_obs.jsonl` 离线重放桌边状态机子集和控制器 |
| `predict_motion_jitter.py` | 根据观测/命令日志粗略预测小车运行抖动风险，并给出主要原因 |
| `plot_control_run.py` | 将视觉观测、速度命令、状态/phase 画成时间曲线，并导出 CSV/JSON 总结 |
| `serial_open_loop_test.py` | 生成或实际发送开环速度表，测底盘死区、速度映射和执行抖动 |
| `compare_replay_sweep.py` | 对同一段视觉日志批量扫参数，按到达状态和抖动风险排序 |

## 1. 先做纯控制器仿真

不需要小车、不需要相机，只测试 `ctrl_code/control/docking_controller.py`。

```bash
python3 tools/sim_docking_controller.py --scenario all --out-dir runs/offline_sim
```

重点看：

- `sim_*_summary.json` 里的 `jitter_prediction.risk_score`
- `phase_counts` 中 `SPIN_ONLY / FORWARD_APPROACH / OBS_GRACE_DECAY` 是否符合预期
- `sim_*.csv` 中 `vx_norm / wz_norm` 是否有频繁换向

## 2. 用真实视觉日志 replay

真实实验只需要采集 `table_edge_obs.jsonl`，之后可以反复离线 replay。

```bash
python3 tools/replay_table_edge_log.py \
  --input runs/xxx/table_edge_obs.jsonl \
  --out-dir runs/offline_replay
```

如果你的 run 目录里已经有 `table_edge_obs.jsonl`，也可以：

```bash
python3 tools/replay_table_edge_log.py --run-dir runs/xxx --out-dir runs/offline_replay
```

输出：

- `replay_cmd_vel.jsonl`：离线重放得到的速度命令
- `replay_state_trace.jsonl`：状态切换、计数器、raw/filtered 误差、phase
- `replay_summary.json`：最终是否到达、状态数量、风险评分

## 3. 画图分析

```bash
python3 tools/plot_control_run.py \
  --table runs/xxx/table_edge_obs.jsonl \
  --cmd runs/offline_replay/replay_cmd_vel.jsonl \
  --out-dir runs/offline_plot
```

输出：

- `obs_timeseries.png`：raw/filtered yaw、raw/filtered dist、confidence
- `cmd_timeseries.png`：vx/vy/wz 归一化速度命令
- `state_timeline.png`：状态和控制 phase 时间线
- `merged_timeline.csv`：统一时间轴数据，方便进一步分析
- `plot_summary.json`：统计量和抖动风险预测

如果运行环境没有 `matplotlib`，工具会跳过 PNG，但仍会输出 CSV 和 JSON。

## 4. 粗略预测运行抖动

对 replay 输出做风险预测：

```bash
python3 tools/predict_motion_jitter.py \
  --input runs/offline_replay/replay_cmd_vel.jsonl \
  --out-dir runs/offline_jitter_report
```

输出 `jitter_prediction.json`，包含：

- `risk_score`：0~100，越高越可能出现肉眼可见抖动/顿挫
- `risk_level`：LOW / MEDIUM / HIGH / VERY_HIGH
- `top_causes`：主要来源，比如 yaw 观测抖、wz 换向频繁、命令变化率过大、观测丢失等

注意：这个风险评分是启发式指标，不是精确动力学模型。它适合快速筛掉明显不好的参数组合。

## 5. 扫参数，选候选方案

```bash
python3 tools/compare_replay_sweep.py \
  --input runs/xxx/table_edge_obs.jsonl \
  --alphas 0.25,0.35,0.45 \
  --approach-vxs 0.08,0.10,0.12 \
  --approach-wzs 0.12,0.16 \
  --out-dir runs/offline_replay_sweep
```

输出：

- `sweep_results.csv`
- `sweep_results.json`
- 每个 case 的 `replay_summary.json`

排序规则：优先能进入 `AT_TABLE_EDGE`，然后按抖动风险分数低的排前面。

## 6. 底盘开环测试

默认 dry-run，不会发串口：

```bash
python3 tools/serial_open_loop_test.py --out-dir runs/open_loop_test
```

真正发给 STM32 前，确保小车架空或放在安全区域：

```bash
python3 tools/serial_open_loop_test.py \
  --real \
  --port /dev/ttyHS1 \
  --baud 115200 \
  --phase-s 2.0 \
  --out-dir runs/open_loop_test_real
```

要记录：

- `vx=0.030/0.050/0.080/0.100/0.120` 哪些能稳定起步
- `wz=0.040/0.060/0.080/0.120/0.160` 哪些能稳定旋转
- 是否存在明显死区、突然起步、停止顿挫
- 用结果反推 `ORCH_DOCKING_*_MAX_*` 和 PID `min_abs_output`

## 推荐调试顺序

```text
1. 固定摄像头，车不动，采集 30s table_edge_obs.jsonl
2. plot_control_run.py 看视觉本身是否稳定
3. replay_table_edge_log.py 用同一段日志重放控制器
4. compare_replay_sweep.py 选 1~2 组候选参数
5. serial_open_loop_test.py 测底盘死区和速度映射
6. 最后才上车完整闭环验证
```

## 常用参数覆盖

这些工具可以直接接受常用控制参数覆盖，例如：

```bash
python3 tools/replay_table_edge_log.py --input table_edge_obs.jsonl \
  --filter-alpha 0.25 \
  --coarse-enter 0.16 \
  --coarse-exit 0.07 \
  --approach-vx 0.10 \
  --approach-wz 0.14
```

也可以使用和主程序一致的环境变量：

```bash
export ORCH_DOCKING_FILTER_ALPHA=0.25
export ORCH_DOCKING_APPROACH_MAX_VX=0.10
export ORCH_DOCKING_APPROACH_MAX_WZ=0.14
```
