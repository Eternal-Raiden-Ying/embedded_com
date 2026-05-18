# ROBOT_MOTION_CONTRACT

## 1. 当前目标
小车用于盲人取物平台。当前重点不是高速运动，而是：
- SC171 能稳定下发视觉闭环速度指令；
- STM32 能稳定解析并执行低速运动；
- 小车能靠近桌边，并在目标距离附近可靠停车；
- 终点附近通过短时速度脉冲 + STOP + 视觉复检实现微调。

## 2. 当前 STM32 主协议
当前采用成熟稳定的三轴速度协议，不使用四轮 `VEL/JOG/STATUS` 协议作为主线。

SC171 → STM32：

```text
MODE SEARCH
MODE RETURN
MODE AUTOSEARCH
MODE AUTOEXPLORE
V <vx_mps> <vy_mps> <wz_radps>
STOP
```

说明：
- 每条命令必须以 `\n` 结束；
- `V` 只有在 `MODE SEARCH` 或 `MODE RETURN` 下生效；
- `vx_mps` 单位为 m/s，正值表示前进；
- `vy_mps` 单位为 m/s，正值表示左移；
- `wz_radps` 单位为 rad/s，正值表示左转 / 逆时针；
- `STOP` 用于停车。

## 3. STM32 回传协议
STM32 → SC171：

```text
FB <原始命令>
```

示例：

```text
FB MODE SEARCH
FB V 0.020 0.000 0.000
FB STOP
```

注意：
- `FB` 只表示 STM32 已收到并解析该行命令；
- `FB` 不表示电机真实执行成功；
- `FB` 不表示小车真实速度、位移、姿态或是否停稳；
- 真实距离和姿态仍由 SC171 视觉闭环判断。

## 4. 当前 STM32 内部执行机制
当前 STM32 主链路为：

```text
UART2 接收 SC171 命令
→ 解析 MODE / V / STOP
→ 麦克纳姆四轮速度解算
→ 低速插帧
→ UART3 输出总线电机控制帧
```

当前关键机制：
- `MODE SEARCH` / `MODE RETURN` 控制速度命令是否生效；
- `MODE AUTOSEARCH` / `MODE AUTOEXPLORE` 由 STM32 内部自动策略使用，SC171 默认不在这两个模式下发送 `V`；
- `V vx vy wz` 表示三轴速度目标；
- STM32 内部将 `vx/vy/wz` 解算成四轮速度；
- 总线驱动板存在离散速度档位限制；
- STM32 通过在 `1500` 与相邻有效档位之间插入停车帧，实现更低平均速度；
- SC171 不需要 1000Hz 下发速度，按视觉闭环频率稳定发送即可。

## 5. 建议 SC171 发送频率
推荐初值：

```text
ORCH_CAR_SEND_PERIOD_MS = 50
```

含义：
- 约 20Hz 发送一次速度命令；
- 若响应偏慢，可尝试 30ms；
- 不建议一开始追求 1ms 或 1000Hz；
- STM32 内部负责低速插帧，SC171 负责稳定视觉闭环。

## 6. SC171 侧控制原则
- 状态机不直接拼底层串口字符串，应通过协议 / bridge 层统一发送；
- 启动运动前先发送 `MODE SEARCH` 或 `MODE RETURN`；
- 接近和对齐阶段持续发送 `V vx vy wz`；
- 停车时发送 `STOP`；
- 不再依赖 `JOG_DONE / STATUS` 作为当前主流程；
- `FB` 仅作为接收确认，不作为物理执行完成确认；
- 终点停靠采用 `STOP → settle → 视觉复检 → 必要时短时 V 脉冲 → STOP`。

## 7. 终点微调策略
当前不使用 STM32 本地原子 JOG 作为主线。SC171 侧用短时速度脉冲模拟微动作：

```text
MODE SEARCH
V <small_vx> <small_vy> <small_wz>
持续 duration_ms
STOP
等待 settle_ms
重新读取视觉 dist / yaw
```

原则：
- 每次微调后必须 STOP；
- STOP 后等待 `settle_ms`；
- 重新读取视觉，而不是假设动作一定到位；
- 多帧稳定后才进入 DONE；
- 不用单帧 edge 结果判断成功。

## 8. 近期必须实测 / 回填的参数
- SC171 实际串口设备路径与波特率；
- `FB` 是否能被 SC171 及时收到；
- `V 0.005 / 0.010 / 0.020 / 0.030 / 0.050` 的实际运动效果；
- 最小稳定前进速度；
- 最小稳定转向速度；
- 短时速度脉冲的位移：如 `V 0.020 0 0` 持续 100/200/300ms；
- STOP 后多走距离；
- 推荐 `stop_margin_cm`；
- 推荐 `settle_ms`；
- ROI preset 与桌边检测稳定性。

## 9. SC171 测试顺序
1. 不启动状态机，只用 motion probe 测：
   - `MODE SEARCH`
   - `V 0.020 0.000 0.000`
   - `STOP`
   - `FB` 回传
2. 确认轮向和停车；
3. 测不同 `vx/vy/wz` 的真实运动效果；
4. 测短时速度脉冲位移；
5. 固定摄像头，调试 ROI 和 edge 输出；
6. 开启 table-edge-only 状态机；
7. 确认 30cm 附近 STOP / settle / 视觉复检；
8. 再考虑目标搜索和抓取流程。

## 10. 已暂停 / 非当前主线
以下内容不作为当前主线依赖：
- 四轮 `VEL <s006> <s007> <s008> <s009> <seq>` 协议；
- STM32 本地 `JOG ... duration_ms seq` 协议；
- `STATUS` 主动查询；
- `[CAR][JOG_DONE]` 作为动作完成依据；
- PS2 手柄实时控制；
- STM32 自动搜索 / 自动探索；
- 超声波急停实时循环。

## 11. 当前一句话说明
当前系统采用“SC171 视觉闭环三轴速度控制 + STM32 麦克纳姆解算与低速插帧”的方案：SC171 发送 `MODE / V / STOP`，STM32 负责低速执行，真实停靠精度依靠视觉复检、提前停车和短时速度脉冲微调实现。
