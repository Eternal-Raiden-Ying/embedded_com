# ROBOT_MOTION_CONTRACT

## 1. 当前目标
小车用于盲人取物平台。当前重点不是高速运动，而是：
- 低速稳定靠近桌边；
- 能稳定停在目标距离附近；
- 支持 SC171 与 STM32 之间统一运动协议；
- 支持终点附近 JOG 微动作。

## 2. STM32 新协议
SC171 发给 STM32：

VEL <s006> <s007> <s008> <s009> <seq>
STOP <seq>
JOG <s006> <s007> <s008> <s009> <duration_ms> <seq>
STATUS

参数：
- s006/s007/s008/s009 是四个轮子的低速控制值；
- 范围：-100 ~ 100；
- 0 表示停止；
- seq 是递增命令号；
- duration_ms 建议范围：20 ~ 1000ms。

STM32 回传：
[CAR][JOG_START] seq=...
[CAR][JOG_DONE] seq=...
[CAR][JOG_BUSY] seq=...
[CAR][TIMEOUT] auto stop

## 3. 四轮 ID
006 / 007 / 008 / 009

## 4. STM32 已有机制
- car_run_raw_pwm()
- car_stop_raw()
- car_run_slow()
- MotorCalib dz_pos / dz_neg / max_offset
- car_set_target_slow()
- car_motion_update_20ms()
- CAR_CMD_TIMEOUT_MS
- car_start_jog()
- CAR_TEST_MODE_* 测试入口

## 5. 明天必须测出的参数
- 每个轮子的正向稳定启动 P
- 每个轮子的反向稳定启动 P
- dz_pos / dz_neg
- max_offset
- 推荐连续低速 speed
- 推荐 JOG speed / duration
- STOP 后多走距离
- 推荐 stop_margin_cm
- 四轮方向映射是否正确

## 6. SC171 侧原则
- 状态机不直接拼底层串口字符串；
- 先用独立脚本测试 STM32 协议；
- 再接入状态机；
- 桌边最终停靠使用 STOP_AND_SETTLE + MICRO_ADJUST。