# Next TODO

更新时间：2026-05-04

## P0 — 方向约束重构（刚完成，待验证）

- [x] 方向约束：`feasible_angle_deg` → `feasible_distance_cm`（approach 直线到参考 Z 线空间距离）
- [x] 过滤顺序统一：碰撞 → 方向过滤 → NMS（只做一次）
- [x] pitch_deg 重定义：P 面投影后的仰角
- [x] roll_deg 重定义：垂直 v_proj 平面内绕 approach 轴旋转
- [x] `build_reposition_proposal`：Step1(dx=0) → Step2(放宽)
- [x] protocol v1.2：`reposition_proposal` 字段、format_version 升级、文档更新
- [ ] 两个基准 bag 回归重跑（确认 feasible 数不减少）
- [ ] 参考线 (lx, ly) 标定值填入 config

## P1 — 深度后处理 & 点云质量

- [ ] 双边滤波补到 `postprocess_depth_image`，与当前中值滤波做 A/B 对比
- [ ] RealSense SDK filter chain vs 自研后处理的点云质量对比
- [ ] **暂不集成到 engine.py**

## P2 — 机械臂约束

- [ ] 机械臂约束补 pitch 硬过滤
- [ ] 调研 IK / reachability 检查的接入方式（可能 blocked）
- [ ] 确认碰撞检测与 debug mesh 参数是否需要统一

## P3 — 可视化补全

- [ ] 碰撞剔除 grasp 单独可视化为 PLY
- [ ] 按剔除原因分类输出

## P4 — 测试基准维护

- [ ] 两个基准 bag 推理结果纳入正式回归用例
- [ ] 后续改动 `engine.py`、`collision_detector.py`、`frames.py` 需重跑两个 bag
