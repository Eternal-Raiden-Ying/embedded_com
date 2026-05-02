# Next TODO

更新时间：2026-05-02

## P0 — 下游输出格式冻结

- [x] protocol 包完成，`format_version: "1.1"`，status 三分法，detection 对象含 `similar_detection_result`
- [x] `summarize_response` 已适配新格式
- [x] `api_protocol.md` 板端通信协议文档已完成
- [ ] 板端联调确认协议字段可解析，无 breaking change

## P1 — 深度后处理 & 点云质量

- [ ] 双边滤波补到 `postprocess_depth_image`，与当前中值滤波做 A/B 对比
- [ ] RealSense SDK filter chain vs 自研后处理的点云质量对比（利用 `compare_bag_pointclouds.py`）
- [ ] 明确当前反投影与 SDK 原生 pointcloud 的偏差量级，写入对比报告
- [ ] **暂不集成到 engine.py**，确认效果后在 test 层再多跑几组 bag

## P2 — gripper 参数 & 机械臂约束

- [x] debug gripper mesh 与 collision 参数已明确为两组独立参数
- [ ] 确认碰撞检测是否需要与 debug mesh 统一（如统一则需评估对基准 bag 的影响）
- [ ] 机械臂约束补 pitch 硬过滤
- [ ] 调研 IK / reachability 检查的接入方式（需机械臂模型，可能 blocked）

## P3 — 可视化补全

- [ ] 碰撞剔除 grasp 单独可视化为 PLY
- [ ] 按剔除原因（collision / angle / score）分类输出

## P4 — 测试基准维护

- [ ] 两个基准 bag 的推理结果纳入正式回归用例
- [ ] 后续改动 `engine.py`、`collision_detector.py`、`frames.py` 需重跑两个 bag 确认无退化
