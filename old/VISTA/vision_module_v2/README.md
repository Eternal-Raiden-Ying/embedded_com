# vision_module_v2

备选视觉链路，不覆盖当前 `vision_module`。

这版的重点不是替换你现在的主链，而是把 `QNN` 推理出来的 `mask` 直接带到 application 层，并把输出格式提前向后续联调协议靠齐。

当前特性:

- 复用现有 `vision_module.backend.vision_engine` 和模型配置，不改动老模块。
- 在 `EDGE_TARGET_SEARCH` 模式下，application 会为当前目标生成带 `mask_ready / mask_shape / mask_area_ratio / mask_bbox` 的 `target_obs`。
- 在 `TABLE_EDGE_SEARCH` 模式下，当前仅输出代理版 `table_edge_obs`。
  说明:
  深度桌边检测仍等待 `Offline_Edge_Test` 验证结果，这里只保留了协议位置和后续插拔点，不默认启用假的边缘控制量。

建议用途:

- 联调前先验证 “请求进来 -> 模型跑起来 -> mask 进入 application -> 目标消息出去” 这条链。
- 后续你完成 `Offline_Edge_Test` 验证后，可以把桌边提取器接到这里，再决定是否替换当前主视觉版本。

运行方式:

```bash
py -3 VISTA/vision_module_v2/app.py
```
