# VISTA 产品功能需求

本文档描述 VISTA 当前仍然成立的产品目标与能力要求。

它不是历史实现说明，也不应携带已经失效的命名假设。

## 1. 产品定位

VISTA 是部署在端侧设备上的视觉能力模块，服务于机器人在真实场景中的：

- 目标搜索与定位
- 抓取前观测与远程抓取协作
- 回航目标观测
- 后续感知扩展

核心原则：

- 端侧优先，尽量减少不必要的资源消耗
- 能力解耦，相机 / 推理 / 深度 / 网络各自独立
- 模式化运行，根据任务目标切换资源组合
- 既支持正式运行，也支持开发调试与验证
- 协议稳定性优先于临时兼容逻辑

## 2. 当前核心功能目标

### 2.1 本地目标定位与循迹

目标：

- 使用彩色相机作为主视觉输入
- 使用本地 NPU 推理模型完成目标检测或分割
- 输出目标观测结果供上层控制模块使用
- 在端侧实现低延迟、较高流畅度的目标跟踪

当前基线要求：

- 本地主线必须保留 `coco80 detect`
- 未来可以扩展或切换到更窄类别集，但不能以删除 `coco80 detect` 为前提
- 本地处理优先，减少带宽消耗
- 输入分辨率当前以 `640x640` 路径为主
- 需要通过曝光/亮度策略减少运动残影并保证可见性

### 2.2 远程抓取预测

目标：

- 使用彩色相机与深度相机联合采样
- 采集抓取所需的 RGB 与 Depth 数据
- 将抓取所需数据发送到远程服务做抓取姿态预测
- 获取抓取结果后供上层执行抓取或微调

当前收口要求：

- remote 路径以 `class_id` 作为目标类别输入
- `class_id` 的真值来源应为外部输入，不应依赖端内猜测
- 支持按需触发，不进行持续高频请求
- depth 数据是抓取链路关键输入
- 上传编码不能过早收缩到单一格式，当前应支持至少 `png/jpeg` 可配置
- remote `service INIT -> PREDICT` 必须是显式有 gate 的顺序，不允许无确认直冲 `PREDICT`
- `GRASP_REMOTE` 需要基于新 mode 的 fresh frame 再触发 `PREDICT`

当前质量要求：

- 远程抓取模式应优先保障数据质量，而不是持续高频运行
- 是否使用 JPEG 压缩，应结合实际带宽与效果评估决定

### 2.3 开发调试能力

目标：

- 提供 camera demo，验证真实相机链路
- 提供 backend smoke test，验证 camera / predictor / pipeline
- 调试时可以查看画面、帧率、基础状态
- 正式运行时可关闭调试显示，减少资源开销

## 3. 已明确的模式需求

### 3.1 `TRACK_LOCAL`

目标：

- 本地目标搜索、定位、循迹

资源组合：

- Color camera
- Local AI infer
- 不启用 depth
- 不启用 remote network grasp

要求：

- 优先低延迟与稳定帧率
- 尽量少占用端侧资源
- 相机参数要支持自动曝光/亮度策略
- 默认 detect 基线必须可直接驱动上游需要的目标观测输出

### 3.2 `GRASP_REMOTE`

目标：

- 远程抓取预测

资源组合：

- Color camera
- Depth camera
- Remote network grasp service
- 本地 AI 推理默认不持续开启

要求：

- 支持获取 remote 所需的 RGB + Depth 数据
- `PREDICT` 必须发生在服务器 `INIT` 完成之后
- `PREDICT` 必须发生在新 mode 的新 frame 就绪之后，而不是仅凭 mode 切换成功立即触发
- 支持将抓取相关图像与元数据发送到远端
- 支持在需要时回收本地无关资源
- 如果 server `INIT` 可在 service start 时完成且不消耗本地关键资源，可以采用该策略

### 3.3 `MICRO_ADJUST`

目标：

- 在抓取预测前后，根据视觉反馈进行位置微调

适用场景：

- 目标距离不合适
- 夹爪姿态受物理限制
- 当前位姿下不能直接抓取

### 3.4 待扩展模式

以下仍属于扩展方向，不是当前默认 runtime 基线：

- `DEPTH_PERCEPTION`
- 其他导航辅助相关 mode

## 4. 相机能力需求

### 4.1 Color camera

- 独立 backend 实现
- 面向本地追踪与高质量抓取采样
- 支持曝光与亮度控制接口
- 支持不同 profile：
  - 低延迟 profile
  - 高质量抓取 profile

当前收口要求：

- 相机参数如分辨率、裁剪、输出格式，应能进入 mode/profile 所有权
- `GRASP_REMOTE` 需要能够单独指定其 capture contract，而不是隐式继承 local tracking 默认值
- remote 上传编码默认也应属于 mode/profile 所有权，而不是散落在 manager 内部常量

### 4.2 IR camera

- 独立 backend 实现
- 输入 / 输出格式必须与实际硬件兼容
- 当前约束：
  - 输入格式可为 `GRAY8` 或 `UYVY`
  - 输出格式可为 `RGB` 或 `BGR`
- 不应再复用 color camera 的参数假设

### 4.3 Depth camera

- 独立 backend 实现
- 优先使用专用 RealSense 通路
- 调试可视化需进行彩色渲染
- 需考虑近距离盲区与有效工作距离
- 在 remote grasp 路径中应作为明确 capability，而不是隐含附属项

## 5. Predictor 能力需求

- Predictor 层应明确继承抽象基类
- 真实实现与 mock 实现接口一致
- 应同时允许 detect 和 segment 两类 predictor profile
- 公共能力命名不应再强制围绕 `Segment` 命名
- 日志统一走 logging，不使用散落的 print
- 支持资源释放与重复调用的幂等性

当前收口要求：

- 默认 detect 主线必须是正式支持的产品能力，不是临时兼容项
- stage 侧目标解析不能假设只有单一全局 class vocabulary
- class vocabulary 应与 active model/profile 语义对齐，并随 `local_perception.class_names` 下发
- detect `local_perception` 需稳定发布显式 box contract；当前基线为 `infer_boxes = [x1, y1, x2, y2, score, class_id]`
- 非 mock 运行中，runtime 主路径不能被包级别的隐式 mock alias 静默掩盖

## 6. Engine 与运行时能力需求

- Engine 不应只是推理循环，还应承担资源编排职责
- Engine 需要支持模式切换
- 模式切换不能全量重建所有资源，应尽量做差量更新
- 共用能力在模式切换时应尽量复用
- 调试预览应作为旁路能力，而不是模式核心逻辑的一部分

当前收口要求：

- 如果 mode profile 中声明了运行时策略字段，例如 `release_cooldown_s`，则运行时应提供真实语义；否则应删除装饰性字段
- remote request 的最小 contract 应在 stage / mode / manager 之间有清晰所有权边界

## 7. 性能与资源要求

- 端侧部署优先考虑开销
- 默认状态下不应打开不必要的 camera / infer / preview / network
- 模式切换时避免重复初始化相机与模型
- 本地追踪模式应优先保障实时性
- 远程抓取模式应优先保障数据质量
- 上传编码策略应允许根据实际带宽和抓取效果进行调整，而不是写死单一路径

## 8. 可靠性要求

- 相机、predictor、engine 都需要支持显式 release
- 初始化失败时不能导致析构阶段二次崩溃
- mock 测试路径要稳定可用
- real 测试路径要能输出明确错误原因
- 模式切换、stop、重复 release 应保持幂等

当前收口要求：

- remote `service INIT -> PREDICT` 时序必须可验证，`RELEASE` 应走 shutdown / disable / explicit reset 路径
- remote `PREDICT` 不能依赖 race timing 才成功
- 未显式提供 `class_id` 的 remote grasp 请求应被清晰拒绝，而不是靠端内猜测补齐
- 非 mock 路径下，真实后端故障不应通过静默 fallback 掩盖

## 9. 测试需求

- 提供独立 backend 测试脚本：
  - `test_sensors.py`
  - `test_predictor.py`
  - `test_pipeline.py`
  - `test_color_controls.py`
- 提供独立 camera demo：
  - `demo_camera.py`
- mock 与 real 路径都应可验证
- 开发阶段优先保证 backend 自身逻辑正确，再逐步推进 app 层适配

对于当前收口重点，还应额外验证：

- detect real path 的真实输出是否能稳定穿过 manager -> stage 边界
- detect class vocabulary 是否始终跟随 active model/profile，而不是退回全局硬编码表
- remote `class_id` 是否始终来自显式外部输入
- remote `INIT` 完成 gate 是否真实存在
- `GRASP_REMOTE` 是否只在 fresh frame 到达后触发 `PREDICT`
- 上传编码切换是否影响结果质量与网络表现
## 2026-04 Requirement Clarifications

- Camera and predictor runtime backend selection must be controlled by `VISTA_BACKEND`, not by `capability_placeholder`.
- Detect fallback must remain operational on the default `coco80` line even when model-profile `class_names` are missing, and that weakening must be explicit in runtime diagnostics.
- Manager boundaries must reject or downgrade malformed detect rows explicitly. Bad detect payloads must not fail only by being swallowed inside stage code.
- Mode switches must not stall frame consumers after scheduler slot reset. Predictor and preview workers must treat `(generation, seq)` as the stream cursor.
- The default camera color baseline is BGR. Detect path validation and debug tooling must follow BGR unless a model-specific adapter explicitly converts it.
- `TRACK_LOCAL`, `MICRO_ADJUST`, and `GRASP_REMOTE` must own distinct RGB-camera capture profiles rather than all inheriting one board-level default unchanged.
- Legacy predictor aliases and debug entrypoints that bypass the supported runtime path should be removed instead of kept as parallel surfaces.
