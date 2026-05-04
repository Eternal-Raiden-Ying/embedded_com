# TODO

更新时间：2026-05-04

状态约定：
- `todo`：未开始
- `doing`：正在推进
- `done`：当前阶段已完成
- `blocked`：存在外部依赖或前置条件未满足

## P0 基础重构

### 1. 坐标系转换模块化
- 状态：`done`
- 优先级：`P0`
- 已推进内容：
  - 已新增 `grasp_module/backend/utils/frames.py`
  - 已封装 `camera -> output -> calibration -> robot` 以及直接 `camera -> robot`
  - 对外协议位置与角度已经切到 `robot` 坐标系
  - debug 点云与 grasp mesh 仍保留在原始相机系
- 测试：
  - `py_compile` 通过
  - 单图推理验证通过

### 2. 下游输出格式完善与冻结
- 状态：`done`
- 优先级：`P0`
- 已推进内容：
  - 新建 `grasp_module/backend/protocol/` 包，提取 `build_downstream_response` 为唯一实现
  - 消除 `server_app.py` 与 `test/utils/reporting.py` 之间的重复代码
  - 响应新增 `format_version: "1.1"` 字段
  - `message` 字段在所有 6 个分支中填充有意义的调试信息
  - status 三分法（success / reposition_required / failure）
  - 所有响应新增 `detection` 对象，含 `similar_detection_result` 标记回退
  - YOLO `no_detection` 与 GraspNet `no_grasp_detected` 分层为 failure
  - 配置中明确了 debug gripper mesh 参数与碰撞检测参数的两组独立角色
  - `grasp_module/docs/api_protocol.md` — 完整板端通信协议文档
- 参考基准（两个 bag 已验证可输出有效抓取）：
  - `grasp_module/test/data/bag/20260426_165604.bag` — class_id=47 直接命中
  - `grasp_module/test/data/bag/20260426_170719.bag` — fallback 55 成功
  - 推理结果保存于 `grasp_module/test/bag_debug/`

### 3. 方向约束重构
- 状态：`done`
- 优先级：`P0`
- 已推进内容：
  - `feasible_angle_deg` → `feasible_distance_cm`：从「approach 在 XZ 平面」改为「approach 直线到参考 Z 线空间距离 < 阈值」
  - 过滤顺序统一：碰撞 → 方向过滤 → NMS（仅一次）
  - pitch_deg 重定义：P 面（含 L+g 的垂直面）投影后的仰角
  - roll_deg 重定义：垂直 v_proj 平面内绕 approach 轴的旋转角
  - 新增 `build_reposition_proposal`：无 feasible 时生成 XY 移动建议（Step1 dx=0 → Step2 放宽）
  - 协议升级 v1.2，新增 `reposition_proposal` 字段
  - 新增配置参数：`reference_line_x_cm`/`reference_line_y_cm`/`reposition_max_distance_cm`
- 测试：两个基准 bag 验证通过（feasible 数略降但正常输出）

### 4. 中间结果可观测性
- 状态：`doing`
- 优先级：`P0`
- 已推进内容：
  - 单图 / bag 调试已支持保存：
    - `yolo_overlay`
    - `masked_cloud.ply`
    - `scene_cloud.ply`
    - `grasps_top15_heatmap.ply`
    - `best_protocol_grasp.ply`
  - 新增 YOLO 预检 summary 和 detection route 落盘
- 待补充：
  - 碰撞剔除 grasp 的单独可视化
  - 按剔除原因分类输出

## P1 上游感知稳定

### 5. YOLO 偶发漏检排查
- 状态：`doing`
- 优先级：`P1`
- 已推进内容：
  - 已扫描多组 bag 导出帧
  - 已完成 `conf / iou / imgsz` 对比
  - 已确认当前主要瓶颈是 `apple=47` 置信度偏低与类别混淆
  - 已新增 fallback 路由：`47 -> 32 -> 55`
  - 两个基准 bag 已验证 fallback 可用
- 后续方向：
  - 补 `require_single_detection`
  - 评估是否做定向微调

### 6. 3D 投影畸变与线性度问题
- 状态：`todo`
- 优先级：`P1`
- 已知方向：
  - 对比当前反投影点云与 RealSense SDK 原生 pointcloud
  - 核查 `align_mode / intrinsics / factor_depth`
- 测试：未开始

### 7. 深度后处理验证
- 状态：`doing`
- 优先级：`P1`
- 已推进内容：
  - 中值滤波 + 零值孔洞填充已在 `test_engine.py` 中实现（`postprocess_depth_image`）
  - RealSense SDK 官方 filter chain 封装在 `bag_io.py`（`apply_rs_official_filters`）
- 待完成：
  - 双边滤波
  - spatial / temporal / hole filling 效果对比
  - **暂不集成到 `engine.py` 主分支**，需更多测试数据验证后再决定
- 测试：test 层进行中

### 8. 边界缺失问题
- 状态：`todo`
- 优先级：`P1`
- 已知方向：
  - 深度后处理
  - 多帧选优 / 多帧融合
  - 局部 ROI 约束
- 测试：未开始

## P2 机器人约束与执行语义

### 9. 夹爪 / 机械臂执行语义重定义
- 状态：`doing`
- 优先级：`P2`
- 已推进内容：
  - 已自建 `grasp_module/backend/utils/gripper_mesh.py`，不再依赖 graspnetAPI 运行时签名
  - 已接入 config 中定义的物理世界夹爪参数
  - 已明确 gripper_mesh **仅作用于 debug PLY 可视化**，不影响碰撞检测
  - 碰撞检测使用独立参数组（`collision_finger_width_m` 等）
- 待实现：
  - 明确 `grasp pose / gripper pose / tcp pose` 三者语义
  - 确认是否需要在碰撞检测中统一使用同一套夹爪几何参数
- 测试：未开始

### 10. 机械臂约束过滤优化
- 状态：`doing`
- 优先级：`P2`
- 已推进内容：
  - 已在机械臂坐标系下计算：
    - `pitch_deg`（P 面投影仰角）
    - `roll_deg`（绕 approach 轴旋转）
    - `feasible_distance_cm`（L-A 线距离）
  - 已接入给定的 `R,t`
- 待补充：
  - `pitch` 区间硬过滤
  - IK / reachability 检查

## P3 碰撞、评分与搜索空间优化

### 11. 可视化被碰撞筛掉的 grasp
- 状态：`todo`
- 优先级：`P3`
- 待实现：
  - kept grasps
  - collision rejected grasps
  - collision scene cloud
- 测试：未开始

### 12. 场景点云选取优化
- 状态：`todo`
- 优先级：`P3`
- 待实现：
  - 比较 bbox 局部点云 / 深度邻域点云 / mask 外扩点云
- 测试：未开始

### 13. 评分与筛选逻辑优化
- 状态：`todo`
- 优先级：`P3`
- 待实现：
  - 结合点云完整度、边界缺失、可达性、碰撞余量
- 测试：未开始

### 14. 调整 seed / M_POINT / 搜索空间
- 状态：`blocked`
- 优先级：`P3`
- 原因：
  - 上游点云质量和机器人约束尚未稳定
  - 当前 `M_POINT = 1024` 与模型结构和训练分布耦合
- 当前结论：
  - 现阶段不优先推进

## P4 自动化与工具

### 15. 自动化手眼标定流程
- 状态：`todo`
- 优先级：`P4`
- 已推进内容：
  - 已有 `handeye_from_bag.py`
  - 已支持 bag 抽帧、点云导出、z 裁剪、浏览版点云
- 测试：
  - 已用真实 bag 数据验证脚本可用
- 待补充：
  - 自动配对与误差报告

### 16. 测试脚本体系整理
- 状态：`doing`
- 优先级：`P4`
- 已推进内容：
  - 已新增 `grasp_module/test/utils/`
  - `test_engine.py` 已改为：
    - 先做 YOLO 预检
    - 再按选中单帧做 grasp
    - 延迟导入重型 grasp 模型
  - 已新增 `test_yolo.py`
  - `handeye_from_bag.py` 已改为复用共享 bag 工具
- 测试：
  - `py_compile` 待本轮统一回归
  - bag 数据回归待本轮统一验证
