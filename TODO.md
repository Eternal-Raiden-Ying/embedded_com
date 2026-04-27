# TODO

更新时间：2026-04-26

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

### 2. 中间结果可观测性
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

### 3. YOLO 偶发漏检排查
- 状态：`doing`
- 优先级：`P1`
- 已推进内容：
  - 已扫描多组 bag 导出帧
  - 已完成 `conf / iou / imgsz` 对比
  - 已确认当前主要瓶颈是 `apple=47` 置信度偏低与类别混淆
  - 已新增 `test_yolo.py` 和测试层 fallback：
    - `47 -> 32 -> 55`
- 测试：
  - 现有 bag 数据已做回归
  - 默认 `bag_top_k=1` 已验证
- 后续方向：
  - 补 `require_single_detection`
  - 评估是否做定向微调

### 4. 3D 投影畸变与线性度问题
- 状态：`todo`
- 优先级：`P1`
- 已知方向：
  - 对比当前反投影点云与 RealSense SDK 原生 pointcloud
  - 核查 `align_mode / intrinsics / factor_depth`
- 测试：未开始

### 5. 深度后处理验证
- 状态：`todo`
- 优先级：`P1`
- 已知方向：
  - 中值滤波
  - 双边滤波
  - spatial / temporal / hole filling
- 测试：未开始

### 6. 边界缺失问题
- 状态：`todo`
- 优先级：`P1`
- 已知方向：
  - 深度后处理
  - 多帧选优 / 多帧融合
  - 局部 ROI 约束
- 测试：未开始

## P2 机器人约束与执行语义

### 7. 夹爪 / 机械臂执行语义重定义
- 状态：`todo`
- 优先级：`P2`
- 已推进内容：
  - 当前已区分模型 grasp 输出与协议输出
- 待实现：
  - 明确 `grasp pose / gripper pose / tcp pose`
  - 引入真实夹爪参数与 TCP 外参
- 测试：未开始

### 8. 机械臂约束过滤优化
- 状态：`doing`
- 优先级：`P2`
- 已推进内容：
  - 已在机械臂坐标系下计算：
    - `pitch_deg`
    - `roll_deg`
    - `feasible_angle_deg`
  - 已接入你给定的 `R,t`
  - 已做 `5 / 8 / 10 / 15 deg` 阈值对比
- 待补充：
  - `pitch` 区间优先或硬过滤
  - IK / reachability 检查

## P3 碰撞、评分与搜索空间优化

### 9. 可视化被碰撞筛掉的 grasp
- 状态：`todo`
- 优先级：`P3`
- 待实现：
  - kept grasps
  - collision rejected grasps
  - collision scene cloud
- 测试：未开始

### 10. 场景点云选取优化
- 状态：`todo`
- 优先级：`P3`
- 待实现：
  - 比较 bbox 局部点云 / 深度邻域点云 / mask 外扩点云
- 测试：未开始

### 11. 评分与筛选逻辑优化
- 状态：`todo`
- 优先级：`P3`
- 待实现：
  - 结合点云完整度、边界缺失、可达性、碰撞余量
- 测试：未开始

### 12. 调整 seed / M_POINT / 搜索空间
- 状态：`blocked`
- 优先级：`P3`
- 原因：
  - 上游点云质量和机器人约束尚未稳定
  - 当前 `M_POINT = 1024` 与模型结构和训练分布耦合
- 当前结论：
  - 现阶段不优先推进

## P4 自动化与工具

### 13. 自动化手眼标定流程
- 状态：`todo`
- 优先级：`P4`
- 已推进内容：
  - 已有 `handeye_from_bag.py`
  - 已支持 bag 抽帧、点云导出、z 裁剪、浏览版点云
- 测试：
  - 已用真实 bag 数据验证脚本可用
- 待补充：
  - 自动配对与误差报告

### 14. 测试脚本体系整理
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
