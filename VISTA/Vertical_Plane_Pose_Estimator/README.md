# Vertical_Plane_Pose_Estimator

一套全新的“基于桌前竖直平面检测”的轻量姿态解算模块。

设计目标：

- 只做轻量深度几何估计，不做大而全点云系统
- 输入：单帧深度图
- 输出：`yaw_err`、`dist_err`、`confidence`
- 满足 5 到 10 Hz 控制级需求
- 先以 Python + NumPy 离线验证为主

## 文件说明

- `schema.py`
  配置与数据结构定义
- `estimator.py`
  核心算法实现，末尾自带 mock test
- `depth_png_runner.py`
  用真实 depth png 直接跑一次估计

## 直接运行 mock test

```bash
python estimator.py
```

它会：

- 自动生成一个带高斯噪声与离群点的桌前竖直平面点云
- 调用估计器
- 打印：
  - 单次耗时
  - 平面方程
  - `yaw_err`
  - `dist_err`
  - `confidence`

## 用真实 depth png 跑

```bash
python depth_png_runner.py --depth-png your_depth.png --calib-json ../Offline_Edge_Test/calib.json
```
