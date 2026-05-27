import cv2
import json
import numpy as np
import matplotlib.pyplot as plt
from TableEdgeDetector import TableEdgeDetector, CameraCalib

def load_config(json_path):
    with open(json_path, 'r') as f:
        data = json.load(f)
    calib = CameraCalib(
        fx=data['fx'], fy=data['fy'], cx=data['cx'], cy=data['cy'], 
        depth_scale=data['depth_scale']
    )
    return calib, data['target_dist_m']

def main():
    # 1. 配置文件和图像路径
    config_path = "calib.json"
    depth_image_path = "test_data/frame_001.png"

    # 2. 读取配置与初始化
    calib, target_dist = load_config(config_path)
    detector = TableEdgeDetector(calib, target_dist_m=target_dist)

    # 3. 读取 16位 深度图 (重要：IMREAD_ANYDEPTH 保证数据不失真)
    depth_raw = cv2.imread(depth_image_path, cv2.IMREAD_ANYDEPTH)
    if depth_raw is None:
        print(f"❌ 找不到深度图文件: {depth_image_path}")
        return

    print("✅ 成功加载深度图，正在进行几何解算...")
    
    # 4. 执行离线测试
    res, depth_meters, pc_all, pc_table = detector.process_offline(depth_raw)

    print(f"\n--- 解算结果 ---")
    print(f"找寻桌边成功: {res.edge_found}")
    print(f"偏航角误差 (Yaw): {res.yaw_err_rad * 180 / np.pi:.2f} 度")
    print(f"距离误差 (Dist): {res.dist_err_m:.2f} 米")
    print(f"置信度: {res.edge_confidence:.2f}")

    # 5. 使用 Matplotlib 强大的可视化功能进行诊断
    fig = plt.figure(figsize=(16, 5))

    # 子图1：深度热力图
    ax1 = fig.add_subplot(131)
    img = ax1.imshow(depth_meters, cmap='jet')
    ax1.set_title("Filtered Depth Map (Meters)")
    fig.colorbar(img, ax=ax1, shrink=0.5)

    if pc_all is not None:
        # 子图2：3D 侧视图 (看高度分布，用于调参 self._find_table_plane)
        ax2 = fig.add_subplot(132)
        # 降采样显示以防卡顿
        step = 5 
        ax2.scatter(pc_all[::step, 2], pc_all[::step, 1], s=1, c='gray', alpha=0.3, label='All Points')
        if pc_table is not None:
            ax2.scatter(pc_table[::step, 2], pc_table[::step, 1], s=2, c='red', label='Table Points')
        
        ax2.set_xlabel('Z (Forward Distance)')
        ax2.set_ylabel('Y (Height - Downward)')
        ax2.invert_yaxis() # Y轴朝下
        ax2.set_title("Side View (Y-Z Plane)")
        ax2.legend()

        # 子图3：2D 俯视雷达图 (看边缘拟合情况)
        ax3 = fig.add_subplot(133)
        if pc_table is not None:
            ax3.scatter(pc_table[::step, 0], pc_table[::step, 2], s=2, c='blue', alpha=0.5)
        
        ax3.plot(0, 0, marker='^', color='red', markersize=15, label='Camera/Robot')
        
        # 绘制雷达拟合线 (如果找边成功)
        if res.edge_found:
            # y = kx + b (这里的 y 是 Z_axis, x 是 X_axis)
            yaw = res.yaw_err_rad
            # 根据计算反推线段端点
            x_vals = np.array([-0.5, 0.5])
            z_vals = x_vals * np.tan(yaw) + (target_dist + res.dist_err_m)
            ax3.plot(x_vals, z_vals, color='green', linewidth=3, label='Fitted Edge')

        ax3.set_xlabel('X (Left/Right)')
        ax3.set_ylabel('Z (Forward Distance)')
        ax3.set_xlim([-1, 1])
        ax3.set_ylim([0, 2])
        ax3.set_title("Top-Down BEV Radar (X-Z Plane)")
        ax3.legend()

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()