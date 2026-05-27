import cv2
import numpy as np
from dataclasses import dataclass
from sklearn.linear_model import RANSACRegressor

@dataclass
class CameraCalib:
    fx: float
    fy: float
    cx: float
    cy: float
    depth_scale: float = 0.001

@dataclass
class EdgeDetectResult:
    edge_found: bool
    yaw_err_rad: float
    dist_err_m: float
    edge_confidence: float

class TableEdgeDetector:
    def __init__(self, calib: CameraCalib, target_dist_m: float = 0.5):
        self.calib = calib
        self.target_dist_m = target_dist_m
        
        # --- 核心调试参数 (你可以在 Windows 上反复调这几个值) ---
        self.roi_y = (100, 380)       # 垂直方向画面截取
        self.roi_x = (100, 540)       # 水平方向画面截取
        self.z_min = 0.2              # 最小有效深度 (米)
        self.z_max = 2.0              # 最大有效深度 (米)
        self.table_height_range = (0.6, 0.9) # 预期桌子高度区间 (相对相机Z轴偏下)

    def process_offline(self, depth_image_16bit: np.ndarray) -> tuple:
        """专供离线测试使用，返回所有中间变量供 matplotlib 渲染"""
        # 1. 深度预处理
        valid_mask, depth_meters = self._preprocess_depth(depth_image_16bit)
        
        # 2. 深度转 3D 点云 (以相机为原点的局部坐标系)
        pc_cam = self._depth_to_3d(depth_meters, valid_mask)
        if len(pc_cam) < 1000:
            return EdgeDetectResult(False, 0, 0, 0), depth_meters, pc_cam, None
            
        # 3. 寻找高度符合桌面的点云
        table_mask = self._find_table_plane(pc_cam)
        table_pc = pc_cam[table_mask]
        if len(table_pc) < 500:
            return EdgeDetectResult(False, 0, 0, 0), depth_meters, pc_cam, table_pc

        # 4. 提取前沿点云 (在 Y 轴高度对齐的情况下，看 X-Z 平面分布)
        edge_points = table_pc # 纯离线验证先不过滤边缘点，看整体拟合

        # 5. 直线拟合
        success, yaw_err, dist_err, conf, line_params = self._fit_edge_line(edge_points)
        
        res = EdgeDetectResult(success, yaw_err, dist_err, conf)
        return res, depth_meters, pc_cam, table_pc

    def _preprocess_depth(self, depth_img):
        depth_roi = depth_img[self.roi_y[0]:self.roi_y[1], self.roi_x[0]:self.roi_x[1]]
        depth_filtered = cv2.medianBlur(depth_roi, 5)
        depth_m = depth_filtered.astype(np.float32) * self.calib.depth_scale
        mask = (depth_m > self.z_min) & (depth_m < self.z_max)
        return mask, depth_m

    def _depth_to_3d(self, depth_m, mask):
        h, w = depth_m.shape
        u, v = np.meshgrid(np.arange(self.roi_x[0], self.roi_x[1]), np.arange(self.roi_y[0], self.roi_y[1]))
        u_valid, v_valid, z_valid = u[mask], v[mask], depth_m[mask]

        # 转换为相机坐标系下的 3D 点 (注意：多数相机坐标系 Z 向前，X 向右，Y 向下)
        x_c = (u_valid - self.calib.cx) * z_valid / self.calib.fx
        y_c = (v_valid - self.calib.cy) * z_valid / self.calib.fy
        
        return np.vstack((x_c, y_c, z_valid)).T

    def _find_table_plane(self, pc_cam):
        """利用 Y 轴 (垂直向下的轴) 进行高度筛选"""
        # 注意：此处高度 Y 取决于相机姿态。如果相机水平正视前方，Y轴就是高度方向
        y = pc_cam[:, 1]  
        # 因为相机Y轴向下，地面可能是个正值，桌子相对地面偏上，Y值较小
        # 你需要在 Windows 上通过画图观察你的具体高度分布并修改这里
        # 这里仅作示意，暂且取 Y 在 -0.2 到 0.2 之间的点
        return (y > -0.2) & (y < 0.2) 

    def _fit_edge_line(self, edge_points):
        # 取 X (左右) 和 Z (前后) 进行俯视图 2D 拟合
        X_axis = edge_points[:, 0].reshape(-1, 1) # 左右
        Z_axis = edge_points[:, 2].reshape(-1, 1) # 距离
        
        try:
            ransac = RANSACRegressor(min_samples=10, residual_threshold=0.05)
            ransac.fit(X_axis, Z_axis)
            k = ransac.estimator_.coef_[0][0]
            b = ransac.estimator_.intercept_[0]
            
            # 计算误差
            yaw_err = np.arctan(k)
            dist_err = b - self.target_dist_m
            conf = np.sum(ransac.inlier_mask_) / len(edge_points)
            
            return True, float(yaw_err), float(dist_err), float(conf), (k, b)
        except Exception:
            return False, 0.0, 0.0, 0.0, None