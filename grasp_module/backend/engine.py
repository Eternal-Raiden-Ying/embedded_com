import os
import sys
import json
import numpy as np
import argparse
import time
import torch
import open3d as o3d
from graspnetAPI.graspnet_eval import GraspGroup

from .models.graspnet import GraspNet, pred_decode
from .utils.preprocess import minkowski_collate_fn
from .utils.collision_detector import ModelFreeCollisionDetector
from .utils.data_utils import CameraInfo, create_point_cloud_from_depth_image


def load_camera_metadata(metadata_path):
    """
    从 JSON 文件加载相机内参
    
    Args:
        metadata_path: 相机元数据 JSON 文件路径
        
    Returns:
        CameraInfo: 相机信息对象
    """
    if not os.path.exists(metadata_path):
        print(f"[Warning] Camera metadata file not found: {metadata_path}")
        return None
    
    with open(metadata_path, 'r') as f:
        metadata = json.load(f)
    
    depth_info = metadata.get('depth', {})
    factor_depth = metadata.get('factor_depth')
    depth_scale = metadata.get('depth_scale')

    if factor_depth is not None:
        scale = float(factor_depth)
        scale_source = 'factor_depth'
    elif depth_scale is not None:
        depth_scale = float(depth_scale)
        # 兼容历史 metadata: 若保存的是 RealSense depth_scale(米/单位)，
        # 这里转换成 create_point_cloud_from_depth_image 需要的除数 factor_depth。
        scale = 1.0 / depth_scale if depth_scale > 0 and depth_scale < 1 else depth_scale
        scale_source = 'depth_scale(converted)' if depth_scale < 1 else 'depth_scale'
    else:
        scale = 1000.0
        scale_source = 'default'
    
    camera_info = CameraInfo(
        width=float(depth_info.get('width', 1280)),
        height=float(depth_info.get('height', 720)),
        fx=float(depth_info.get('fx', 631.5)),
        fy=float(depth_info.get('fy', 631.2)),
        cx=float(depth_info.get('cx', 639.5)),
        cy=float(depth_info.get('cy', 359.5)),
        scale=scale
    )
    
    print(f"[Camera] Loaded intrinsics from {metadata_path}")
    print(f"  fx={camera_info.fx}, fy={camera_info.fy}")
    print(f"  cx={camera_info.cx}, cy={camera_info.cy}")
    print(f"  scale={camera_info.scale} ({scale_source})")
    
    return camera_info


class RealSenseGraspPredictor:
    def __init__(self, cfgs):
        self.cfgs = cfgs
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        
        camera_info = None
        
        if hasattr(cfgs, 'camera_metadata') and cfgs.camera_metadata:
            camera_info = load_camera_metadata(cfgs.camera_metadata)
        
        if camera_info is None:
            camera_info = CameraInfo(
                width=1280.0, 
                height=720.0, 
                fx=631.55,   
                fy=631.21,   
                cx=638.43,   
                cy=366.50,   
                scale=1000.0
            )
            print("[Camera] Using default camera intrinsics (graspnet kinect defaults)")
        
        self.camera_info = camera_info
        
        # 2. 初始化并加载模型
        print("Loading GraspNet model...")
        self.net = GraspNet(seed_feat_dim=cfgs.seed_feat_dim, is_training=False)
        self.net.to(self.device)
        
        checkpoint = torch.load(cfgs.checkpoint_path, map_location=self.device)
        self.net.load_state_dict(checkpoint['model_state_dict'])
        self.net.eval()
        print("-> Model loaded successfully.")

        if cfgs.debug and not os.path.exists(cfgs.dump_dir):
            os.makedirs(cfgs.dump_dir)

    def preprocess(self, color_img, depth_img, seg_mask):
        """
        处理输入的 RGB, Depth, Seg 为模型所需的点云数据
        color_img: np.ndarray (H, W, 3) 
        depth_img: np.ndarray (H, W) 
        seg_mask: np.ndarray (H, W) 背景通常为0，目标物体>0
        """
        # 生成点云
        cloud = create_point_cloud_from_depth_image(depth_img, self.camera_info, organized=True)

        # 获取有效点掩码：有深度，且在分割掩码内
        depth_mask = (depth_img > 0)
        # 实际使用中可能还需要一个固定 Workspace Bounding Box 过滤掉桌子或背景
        # workspace_mask = (cloud[:,:,2] < 1.0) & (cloud[:,:,2] > 0.1) # 简单的深度截断示例
        mask = (depth_mask & (seg_mask > 0)) 
        
        cloud_masked = cloud[mask]
        color_masked = color_img.reshape(-1, 3)[mask.flatten()]

        if len(cloud_masked) == 0:
            return None

        # 采样到固定点数 (默认 15000)
        if len(cloud_masked) >= self.cfgs.num_point:
            idxs = np.random.choice(len(cloud_masked), self.cfgs.num_point, replace=False)
        else:
            idxs1 = np.arange(len(cloud_masked))
            idxs2 = np.random.choice(len(cloud_masked), self.cfgs.num_point - len(cloud_masked), replace=True)
            idxs = np.concatenate([idxs1, idxs2], axis=0)
            
        cloud_sampled = cloud_masked[idxs]
        color_sampled = color_masked[idxs]
        
        ret_dict = {
            'point_clouds': cloud_sampled.astype(np.float32),
            'coors': cloud_sampled.astype(np.float32) / self.cfgs.voxel_size,
            'feats': np.ones_like(cloud_sampled).astype(np.float32),
            'colors': color_sampled.astype(np.float32) / 255.0,
        }
        return ret_dict

    def post_process_grasps(self, grasp_group):
        """
        【预留接口】：处理网络输出的抓取预测 (过滤、排序、NMS)
        你可以根据实际需求，在此处增加抓取宽度限制、方向限制等。
        """
        if grasp_group is None or grasp_group.__len__() == 0:
            return grasp_group

        # 1. NMS 过滤重叠抓取
        grasp_group = grasp_group.nms()
        
        # 2. 按分数排序
        grasp_group = grasp_group.sort_by_score()
        
        # 3. 过滤出置信度最高的前 N 个 (例如前 10 个)
        # if grasp_group.__len__() > 10:
        #     grasp_group = grasp_group[:10]
            
        return grasp_group

    def infer(self, color_img, depth_img, seg_mask):
        """
        供外部调用的实际推理接口
        """
        tic = time.time()
        
        # 1. 数据预处理
        data_dict = self.preprocess(color_img, depth_img, seg_mask)
        if data_dict is None:
            print("No valid points found in the masked region.")
            return None

        batch_data = minkowski_collate_fn([data_dict])
        for key in batch_data:
            if 'list' in key:
                for i in range(len(batch_data[key])):
                    for j in range(len(batch_data[key][i])):
                        batch_data[key][i][j] = batch_data[key][i][j].to(self.device)
            else:
                batch_data[key] = batch_data[key].to(self.device)

        # 2. 网络前向推理
        with torch.no_grad():
            end_points = self.net(batch_data)
            grasp_preds_list = pred_decode(end_points)
            
        preds = grasp_preds_list[0].detach().cpu().numpy()
        gg = GraspGroup(preds)

        # 3. 碰撞检测 (如果启用)
        if self.cfgs.collision_thresh > 0:
            cloud = data_dict['point_clouds']
            mfcdetector = ModelFreeCollisionDetector(cloud, voxel_size=self.cfgs.voxel_size_cd)
            collision_mask = mfcdetector.detect(gg, approach_dist=0.05, collision_thresh=self.cfgs.collision_thresh)
            gg = gg[~collision_mask]

        # 4. 后处理逻辑预留 (自定义过滤)
        gg = self.post_process_grasps(gg)

        print(f'-> Inference finished. Found {gg.__len__()} grasps. Time: {time.time() - tic:.4f}s')

        # 5. Debug 模式可视化
        if self.cfgs.debug:
            data_dict = self.preprocess(color_img, depth_img, np.ones_like(depth_img))
            self.save_debug_visualizations(data_dict, gg)

        return gg

    def save_debug_visualizations(self, data_dict, gg):
        """
        在 Debug 模式下将结果存盘，方便用 MeshLab 或 CloudCompare 查看
        """
        pc = data_dict['point_clouds']
        colors = data_dict['colors']
        
        # 如果抓取位姿太多，只画前30个避免文件过大
        vis_gg = gg[:30] if gg.__len__() > 30 else gg
        grippers = vis_gg.to_open3d_geometry_list()
        
        cloud = o3d.geometry.PointCloud()
        cloud.points = o3d.utility.Vector3dVector(pc.astype(np.float32))
        cloud.colors = o3d.utility.Vector3dVector(colors)
        
        # 合并所有夹爪为一个 Mesh
        combined_grippers = o3d.geometry.TriangleMesh()
        for g in grippers:
            combined_grippers += g
            
        cloud_path = os.path.join(self.cfgs.dump_dir, "debug_cloud.ply")
        mesh_path = os.path.join(self.cfgs.dump_dir, "debug_grasps.ply")
        
        o3d.io.write_point_cloud(cloud_path, cloud)
        o3d.io.write_triangle_mesh(mesh_path, combined_grippers)
        
        print(f"[DEBUG] Visualizations saved to:\n - {cloud_path}\n - {mesh_path}")

# ================= 测试运行逻辑 =================
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint_path', default='./weights/minkuresunet_kinect.tar')
    parser.add_argument('--dump_dir', help='Dump dir to save outputs', default='./debug_res')
    parser.add_argument('--seed_feat_dim', default=512, type=int, help='Point wise feature dim')
    parser.add_argument('--num_point', type=int, default=15000, help='Point Number [default: 15000]')
    parser.add_argument('--voxel_size', type=float, default=0.005, help='Voxel Size for sparse convolution')
    parser.add_argument('--collision_thresh', type=float, default=-1, help='Collision Threshold in collision detection [default: 0.01]')
    parser.add_argument('--voxel_size_cd', type=float, default=0.01, help='Voxel Size for collision detection')
    
    # 增加的 Debug 开关
    parser.add_argument('--debug', action='store_true', default=False, help='Enable debug mode to save point cloud and grasp meshes')
    cfgs = parser.parse_args()

    # 初始化推理引擎（只需执行一次）
    predictor = RealSenseGraspPredictor(cfgs)

    # ---------------- 模拟你的外部调用环境 ----------------
    # 假设以下数据是通过 ROS / OpenCV 相机驱动获取的 numpy 数组
    H, W = 720, 1280
    dummy_rgb = np.random.randint(0, 255, (H, W, 3), dtype=np.uint8)
    dummy_depth = (np.random.rand(H, W) * 1000).astype(np.uint16)
    dummy_seg = np.random.randint(0, 2, (H, W), dtype=np.uint8) # 0为背景，1为前景物体

    # 执行推理 (每次相机给到新帧时调用)
    grasp_results = predictor.infer(dummy_rgb, dummy_depth, dummy_seg)
    
    # 实际机械臂执行逻辑示例:
    # if grasp_results is not None and len(grasp_results) > 0:
    #     best_grasp = grasp_results[0]
    #     translation = best_grasp.translation
    #     rotation = best_grasp.rotation_matrix
    #     # ... 转换为 Robot TCP Pose 并执行 ...
