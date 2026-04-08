import os
import sys
import numpy as np
import argparse
import time
import torch
import open3d as o3d
from graspnetAPI.graspnet_eval import GraspGroup

from .models.graspnet import GraspNet, pred_decode
from .utils.preprocess import minkowski_collate_fn
from .utils.collision_detector import ModelFreeCollisionDetector
from .utils.data_utils import (
    CameraInfo,
    build_ply_output_path,
    create_colored_point_cloud_from_rgbd,
    filter_point_cloud_by_z,
    load_camera_info_from_metadata,
    write_open3d_point_cloud,
)


class RealSenseGraspPredictor:
    def __init__(self, cfgs):
        self.cfgs = cfgs
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        
        default_camera_info = CameraInfo(
            width=1280.0,
            height=720.0,
            fx=631.55,
            fy=631.21,
            cx=638.43,
            cy=366.50,
            scale=1000.0,
        )

        camera_info = load_camera_info_from_metadata(
            getattr(cfgs, 'camera_metadata', None),
            default_camera=default_camera_info,
        )

        if camera_info is None:
            camera_info = default_camera_info
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
        cloud_masked, color_masked = create_colored_point_cloud_from_rgbd(
            color_img,
            depth_img,
            self.camera_info,
            mask=seg_mask,
        )

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
            'colors': color_sampled.astype(np.float32),
        }
        return ret_dict

    def _score_to_color(self, score, min_score, max_score):
        if max_score - min_score < 1e-6:
            normalized = 1.0
        else:
            normalized = float((score - min_score) / (max_score - min_score))
        return (normalized, 0.0, 1.0 - normalized)

    def _build_grasp_mesh(self, grasp_group):
        combined_grippers = o3d.geometry.TriangleMesh()
        if grasp_group is None or len(grasp_group) == 0:
            return combined_grippers

        scores = grasp_group.scores
        min_score = float(scores.min())
        max_score = float(scores.max())
        for i in range(len(grasp_group)):
            grasp = grasp_group[i]
            color = self._score_to_color(float(grasp.score), min_score, max_score)
            combined_grippers += grasp.to_open3d_geometry(color=color)
        return combined_grippers

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
            self.save_debug_visualizations(color_img, depth_img, seg_mask, gg)

        return gg

    def save_debug_visualizations(self, color_img, depth_img, seg_mask, gg):
        """
        在 Debug 模式下将结果存盘，方便用 MeshLab 或 CloudCompare 查看
        """
        masked_points, masked_colors = create_colored_point_cloud_from_rgbd(
            color_img,
            depth_img,
            self.camera_info,
            mask=seg_mask,
        )
        scene_points, scene_colors = create_colored_point_cloud_from_rgbd(
            color_img,
            depth_img,
            self.camera_info,
            mask=None,
        )
        scene_points, scene_colors = filter_point_cloud_by_z(
            scene_points,
            scene_colors,
            z_min=0.0,
            z_max=getattr(self.cfgs, 'scene_max_depth', 3.0),
        )

        top_k = getattr(self.cfgs, 'debug_grasp_count', 15)
        vis_gg = gg[:top_k] if gg.__len__() > top_k else gg
        combined_grippers = self._build_grasp_mesh(vis_gg)

        masked_cloud_path = build_ply_output_path(self.cfgs.dump_dir, 'masked_cloud.ply')
        scene_cloud_path = build_ply_output_path(self.cfgs.dump_dir, 'scene_cloud.ply')
        grasp_mesh_path = build_ply_output_path(self.cfgs.dump_dir, 'grasps_top15_heatmap.ply')

        write_open3d_point_cloud(masked_cloud_path, masked_points, masked_colors)
        write_open3d_point_cloud(scene_cloud_path, scene_points, scene_colors)
        o3d.io.write_triangle_mesh(grasp_mesh_path, combined_grippers)

        print(
            f"[DEBUG] Visualizations saved to:\n"
            f" - {masked_cloud_path}\n"
            f" - {scene_cloud_path}\n"
            f" - {grasp_mesh_path}"
        )

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
    parser.add_argument('--scene_max_depth', type=float, default=3.0, help='Maximum depth in meters for debug scene cloud')
    parser.add_argument('--debug_grasp_count', type=int, default=15, help='Number of top grasps to export in debug mesh')
    
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
