import os
import sys
import logging
import numpy as np
import argparse
import time
import cv2
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
from .utils.yolo_utils import load_yolo_model, predict_target_masks


logger = logging.getLogger("vision.grasp")


class RealSenseGraspPredictor:
    def __init__(self, cfgs):
        self.cfgs = cfgs
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.seg_model = None

        self.camera_info = self._load_camera_info()
        self.net = self._load_grasp_model()

        if cfgs.debug and not os.path.exists(cfgs.dump_dir):
            os.makedirs(cfgs.dump_dir)

    def _default_camera_info(self):
        return CameraInfo(
            width=1280.0,
            height=720.0,
            fx=631.55,
            fy=631.21,
            cx=638.43,
            cy=366.50,
            scale=1000.0,
        )

    def _load_camera_info(self):
        default_camera_info = self._default_camera_info()
        camera_info = load_camera_info_from_metadata(
            getattr(self.cfgs, 'camera_metadata', None),
            default_camera=default_camera_info,
        )
        if camera_info is None:
            logger.info("Using default camera intrinsics (graspnet kinect defaults)")
            return default_camera_info
        return camera_info

    def _load_grasp_model(self):
        logger.info("Loading GraspNet model from %s", self.cfgs.checkpoint_path)
        net = GraspNet(seed_feat_dim=self.cfgs.seed_feat_dim, is_training=False)
        net.to(self.device)

        checkpoint = torch.load(self.cfgs.checkpoint_path, map_location=self.device)
        net.load_state_dict(checkpoint['model_state_dict'])
        net.eval()
        logger.info("GraspNet model loaded")
        return net

    def _sample_point_cloud(self, cloud_masked, color_masked):
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

        return {
            'point_clouds': cloud_sampled.astype(np.float32),
            'coors': cloud_sampled.astype(np.float32) / self.cfgs.voxel_size,
            'feats': np.ones_like(cloud_sampled).astype(np.float32),
            'colors': color_sampled.astype(np.float32),
        }

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
        return self._sample_point_cloud(cloud_masked, color_masked)

    def preprocess_points(self, points, colors):
        return self._sample_point_cloud(points, colors)

    def _get_yolo_model(self):
        if self.seg_model is None:
            model_name = getattr(self.cfgs, 'yolo_model', 'yolo26m-seg.pt')
            weights_dir = getattr(self.cfgs, 'yolo_weights_dir', None)
            logger.info("Loading YOLO segmentation model: %s", model_name)
            self.seg_model = load_yolo_model(model_name, weights_dir)
        return self.seg_model

    def _predict_target_masks(self, color_img, target_class_id):
        bgr_image = cv2.cvtColor(color_img, cv2.COLOR_RGB2BGR)
        seg_mask, bbox_mask, overlay_img, yolo_info = predict_target_masks(
            self._get_yolo_model(),
            bgr_image,
            target_class_id,
            conf=getattr(self.cfgs, 'yolo_conf', 0.25),
            iou=getattr(self.cfgs, 'yolo_iou', 0.7),
            bbox_scale=getattr(self.cfgs, 'bbox_expand_scale', 2.0),
        )
        return seg_mask, bbox_mask, overlay_img, yolo_info

    def _build_scene_cloud(self, color_img, depth_img, seg_mask, bbox_mask):
        grasp_points, grasp_colors = create_colored_point_cloud_from_rgbd(
            color_img,
            depth_img,
            self.camera_info,
            mask=seg_mask,
        )
        if len(grasp_points) == 0:
            return grasp_points, grasp_colors

        bbox_points, bbox_colors = create_colored_point_cloud_from_rgbd(
            color_img,
            depth_img,
            self.camera_info,
            mask=bbox_mask,
        )
        if len(bbox_points) == 0:
            return grasp_points, grasp_colors

        z_margin = getattr(self.cfgs, 'collision_depth_margin', 0.15)
        z_min = max(0.0, float(grasp_points[:, 2].min()) - z_margin)
        z_max = float(grasp_points[:, 2].max()) + z_margin
        z_max = min(z_max, getattr(self.cfgs, 'scene_max_depth', 3.0))
        bbox_points, bbox_colors = filter_point_cloud_by_z(
            bbox_points,
            bbox_colors,
            z_min=z_min,
            z_max=z_max,
        )
        if len(bbox_points) == 0:
            return grasp_points, grasp_colors
        return bbox_points, bbox_colors

    def _build_input_clouds(self, color_img, depth_img, seg_mask, bbox_mask):
        masked_points, masked_colors = create_colored_point_cloud_from_rgbd(
            color_img,
            depth_img,
            self.camera_info,
            mask=seg_mask,
        )
        scene_points, scene_colors = self._build_scene_cloud(color_img, depth_img, seg_mask, bbox_mask)
        return masked_points, masked_colors, scene_points, scene_colors

    def _resolve_masks(self, color_img, target_class_id):
        seg_mask, bbox_mask, overlay_img, yolo_info = self._predict_target_masks(color_img, target_class_id)
        yolo_info['source'] = 'yolo'
        return seg_mask, bbox_mask, overlay_img, yolo_info

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

    def _print_debug_timings(self, timings, extras=None):
        if not getattr(self.cfgs, 'debug', False):
            return

        ordered_keys = [
            'mask',
            'clouds',
            'preprocess',
            'transfer',
            'forward',
            'collision',
            'postprocess',
            'debug_export',
            'total',
        ]
        logger.info('[DEBUG] Timings (s):')
        for key in ordered_keys:
            if key in timings:
                logger.info(" - %s: %.4f", key, timings[key])
        if extras:
            logger.info('[DEBUG] Stats:')
            for key, value in extras.items():
                logger.info(" - %s: %s", key, value)

    def _prepare_batch_data(self, data_dict):
        batch_data = minkowski_collate_fn([data_dict])
        for key in batch_data:
            if 'list' in key:
                for i in range(len(batch_data[key])):
                    for j in range(len(batch_data[key][i])):
                        batch_data[key][i][j] = batch_data[key][i][j].to(self.device)
            else:
                batch_data[key] = batch_data[key].to(self.device)
        return batch_data

    def _forward_grasps(self, batch_data):
        with torch.no_grad():
            end_points = self.net(batch_data)
            grasp_preds_list = pred_decode(end_points)
        preds = grasp_preds_list[0].detach().cpu().numpy()
        return GraspGroup(preds)

    def _apply_collision_detection(self, grasp_group, scene_points, fallback_cloud):
        if self.cfgs.collision_thresh <= 0:
            if self.cfgs.debug:
                logger.info("Collision detection skipped. collision_thresh=%s", self.cfgs.collision_thresh)
            return grasp_group

        collision_cloud = scene_points if scene_points is not None and len(scene_points) > 0 else fallback_cloud
        mfcdetector = ModelFreeCollisionDetector(collision_cloud, voxel_size=self.cfgs.voxel_size_cd)
        collision_mask = mfcdetector.detect(
            grasp_group,
            approach_dist=0.05,
            collision_thresh=self.cfgs.collision_thresh,
        )
        if self.cfgs.debug:
            logger.info(
                "Collision detection enabled. threshold=%s, collision_cloud_points=%s",
                self.cfgs.collision_thresh,
                len(collision_cloud),
            )
        return grasp_group[~collision_mask]

    def _log_target_info(self, yolo_info):
        if yolo_info is None:
            return
        logger.info(
            "Target mask source=%s bbox=%s conf=%s",
            yolo_info.get('source'),
            yolo_info.get('bbox'),
            yolo_info.get('confidence'),
        )

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

    def infer(self, color_img, depth_img, class_id):
        """
        供外部调用的实际推理接口。
        target: 目标类别 id，或已有的二值 seg mask
        """
        tic = time.perf_counter()
        timings = {}

        stage_tic = time.perf_counter()
        seg_mask, bbox_mask, overlay_img, yolo_info = self._resolve_masks(color_img, int(class_id))
        timings['mask'] = time.perf_counter() - stage_tic
        if seg_mask.sum() == 0:
            logger.warning("No valid target mask found")
            return None

        stage_tic = time.perf_counter()
        masked_points, masked_colors, scene_points, scene_colors = self._build_input_clouds(
            color_img,
            depth_img,
            seg_mask,
            bbox_mask,
        )
        timings['clouds'] = time.perf_counter() - stage_tic
        if getattr(self.cfgs, 'debug', False):
            logger.info("Masked cloud points: %s", len(masked_points))
            logger.info("Scene cloud points: %s", len(scene_points))
        
        stage_tic = time.perf_counter()
        data_dict = self.preprocess_points(masked_points, masked_colors)
        timings['preprocess'] = time.perf_counter() - stage_tic
        if data_dict is None:
            logger.warning("No valid points found in the masked region")
            return None

        stage_tic = time.perf_counter()
        batch_data = self._prepare_batch_data(data_dict)
        timings['transfer'] = time.perf_counter() - stage_tic

        stage_tic = time.perf_counter()
        gg = self._forward_grasps(batch_data)
        timings['forward'] = time.perf_counter() - stage_tic

        stage_tic = time.perf_counter()
        gg = self._apply_collision_detection(gg, scene_points, data_dict['point_clouds'])
        timings['collision'] = time.perf_counter() - stage_tic

        stage_tic = time.perf_counter()
        gg = self.post_process_grasps(gg)
        timings['postprocess'] = time.perf_counter() - stage_tic

        timings['debug_export'] = 0.0
        self._log_target_info(yolo_info)

        if self.cfgs.debug:
            stage_tic = time.perf_counter()
            self.save_debug_visualizations(
                color_img,
                depth_img,
                seg_mask,
                bbox_mask,
                gg,
                scene_points,
                scene_colors,
                overlay_img,
                masked_points,
                masked_colors,
            )
            timings['debug_export'] = time.perf_counter() - stage_tic

        timings['total'] = time.perf_counter() - tic
        self._print_debug_timings(
            timings,
            extras={
                'masked_cloud_points': len(masked_points),
                'scene_cloud_points': len(scene_points),
                'collision_thresh': self.cfgs.collision_thresh,
            },
        )

        logger.info('Inference finished. Found %s grasps. Time: %.4fs', gg.__len__(), timings['total'])

        return gg

    def save_debug_visualizations(self, color_img, depth_img, seg_mask, bbox_mask, gg, scene_points=None, scene_colors=None, overlay_img=None, masked_points=None, masked_colors=None):
        """
        在 Debug 模式下将结果存盘，方便用 MeshLab 或 CloudCompare 查看
        """
        if masked_points is None or masked_colors is None:
            masked_points, masked_colors = create_colored_point_cloud_from_rgbd(
                color_img,
                depth_img,
                self.camera_info,
                mask=seg_mask,
            )
        if scene_points is None or scene_colors is None:
            scene_points, scene_colors = self._build_scene_cloud(color_img, depth_img, seg_mask, bbox_mask)

        top_k = getattr(self.cfgs, 'debug_grasp_count', 15)
        vis_gg = gg[:top_k] if gg.__len__() > top_k else gg
        combined_grippers = self._build_grasp_mesh(vis_gg)

        masked_cloud_path = build_ply_output_path(self.cfgs.dump_dir, 'masked_cloud.ply')
        scene_cloud_path = build_ply_output_path(self.cfgs.dump_dir, 'scene_cloud.ply')
        grasp_mesh_path = build_ply_output_path(self.cfgs.dump_dir, 'grasps_top15_heatmap.ply')

        write_open3d_point_cloud(masked_cloud_path, masked_points, masked_colors)
        write_open3d_point_cloud(scene_cloud_path, scene_points, scene_colors)
        o3d.io.write_triangle_mesh(grasp_mesh_path, combined_grippers)
        if overlay_img is not None:
            overlay_path = os.path.join(self.cfgs.dump_dir, 'ply', 'yolo_overlay.jpg')
            cv2.imwrite(overlay_path, overlay_img)

        logger.info(
            "[DEBUG] Visualizations saved to:\n - %s\n - %s\n - %s",
            masked_cloud_path,
            scene_cloud_path,
            grasp_mesh_path,
        )

# ================= 测试运行逻辑 =================
if __name__ == '__main__':
    from ..config.logging_config import configure_grasp_logger
    from ..config.predictor_config import build_predictor_arg_parser

    configure_grasp_logger()
    parser = build_predictor_arg_parser(description='Standalone grasp predictor debug runner')
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
    grasp_results = predictor.infer(dummy_rgb, dummy_depth, int(getattr(cfgs, 'yolo_class_id', 46)))
    
    # 实际机械臂执行逻辑示例:
    # if grasp_results is not None and len(grasp_results) > 0:
    #     best_grasp = grasp_results[0]
    #     translation = best_grasp.translation
    #     rotation = best_grasp.rotation_matrix
    #     # ... 转换为 Robot TCP Pose 并执行 ...
