""" Tools for data processing.
    Author: chenxi-wang
"""

import json
import logging
import os

import cv2
import numpy as np


logger = logging.getLogger("vision.grasp")


class CameraInfo():
    """ Camera intrisics for point cloud creation. """

    def __init__(self, width, height, fx, fy, cx, cy, scale):
        self.width = width
        self.height = height
        self.fx = fx
        self.fy = fy
        self.cx = cx
        self.cy = cy
        self.scale = scale


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)
    return path


def get_ply_output_dir(base_dir):
    return ensure_dir(os.path.join(base_dir, 'ply'))


def build_ply_output_path(base_dir, filename):
    return os.path.join(get_ply_output_dir(base_dir), filename)


def load_camera_info_from_metadata(metadata_path, default_camera=None):
    """Load CameraInfo from exported metadata json."""
    if not metadata_path or not os.path.exists(metadata_path):
        if metadata_path:
            logger.warning("Camera metadata file not found: %s", metadata_path)
        return default_camera

    with open(metadata_path, 'r', encoding='utf-8') as f:
        metadata = json.load(f)

    depth_info = metadata.get('depth', {})
    factor_depth = metadata.get('factor_depth')
    depth_scale = metadata.get('depth_scale')

    if factor_depth is not None:
        scale = float(factor_depth)
        scale_source = 'factor_depth'
    elif depth_scale is not None:
        depth_scale = float(depth_scale)
        scale = 1.0 / depth_scale if 0 < depth_scale < 1 else depth_scale
        scale_source = 'depth_scale(converted)' if 0 < depth_scale < 1 else 'depth_scale'
    elif default_camera is not None:
        scale = float(default_camera.scale)
        scale_source = 'default_camera'
    else:
        scale = 1000.0
        scale_source = 'default'

    camera_info = CameraInfo(
        width=float(depth_info.get('width', default_camera.width if default_camera else 1280)),
        height=float(depth_info.get('height', default_camera.height if default_camera else 720)),
        fx=float(depth_info.get('fx', default_camera.fx if default_camera else 631.55)),
        fy=float(depth_info.get('fy', default_camera.fy if default_camera else 631.21)),
        cx=float(depth_info.get('cx', default_camera.cx if default_camera else 638.43)),
        cy=float(depth_info.get('cy', default_camera.cy if default_camera else 366.50)),
        scale=scale,
    )

    logger.info("Loaded camera metadata: %s", metadata_path)
    logger.info(
        "Camera intrinsics fx=%.4f fy=%.4f cx=%.4f cy=%.4f scale=%.6f (%s)",
        camera_info.fx,
        camera_info.fy,
        camera_info.cx,
        camera_info.cy,
        camera_info.scale,
        scale_source,
    )
    return camera_info


def create_point_cloud_from_depth_image(depth, camera, organized=True):
    """ Generate point cloud using depth image only.

        Input:
            depth: [numpy.ndarray, (H,W), numpy.float32]
                depth image
            camera: [CameraInfo]
                camera intrinsics
            organized: bool
                whether to keep the cloud in image shape (H,W,3)

        Output:
            cloud: [numpy.ndarray, (H,W,3)/(H*W,3), numpy.float32]
                generated cloud, (H,W,3) for organized=True, (H*W,3) for organized=False
    """
    assert (depth.shape[0] == camera.height and depth.shape[1] == camera.width)
    xmap = np.arange(camera.width)
    ymap = np.arange(camera.height)
    xmap, ymap = np.meshgrid(xmap, ymap)
    points_z = depth / camera.scale
    points_x = (xmap - camera.cx) * points_z / camera.fx
    points_y = (ymap - camera.cy) * points_z / camera.fy
    cloud = np.stack([points_x, points_y, points_z], axis=-1)
    if not organized:
        cloud = cloud.reshape([-1, 3])
    return cloud


def create_colored_point_cloud_from_rgbd(color, depth, camera, mask=None):
    """Project RGB-D into camera coordinates with optional mask."""
    cloud = create_point_cloud_from_depth_image(depth, camera, organized=True)
    valid_mask = (depth > 0)
    if mask is not None:
        valid_mask &= (mask > 0)

    points = cloud[valid_mask].astype(np.float32)
    colors = None
    if color is not None:
        colors = color.reshape(-1, 3)[valid_mask.reshape(-1)].astype(np.float32) / 255.0
    return points, colors


def filter_point_cloud_by_z(points, colors=None, z_min=None, z_max=None):
    if points.size == 0:
        return points, colors

    mask = np.ones(points.shape[0], dtype=bool)
    if z_min is not None:
        mask &= points[:, 2] >= z_min
    if z_max is not None:
        mask &= points[:, 2] <= z_max

    points = points[mask]
    if colors is not None:
        colors = colors[mask]
    return points, colors


def write_open3d_point_cloud(ply_path, points, colors=None):
    """Write point cloud using Open3D without changing axes."""
    import open3d as o3d

    ensure_dir(os.path.dirname(ply_path))
    cloud = o3d.geometry.PointCloud()
    cloud.points = o3d.utility.Vector3dVector(points.astype(np.float32))
    if colors is not None:
        cloud.colors = o3d.utility.Vector3dVector(colors.astype(np.float32))
    o3d.io.write_point_cloud(ply_path, cloud)


def transform_point_cloud(cloud, transform, format='4x4'):
    """ Transform points to new coordinates with transformation matrix.

        Input:
            cloud: [np.ndarray, (N,3), np.float32]
                points in original coordinates
            transform: [np.ndarray, (3,3)/(3,4)/(4,4), np.float32]
                transformation matrix, could be rotation only or rotation+translation
            format: [string, '3x3'/'3x4'/'4x4']
                the shape of transformation matrix
                '3x3' --> rotation matrix
                '3x4'/'4x4' --> rotation matrix + translation matrix

        Output:
            cloud_transformed: [np.ndarray, (N,3), np.float32]
                points in new coordinates
    """
    if not (format == '3x3' or format == '4x4' or format == '3x4'):
        raise ValueError('Unknown transformation format, only support \'3x3\' or \'4x4\' or \'3x4\'.')
    if format == '3x3':
        cloud_transformed = np.dot(transform, cloud.T).T
    elif format == '4x4' or format == '3x4':
        ones = np.ones(cloud.shape[0])[:, np.newaxis]
        cloud_ = np.concatenate([cloud, ones], axis=1)
        cloud_transformed = np.dot(transform, cloud_.T).T
        cloud_transformed = cloud_transformed[:, :3]
    return cloud_transformed


def compute_point_dists(A, B):
    """ Compute pair-wise point distances in two matrices.

        Input:
            A: [np.ndarray, (N,3), np.float32]
                point cloud A
            B: [np.ndarray, (M,3), np.float32]
                point cloud B

        Output:
            dists: [np.ndarray, (N,M), np.float32]
                distance matrix
    """
    A = A[:, np.newaxis, :]
    B = B[np.newaxis, :, :]
    dists = np.linalg.norm(A - B, axis=-1)
    return dists


def remove_invisible_grasp_points(cloud, grasp_points, pose, th=0.01):
    """ Remove invisible part of object model according to scene point cloud.

        Input:
            cloud: [np.ndarray, (N,3), np.float32]
                scene point cloud
            grasp_points: [np.ndarray, (M,3), np.float32]
                grasp point label in object coordinates
            pose: [np.ndarray, (4,4), np.float32]
                transformation matrix from object coordinates to world coordinates
            th: [float]
                if the minimum distance between a grasp point and the scene points is greater than outlier, the point will be removed

        Output:
            visible_mask: [np.ndarray, (M,), np.bool]
                mask to show the visible part of grasp points
    """
    grasp_points_trans = transform_point_cloud(grasp_points, pose)
    dists = compute_point_dists(grasp_points_trans, cloud)
    min_dists = dists.min(axis=1)
    visible_mask = (min_dists < th)
    return visible_mask


def get_workspace_mask(cloud, seg, trans=None, organized=True, outlier=0):
    """ Keep points in workspace as input.

        Input:
            cloud: [np.ndarray, (H,W,3), np.float32]
                scene point cloud
            seg: [np.ndarray, (H,W,), np.uint8]
                segmantation label of scene points
            trans: [np.ndarray, (4,4), np.float32]
                transformation matrix for scene points, default: None.
            organized: [bool]
                whether to keep the cloud in image shape (H,W,3)
            outlier: [float]
                if the distance between a point and workspace is greater than outlier, the point will be removed
                
        Output:
            workspace_mask: [np.ndarray, (H,W)/(H*W,), np.bool]
                mask to indicate whether scene points are in workspace
    """
    if organized:
        h, w, _ = cloud.shape
        cloud = cloud.reshape([h * w, 3])
        seg = seg.reshape(h * w)
    if trans is not None:
        cloud = transform_point_cloud(cloud, trans)
     # here graspnet use 0 for background, when only selected item is required to be grasped, replace this with 'cloud[seg == target]'
    foreground = cloud[seg > 0]   
    xmin, ymin, zmin = foreground.min(axis=0)
    xmax, ymax, zmax = foreground.max(axis=0)
    mask_x = ((cloud[:, 0] > xmin - outlier) & (cloud[:, 0] < xmax + outlier))
    mask_y = ((cloud[:, 1] > ymin - outlier) & (cloud[:, 1] < ymax + outlier))
    mask_z = ((cloud[:, 2] > zmin - outlier) & (cloud[:, 2] < zmax + outlier))
    workspace_mask = (mask_x & mask_y & mask_z)
    if organized:
        workspace_mask = workspace_mask.reshape([h, w])

    return workspace_mask


# ============================================================
# 方案 B: 深度→颜色 3D 投影对齐
# ============================================================

def load_color_camera_info_from_metadata(metadata_path, default_camera=None):
    """从 metadata JSON 的 ``color`` 段加载彩色相机内参。

    Args:
        metadata_path: metadata JSON 文件路径。
        default_camera: 如果 metadata 中没有 color 段，回退到此值。

    Returns:
        CameraInfo for the color sensor, or default_camera if unavailable.
    """
    if not metadata_path or not os.path.exists(metadata_path):
        return default_camera

    with open(metadata_path, 'r', encoding='utf-8') as f:
        metadata = json.load(f)

    color_info = metadata.get('color')
    if not color_info:
        return default_camera

    fallback = default_camera
    camera_info = CameraInfo(
        width=float(color_info.get('width', fallback.width if fallback else 1280)),
        height=float(color_info.get('height', fallback.height if fallback else 720)),
        fx=float(color_info.get('fx', fallback.fx if fallback else 906.98)),
        fy=float(color_info.get('fy', fallback.fy if fallback else 905.03)),
        cx=float(color_info.get('cx', fallback.cx if fallback else 646.37)),
        cy=float(color_info.get('cy', fallback.cy if fallback else 369.08)),
        scale=1000.0,
    )

    logger.info("Loaded color camera metadata from %s", metadata_path)
    logger.info(
        "Color intrinsics wxh=%sx%s fx=%.4f fy=%.4f cx=%.4f cy=%.4f",
        int(camera_info.width), int(camera_info.height),
        camera_info.fx, camera_info.fy,
        camera_info.cx, camera_info.cy,
    )
    return camera_info


def load_depth_to_color_extrinsic(metadata_path):
    """从 metadata JSON 的 ``extrinsics_depth_to_color`` 段加载 4x4 外参矩阵。

    Args:
        metadata_path: metadata JSON 文件路径。

    Returns:
        4x4 numpy float64 齐次变换矩阵 [R|t; 0 0 0 1]，无此字段返回 None。
    """
    if not metadata_path or not os.path.exists(metadata_path):
        return None

    with open(metadata_path, 'r', encoding='utf-8') as f:
        metadata = json.load(f)

    extrin = metadata.get('extrinsics_depth_to_color')
    if not extrin:
        return None

    rotation = np.array(extrin['rotation'], dtype=np.float64).reshape(3, 3)
    translation = np.array(extrin['translation'], dtype=np.float64)

    extrinsic_4x4 = np.eye(4, dtype=np.float64)
    extrinsic_4x4[:3, :3] = rotation
    extrinsic_4x4[:3, 3] = translation

    logger.info("Loaded depth-to-color extrinsics (4x4) from %s", metadata_path)
    return extrinsic_4x4


def map_depth_cloud_to_color_image(depth, color, depth_cam, color_cam, depth_to_color_extrinsic=None):
    """将完整 depth 点云投影到 color 2D 像素系 — 方案 B 的核心。

    1. 用 depth 内参生成完整 3D 点云。
    2. 用 depth→color 外参变换到 color 坐标系。
    3. 用 color 内参投影到 2D 像素坐标。
    4. 返回 UV + in_view 掩码，由调用方按 YOLO mask 划分点云。

    Args:
        depth: (H, W) uint16 深度图。
        color: (H, W, 3) uint8 RGB 图像。
        depth_cam: CameraInfo — 深度传感器内参。
        color_cam: CameraInfo — 彩色传感器内参。
        depth_to_color_extrinsic: (4, 4) np.float64 外参矩阵，None 时假设共轴。

    Returns:
        points: (N, 3) float32 — depth 坐标系下的 3D 点。
        colors: (N, 3) float32 — RGB 颜色 [0,1]，超出 color FOV 的点为黑色。
        u: (N,) int32 — color 图像上的列坐标，无效时为负值。
        v: (N,) int32 — color 图像上的行坐标，无效时为负值。
        in_view: (N,) bool — 该 3D 点投影后是否落在 color 图像范围内。
    """
    # 1. 生成完整 depth 点云 (depth 坐标系)
    all_points = create_point_cloud_from_depth_image(depth, depth_cam, organized=False)
    valid_z = depth.reshape(-1) > 0
    points = all_points[valid_z]
    n = len(points)

    # 默认：黑色，UV 无效
    colors = np.zeros((n, 3), dtype=np.float32)
    u = np.full(n, -1, dtype=np.int32)
    v = np.full(n, -1, dtype=np.int32)

    # 2. 变换到 color 坐标系
    if depth_to_color_extrinsic is not None:
        pts_c = transform_point_cloud(points, depth_to_color_extrinsic, format='4x4')
    else:
        pts_c = points

    # 3. 针孔投影到 color 2D 平面
    front_mask = pts_c[:, 2] > 0
    z_front = pts_c[front_mask, 2]
    u[front_mask] = np.round(
        (pts_c[front_mask, 0] * color_cam.fx / z_front) + color_cam.cx
    ).astype(np.int32)
    v[front_mask] = np.round(
        (pts_c[front_mask, 1] * color_cam.fy / z_front) + color_cam.cy
    ).astype(np.int32)

    # 4. 检查投影点是否在 color 图像范围内
    h, w = color.shape[:2]
    in_view = front_mask & (u >= 0) & (u < w) & (v >= 0) & (v < h)

    # 5. 对视野内的点提取 RGB 颜色
    colors[in_view] = color[v[in_view], u[in_view]].astype(np.float32) / 255.0

    logger.debug(
        "map_depth_cloud: %d valid depth points, %d in color FOV, "
        "%d outside color FOV (preserved for collision)",
        n, int(in_view.sum()), int((~in_view & front_mask).sum()),
    )
    return points, colors, u, v, in_view


# ============================================================
# 深度图预处理工具
# ============================================================

def normalize_depth_shape(depth_img):
    """Squeeze single-channel trailing dimension: (H,W,1) -> (H,W)."""
    if depth_img is None:
        return depth_img
    if depth_img.ndim == 3 and depth_img.shape[2] == 1:
        return depth_img[:, :, 0]
    return depth_img


def sanitize_depth_image(depth_img, depth_min_mm=1, depth_max_mm=2000):
    """Clip depth to [depth_min_mm, depth_max_mm], zero out invalid pixels."""
    depth_img = normalize_depth_shape(depth_img)
    if depth_img is None:
        return depth_img

    depth = depth_img.astype(np.uint16, copy=True)
    valid_mask = depth > 0
    if depth_min_mm is not None:
        valid_mask &= depth >= int(depth_min_mm)
    if depth_max_mm is not None:
        valid_mask &= depth <= int(depth_max_mm)
    depth[~valid_mask] = 0
    return depth


def _fill_zero_holes_with_median(depth_img, kernel_size=5, iterations=1):
    """Iterative median-based hole filling: only fills zero pixels that have non-zero neighbours."""
    if kernel_size <= 1 or iterations <= 0:
        return depth_img

    filled = depth_img.astype(np.uint16, copy=True)
    for _ in range(iterations):
        candidate = cv2.medianBlur(filled, kernel_size)
        hole_mask = (filled == 0) & (candidate > 0)
        if not np.any(hole_mask):
            break
        filled[hole_mask] = candidate[hole_mask]
    return filled


def postprocess_depth_image(depth_img, cfgs):
    """Full depth pre-processing pipeline used by both test_engine and server.

    Stages: clip range -> median denoise -> hole-fill -> bilateral (optional) -> clip.
    """
    depth = sanitize_depth_image(
        depth_img,
        depth_min_mm=getattr(cfgs, "depth_min_mm", 1),
        depth_max_mm=getattr(cfgs, "depth_max_mm", 2000),
    )

    if not getattr(cfgs, "depth_postprocess", True):
        return depth

    smooth_method = str(getattr(cfgs, "depth_smooth_method", "median"))
    smooth_kernel = int(getattr(cfgs, "depth_smooth_kernel", 5))
    hole_fill_kernel = int(getattr(cfgs, "depth_hole_fill_kernel", 5))
    hole_fill_iterations = int(getattr(cfgs, "depth_hole_fill_iterations", 2))

    if smooth_kernel % 2 == 0:
        smooth_kernel += 1
    if hole_fill_kernel % 2 == 0:
        hole_fill_kernel += 1

    processed = depth.copy()

    # ── 阶段 1: median 去飞点 ──
    if smooth_kernel > 1 and smooth_method != "none":
        median_smoothed = cv2.medianBlur(processed, smooth_kernel)
        processed = np.where(processed > 0, median_smoothed, 0).astype(np.uint16)

    # ── 阶段 2: 孔洞填充 ──
    processed = _fill_zero_holes_with_median(
        processed,
        kernel_size=hole_fill_kernel,
        iterations=hole_fill_iterations,
    )

    # ── 阶段 3: bilateral 保边平滑 ──
    if smooth_method == "bilateral":
        bilat_d = int(getattr(cfgs, "depth_bilateral_d", 9))
        sigma_color = float(getattr(cfgs, "depth_bilateral_sigma_color", 75.0))
        sigma_space = float(getattr(cfgs, "depth_bilateral_sigma_space", 75.0))
        bilat_smoothed = cv2.bilateralFilter(
            processed.astype(np.float32), bilat_d,
            sigma_color, sigma_space,
        ).astype(np.uint16)
        processed = np.where(processed > 0, bilat_smoothed, 0).astype(np.uint16)

    processed = sanitize_depth_image(
        processed,
        depth_min_mm=getattr(cfgs, "depth_min_mm", 1),
        depth_max_mm=getattr(cfgs, "depth_max_mm", 2000),
    )
    return processed


# ============================================================
# 薄壳点云补充 — 沿视线方向在边缘/空洞处延伸
# ============================================================

def generate_shell_points(depth, camera, shell_thickness_m=0.02, shell_steps=2, depth_diff_threshold_mm=20, edge_kernel=1):
    """在深度不连续处和孔洞边界，沿视线方向生成额外的薄壳点。

    用于弥补 depth 图中薄表面/边缘/遮挡区域采集不足的问题。
    每个标记点沿 camera ray 方向延伸 shell_thickness_m 米，
    等距生成 shell_steps 个补充点。

    Args:
        depth: (H,W) uint16 深度图 (mm)。
        camera: CameraInfo。
        shell_thickness_m: 沿视线延伸的总厚度 (米)。
        shell_steps: 延伸点数。
        depth_diff_threshold_mm: 判定为边的深度差阈值 (mm)。
        edge_kernel: 边缘检测步长 (像素), >=1。k=1 比较相邻像素,
                     k=N 比较相距 N 像素的两个点。

    Returns:
        shell_points: (K, 3) float32 — 补充的 3D 点。
        shell_mask:  (H,W) bool — 哪些像素生成了壳点。
    """
    h, w = depth.shape
    valid = (depth > 0)
    k = max(1, int(edge_kernel))

    # ── 边缘检测: 相距 k 像素的深度差 > 阈值 ──
    edge = np.zeros((h, w), dtype=bool)
    # 水平方向: pixel[x] vs pixel[x+k]
    if w > k:
        valid_h = valid[:, k:] & valid[:, :-k]
        diff_h = np.abs(depth[:, k:].astype(np.int32) - depth[:, :-k].astype(np.int32))
        h_edge = valid_h & (diff_h > depth_diff_threshold_mm)
        edge[:, k:] |= h_edge
        edge[:, :-k] |= h_edge
    # 垂直方向: pixel[y] vs pixel[y+k]
    if h > k:
        valid_v = valid[k:, :] & valid[:-k, :]
        diff_v = np.abs(depth[k:, :].astype(np.int32) - depth[:-k, :].astype(np.int32))
        v_edge = valid_v & (diff_v > depth_diff_threshold_mm)
        edge[k:, :] |= v_edge
        edge[:-k, :] |= v_edge

    # ── 孔洞边界: 有效像素邻接零像素 ──
    kernel = np.ones((3, 3), dtype=np.uint8)
    zero_neighbors = cv2.filter2D(
        (depth == 0).astype(np.uint8), -1, kernel,
    )
    hole_boundary = valid & (zero_neighbors > 0)

    # ── 合并 ──
    shell_mask = edge | hole_boundary
    if not np.any(shell_mask):
        return np.empty((0, 3), dtype=np.float32), shell_mask

    # ── 生成 3D 点 ──
    xmap = np.arange(w, dtype=np.float32)
    ymap = np.arange(h, dtype=np.float32)
    xmap, ymap = np.meshgrid(xmap, ymap)

    z_valid = depth[shell_mask].astype(np.float32) / camera.scale
    x_valid = (xmap[shell_mask] - camera.cx) * z_valid / camera.fx
    y_valid = (ymap[shell_mask] - camera.cy) * z_valid / camera.fy

    # 视线方向 (camera center → point)
    ray_x = x_valid / camera.fx  # 未归一化，direction ~ (x/fx, y/fy, z)
    ray_y = y_valid / camera.fy
    ray_z = z_valid
    ray_norm = np.sqrt(ray_x**2 + ray_y**2 + ray_z**2) + 1e-9
    ray_x /= ray_norm
    ray_y /= ray_norm
    ray_z /= ray_norm

    # 沿视线方向延伸 shell_thickness_m，等距生成 shell_steps 层
    n_base = int(np.sum(shell_mask))
    shell_points_list = []
    for step in range(1, shell_steps + 1):
        delta = shell_thickness_m * step / shell_steps
        shell_points_list.append(np.stack([
            x_valid + ray_x * delta,
            y_valid + ray_y * delta,
            z_valid + ray_z * delta,
        ], axis=-1))

    shell_points = np.concatenate(shell_points_list, axis=0).astype(np.float32)
    return shell_points, shell_mask
