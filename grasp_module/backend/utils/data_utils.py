""" Tools for data processing.
    Author: chenxi-wang
"""

import json
import logging
import os

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
