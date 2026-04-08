"""
导出 RealSense 相机內参到 metadata 文件
使用方法: python export_camera_intrinsics.py <path_to_bag_file>

注意: 导出原始 depth 相机的内参，用于与原始模型处理方式一致
"""

import os
import sys
import json
import numpy as np
import argparse

try:
    import pyrealsense2 as rs
except ImportError:
    print("Error: pyrealsense2 not installed. Run: pip install pyrealsense2")
    sys.exit(1)


def export_camera_intrinsics(bag_file, output_path=None, align_to_depth=True):
    """
    从 bag 文件中提取相机內参并保存为 JSON
    
    Args:
        bag_file: RealSense bag 文件路径
        output_path: 输出 JSON 文件路径
        align_to_depth: True=color对齐到depth (保持depth内参), False=depth对齐到color
    """
    if not os.path.exists(bag_file):
        print(f"Error: File not found: {bag_file}")
        return None
    
    pipeline = rs.pipeline()
    config = rs.config()
    
    try:
        config.enable_device_from_file(bag_file, repeat_playback=False)
        profile = pipeline.start(config)
    except Exception as e:
        print(f"Error: Failed to open bag file: {e}")
        return None
    
    # 获取对齐前的原始帧，以获取原始内参
    frames = pipeline.wait_for_frames()
    
    # 获取原始（未对齐）的深度帧和彩色帧
    raw_depth_frame = frames.get_depth_frame()
    raw_color_frame = frames.get_color_frame()
    
    if not raw_depth_frame or not raw_color_frame:
        print("Error: No frames available")
        pipeline.stop()
        return None
    
    # 获取原始帧的 profile 和内参
    raw_depth_profile = raw_depth_frame.get_profile()
    raw_color_profile = raw_color_frame.get_profile()
    
    depth_intrinsics = raw_depth_profile.as_video_stream_profile().get_intrinsics()
    color_intrinsics = raw_color_profile.as_video_stream_profile().get_intrinsics()
    
    # RealSense depth_scale 的语义是: 1 个 depth unit 对应多少米。
    # graspness/graspnet 里的 factor_depth 语义相反: depth_raw / factor_depth -> 米。
    # 二者互为倒数，通常 depth_scale=0.001 时 factor_depth=1000.0。
    depth_sensor = profile.get_device().first_depth_sensor()
    depth_scale = depth_sensor.get_depth_scale()
    factor_depth = 1.0 / depth_scale if depth_scale > 0 else 1000.0
    
    pipeline.stop()
    
    camera_metadata = {
        "camera_type": "realsense",
        "align_mode": "depth" if align_to_depth else "color",
        "depth": {
            "width": depth_intrinsics.width,
            "height": depth_intrinsics.height,
            "fx": depth_intrinsics.fx,
            "fy": depth_intrinsics.fy,
            "cx": depth_intrinsics.ppx,
            "cy": depth_intrinsics.ppy,
            "model": str(depth_intrinsics.model),
            "coeffs": list(depth_intrinsics.coeffs)
        },
        "color": {
            "width": color_intrinsics.width,
            "height": color_intrinsics.height,
            "fx": color_intrinsics.fx,
            "fy": color_intrinsics.fy,
            "cx": color_intrinsics.ppx,
            "cy": color_intrinsics.ppy,
            "model": str(color_intrinsics.model),
            "coeffs": list(color_intrinsics.coeffs)
        },
        "depth_scale": depth_scale,
        "factor_depth": factor_depth,
        "description": "Camera intrinsics from raw depth frame (matching graspnet model processing)"
    }
    
    if output_path is None:
        base_name = os.path.splitext(os.path.basename(bag_file))[0]
        output_path = os.path.join(os.path.dirname(bag_file), f"{base_name}_camera_metadata.json")
    
    with open(output_path, 'w') as f:
        json.dump(camera_metadata, f, indent=2)
    
    print(f"Camera intrinsics saved to: {output_path}")
    print(f"\n[Depth Camera - Use this for point cloud generation]")
    print(f"  Resolution: {camera_metadata['depth']['width']}x{camera_metadata['depth']['height']}")
    print(f"  fx: {camera_metadata['depth']['fx']:.4f}")
    print(f"  fy: {camera_metadata['depth']['fy']:.4f}")
    print(f"  cx: {camera_metadata['depth']['cx']:.4f}")
    print(f"  cy: {camera_metadata['depth']['cy']:.4f}")
    print(f"  depth_scale: {camera_metadata['depth_scale']}")
    print(f"  factor_depth: {camera_metadata['factor_depth']}")
    print(f"\n[Color Camera - For reference]")
    print(f"  Resolution: {camera_metadata['color']['width']}x{camera_metadata['color']['height']}")
    print(f"  fx: {camera_metadata['color']['fx']:.4f}")
    print(f"  fy: {camera_metadata['color']['fy']:.4f}")
    print(f"  cx: {camera_metadata['color']['cx']:.4f}")
    print(f"  cy: {camera_metadata['color']['cy']:.4f}")
    
    return camera_metadata


def export_camera_intrinsics_from_live(device_id=None, output_path="camera_metadata.json"):
    """
    从实时相机设备中提取內参 (原始 depth 内参)
    """
    pipeline = rs.pipeline()
    config = rs.config()
    
    if device_id is not None:
        config.enable_device(device_id)
    
    profile = pipeline.start(config)
    
    frames = pipeline.wait_for_frames()
    raw_depth_frame = frames.get_depth_frame()
    raw_color_frame = frames.get_color_frame()
    
    depth_profile = raw_depth_frame.get_profile()
    color_profile = raw_color_frame.get_profile()
    
    depth_intrinsics = depth_profile.as_video_stream_profile().get_intrinsics()
    color_intrinsics = color_profile.as_video_stream_profile().get_intrinsics()
    
    depth_sensor = profile.get_device().first_depth_sensor()
    depth_scale = depth_sensor.get_depth_scale()
    factor_depth = 1.0 / depth_scale if depth_scale > 0 else 1000.0
    
    pipeline.stop()
    
    camera_metadata = {
        "camera_type": "realsense_live",
        "align_mode": "depth",
        "depth": {
            "width": depth_intrinsics.width,
            "height": depth_intrinsics.height,
            "fx": depth_intrinsics.fx,
            "fy": depth_intrinsics.fy,
            "cx": depth_intrinsics.ppx,
            "cy": depth_intrinsics.ppy,
            "model": str(depth_intrinsics.model),
            "coeffs": list(depth_intrinsics.coeffs)
        },
        "color": {
            "width": color_intrinsics.width,
            "height": color_intrinsics.height,
            "fx": color_intrinsics.fx,
            "fy": color_intrinsics.fy,
            "cx": color_intrinsics.ppx,
            "cy": color_intrinsics.ppy,
            "model": str(color_intrinsics.model),
            "coeffs": list(color_intrinsics.coeffs)
        },
        "depth_scale": depth_scale,
        "factor_depth": factor_depth,
        "description": "Camera intrinsics from live RealSense device (raw depth)"
    }
    
    with open(output_path, 'w') as f:
        json.dump(camera_metadata, f, indent=2)
    
    print(f"Camera intrinsics saved to: {output_path}")
    print(f"\n[Depth Camera]")
    print(f"  fx: {camera_metadata['depth']['fx']:.4f}")
    print(f"  fy: {camera_metadata['depth']['fy']:.4f}")
    print(f"  cx: {camera_metadata['depth']['cx']:.4f}")
    print(f"  cy: {camera_metadata['depth']['cy']:.4f}")
    print(f"  depth_scale: {camera_metadata['depth_scale']}")
    print(f"  factor_depth: {camera_metadata['factor_depth']}")
    
    return camera_metadata


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Export RealSense camera intrinsics (raw depth)")
    parser.add_argument("bag_file", nargs="?", help="Path to bag file")
    parser.add_argument("-o", "--output", help="Output JSON file path")
    parser.add_argument("--live", action="store_true", help="Use live camera instead of bag file")
    parser.add_argument("--device-id", help="Device serial number for live mode")
    
    args = parser.parse_args()
    
    if args.live:
        export_camera_intrinsics_from_live(args.device_id, args.output or "camera_metadata.json")
    elif args.bag_file:
        export_camera_intrinsics(args.bag_file, args.output)
    else:
        parser.print_help()
        print("\nExamples:")
        print("  python export_camera_intrinsics.py camera.bag")
        print("  python export_camera_intrinsics.py camera.bag -o my_camera.json")
        print("  python export_camera_intrinsics.py --live")
