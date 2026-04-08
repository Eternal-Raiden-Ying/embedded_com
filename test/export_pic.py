import pyrealsense2 as rs
import numpy as np
import cv2
import os
import sys

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(CURRENT_DIR)
sys.path.append(ROOT_DIR)

from grasp_module.backend.utils.data_utils import (
    CameraInfo,
    build_ply_output_path,
    create_colored_point_cloud_from_rgbd,
    write_open3d_point_cloud,
)

# 配置路径
bag_file = r"E:\Documents_E\embedded_competition\Fibocom\data\camera_behind.bag"  # 你的 .bag 文件路径
output_base_path = os.path.join(ROOT_DIR, "output_dataset", "camera_behind") # 存放照片的文件夹

paths = {
    "color": os.path.join(output_base_path, "color"),
    "depth_raw": os.path.join(output_base_path, "depth_raw_16bit"), # 用于算法
    "depth_vis_color": os.path.join(output_base_path, "depth_vis_colored"), # 用于人工查看 (伪彩色)
}

for p in paths.values():
    if not os.path.exists(p):
        os.makedirs(p)

# 1. 初始化 Pipeline
pipeline = rs.pipeline()
config = rs.config()

# 从文件加载
config.enable_device_from_file(bag_file, repeat_playback=False)
profile = pipeline.start(config)

# 2. 初始化对齐对象 - 让 Color 对齐到 Depth (保持 depth 原始分辨率和内参)
align_to = rs.stream.depth
align = rs.align(align_to)

# 创建 RealSense 的色彩化过滤器（默认生成 Jet 伪彩色）
colorizer = rs.colorizer()
# 你可以自定义色彩化模式：0=Jet, 1=Classic, 2=Fire, 3=Smooth, 4=Quantized...
colorizer.set_option(rs.option.color_scheme, 0) 

frame_count = 0
try:
    for i in range(1):
        # 等待下一组帧
        frames = pipeline.wait_for_frames()
        
        # 对齐帧
        aligned_frames = align.process(frames)
        
        # 获取对齐后的帧
        aligned_depth_frame = aligned_frames.get_depth_frame()
        color_frame = aligned_frames.get_color_frame()
        
        if not aligned_depth_frame or not color_frame:
            continue

        # --- 3. 颜色图片修复 ---
        # 获取彩色帧的原始数据
        color_data = np.asanyarray(color_frame.get_data())

        # 统一得到 RGB 数据用于点云着色，再转换成 BGR 保存到磁盘。
        if color_frame.get_profile().as_video_stream_profile().format() == rs.format.rgb8:
            color_image_rgb = color_data
        else:
            color_image_rgb = cv2.cvtColor(color_data, cv2.COLOR_BGR2RGB)

        color_image_correct = cv2.cvtColor(color_image_rgb, cv2.COLOR_RGB2BGR)

        # 保存修复后的彩色图
        color_filename = os.path.join(paths["color"], f"color_{frame_count:05d}.png")
        cv2.imwrite(color_filename, color_image_correct)
        
        # --- 4. 深度图处理与渲染 ---
        
        # A. 保存原始 16位数据 (用于算法训练)
        depth_data = np.asanyarray(aligned_depth_frame.get_data())
        depth_raw_filename = os.path.join(paths["depth_raw"], f"depth_raw_{frame_count:05d}.png")
        cv2.imwrite(depth_raw_filename, depth_data) # OpenCV 会自动将其保存为单通道 16-bit PNG

        # --- 5. 使用与 engine 完全一致的反投影逻辑导出点云 ---
        depth_intrinsics = aligned_depth_frame.get_profile().as_video_stream_profile().get_intrinsics()
        depth_scale = profile.get_device().first_depth_sensor().get_depth_scale()
        camera_info = CameraInfo(
            width=float(depth_intrinsics.width),
            height=float(depth_intrinsics.height),
            fx=float(depth_intrinsics.fx),
            fy=float(depth_intrinsics.fy),
            cx=float(depth_intrinsics.ppx),
            cy=float(depth_intrinsics.ppy),
            scale=float(1.0 / depth_scale),
        )
        full_points, full_colors = create_colored_point_cloud_from_rgbd(
            color_image_rgb,
            depth_data,
            camera_info,
            mask=None,
        )
        point_cloud_path = build_ply_output_path(output_base_path, f"cloud_{frame_count:05d}.ply")
        write_open3d_point_cloud(point_cloud_path, full_points, full_colors)
        print(f"点云已保存，请使用 MeshLab 打开 {point_cloud_path}")

        # C. 伪彩色渲染 (使用 RealSense 内置色彩器)
        # 这种方法最清晰，蓝色代表远，红色代表近（取决于模式）
        colorized_depth = colorizer.colorize(aligned_depth_frame)
        depth_color_data = np.asanyarray(colorized_depth.get_data())
        
        # 注意：RealSense colorizer 输出的是 RGB，用于 opencv 保存需要转为 BGR
        depth_color_data_bgr = cv2.cvtColor(depth_color_data, cv2.COLOR_RGB2BGR)
        
        depth_vis_color_filename = os.path.join(paths["depth_vis_color"], f"depth_color_{frame_count:05d}.png")
        cv2.imwrite(depth_vis_color_filename, depth_color_data_bgr)

        

        frame_count += 1
        print(f"正在处理第 {frame_count} 帧...")

except RuntimeError:
    print("文件处理完毕。")

finally:
    pipeline.stop()
