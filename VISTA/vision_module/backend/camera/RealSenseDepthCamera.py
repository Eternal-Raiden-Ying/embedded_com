import pyrealsense2 as rs
import numpy as np
import logging

from .base import ICamera


logger = logging.getLogger("vision.camera")

class RealSenseDepthCamera(ICamera):
    """
    AidLux 专用的 RealSense 深度相机底层直读类
    绕过 V4L2 pipeline，直接使用 Sensor API 获取极低延迟的 16位 深度图
    """
    def __init__(self, width: int = 424, height: int = 240, fps: int = 15):
        """
        初始化深度相机。
        注意：必须传入相机硬件原生支持的分辨率和帧率，否则会报错。
        推荐低负载配置：424x240@15fps, 640x360@30fps, 640x480@30fps
        """
        self.width = width
        self.height = height
        self.fps = fps
        self.is_running = False
        
        # 1. 初始化上下文并寻找设备
        self.ctx = rs.context()
        devices = self.ctx.query_devices()
        if len(devices) == 0:
            raise RuntimeError("🚨 未检测到 RealSense 设备，请检查 USB 连接！")
            
        self.dev = devices[0]
        self.depth_sensor = self.dev.first_depth_sensor()
        
        # 2. 查找匹配的底层数据流配置
        target_profile = None
        available_profiles = set() # 记录可用配置以便报错时提示用户
        
        for p in self.depth_sensor.get_stream_profiles():
            if p.stream_type() == rs.stream.depth and p.format() == rs.format.z16:
                vp = p.as_video_stream_profile()
                available_profiles.add(f"{vp.width()}x{vp.height()} @ {vp.fps()}fps")
                
                # 寻找精确匹配的配置
                if vp.width() == width and vp.height() == height and vp.fps() == fps:
                    target_profile = p
                    break
                    
        if not target_profile:
            # 格式化可用配置列表，方便用户排错
            profiles_str = "\n".join(sorted(list(available_profiles)))
            raise ValueError(f"🚨 找不到支持的流配置: {width}x{height} @ {fps}fps。\n"
                             f"当前硬件支持的 16位 深度配置有:\n{profiles_str}")

        # 3. 创建高性能帧队列 (容量为1，永远只取最新一帧)
        self.frame_queue = rs.frame_queue(1)
        
        # 4. 打开传感器并启动流
        self.depth_sensor.open(target_profile)
        self.depth_sensor.start(self.frame_queue)
        self.is_running = True
        logger.info("realsense stream started: %sx%s @ %sfps", width, height, fps)

    def read_frame(self) -> np.ndarray:
        """
        读取一帧 16 位深度图
        :return: 形状为 (H, W) 的 numpy.uint16 数组。如果超时则返回空数组。
        """
        if not self.is_running:
            return np.array([])
            
        try:
            # 设置 1000ms 超时时间，防止 USB 意外断开导致主线程永久卡死
            frame = self.frame_queue.wait_for_frame(1000)
            if not frame:
                return np.array([])
            
            # asanyarray 可以实现近乎零拷贝的内存映射，CPU 开销极低
            depth_image = np.asanyarray(frame.get_data())
            return depth_image
            
        except RuntimeError:
            # 捕获帧超时异常，返回空数组，让主程序继续运行
            return np.array([])

    def release(self):
        """安全释放底层传感器资源"""
        if self.is_running:
            self.is_running = False
            try:
                self.depth_sensor.stop()
                self.depth_sensor.close()
                logger.info("realsense stream closed")
            except:
                pass

    def __del__(self):
        self.release()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.release()
