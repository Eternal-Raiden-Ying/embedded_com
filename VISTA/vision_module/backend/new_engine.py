import time
import queue
import json
import threading
import traceback
import requests
from typing import Optional, Dict, List, Tuple
import cv2
import numpy as np

from .camera import HardwareCamera, RealSenseDepthCamera
from .predictor import QNN_YOLO_Segment_Predictor  # 不同模型的predcitor框架不同
from ..config.schema import VisionServiceConfig

class VisionEngine:
    """
    纯粹的视觉引擎能力层 (SDK)
    外部暴露: init(), start(), stop(), set_cameras(), set_inference(), get_latest_data()
    云端抓取: init_server(), release_server(), predict_grasp()
    """
    def __init__(self, cfg: VisionServiceConfig, logger):
        self.cfg = cfg
        self.log = logger
        
        # 资源句柄
        self.cams: Dict[str, HardwareCamera] = {}
        self.predictor: Optional[QNN_YOLO_Segment_Predictor] = None
        
        # 线程与并发控制
        self.running = False
        self.lock = threading.Lock()
        self.infer_queue = queue.Queue(maxsize=2)
        self._pause_infer_worker = False  # 抓取时临时挂起 NPU 持续推理的标记
        
        # 状态控制 (App层随时修改)
        self.active_cams: List[str] = ["rgb"]  # 默认只开 RGB
        self.infer_enabled = False             # 默认不开 NPU
        
        # 数据缓存 (供 App 层随时无阻塞读取)
        self.latest_cpu_frames: Dict[str, cv2.Mat] = {}
        self.latest_infer_res: Dict[str, list] = {"boxes": [], "masks": []}
        
        # HTTP 云端长连接
        self.cloud_session = requests.Session()

    def init(self):
        """1. 初始化所有硬件外设与模型内存"""
        self.log.info("⚙️ 引擎初始化：加载相机节点与 AI 模型...")
        
        for name, cam_cfg in self.cfg.camera.streams.items():
            if not cam_cfg.enable: continue
            video_node = f"/dev/video{cam_cfg.source}" if str(cam_cfg.source).isdigit() else cam_cfg.source
            
            self.cams[name] = HardwareCamera(
                device=video_node, in_w=cam_cfg.in_w, in_h=cam_cfg.in_h, 
                out_w=cam_cfg.out_w, out_h=cam_cfg.out_h, format=cam_cfg.format, 
                crop_x=cam_cfg.crop_x, crop_y=cam_cfg.crop_y, 
                crop_w=cam_cfg.crop_w, crop_h=cam_cfg.crop_h
            )
            self.log.info(f"📸 挂载相机 [{name.upper()}] -> {video_node} (Format: {cam_cfg.format})")

        active_model_name = self.cfg.model.active_model
        model_profile = self.cfg.model.profiles.get(active_model_name)
        if model_profile:
            self.predictor = QNN_YOLO_Segment_Predictor(model_profile)
            self.log.info(f"🧠 加载 NPU 模型 [{active_model_name}]")

    def start(self):
        """2. 启动引擎流水线"""
        if self.running: return
        self.running = True
        
        threading.Thread(target=self._capture_worker, name="Engine_Capture", daemon=True).start()
        threading.Thread(target=self._infer_worker, name="Engine_Infer", daemon=True).start()
        self.log.info("🚀 视觉底层引擎流水线已全速运转")

    def stop(self):
        """3. 安全销毁所有资源"""
        self.running = False
        time.sleep(0.5) 
        self.cams.clear() 
        self.log.info("🛑 视觉引擎已断开所有相机连接")
        
        if self.predictor is not None:
            self.predictor.release()
            self.predictor = None
            self.log.info("🛑 视觉引擎已彻底卸载 NPU 模型")
            
        self.cloud_session.close()
        self.log.info("✅ 底层硬件资源已全部安全释放")

    # ==========================================
    # 外部控制 API (App层调用)
    # ==========================================
    def set_camera(self, name: str, enable: bool, cfg: dict = None):
        with self.lock:
            if enable:
                if name not in self.cams:
                    # 获取默认配置对象
                    cam_cfg = self.cfg.camera.streams.get(name)
                    
                    # 容错：如果既没有默认配置，也没有传入自定义配置，则报错
                    if not cam_cfg and not cfg:
                        self.log.error(f"❌ 找不到相机 [{name}] 的配置参数")
                        return
                    
                    # 初始化自定义配置为字典，方便后续统一处理
                    cfg = cfg or {}

                    # 内部辅助函数：优先从传入的 cfg 获取参数，没有则回退到 cam_cfg
                    def get_param(key, default=None):
                        if key in cfg:
                            return cfg[key]
                        else:
                            return cam_cfg[key]

                    if name == 'depth':
                        self.cams[name] = RealSenseDepthCamera(
                            height=get_param('height'), 
                            width=get_param('width'), 
                            fps=get_param('fps')
                        )
                        log_target = "RealSense Depth" # 修复原代码这里没有 video_node 的 Bug
                    else:
                        source = get_param('source')
                        video_node = f"/dev/video{source}" if str(source).isdigit() else source
                        
                        self.cams[name] = HardwareCamera(
                            device=video_node, 
                            format=get_param('format'), 
                            fps=get_param('fps'),
                            in_w=get_param('in_w'), 
                            in_h=get_param('in_h'),
                            out_w=get_param('out_w'), 
                            out_h=get_param('out_h'), 
                            crop_x=get_param('crop_x'), 
                            crop_y=get_param('crop_y'),
                            crop_w=get_param('crop_w'), 
                            crop_h=get_param('crop_h')
                        )
                        log_target = video_node

                    self._clear_latest()
                    self.log.info(f"📸 挂载并启动相机 [{name.upper()}] -> {log_target}")
                return
            
            # --- 卸载相机的逻辑保持不变 ---
            if name in self.cams:
                del self.cams[name]
                self._clear_latest()
                self._drain_infer_queue()
                self.log.info(f"🛑 卸载并释放相机 [{name.upper()}]")

    def set_inference(self, enable: bool):
        if self.infer_enabled != enable:
            self.infer_enabled = enable
            state = "开启" if enable else "休眠"
            self.log.info(f"⚡ NPU 推理已{state}")
            if not enable:
                with self.lock:
                    self.latest_infer_res = {"boxes": [], "masks": []}

    def get_latest_data(self) -> Tuple[Dict[str, cv2.Mat], Dict[str, list]]:
        with self.lock:
            return self.latest_cpu_frames.copy(), self.latest_infer_res.copy()

    # ==========================================
    # 云端抓取网络集成 API
    # ==========================================
    def init_server(self, server_url: str) -> bool:
        """发送初始化请求，唤醒云端模型并加载到 GPU"""
        self.log.info("🌐 发送 INIT 请求唤醒云端模型...")
        try:
            response = self.cloud_session.post(f"{server_url.rstrip('/')}/init", timeout=15.0)
            response.raise_for_status()
            self.log.info(f"云端响应: {response.json().get('message')}")
            return True
        except requests.exceptions.RequestException as e:
            self.log.error(f"云端模型初始化失败: {e}")
            return False

    def release_server(self, server_url: str):
        """发送释放请求，清空云端显存"""
        self.log.info("🌐 发送 RELEASE 请求释放云端显存...")
        try:
            response = self.cloud_session.post(f"{server_url.rstrip('/')}/release", timeout=5.0)
            response.raise_for_status()
            self.log.info(f"云端响应: {response.json().get('message')}")
        except requests.exceptions.RequestException as e:
            self.log.error(f"释放云端模型失败: {e}")

    def predict_grasp(self, server_url: str, robot_id: str = "arm_001"):
        """
        抓取核心接口
        """
        if not self.infer_enabled:
            self.log.error("NPU 未开启，无法抓取队列数据！请先调用 set_inference(True)")
            return None
            
        self.log.info("🎯 开始执行同步抓取推理流程...")
        self._pause_infer_worker = True  # 挂起 infer_worker
        
        try:
            # 1. 清空旧队列，确保拿到的是最新鲜的画面
            while not self.infer_queue.empty():
                try:
                    old_data = self.infer_queue.get_nowait()
                    old_hw = old_data[0] # 修复点 1：解包 Tuple
                    del old_hw           # 强制释放旧硬件指针
                except queue.Empty: pass
            
            # 2. 等待获取最新的同步数据包
            data = self.infer_queue.get(timeout=2.0)
            hw_frame, rgb_cpu, depth_cpu = data
            
            if depth_cpu is None:
                self.log.error("没有取到深度图，无法进行 3D 抓取！检查 active_cams")
                del hw_frame
                return None
                
            # 3. 运行 NPU 获取 Seg 掩码
            _, masks = self.predictor.predict_frame(hw_frame)
            del hw_frame  # 必须立刻释放硬件显存！
            
            # 【根据你的模型输出自行调整】这里假设获取到的 masks 需要合并或转换成单通道图像
            if isinstance(masks, np.ndarray) and len(masks.shape) == 2:
                seg = masks.astype(np.uint8)
            elif isinstance(masks, list) and len(masks) > 0:
                seg = masks[0].astype(np.uint8) # 取最主要的目标
            else:
                seg = np.zeros(rgb_cpu.shape[:2], dtype=np.uint8)
                
            # 4. 图像压缩 (RGB 在 capture_worker 里已经被转成 BGR 了，可以直接用)
            tic_comp = time.time()
            _, rgb_enc = cv2.imencode('.jpg', rgb_cpu, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
            _, depth_enc = cv2.imencode('.png', depth_cpu, [int(cv2.IMWRITE_PNG_COMPRESSION), 3])
            _, seg_enc = cv2.imencode('.png', seg, [int(cv2.IMWRITE_PNG_COMPRESSION), 3])
            
            self.log.info(f"🗜️ 图像压缩完成，耗时 {time.time() - tic_comp:.3f}s. "
                          f"RGB:{len(rgb_enc)/1024:.1f}KB Depth:{len(depth_enc)/1024:.1f}KB Seg:{len(seg_enc)/1024:.1f}KB")
            
            # 5. 上传至云端进行 Grasp 推理
            files = {
                'rgb_file': ('rgb.jpg', rgb_enc.tobytes(), 'image/jpeg'),
                'depth_file': ('depth.png', depth_enc.tobytes(), 'image/png'),
                'seg_file': ('seg.png', seg_enc.tobytes(), 'image/png')
            }
            metadata = {
                "robot_id": robot_id,
                "cmd": "get_grasp",
                "timestamp": time.time()
            }
            
            self.log.info("☁️ 正在向云算力集群请求位姿...")
            tic_net = time.time()
            response = self.cloud_session.post(
                f"{server_url.rstrip('/')}/predict", 
                files=files, 
                data={'metadata': json.dumps(metadata)}, 
                timeout=10.0
            )
            response.raise_for_status()
            
            self.log.info(f"✅ 云端推理完成，请求总耗时 {time.time() - tic_net:.3f}s")
            return response.json()
            
        except Exception as e:
            self.log.error(f"抓取流程崩溃:\n{traceback.format_exc()}")
            return None
            
        finally:
            self._pause_infer_worker = False # 恢复常规推理引擎

    # ==========================================
    # 内部工作车间 (完全解耦)
    # ==========================================
    def _capture_worker(self):
        """
        纯粹的拉流车间：保证 UI 永远有最高帧率的图，不受 AI 拖累。
        同时为推理车间 (NPU) 和云端抓取提供【时间绝对同步】的画面包。
        """
        while self.running:
            # 1. 安全获取当前激活的相机列表
            with self.lock:
                active_list = list(self.active_cams)
                
            has_data = False
            current_cpu = {}
            rgb_hw = None
            
            # 2. 遍历所有激活的相机，拉取最新硬件帧
            for name in active_list:
                hw_frame = self.cams[name].read_frame()
                if hw_frame is None or hw_frame.size == 0: 
                    continue
                has_data = True
                
                # 3. 瞬间生成安全的 CPU 副本，供 App 层或抓取压缩使用
                if name == "rgb":
                    # 将硬件的 RGB 格式转为 BGR，供 OpenCV 预览和后续 JPEG 压缩使用
                    cpu_copy = cv2.cvtColor(hw_frame, cv2.COLOR_RGB2BGR) 
                else:
                    # Depth 或 Grey 图像直接在 CPU 内存中拷贝
                    cpu_copy = hw_frame.copy() 
                
                current_cpu[name] = cpu_copy
                
                # 更新全局最新帧缓存 (供 get_latest_data 无阻塞读取)
                with self.lock:
                    self.latest_cpu_frames[name] = cpu_copy

                # 4. 硬件显存管理：若开启推理，扣留 RGB 硬件指针；否则用完即抛
                if name == "rgb" and self.infer_enabled:
                    rgb_hw = hw_frame
                else:
                    # 如果不需要丢给 NPU，必须立刻释放底层硬件内存，防止泄漏！
                    del hw_frame

            # 5. 组装同步帧包并推入推理队列
            if rgb_hw is not None:
                # 检查队列是否阻塞 (maxsize=2)
                if self.infer_queue.full():
                    try:
                        # 弹出最旧的数据包
                        old_data = self.infer_queue.get_nowait()
                        # 【核心修复】：解包 Tuple，拿到硬件指针后强制删除
                        old_hw = old_data[0] 
                        del old_hw           
                    except queue.Empty: 
                        pass
                
                # 打包严格时间同步的 Tuple: (硬件RGB，CPU端RGB，CPU端Depth)
                # 注：如果 depth 没开启，get("depth") 会安全地返回 None
                sync_packet = (rgb_hw, current_cpu.get("rgb"), current_cpu.get("depth"))
                self.infer_queue.put(sync_packet)

            # 6. 防空转保护
            if not has_data:
                time.sleep(0.01)

    def _infer_worker(self):
        """纯粹的推理车间：NPU 只吃硬件原始指针"""
        while self.running:
            # 如果被抓取流程挂起，则原地待命
            if getattr(self, '_pause_infer_worker', False):
                time.sleep(0.01)
                continue
                
            try:
                data = self.infer_queue.get(timeout=0.5)
                hw_frame = data[0] # 解包 tuple
            except queue.Empty:
                continue
                
            try:
                out_boxes, masks = self.predictor.predict_frame(hw_frame)
                del hw_frame # 🚨 NPU 用完后强制释放硬件显存
                
                with self.lock:
                    self.latest_infer_res = {"boxes": out_boxes, "masks": masks}
                    
            except Exception as e:
                self.log.error(f"推理崩溃:\n{traceback.format_exc()}")
                if 'hw_frame' in locals(): del hw_frame