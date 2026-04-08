import time
import argparse
import threading
import queue
import aidcv as cv2  # AidLux 专用的推流与可视化库

# 导入我们全新封装的底层加速模块
from ..backend.camera import HardwareCamera
from ..backend.predictor import QNNPredictor

# 导入工具与监控
from ..utils.plot import draw_detect_res_fast
from ..utils.statistic import HardwareMonitor

def parser_args():
    parser = argparse.ArgumentParser(description="AidLux QNN 零拷贝极速流水线")
    parser.add_argument(
        '--target_model', type=str, 
        default='/home/aidlux/2026/VISTA/vision_module/model/yolo26s-seg-grasp/yolo26s-seg-grasp_split_qcs6490_w8a8.qnn236.ctx.bin.amf',
        help="量化模型路径"
    )
    parser.add_argument('--source', type=str, default='6', help="视频源: 填入 0 调用摄像头，或传入具体节点名")
    parser.add_argument('--max_fps', type=int, default=30, help="最高渲染帧率，默认放开到 30")
    parser.add_argument('--width', type=int, default=640, help="模型输入宽")
    parser.add_argument('--height', type=int, default=640, help="模型输入高")
    parser.add_argument('--conf_thres', type=float, default=0.25, help="置信度阈值")
    parser.add_argument('--iou_thres', type=float, default=0.3, help="NMS IOU阈值")
    parser.add_argument('--class_num', type=int, default=20, help="数据集类别数")
    return parser.parse_args()


class AsyncStreamPipeline:
    def __init__(self, args):
        self.max_fps = args.max_fps
        self.hw_monitor = HardwareMonitor()
        
        # 1. 格式化设备源
        source_str = str(args.source)
        video_node = f"/dev/video{source_str}" if source_str.isdigit() else source_str

        # 2. 初始化硬件相机 (激活 Center Crop，1280x720 -> 完美比例 640x640 RGB)
        print("🎥 正在拉起 C++ DMA-BUF 零拷贝相机...")
        self.cam = HardwareCamera(
            device=video_node, 
            in_w=1280, in_h=720, 
            out_w=args.width, out_h=args.height, 
            format="RGB", 
            crop_x=280, crop_y=0, crop_w=720, crop_h=720 
        )

        # 3. 初始化 AI 推理引擎
        self.model = QNNPredictor(args)
        
        # 4. 建立线程安全的流水线队列 (限制 maxsize 防止爆内存和累计延迟)
        self.frame_queue = queue.Queue(maxsize=2)
        self.result_queue = queue.Queue(maxsize=2)
        self.running = True

        # 5. 启动后台独立工作车间
        self.capture_thread = threading.Thread(target=self._capture_worker, daemon=True)
        self.infer_thread = threading.Thread(target=self._infer_worker, daemon=True)
        
        self.capture_thread.start()
        self.infer_thread.start()

    def _capture_worker(self):
        """线程 1：拉流车间 - 专职向底层硬件索要画面"""
        print("🟢 拉流线程已启动，底层跑在 C++ 隔离态...")
        while self.running:
            hw_frame = self.cam.read_frame()
            if hw_frame is None or hw_frame.size == 0:
                time.sleep(0.01)
                continue
            
            # 队列满了就扔掉旧画面，永远只给 AI 看"最新"的一瞬
            if self.frame_queue.full():
                try:
                    old_frame = self.frame_queue.get_nowait()
                    del old_frame # 🚨 极其关键：将作废的显存还给底层池！
                except queue.Empty:
                    pass
            
            self.frame_queue.put(hw_frame)

    def _infer_worker(self):
        """线程 2：推理车间 - 专职调用 NPU 与画图引擎"""
        print("🧠 推理线程已启动，QNN 引擎接管...")
        while self.running:
            try:
                # 阻塞等待新画面 (设置 timeout 防止主线程退出时卡死)
                hw_frame = self.frame_queue.get(timeout=1)
            except queue.Empty:
                continue

            # ==========================================
            # ⚡ 核心推理：将原汁原味的 RGB 硬件显存地址直接喂给 NPU
            # ==========================================
            out_boxes, masks = self.model.predict_frame(hw_frame)

            # 准备显示：转换颜色并生成普通的 CPU 内存副本
            display_bgr = cv2.cvtColor(hw_frame, cv2.COLOR_RGB2BGR)
            
            # 🚨 保命操作：立刻释放原有的硬件内存，防止管道死锁
            del hw_frame 

            # 极速向量化画框与掩码渲染
            res_frame = draw_detect_res_fast(display_bgr, out_boxes, masks)

            # 将画好的成品塞入结果队列
            if self.result_queue.full():
                try:
                    self.result_queue.get_nowait()
                except queue.Empty:
                    pass
            self.result_queue.put(res_frame)

    def run(self):
        """主线程：负责与网页 CVS 交互并绘制硬件监控 UI"""
        print("🚀 主推流线程已启动！请打开网页 [cvs] 查看。")
        cv2.namedWindow("AidLux AI Stream")
        
        target_frame_time = 1.0 / self.max_fps
        prev_frame_time = time.time()
        
        while self.running:
            loop_start_time = time.time()
            
            try:
                # 瞬间拿到已经推理好的成品图像
                res_frame = self.result_queue.get(timeout=0.1)
                
                # --- 获取硬件监控状态 ---
                stats = self.hw_monitor.get_all_stats()
                cv2.putText(res_frame, f"CPU: {stats['CPU']}%", (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
                cv2.putText(res_frame, f"GPU: {stats['GPU']}%", (20, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
                dsp_val = stats['DSP']
                cv2.putText(res_frame, f"DSP: {dsp_val}%" if dsp_val >= 0 else "DSP: N/A", (20, 140), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
                
                # --- 计算全链路综合 FPS ---
                now = time.time()
                real_fps = 1.0 / (now - prev_frame_time + 1e-5)
                prev_frame_time = now 
                cv2.putText(res_frame, f"Real FPS: {real_fps:.1f} (Cap: {self.max_fps})", (20, 40), 
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

                # --- 网页推流显示 ---
                cv2.imshow("AidLux AI Stream", res_frame)
                
            except queue.Empty:
                # 拿不到图时也要维持 cv2.waitKey 心跳，防止网页断连
                pass
            
            # --- 处理退出事件 ---
            if cv2.waitKey(1) & 0xFF in [27, ord('q')]:
                print("🛑 收到退出信号，正在关闭系统...")
                self.running = False
                break

            # --- 智能锁帧逻辑 ---
            process_time = time.time() - loop_start_time
            if process_time < target_frame_time:
                time.sleep(target_frame_time - process_time)


if __name__ == "__main__":
    args = parser_args()
    pipeline = AsyncStreamPipeline(args)
    pipeline.run()
