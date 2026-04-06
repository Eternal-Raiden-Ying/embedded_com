import os
import sys
import subprocess

# 1. 动态获取当前 __init__.py 所在的绝对路径
_CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))

# 2. 将当前目录临时加入系统路径，确保 Python 能找到同目录下的 .so 文件
if _CURRENT_DIR not in sys.path:
    sys.path.insert(0, _CURRENT_DIR)

try:
    # 3. 导入底层的 C++ 拓展库
    import fast_cam
except ImportError as e:
    raise ImportError(
        f"🚨 无法导入底层 C++ 扩展库！\n"
        f"请确保你已经进入 {_CURRENT_DIR}/csrc 目录完成了 CMake 编译，\n"
        f"并将生成的 fast_cam.*.so 文件复制到了 {_CURRENT_DIR} 目录下。\n"
        f"底层错误信息: {e}"
    )

# 4. 导出 Camera 类，对外暴露干净的 API

# 查看硬件支持的参数配置，请在终端运行 v4l2-ctl -d /dev/video[id of your camera] --list-formats-ext
class HardwareCamera:
    """
    AidLux 硬件加速零拷贝相机 (基于 GStreamer & qtivtransform)
    支持动态调节 V4L2 寄存器 (曝光、亮度等)
    """
    def __init__(self, 
                 device: str = '/dev/video0', 
                 in_w: int = 1280, in_h: int = 720, 
                 out_w: int = 640, out_h: int = 640, 
                 fps: int = 30,                 # ==== 同步新增 ====
                 format: str = "RGB",           
                 in_format: str = "YUY2",       
                 flip_h: bool = False, flip_v: bool = False, 
                 rotate: int = 0,
                 crop_x: int = 0, crop_y: int = 0, 
                 crop_w: int = 0, crop_h: int = 0,
                 auto_exposure: bool = None,    
                 exposure: int = None,          
                 brightness: int = None):       
        
        self.device = device
        self._cam = None 

        # 1. 提前配置硬件寄存器...
        if auto_exposure is not None:
            self.set_auto_exposure(auto_exposure)
        if exposure is not None:
            self.set_exposure(exposure)
        if brightness is not None:
            self.set_brightness(brightness)

        # 2. 实例化 C++ 底层类 (严格按 C++ 绑定的顺序传入)
        self._cam = fast_cam.Camera(
            device, in_w, in_h, out_w, out_h, fps,  # ==== 传入 fps ====
            in_format, format, 
            flip_h, flip_v, rotate, crop_x, crop_y, crop_w, crop_h
        )
    # ==========================================
    # 动态控制接口 (基于 subprocess 包装)
    # ==========================================
    def _v4l2_set_ctrl(self, ctrl_name: str, value: int):
        """内部通用方法：调用 v4l2-ctl 设置底层参数，自带异常保护"""
        try:
            subprocess.run(
                ["v4l2-ctl", "-d", self.device, "-c", f"{ctrl_name}={value}"],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.PIPE,
                check=True,
                text=True
            )
        except subprocess.CalledProcessError as e:
            # 针对不支持该控制的相机（如红外/深度相机）给出友好提示而非崩溃
            print(f"⚠️ [硬件相机提示] 设备 {self.device} 不支持/或越界配置 '{ctrl_name}'。")
        except FileNotFoundError:
            print(f"🚨 [系统错误] 未找到 v4l2-ctl 命令，请执行 'apt install v4l2-utils' 安装。")

    def set_auto_exposure(self, enable: bool):
        """动态开关自动曝光 (True: 自动, False: 手动)"""
        val = 3 if enable else 1  # 3 代表 Auto，1 代表 Manual
        self._v4l2_set_ctrl("exposure_auto", val)

    def set_exposure(self, value: int):
        """
        动态设置绝对曝光时间 (值越小曝光越短，防拖影)。
        必须在 auto_exposure=False 时才生效。
        """
        self._v4l2_set_ctrl("exposure_absolute", value)

    def set_brightness(self, value: int):
        """动态设置画面亮度 (例如 D435 RGB 相机范围通常是 -64 ~ 64)"""
        self._v4l2_set_ctrl("brightness", value)


    # ==========================================
    # 数据读取与资源释放接口
    # ==========================================
    def read_frame(self):
        """
        读取一帧零拷贝硬件图像。
        :return: NumPy Array，如果读取失败或预热中则返回空数组 (size==0)
        """
        if self._cam is None:
            raise RuntimeError("相机已被释放或未正确初始化，无法读取帧。")
        return self._cam.read_frame()

    def release(self):
        """主动释放底层 C++ 资源，安全关闭 GStreamer 流水线"""
        if self._cam is not None:
            # 删除 C++ 对象的引用，触发 C++ 端的析构函数 (~HardwareCamera)
            del self._cam
            self._cam = None
            print(f"✅ [硬件相机] {self.device} 资源已安全释放。")

    def __del__(self):
        """Python 垃圾回收触发机制，防止用户忘记 release"""
        self.release()

    # ------------------------------------------
    # 魔法方法: 支持 with 上下文管理器
    # ------------------------------------------
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.release()
