import os
import psutil

class HardwareMonitor:
    def __init__(self):
        # 高通 Adreno GPU 占用率的经典内核节点路径
        self.gpu_path_1 = "/sys/class/kgsl/kgsl-3d0/gpu_busy_percentage"
        self.gpu_path_2 = "/sys/class/kgsl/kgsl-3d0/gpubusy"
        
        # 高通 Hexagon DSP/NPU 相关的可能节点 (不同内核版本路径会有差异)
        # 这里列出 QCS6490/骁龙系列 常见的 CDSP (Compute DSP) 频率/负载节点
        self.dsp_paths = [
            "/sys/class/devfreq/soc:qcom,cdsp-cdsp-l3-lat/device/load",
            "/sys/class/devfreq/soc:qcom,npu-npu-ddr-bw/bw_hwmon/io_percent"
        ]

    def read_sysfs(self, path):
        """安全读取 Linux sysfs 节点的数值"""
        if os.path.exists(path):
            try:
                with open(path, 'r') as f:
                    return f.read().strip()
            except Exception:
                return None
        return None

    def get_cpu_usage(self):
        """获取 CPU 整体占用率"""
        return psutil.cpu_percent(interval=None)

    def get_gpu_usage(self):
        """获取高通 Adreno GPU 占用率"""
        # 尝试路径 1 (直接返回百分比，如 "45 %")
        val = self.read_sysfs(self.gpu_path_1)
        if val:
            return float(val.replace('%', '').strip())
        
        # 尝试路径 2 (返回形式通常是 "繁忙周期 总周期"，例如 "4500 10000")
        val = self.read_sysfs(self.gpu_path_2)
        if val:
            parts = val.split()
            if len(parts) >= 2 and float(parts[1]) > 0:
                return round((float(parts[0]) / float(parts[1])) * 100, 1)
        return 0.0

    def get_dsp_usage(self):
        """获取 Hexagon DSP/NPU 占用率 (受限于内核权限)"""
        for path in self.dsp_paths:
            val = self.read_sysfs(path)
            if val and val.isdigit():
                return float(val)
        return -1.0  # -1 表示该内核节点未暴露或无权限读取

    def get_all_stats(self):
        return {
            "CPU": self.get_cpu_usage(),
            "GPU": self.get_gpu_usage(),
            "DSP": self.get_dsp_usage()
        }
