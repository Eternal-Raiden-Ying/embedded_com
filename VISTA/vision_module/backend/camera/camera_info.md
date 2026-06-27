# 相机设备参数能力信息 (Camera Capability Info)

## 摘要 (Summary)
*   **深度相机 (Depth)**：不支持通过 GStreamer (gst) 管道进行捕获。
*   **红外相机 (IR)**：支持输出格式为 `GRAY8` 或 `UYVY`，并在后端可转换为 `RGB/BGR`。
*   **彩色相机 (Color)**：支持输出格式为 `YUY2`，并在后端可转换为 `RGB/BGR`。

---

## 1. 深度相机能力 (Depth Cam Capabilities)
*   **系统调用接口**：`ioctl: VIDIOC_ENUM_FMT`
*   **采集类型**：Video Capture（视频捕获）
*   **支持格式与分辨率组合**：
    *   **格式 `[0]: Z16` (16-bit 原始深度值)**
        *   分辨率 `256x144`：支持 300 fps / 90 fps
        *   分辨率 `424x240`：支持 90 fps / 60 fps / 30 fps / 15 fps / 6 fps
        *   分辨率 `480x270`：支持 90 fps / 60 fps / 30 fps / 15 fps / 6 fps
        *   分辨率 `640x360`：支持 90 fps / 60 fps / 30 fps / 15 fps / 6 fps
        *   分辨率 `640x480`（**推荐工作模式**）：支持 90 fps / 60 fps / 30 fps / 15 fps / 6 fps
        *   分辨率 `848x100`：支持 300 fps / 100 fps
        *   分辨率 `848x480`：支持 90 fps / 60 fps / 30 fps / 15 fps / 6 fps
        *   分辨率 `1280x720`：支持 30 fps / 15 fps / 6 fps

---

## 2. 红外相机能力 (IR Cam Capabilities)
*   **系统调用接口**：`ioctl: VIDIOC_ENUM_FMT`
*   **采集类型**：Video Capture（视频捕获）
*   **支持格式与分辨率组合**：
    *   **格式 `[0]: GREY` (8-bit 灰度图)**
        *   分辨率 `640x480`：支持 90 fps / 60 fps / 30 fps / 15 fps / 6 fps
        *   分辨率 `1280x720`：支持 30 fps / 15 fps / 6 fps
    *   **格式 `[1]: UYVY` (UYVY 4:2:2 压缩采样)**
        *   分辨率 `640x480`：支持 90 fps / 60 fps / 30 fps / 15 fps / 6 fps
        *   分辨率 `1280x720`：支持 30 fps / 15 fps / 6 fps
    *   **格式 `[2]: GREY` (同格式 [0])**
    *   **格式 `[3]: Y8I` (交叉 8-bit 灰度采样)**
        *   分辨率 `640x480`：支持 90 fps / 60 fps / 30 fps / 15 fps / 6 fps
        *   分辨率 `1280x720`：支持 30 fps / 15 fps / 6 fps
        *   分辨率 `1280x800`：支持 30 fps / 15 fps
    *   **格式 `[4]: Y12I` (交叉 12-bit 灰度采样)**
        *   分辨率 `640x400`：支持 25 fps / 15 fps
        *   分辨率 `1280x800`：支持 25 fps / 15 fps

---

## 3. 彩色相机能力 (Color Cam Capabilities)
*   **系统调用接口**：`ioctl: VIDIOC_ENUM_FMT`
*   **采集类型**：Video Capture（视频捕获）
*   **支持格式与分辨率组合**：
    *   **格式 `[0]: YUYV` (YUYV 4:2:2 图像数据)**
        *   分辨率 `320x240`：支持 60 fps / 30 fps / 6 fps
        *   分辨率 `640x480`：支持 60 fps / 30 fps / 15 fps / 6 fps
        *   分辨率 `960x540`：支持 60 fps / 30 fps / 15 fps / 6 fps
        *   分辨率 `1280x720`（**推荐工作模式**）：支持 30 fps / 15 fps / 6 fps
        *   分辨率 `1920x1080`：支持 30 fps / 15 fps / 6 fps