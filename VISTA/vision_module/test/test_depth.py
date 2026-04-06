import pyrealsense2 as rs
import numpy as np
import aidcv as cv2
import time

def main():
    print("="*50)
    print("🚀 RealSense 底层 Sensor API 视频流启动")
    print("="*50)

    # 1. 寻找设备并获取传感器
    ctx = rs.context()
    devices = ctx.query_devices()
    if not devices:
        print("❌ 找不到设备！")
        return
        
    dev = devices[0]
    depth_sensor = dev.first_depth_sensor()

    # 2. 找到指定的低负载配置 (424x240 @ 15fps Z16)
    target_profile = None
    for p in depth_sensor.get_stream_profiles():
        if p.stream_type() == rs.stream.depth and p.format() == rs.format.z16:
            vp = p.as_video_stream_profile()
            if vp.width() == 424 and vp.fps() == 15:
                target_profile = p
                break
                
    if not target_profile:
        print("❌ 找不到对应的流配置。")
        return

    # 3. 核心机制：创建一个容量为 1 的高性能帧队列
    # 这就像我们之前 GStreamer 里的 max-buffers=1，保证拿到最新的一帧
    frame_queue = rs.frame_queue(1)

    try:
        # 4. 启动传感器，并将抓到的帧直接塞进队列
        print("🔗 正在打开深度传感器...")
        depth_sensor.open(target_profile)
        
        print("🚀 正在启动流，挂载队列接收器...")
        # 注意：这里我们不是用回调函数打印了，而是让它把帧存进 frame_queue
        depth_sensor.start(frame_queue)
        
        cv2.namedWindow("RealSense Depth (Sensor API)")
        print("✅ 视频流已启动！请打开网页 [cvs] 查看。")
        
        prev_time = time.time()
        
        # 5. 主循环：从队列中提取帧并显示
        while True:
            # 从队列中提取一帧 (如果队列是空的，它会阻塞等待，但不会像 pipeline 那样死锁)
            frame = frame_queue.wait_for_frame()
            
            # 转为 numpy 数组 (16位深度图)
            depth_image = np.asanyarray(frame.get_data())
            
            # 渲染成伪彩图以便观察
            depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET)
            
            # 计算 FPS
            curr_time = time.time()
            fps = 1.0 / (curr_time - prev_time)
            prev_time = curr_time
            
            cv2.putText(depth_colormap, f"FPS: {fps:.1f} (Raw Sensor API)", (20, 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            cv2.imshow("RealSense Depth (Sensor API)", depth_colormap)
            
            if cv2.waitKey(1) == 27: # ESC 退出
                break

    except Exception as e:
        print(f"❌ 发生错误: {e}")
        
    finally:
        # 6. 安全关闭
        print("🛑 正在停止传感器...")
        try:
            depth_sensor.stop()
            depth_sensor.close()
        except:
            pass
        print("✅ 退出成功。")

if __name__ == "__main__":
    main()