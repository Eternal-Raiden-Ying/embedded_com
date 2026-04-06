#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import time
import logging
import numpy as np

try:
    import aidcv as cv2
except ImportError:
    import cv2

# 根据你的包结构导入
from ..config.board_config import CONFIG
from ..backend.new_engine import VisionEngine
from ..utils.plot import draw_detect_res_fast

# 云端算力公网地址 (替换为你的真实映射地址)
CLOUD_SERVER_URL = "https://u610261-9f44-2435fe5e.westc.seetacloud.com:8443/api/v1"

def setup_logger():
    logger = logging.getLogger("TestCloudGrasp")
    logger.setLevel(logging.INFO)
    if not logger.handlers:
        handler = logging.StreamHandler()
        fmt = logging.Formatter("%(asctime)s | %(levelname)-5s | %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
        handler.setFormatter(fmt)
        logger.addHandler(handler)
    return logger

def main():
    log = setup_logger()
    log.info("=========================================")
    log.info("  VISTA 端云协同抓取测试脚本启动...")
    log.info("=========================================")

    engine = VisionEngine(CONFIG, log)

    try:
        # 1. 初始化本地硬件与 NPU 模型
        engine.init()
        engine.start()

        # 2. 初始化云端模型 (非常重要：此时预热分配显存)
        success = engine.init_server(CLOUD_SERVER_URL)
        if not success:
            log.error("❌ 云端初始化失败，退出测试！")
            return

        if CONFIG.debug.preview:
            cv2.namedWindow("Test Dashboard", cv2.WINDOW_NORMAL)

        # ==================================================
        # 阶段 1：仅开启相机拉流 (5秒)
        # ==================================================
        log.info("📸 [阶段1] 开启相机 RGB + Depth (持续 5 秒)...")
        # 抓取必须用到深度图，因此这里需同时激活 rgb 和 depth
        engine.set_camera("rgb", True, cfg={"out_w":1280, "out_h":720}) 
        engine.set_camera("depth", True, cfg={"width":1280, "height":720}) 
        engine.set_inference(False)

        t_start = time.time()
        while time.time() - t_start < 5.0:
            frames, _ = engine.get_latest_data()
            rgb_cpu = frames.get("rgb")
            
            if rgb_cpu is not None and CONFIG.debug.preview:
                bgr_canvas = rgb_cpu.copy() # VisionEngine内部已转为 BGR
                cv2.putText(bgr_canvas, "Phase 1: Camera Only", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
                cv2.imshow("Test Dashboard", bgr_canvas)
                cv2.waitKey(1)
            time.sleep(0.03)

        # ==================================================
        # 阶段 2：开启本地 NPU 连续推理 (5秒)
        # ==================================================
        log.info("🧠 [阶段2] 开启本地 NPU 连续推理 (持续 5 秒)...")
        engine.set_inference(True)

        t_start = time.time()
        while time.time() - t_start < 5.0:
            frames, infer_res = engine.get_latest_data()
            rgb_cpu = frames.get("rgb")
            
            if rgb_cpu is not None and CONFIG.debug.preview:
                bgr_canvas = rgb_cpu.copy()
                
                # 修复点 2：使用 len() 安全判断 Numpy 数组是否为空
                if infer_res:
                    boxes = infer_res.get("boxes", [])
                    masks = infer_res.get("masks", [])
                    # 如果 boxes 是 Numpy 数组，len(boxes) 可以完美判断它内部有没有目标
                    if len(boxes) > 0:
                        bgr_canvas = draw_detect_res_fast(bgr_canvas, boxes, masks)
                
                cv2.putText(bgr_canvas, "Phase 2: Local NPU Active", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 165, 255), 2)
                cv2.imshow("Test Dashboard", bgr_canvas)
                cv2.waitKey(1)
            time.sleep(0.03)

        # ==================================================
        # 阶段 3：执行云端同步抓取
        # ==================================================
        log.info("☁️ [阶段3] 触发云端抓取推理...")
        
        # 引擎内部会自动挂起 NPU 连续流，安全拿取一帧图像跑出 Seg 并压缩上传
        result = engine.predict_grasp(CLOUD_SERVER_URL, robot_id="test_arm_01")
        
        if result and result.get("status") == "success":
            log.info(f"✅ 云端抓取成功！获取到位姿数据: {result.get('grasps')}")
        else:
            log.error("❌ 云端抓取失败或超时。")

        # ==================================================
        # 阶段 4：关闭 AI 推理进入待机
        # ==================================================
        log.info("💤 抓取完成，关闭本地 NPU 推理节省功耗...")
        engine.set_inference(False)

        # 保持画面显示 2 秒，观察 NPU 框是否消失
        t_start = time.time()
        while time.time() - t_start < 2.0:
            frames, _ = engine.get_latest_data()
            rgb_cpu = frames.get("rgb")
            if rgb_cpu is not None and CONFIG.debug.preview:
                bgr_canvas = rgb_cpu.copy()
                cv2.putText(bgr_canvas, "Phase 4: Standby", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (150, 150, 150), 2)
                cv2.imshow("Test Dashboard", bgr_canvas)
                cv2.waitKey(1)
            time.sleep(0.03)

    except KeyboardInterrupt:
        log.info("🛑 收到用户中断信号 (Ctrl+C)")
    except Exception as e:
        log.error(f"❌ 测试脚本崩溃: {e}", exc_info=True)
    finally:
        log.info("🧹 清理资源并释放云端显存...")
        # 释放云端算力 (非常关键，防止按小时计费平台被白嫖显存)
        engine.release_server(CLOUD_SERVER_URL)
        engine.stop()
        if CONFIG.debug.preview:
            cv2.destroyAllWindows()
        log.info("✅ 测试平稳结束。")

if __name__ == "__main__":
    main()