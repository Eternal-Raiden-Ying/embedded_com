import os
import gc
import json
import time
import logging
import cv2
import numpy as np
import torch
import uvicorn
from fastapi import FastAPI, UploadFile, File, Form, HTTPException

from .server_log import log_msg, log_recv, log_send
from ..config.global_config import cfgs
from ..backend.engine import RealSenseGraspPredictor

# ==========================================
# 3. FastAPI 服务
# ==========================================
app = FastAPI()

# 全局变量，用于保存模型实例
global_predictor = None

@app.on_event("startup")
async def startup_event():
    log_msg(f"Server starting. Configs loaded: {cfgs}")

@app.post("/api/v1/init")
async def init_model():
    global global_predictor
    log_recv("Received request to INIT model.")
    
    if global_predictor is not None:
        msg = "Predictor is already running."
        log_msg(msg, level=logging.WARNING)
        return {"status": "already_loaded", "message": msg}
    
    log_msg("Initializing predictor and loading weights to GPU...")
    try:
        # 1. 实例化模型并加载权重
        global_predictor = RealSenseGraspPredictor(cfgs)
        
        # # ==========================================
        # # 2. 新增：模型 Warm-up (热身)，彻底消除首次推理延迟
        # # ==========================================
        # log_msg("Performing model warm-up (dummy inference) to pre-allocate activation memory...")
        # tic_warmup = time.time()
        
        # # 生成与真实场景完全相同尺寸的假数据
        # H, W = 720, 1280
        # dummy_rgb = np.ones((H, W, 3), dtype=np.uint8)
        # # 用 1000 填充深度图(模拟1米距离)，用 1 填充掩码，确保数据能通过预处理逻辑进入网络
        # dummy_depth = np.ones((H, W), dtype=np.uint16) * 100 
        # dummy_seg = np.ones((H, W), dtype=np.uint8)
        
        # # 强制跑一次完整推理
        # _ = global_predictor.infer(dummy_rgb, dummy_depth, dummy_seg)
        
        # # 【注意】如果不清理这次推理的输出，它会占用少许显存，但通常可以忽略。
        # # 如果追求极致干净，可以加一句 torch.cuda.empty_cache()，但别加，因为清了缓存下次又要重新分配显存。
        
        # log_msg(f"Warm-up complete in {time.time() - tic_warmup:.3f}s. Model is fully ready for zero-latency inference.")
        # # ==========================================
        
        response = {"status": "success", "message": "Predictor loaded and warmed up successfully."}
        log_send("Model initialization complete.")
        return response
        
    except Exception as e:
        log_msg(f"Failed to load model: {str(e)}", level=logging.ERROR)
        raise HTTPException(status_code=500, detail=f"Init failed: {str(e)}")

@app.post("/api/v1/predict")
async def predict_grasp(
    rgb_file: UploadFile = File(...),
    depth_file: UploadFile = File(...),
    seg_file: UploadFile = File(...),
    metadata: str = Form(...)
):
    global global_predictor
    
    # 记录收到请求
    meta_info = json.loads(metadata)
    robot_id = meta_info.get('robot_id', 'unknown')
    cmd = meta_info.get('cmd', 'unknown')
    log_recv(f"Data received from '{robot_id}'. Command: '{cmd}'. File sizes: RGB({rgb_file.size}B)")

    if global_predictor is None:
        log_msg("Prediction rejected: Predictor not initialized.", level=logging.ERROR)
        raise HTTPException(status_code=400, detail="Call /init first.")

    log_msg(f"Starting inference pipeline for {robot_id}...")
    tic = time.time()
    
    # --- 您的解码和推理逻辑 ---
    rgb_bytes = await rgb_file.read()
    rgb = cv2.imdecode(np.frombuffer(rgb_bytes, np.uint8), cv2.IMREAD_COLOR)
    rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)
    depth_bytes = await depth_file.read()
    depth = cv2.imdecode(np.frombuffer(depth_bytes, np.uint8), cv2.IMREAD_UNCHANGED)
    seg_bytes = await seg_file.read()
    seg = cv2.imdecode(np.frombuffer(seg_bytes, np.uint8), cv2.IMREAD_UNCHANGED)
    grasp_results = global_predictor.infer(rgb, depth, seg)
    # ---------------------------
    
    time.sleep(0.5) # 模拟推理耗时
    
    toc = time.time()
    log_msg(f"Inference finished in {toc - tic:.3f}s")
    
    response = {"status": "success", "grasps": "mock_data_array"}
    log_send(f"Sending results back to '{robot_id}'")
    
    return response

@app.post("/api/v1/release")
async def release_model():
    global global_predictor
    log_recv("Received request to RELEASE model.")
    
    if global_predictor is None:
        msg = "Predictor is not running."
        log_msg(msg, level=logging.WARNING)
        response = {"status": "already_released", "message": msg}
        log_send(f"Response: {response['status']}")
        return response
    
    log_msg("Releasing predictor and freeing GPU memory...")
    
    # 释放显存逻辑
    del global_predictor
    global_predictor = None
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        log_msg("CUDA cache cleared.")
        
    response = {"status": "success", "message": "GPU memory freed."}
    log_send("Model released successfully.")
    return response

if __name__ == "__main__":
    # 使用 dataclass 中的参数控制
    # uvicorn_logger 设为 warning 防止其自带的格式打乱我们的清晰日志
    uvicorn.run(app, host="127.0.0.1", port=6006, log_level="warning")