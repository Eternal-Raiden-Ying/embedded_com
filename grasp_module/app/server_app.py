import os
import gc
import json
import time
import logging
import cv2
import numpy as np
import torch
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
APP_DIR = os.path.dirname(os.path.abspath(__file__))
MODULE_DIR = os.path.dirname(APP_DIR)


def _normalize_depth(depth):
    if depth is None:
        return None
    if depth.ndim == 3 and depth.shape[2] == 1:
        return depth[:, :, 0]
    return depth


def _decode_rgb_image(image_bytes):
    image = cv2.imdecode(np.frombuffer(image_bytes, np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise HTTPException(status_code=400, detail="Failed to decode rgb_file")
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def _decode_depth_image(image_bytes):
    depth = _normalize_depth(cv2.imdecode(np.frombuffer(image_bytes, np.uint8), cv2.IMREAD_UNCHANGED))
    if depth is None:
        raise HTTPException(status_code=400, detail="Failed to decode depth_file")
    if depth.ndim != 2:
        raise HTTPException(status_code=400, detail=f"depth_file must decode to a 2D image, got shape {depth.shape}")
    return depth


def load_warmup_sample_inputs():
    candidate_sets = [
        {
            'name': 'app_dummy_inputs',
            'rgb': os.path.join(APP_DIR, 'dummy_inputs', 'color_00000.png'),
            'depth': os.path.join(APP_DIR, 'dummy_inputs', 'depth_raw_00000.png'),
        },
        {
            'name': 'test_data',
            'rgb': os.path.join(MODULE_DIR, 'test', 'data', 'color', 'color_00000.png'),
            'depth': os.path.join(MODULE_DIR, 'test', 'data', 'depth', 'depth_raw_00000.png'),
        },
    ]

    for candidate in candidate_sets:
        if not all(os.path.exists(candidate[key]) for key in ('rgb', 'depth')):
            continue

        rgb_bgr = cv2.imread(candidate['rgb'], cv2.IMREAD_COLOR)
        depth = _normalize_depth(cv2.imread(candidate['depth'], cv2.IMREAD_UNCHANGED))
        if rgb_bgr is None or depth is None:
            continue
        if depth.ndim != 2:
            continue

        rgb = cv2.cvtColor(rgb_bgr, cv2.COLOR_BGR2RGB)
        log_msg(f"Warm-up using sample inputs from {candidate['name']}")
        return rgb, depth.astype(np.uint16)

    return None


def build_generated_warmup_inputs(height=720, width=1280, seed=0):
    rng = np.random.default_rng(seed)
    yy, xx = np.indices((height, width), dtype=np.float32)

    # 生成带梯度和轻微噪声的 RGB，避免完全一致输入。
    rgb = np.zeros((height, width, 3), dtype=np.uint8)
    rgb[..., 0] = np.clip(40 + (xx / width) * 180 + rng.normal(0, 6, size=(height, width)), 0, 255).astype(np.uint8)
    rgb[..., 1] = np.clip(60 + (yy / height) * 120 + rng.normal(0, 6, size=(height, width)), 0, 255).astype(np.uint8)
    rgb[..., 2] = np.clip(90 + ((xx + yy) / (width + height)) * 100 + rng.normal(0, 6, size=(height, width)), 0, 255).astype(np.uint8)

    depth = 470.0 + 45.0 * np.sin(xx / width * np.pi * 2.0) + 25.0 * np.cos(yy / height * np.pi)
    depth += rng.normal(0, 4.0, size=(height, width))
    depth = np.clip(depth, 360.0, 620.0).astype(np.uint16)

    log_msg("Warm-up using generated structured dummy inputs")
    return rgb, depth


def build_warmup_inputs():
    sample_inputs = load_warmup_sample_inputs()
    if sample_inputs is not None:
        return sample_inputs
    return build_generated_warmup_inputs()


def warmup_predictor(predictor):
    log_msg("Performing predictor warm-up inference...")
    tic = time.time()

    original_debug = predictor.cfgs.debug
    predictor.cfgs.debug = False
    try:
        warmup_rgb, warmup_depth = build_warmup_inputs()
        warmup_class_id = int(getattr(predictor.cfgs, 'yolo_class_id', 46))
        _ = predictor.infer(warmup_rgb, warmup_depth, warmup_class_id)
    finally:
        predictor.cfgs.debug = original_debug

    log_msg(f"Warm-up complete in {time.time() - tic:.3f}s")


def build_downstream_response(grasp_results, protocol_targets, predictor_cfgs):
    raw_grasp_count = 0 if grasp_results is None else len(grasp_results)
    if raw_grasp_count == 0:
        return {
            "status": "reposition_required",
            "grasp_count": 0,
            "feasible_count": 0,
            "output_count": 0,
            "targets": [],
            "reason": "no_grasp_detected",
            "message": "placeholder",
        }

    feasible_count = len(protocol_targets)
    if feasible_count == 0:
        return {
            "status": "reposition_required",
            "grasp_count": raw_grasp_count,
            "feasible_count": 0,
            "output_count": 0,
            "targets": [],
            "reason": "no_feasible_grasp",
            "message": "placeholder",
        }

    min_score = float(getattr(predictor_cfgs, 'protocol_min_score', 0.0))
    max_targets = max(1, int(getattr(predictor_cfgs, 'response_max_targets', 5)))
    output_targets = [target for target in protocol_targets if target["confidence"] >= min_score][:max_targets]
    if not output_targets:
        return {
            "status": "reposition_required",
            "grasp_count": raw_grasp_count,
            "feasible_count": feasible_count,
            "output_count": 0,
            "targets": [],
            "reason": "score_below_threshold",
            "message": "placeholder",
        }

    return {
        "status": "success",
        "grasp_count": raw_grasp_count,
        "feasible_count": feasible_count,
        "output_count": len(output_targets),
        "targets": output_targets,
    }

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
        warmup_predictor(global_predictor)
        
        response = {"status": "success", "message": "Predictor loaded and warmed up successfully."}
        log_send("Model initialization complete.")
        return response
        
    except Exception as e:
        if global_predictor is not None:
            del global_predictor
            global_predictor = None
        log_msg(f"Failed to load model: {str(e)}", level=logging.ERROR)
        raise HTTPException(status_code=500, detail=f"Init failed: {str(e)}")

@app.post("/api/v1/predict")
async def predict_grasp(
    rgb_file: UploadFile = File(...),
    depth_file: UploadFile = File(...),
    class_id: int = Form(...),
    metadata: str = Form(...)
):
    global global_predictor
    
    # 记录收到请求
    try:
        meta_info = json.loads(metadata)
    except json.JSONDecodeError as exc:
        raise HTTPException(status_code=400, detail="metadata must be valid JSON") from exc

    if class_id < 0:
        raise HTTPException(status_code=400, detail="class_id must be a non-negative integer")
    robot_id = meta_info.get('robot_id', 'unknown')
    cmd = meta_info.get('cmd', 'unknown')
    log_recv(
        f"Data received from '{robot_id}'. Command: '{cmd}'. "
        f"File sizes: RGB({rgb_file.size}B) Depth({depth_file.size}B). class_id={class_id}"
    )

    if global_predictor is None:
        log_msg("Prediction rejected: Predictor not initialized.", level=logging.ERROR)
        raise HTTPException(status_code=400, detail="Call /init first.")

    log_msg(f"Starting inference pipeline for {robot_id}...")
    tic = time.time()
    
    # --- 您的解码和推理逻辑 ---
    rgb = _decode_rgb_image(await rgb_file.read())
    depth = _decode_depth_image(await depth_file.read())
    grasp_results = global_predictor.infer(rgb, depth, int(class_id))

    toc = time.time()
    log_msg(f"Inference finished in {toc - tic:.3f}s")

    protocol_targets = global_predictor.build_protocol_targets(grasp_results)
    response = build_downstream_response(grasp_results, protocol_targets, global_predictor.cfgs)
    if response["status"] == "success":
        for idx, target in enumerate(response["targets"], start=1):
            log_msg(
                f"Protocol target #{idx}: "
                f"x={target['x_cm']:.2f}cm y={target['y_cm']:.2f}cm z={target['z_cm']:.2f}cm "
                f"pitch={target['pitch_deg']:.2f}deg roll={target['roll_deg']:.2f}deg "
                f"width={target['gripper_width_cm']:.2f}cm depth={target['approach_depth_cm']:.2f}cm "
                f"score={target['confidence']:.4f} feasible_angle={target['feasible_angle_deg']:.2f}deg"
            )
    else:
        log_msg(
            "Protocol output requires reposition. "
            f"reason={response.get('reason')} grasp_count={response.get('grasp_count')} "
            f"feasible_count={response.get('feasible_count')}",
            level=logging.WARNING,
        )
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
    import uvicorn

    # 使用 dataclass 中的参数控制
    # uvicorn_logger 设为 warning 防止其自带的格式打乱我们的清晰日志
    uvicorn.run(app, host="127.0.0.1", port=6006, log_level="warning")
