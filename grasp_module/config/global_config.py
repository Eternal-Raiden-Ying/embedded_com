import os
import gc
import json
import time
import logging
import argparse
from datetime import datetime
from dataclasses import dataclass


# ==========================================
# 1. 配置管理 (Dataclass)
# ==========================================
@dataclass
class AppConfig:
    checkpoint_path: str = '/root/autodl-tmp/vista/grasp_module/weights/minkuresunet_kinect.tar'
    dump_dir: str = '/root/autodl-tmp/vista/grasp_module/debug_res'
    seed_feat_dim: int = 512
    num_point: int = 15000
    voxel_size: float = 0.005
    collision_thresh: float = -1.0
    voxel_size_cd: float = 0.01
    debug: bool = False
    log_path: str = '/root/autodl-tmp/vista/grasp_module/log/server.log'  # 新增的日志路径参数

def get_config() -> AppConfig:
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed_feat_dim', default=512, type=int)
    parser.add_argument('--num_point', type=int, default=15000)
    parser.add_argument('--voxel_size', type=float, default=0.005)
    parser.add_argument('--collision_thresh', type=float, default=-1)
    parser.add_argument('--voxel_size_cd', type=float, default=0.01)
    parser.add_argument('--debug', action='store_true', default=False)
    
    args = parser.parse_args()
    return AppConfig(**vars(args))

cfgs = get_config()
