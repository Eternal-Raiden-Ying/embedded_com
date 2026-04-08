import os
import gc
import json
import time
import logging
import argparse
from datetime import datetime
from dataclasses import dataclass

from ..config.global_config import cfgs


# ==========================================
# 2. 日志系统 (Custom Logger)
# ==========================================
class ServerLogFormatter(logging.Formatter):
    def format(self, record):
        # 提取动态传入的 type，默认为 msg
        msg_type = getattr(record, 'msg_type', 'msg')
        
        # 控制宽度：级别定宽 5，类型定宽 4，居中对齐
        level_str = f"{record.levelname:^5}"
        type_str = f"{msg_type:^4}"
        
        # 格式化时间：年月日时分秒
        dt_str = datetime.fromtimestamp(record.created).strftime('%Y-%m-%d %H:%M:%S')
        
        # 拼接最终格式
        return f"[Server][{level_str}]|{type_str}|{dt_str}| {record.getMessage()}"

def setup_logger(log_file: str):
    logger = logging.getLogger("AppLogger")
    logger.setLevel(logging.INFO)
    logger.propagate = False  # 防止向上传递导致 FastAPI 重复打印
    
    formatter = ServerLogFormatter()
    
    # 文件输出
    os.makedirs(os.path.dirname(os.path.abspath(log_file)), exist_ok=True)
    file_handler = logging.FileHandler(log_file, encoding='utf-8')
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    
    # 控制台输出
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    return logger

# 实例化全局 logger
log = setup_logger(cfgs.log_path)


# 提供便捷的封装函数，省去每次写 extra 的麻烦
def log_msg(msg, level=logging.INFO):   log.log(level, msg, extra={'msg_type': 'msg'})
def log_recv(msg, level=logging.INFO):  log.log(level, msg, extra={'msg_type': 'recv'})
def log_send(msg, level=logging.INFO):  log.log(level, msg, extra={'msg_type': 'send'})