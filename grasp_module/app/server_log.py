import logging
import os
import json
from datetime import datetime

from ..config.global_config import cfgs
from ..config.logging_config import configure_grasp_logger

# 服务启动时间戳，用于日志文件命名
_startup_timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

# 请求记录 & 预测图片目录
LOG_DIR = os.path.dirname(os.path.abspath(cfgs.log_path))
PREDICT_IMG_DIR = os.path.join(LOG_DIR, 'predict_images')
REQUEST_LOG_PATH = os.path.join(LOG_DIR, 'requests.jsonl')


class ServerLogFormatter(logging.Formatter):
    def format(self, record):
        msg_type = getattr(record, 'msg_type', 'msg')
        level_str = f"{record.levelname:^5}"
        type_str = f"{msg_type:^4}"
        dt_str = datetime.fromtimestamp(record.created).strftime('%Y-%m-%d %H:%M:%S')
        return f"[Server][{level_str}]|{type_str}|{dt_str}| {record.getMessage()}"


def _build_timestamped_log_path(original_path: str) -> str:
    """在日志文件名中插入启动时间戳，例如 server.log → server_20260630_143025.log"""
    dirname = os.path.dirname(original_path)
    basename = os.path.basename(original_path)
    name, ext = os.path.splitext(basename)
    return os.path.join(dirname, f'{name}_{_startup_timestamp}{ext}')


def setup_logger(log_file: str):
    logger = logging.getLogger("AppLogger")
    logger.setLevel(logging.INFO)
    logger.propagate = False

    if logger.handlers:
        logger.handlers.clear()

    formatter = ServerLogFormatter()
    os.makedirs(os.path.dirname(os.path.abspath(log_file)), exist_ok=True)

    file_handler = logging.FileHandler(log_file, encoding='utf-8')
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    return logger


# 使用带启动时间戳的日志文件路径
_timestamped_log = _build_timestamped_log_path(cfgs.log_path)
log = setup_logger(_timestamped_log)
configure_grasp_logger(level=log.level, handlers=log.handlers, propagate=False)


def log_msg(msg, level=logging.INFO):
    log.log(level, msg, extra={'msg_type': 'msg'})


def log_recv(msg, level=logging.INFO):
    log.log(level, msg, extra={'msg_type': 'recv'})


def log_send(msg, level=logging.INFO):
    log.log(level, msg, extra={'msg_type': 'send'})


# ==========================================
# 请求 JSON 记录器
# ==========================================
class RequestLogger:
    """将请求元数据追加写入 JSONL 文件。"""

    def __init__(self, path: str):
        self.path = path
        os.makedirs(os.path.dirname(path), exist_ok=True)

    def log(self, entry: dict):
        entry['timestamp'] = datetime.now().isoformat()
        with open(self.path, 'a', encoding='utf-8') as f:
            f.write(json.dumps(entry, ensure_ascii=False) + '\n')


request_logger = RequestLogger(REQUEST_LOG_PATH)


# ==========================================
# 预测图片保存（仅保留最新，每次覆盖）
# ==========================================
def save_predict_images(rgb_bytes: bytes, depth_bytes: bytes):
    """保存最新的预测请求原始图片，每次覆盖。"""
    os.makedirs(PREDICT_IMG_DIR, exist_ok=True)
    rgb_path = os.path.join(PREDICT_IMG_DIR, 'latest_rgb.jpg')
    depth_path = os.path.join(PREDICT_IMG_DIR, 'latest_depth.png')
    with open(rgb_path, 'wb') as f:
        f.write(rgb_bytes)
    with open(depth_path, 'wb') as f:
        f.write(depth_bytes)
