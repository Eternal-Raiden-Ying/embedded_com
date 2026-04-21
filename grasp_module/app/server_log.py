import logging
import os
from datetime import datetime

from ..config.global_config import cfgs
from ..config.logging_config import configure_grasp_logger


class ServerLogFormatter(logging.Formatter):
    def format(self, record):
        msg_type = getattr(record, 'msg_type', 'msg')
        level_str = f"{record.levelname:^5}"
        type_str = f"{msg_type:^4}"
        dt_str = datetime.fromtimestamp(record.created).strftime('%Y-%m-%d %H:%M:%S')
        return f"[Server][{level_str}]|{type_str}|{dt_str}| {record.getMessage()}"


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


log = setup_logger(cfgs.log_path)
configure_grasp_logger(level=log.level, handlers=log.handlers, propagate=False)


def log_msg(msg, level=logging.INFO):
    log.log(level, msg, extra={'msg_type': 'msg'})


def log_recv(msg, level=logging.INFO):
    log.log(level, msg, extra={'msg_type': 'recv'})


def log_send(msg, level=logging.INFO):
    log.log(level, msg, extra={'msg_type': 'send'})
