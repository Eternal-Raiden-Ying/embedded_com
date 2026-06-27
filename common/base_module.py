#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import logging
from abc import ABC, abstractmethod
from typing import Any, Dict, Literal, Optional

from .runtime_logging import configure_stream_logger


class BaseModule(ABC):
    LOG_ENABLED: bool = True
    _log: Optional[logging.Logger] = None

    def __init__(self, module_name: str, log_enabled: bool = True, log_mode: str = "concise"):
        self._module_name = module_name
        self._log: Optional[logging.Logger] = None
        self._init_logger(log_enabled, log_mode)

    def _init_logger(self, log_enabled: bool = True, log_mode: str = "concise"):
        if not log_enabled:
            self._log = None
            return
        self._log = configure_stream_logger(self._module_name, mode=log_mode, enabled=log_enabled)

    def child_logger(self, suffix: str) -> logging.Logger:
        suffix = str(suffix or "").strip()
        return logging.getLogger(f"{self._module_name}.{suffix}" if suffix else self._module_name)

    def _logger_name(self, src: str) -> str:
        src = str(src or "").strip()
        if not src or src == self._module_name:
            return self._module_name
        return f"{self._module_name}.{src.replace(':', '.').replace(' ', '_')}"

    def _format_msg(self, msg: str, data: Optional[Dict[str, Any]] = None) -> str:
        if data:
            return f"{msg} | {data}"
        return msg

    def _levelno(self, level: str) -> int:
        level = str(level or "info").strip().lower()
        if level == "warn":
            return logging.WARNING
        return getattr(logging, level.upper(), logging.INFO)

    def log(self, level: str, src: str, msg: str, data: Optional[Dict[str, Any]] = None):
        if not self._log or not self.LOG_ENABLED:
            return
        logger = logging.getLogger(self._logger_name(src))
        logger.log(self._levelno(level), self._format_msg(msg, data))

    def log_debug(self, src: str, msg: str, data: Optional[Dict[str, Any]] = None):
        self.log("debug", src, msg, data)

    def log_info(self, src: str, msg: str, data: Optional[Dict[str, Any]] = None):
        self.log("info", src, msg, data)

    def log_warn(self, src: str, msg: str, data: Optional[Dict[str, Any]] = None):
        self.log("warn", src, msg, data)

    def log_error(self, src: str, msg: str, data: Optional[Dict[str, Any]] = None):
        self.log("error", src, msg, data)

    def debug(self, msg: str, *args, **kwargs):
        self.log_debug(self._module_name, msg, kwargs or None)

    def info(self, msg: str, *args, **kwargs):
        self.log_info(self._module_name, msg, kwargs or None)

    def warning(self, msg: str, *args, **kwargs):
        self.log_warn(self._module_name, msg, kwargs or None)

    def warn(self, msg: str, *args, **kwargs):
        self.warning(msg, *args, **kwargs)

    def error(self, msg: str, *args, **kwargs):
        self.log_error(self._module_name, msg, kwargs or None)

    def log_ipc(
        self,
        direction: Literal["TX", "RX"],
        target: str,
        message: str,
        data: Optional[Dict[str, Any]] = None
    ):
        if not self._log or not self.LOG_ENABLED:
            return
        src = f"ipc:{direction}"
        msg = f"{target} {message}"
        self.log("info", src, msg, data)

    @abstractmethod
    def start(self):
        pass

    @abstractmethod
    def stop(self):
        pass
