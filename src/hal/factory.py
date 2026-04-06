#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
HAL 工厂模块
通过环境变量 ENV 决定加载哪套硬件实现。

  ENV=mock  -> 加载 mock 实现（Windows 本地开发，无硬件依赖）
  ENV=prod  -> 加载 aidlux 实现（AidLux 硬件，默认）
"""
import os


def get_env() -> str:
    return os.environ.get("ENV", "prod").lower()


def is_mock() -> bool:
    """返回 True 表示当前运行在 mock 模式（Windows 本地开发）"""
    return get_env() == "mock"
