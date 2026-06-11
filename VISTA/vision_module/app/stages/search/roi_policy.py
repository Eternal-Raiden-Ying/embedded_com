#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from typing import Any, Dict, Optional


def target_roi(payload: Optional[Dict[str, Any]], results: Optional[Dict[str, Any]] = None):
    payload = payload or {}
    results = results or {}
    return payload.get("target_roi") or results.get("target_roi")


def table_roi(payload: Optional[Dict[str, Any]], results: Optional[Dict[str, Any]] = None):
    payload = payload or {}
    results = results or {}
    return payload.get("table_roi") or results.get("table_roi")


def history_roi(stage_state: Optional[Dict[str, Any]]):
    stage_state = stage_state or {}
    return stage_state.get("history_roi") or stage_state.get("locked_roi")


def lost_table_search_roi(stage_state: Optional[Dict[str, Any]], fallback=None):
    stage_state = stage_state or {}
    return stage_state.get("lost_table_search_roi") or history_roi(stage_state) or fallback
