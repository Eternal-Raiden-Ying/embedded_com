#!/usr/bin/env python3
# -*- coding: utf-8 -*-

RUNNING = "RUNNING"
WAITING_RESPONSE = "WAITING_RESPONSE"
RESULT_READY = "RESULT_READY"
FAILED = "FAILED"
RELAXING = "RELAXING"


def invalid_search_kind_result(reason: str) -> dict:
    return {"reason": reason}
