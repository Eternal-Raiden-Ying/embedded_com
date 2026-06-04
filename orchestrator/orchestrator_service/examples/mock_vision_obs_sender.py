#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
import socket
import sys
import time

HOST = "127.0.0.1"
PORT = 9002


def send(payload):
    line = json.dumps(payload, ensure_ascii=False, separators=(",", ":")) + "\n"
    with socket.create_connection((HOST, PORT), timeout=2.0) as sock:
        sock.sendall(line.encode("utf-8"))
    print("sent", payload)


def wrap_vision_obs(kind: str, base: dict, perception: dict, *, mode: str = "FIND_OBJECT", status: str = "RUNNING") -> dict:
    stage = "RETURN" if kind == "home_tag_obs" else "SEARCH"
    return {
        **base,
        "type": "vision_obs",
        "stage": stage,
        "mode": mode,
        "status": status,
        "perception": {kind: perception},
    }


if __name__ == "__main__":
    kind = sys.argv[1] if len(sys.argv) > 1 else "table"
    extra_args = [str(arg).strip() for arg in sys.argv[2:]]
    legacy = any(arg.lower() == "legacy" for arg in extra_args)
    value_args = [arg for arg in extra_args if arg.lower() != "legacy"]
    base = {"ts": time.time(), "session_id": "sess_debug", "epoch": 1}
    if kind == "home":
        obs = {
            "found": True,
            "yaw_err_rad": 0.05,
            "distance_m": 0.6,
        }
        payload = {**base, "type": "home_tag_obs", **obs} if legacy else wrap_vision_obs("home_tag_obs", base, obs)
    elif kind == "target":
        obs = {
            "found": True,
            "target": value_args[0] if value_args else "bottle",
            "confidence": 0.88,
            "cx_norm": 0.03,
            "size_norm": 0.30,
            "bbox": [100, 80, 180, 200],
        }
        payload = {**base, "type": "target_obs", **obs} if legacy else wrap_vision_obs("target_obs", base, obs)
    else:
        obs = {
            "table_found": True,
            "edge_found": True,
            "confidence": 0.92,
            "yaw_err_rad": 0.06,
            "dist_err_m": 0.18,
            "lateral_err_m": 0.02,
            "table_cx_norm": 0.05,
            "table_size_norm": 0.42,
            "edge_ready": False,
        }
        payload = {**base, "type": "table_edge_obs", **obs} if legacy else wrap_vision_obs(
            "table_edge_obs",
            base,
            obs,
            mode="FIND_EDGE",
        )
    send(payload)
