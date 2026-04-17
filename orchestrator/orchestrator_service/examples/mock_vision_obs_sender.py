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


if __name__ == "__main__":
    kind = sys.argv[1] if len(sys.argv) > 1 else "table"
    base = {"ts": time.time(), "session_id": "sess_debug", "epoch": 1}
    if kind == "home":
        payload = {
            **base,
            "type": "home_tag_obs",
            "found": True,
            "yaw_err_rad": 0.05,
            "distance_m": 0.6,
        }
    elif kind == "target":
        payload = {
            **base,
            "type": "target_obs",
            "found": True,
            "target": sys.argv[2] if len(sys.argv) > 2 else "cup",
            "confidence": 0.88,
            "cx_norm": 0.03,
            "size_norm": 0.30,
            "bbox": [100, 80, 180, 200],
        }
    else:
        payload = {
            **base,
            "type": "table_edge_obs",
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
    send(payload)
