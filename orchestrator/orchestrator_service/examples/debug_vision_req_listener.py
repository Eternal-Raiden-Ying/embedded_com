#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""监听 orchestrator 发给视觉模块的请求消息（vision_req / home_tag_req）。
用途：
1. 视觉同学可先不接真实摄像头，只验证自己有没有收到 target / mode。
2. 你自己可以独立验证 orchestrator 是否真的把当前 target 发出去了。
"""

import argparse
import json
import os
import socket


def run_tcp(host: str, port: int):
    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server.bind((host, port))
    server.listen(1)
    print(f"[debug_vision_req_listener] listen tcp on {host}:{port}")
    conn, addr = server.accept()
    print(f"[debug_vision_req_listener] orchestrator connected from {addr}")
    with conn:
        buf = ""
        while True:
            data = conn.recv(4096)
            if not data:
                break
            buf += data.decode("utf-8", errors="ignore")
            while "\n" in buf:
                line, buf = buf.split("\n", 1)
                line = line.strip()
                if not line:
                    continue
                try:
                    print("[debug_vision_req_listener]", json.loads(line))
                except Exception:
                    print("[debug_vision_req_listener] raw:", line)


def run_uds(uds_path: str):
    if os.path.exists(uds_path):
        os.unlink(uds_path)
    server = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    server.bind(uds_path)
    server.listen(1)
    print(f"[debug_vision_req_listener] listen uds on {uds_path}")
    conn, _ = server.accept()
    print("[debug_vision_req_listener] orchestrator connected")
    with conn:
        buf = ""
        while True:
            data = conn.recv(4096)
            if not data:
                break
            buf += data.decode("utf-8", errors="ignore")
            while "\n" in buf:
                line, buf = buf.split("\n", 1)
                line = line.strip()
                if not line:
                    continue
                try:
                    print("[debug_vision_req_listener]", json.loads(line))
                except Exception:
                    print("[debug_vision_req_listener] raw:", line)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="监听 orchestrator 发给视觉的请求")
    ap.add_argument("--transport", default="tcp", choices=["tcp", "uds"])
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=9003)
    ap.add_argument("--uds-path", default="/tmp/robot_stack/vision_req.sock")
    args = ap.parse_args()
    if args.transport == "uds":
        run_uds(args.uds_path)
    else:
        run_tcp(args.host, args.port)
