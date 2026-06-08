#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import json
import socket
import time
from typing import Any, Dict


def send_jsonl_tcp(host: str, port: int, payload: Dict[str, Any]) -> None:
    data = (json.dumps(payload, ensure_ascii=False) + "\n").encode("utf-8")
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        sock.connect((host, port))
        sock.sendall(data)
    finally:
        sock.close()


def send(transport: str, host: str, port: int, payload: Dict[str, Any]) -> None:
    if transport != "tcp":
        raise ValueError(f"unsupported transport: {transport}")
    send_jsonl_tcp(host, port, payload)


def send_task_cmd(host: str, port: int, intent: str, target: str = "", confidence: float = 0.95) -> None:
    payload = {
        "ts": time.time(),
        "intent": intent,
        "confidence": confidence,
        "session_id": "sess_debug_motion",
        "epoch": 1,
        "source": "debug_script",
    }
    if intent == "FIND":
        payload["target"] = target
    send("tcp", host, port, payload)
    print("[debug_motion_sequence] send task_cmd:", payload)


def send_table_obs(host: str, port: int, yaw: float, dist: float, edge_ready: bool, table_cx: float = 0.0, table_size: float = 0.35) -> None:
    payload = {
        "ts": time.time(),
        "type": "table_edge_obs",
        "session_id": "sess_debug_motion",
        "epoch": 1,
        "table_found": True,
        "edge_found": True,
        "confidence": 0.92,
        "yaw_err_rad": yaw,
        "dist_err_m": dist,
        "lateral_err_m": 0.0,
        "table_cx_norm": table_cx,
        "table_size_norm": table_size,
        "edge_ready": edge_ready,
    }
    send("tcp", host, port, payload)
    print("[debug_motion_sequence] send table_edge_obs:", payload)


def send_target_obs(host: str, port: int, target: str, found: bool, cx_norm: float = 0.0, size_norm: float = 0.0, vy: float = 0.0) -> None:
    payload = {
        "ts": time.time(),
        "type": "target_obs",
        "session_id": "sess_debug_motion",
        "epoch": 1,
        "found": found,
        "target": target if found else None,
    }
    if found:
        payload.update(
            {
                "confidence": 0.92,
                "cx_norm": cx_norm,
                "size_norm": size_norm,
                "bbox": [100, 80, 180, 200],
                "vy_mps": vy,
            }
        )
    send("tcp", host, port, payload)
    print("[debug_motion_sequence] send target_obs:", payload)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="通过真实 IPC 端口模拟找桌、停靠和沿边搜目标流程")
    p.add_argument("--task-host", default="127.0.0.1")
    p.add_argument("--task-port", type=int, default=9001)
    p.add_argument("--vision-host", default="127.0.0.1")
    p.add_argument("--vision-port", type=int, default=9002)
    p.add_argument("--target", default="bottle")
    p.add_argument("--period", type=float, default=0.35)
    p.add_argument("--send-stop", action="store_true")
    return p


def main() -> None:
    args = build_parser().parse_args()

    print("[debug_motion_sequence] step 1/6: 发送 FIND")
    send_task_cmd(args.task_host, args.task_port, "FIND", target=args.target)
    time.sleep(args.period)

    print("[debug_motion_sequence] step 2/6: 模拟搜索到桌边")
    for _ in range(2):
        send_table_obs(args.vision_host, args.vision_port, yaw=0.18, dist=0.35, edge_ready=False, table_cx=0.18, table_size=0.20)
        time.sleep(args.period)

    print("[debug_motion_sequence] step 3/6: 粗对齐完成，开始接近")
    for _ in range(3):
        send_table_obs(args.vision_host, args.vision_port, yaw=0.04, dist=0.16, edge_ready=False, table_cx=0.04, table_size=0.40)
        time.sleep(args.period)

    print("[debug_motion_sequence] step 4/6: 锁边完成，进入桌边工作带")
    for _ in range(3):
        send_table_obs(args.vision_host, args.vision_port, yaw=0.01, dist=0.02, edge_ready=True, table_cx=0.0, table_size=0.55)
        time.sleep(args.period)

    print("[debug_motion_sequence] step 5/6: 沿边滑动搜索")
    for vy in (0.14, 0.14, -0.14, -0.14):
        send_target_obs(args.vision_host, args.vision_port, args.target, found=False, vy=vy)
        time.sleep(args.period)

    print("[debug_motion_sequence] step 6/6: 发现并锁定目标")
    for _ in range(3):
        send_target_obs(args.vision_host, args.vision_port, args.target, found=True, cx_norm=0.02, size_norm=0.14)
        time.sleep(args.period)

    if args.send_stop:
        print("[debug_motion_sequence] 追加 STOP")
        send_task_cmd(args.task_host, args.task_port, "STOP", confidence=0.98)

    print("[debug_motion_sequence] done")


if __name__ == "__main__":
    main()
