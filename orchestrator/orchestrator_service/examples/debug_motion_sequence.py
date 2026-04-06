#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
debug_motion_sequence.py

用途：
1. 模拟 ASR 发给 orchestrator 的 task_cmd
2. 模拟视觉发给 orchestrator 的 target_obs
3. 不接真实语音、不接真实视觉，只验证：
   task_cmd + target_obs -> 状态机 -> cmd_vel -> 小车串口命令

推荐用途：
- 测小车动作是否符合“搜索 -> 对准 -> 前进 -> 到达停车”
- 测状态机在 target_obs 变化时是否会正确切换状态
"""

import argparse
import json
import socket
import time
from typing import Dict, Any


def send_jsonl_tcp(host: str, port: int, payload: Dict[str, Any]) -> None:
    data = (json.dumps(payload, ensure_ascii=False) + "\n").encode("utf-8")
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        sock.connect((host, port))
        sock.sendall(data)
    finally:
        sock.close()


def send_jsonl_uds(uds_path: str, payload: Dict[str, Any]) -> None:
    data = (json.dumps(payload, ensure_ascii=False) + "\n").encode("utf-8")
    sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    try:
        sock.connect(uds_path)
        sock.sendall(data)
    finally:
        sock.close()


def send(
    transport: str,
    host: str,
    port: int,
    uds_path: str,
    payload: Dict[str, Any],
) -> None:
    if transport == "tcp":
        send_jsonl_tcp(host, port, payload)
    elif transport == "uds":
        send_jsonl_uds(uds_path, payload)
    else:
        raise ValueError(f"unsupported transport: {transport}")


def send_task_cmd(
    task_transport: str,
    task_host: str,
    task_port: int,
    task_uds: str,
    intent: str,
    target: str = "",
    confidence: float = 0.95,
) -> None:
    payload = {
        "ts": time.time(),
        "intent": intent,
        "confidence": confidence,
    }
    if intent == "FIND":
        payload["target"] = target

    send(task_transport, task_host, task_port, task_uds, payload)
    print("[debug_motion_sequence] send task_cmd:", payload)


def send_target_obs(
    vision_transport: str,
    vision_host: str,
    vision_port: int,
    vision_uds: str,
    target: str,
    found: bool,
    cx_norm: float = 0.0,
    size_norm: float = 0.0,
    confidence: float = 0.90,
) -> None:
    payload = {
        "ts": time.time(),
        "type": "target_obs",
        "found": found,
        "target": target,
    }
    if found:
        payload.update(
            {
                "confidence": confidence,
                "cx_norm": cx_norm,
                "size_norm": size_norm,
                "bbox": [100, 80, 180, 200],
            }
        )

    send(vision_transport, vision_host, vision_port, vision_uds, payload)
    print("[debug_motion_sequence] send target_obs:", payload)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="模拟 task_cmd + target_obs，验证状态机驱动小车动作"
    )

    p.add_argument("--task-transport", default="tcp", choices=["tcp", "uds"])
    p.add_argument("--task-host", default="127.0.0.1")
    p.add_argument("--task-port", type=int, default=9001)
    p.add_argument("--task-uds", default="/tmp/robot_stack/task_cmd.sock")

    p.add_argument("--vision-transport", default="tcp", choices=["tcp", "uds"])
    p.add_argument("--vision-host", default="127.0.0.1")
    p.add_argument("--vision-port", type=int, default=9002)
    p.add_argument("--vision-uds", default="/tmp/robot_stack/vision_obs.sock")

    p.add_argument("--target", default="cup")
    p.add_argument("--task-confidence", type=float, default=0.95)

    p.add_argument(
        "--startup-wait",
        type=float,
        default=0.8,
        help="发送 FIND 之后，等待状态机进入 SEARCH 的时间",
    )
    p.add_argument(
        "--period",
        type=float,
        default=0.35,
        help="两次 target_obs 之间的间隔",
    )
    p.add_argument(
        "--send-stop",
        action="store_true",
        help="到达后补发一条 STOP",
    )
    return p


def main() -> None:
    args = build_parser().parse_args()

    print("[debug_motion_sequence] step 1/5: 发送 FIND task_cmd")
    send_task_cmd(
        task_transport=args.task_transport,
        task_host=args.task_host,
        task_port=args.task_port,
        task_uds=args.task_uds,
        intent="FIND",
        target=args.target,
        confidence=args.task_confidence,
    )

    time.sleep(args.startup_wait)

    print("[debug_motion_sequence] step 2/5: 连续几帧未找到，观察 SEARCH 原地转")
    for _ in range(3):
        send_target_obs(
            vision_transport=args.vision_transport,
            vision_host=args.vision_host,
            vision_port=args.vision_port,
            vision_uds=args.vision_uds,
            target=args.target,
            found=False,
        )
        time.sleep(args.period)

    print("[debug_motion_sequence] step 3/5: 目标偏左，观察对准转向")
    for _ in range(3):
        send_target_obs(
            vision_transport=args.vision_transport,
            vision_host=args.vision_host,
            vision_port=args.vision_port,
            vision_uds=args.vision_uds,
            target=args.target,
            found=True,
            cx_norm=0.45,
            size_norm=0.10,
            confidence=0.92,
        )
        time.sleep(args.period)

    print("[debug_motion_sequence] step 4/5: 基本对准且距离较远，观察前进")
    for _ in range(4):
        send_target_obs(
            vision_transport=args.vision_transport,
            vision_host=args.vision_host,
            vision_port=args.vision_port,
            vision_uds=args.vision_uds,
            target=args.target,
            found=True,
            cx_norm=0.04,
            size_norm=0.22,
            confidence=0.94,
        )
        time.sleep(args.period)

    print("[debug_motion_sequence] step 5/5: 目标很近，观察 ARRIVED + 停车")
    for _ in range(3):
        send_target_obs(
            vision_transport=args.vision_transport,
            vision_host=args.vision_host,
            vision_port=args.vision_port,
            vision_uds=args.vision_uds,
            target=args.target,
            found=True,
            cx_norm=0.01,
            size_norm=0.52,
            confidence=0.96,
        )
        time.sleep(args.period)

    if args.send_stop:
        print("[debug_motion_sequence] 补发 STOP")
        send_task_cmd(
            task_transport=args.task_transport,
            task_host=args.task_host,
            task_port=args.task_port,
            task_uds=args.task_uds,
            intent="STOP",
            confidence=0.98,
        )

    print("[debug_motion_sequence] done")


if __name__ == "__main__":
    main()