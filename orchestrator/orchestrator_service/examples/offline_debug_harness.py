#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import json
import socket
import threading
import time
from typing import Dict


DEFAULT_PORTS = {
    "task_cmd_in": 9001,
    "vision_obs_in": 9002,
    "vision_req_out": 9003,
    "tts_event_out": 9011,
    "task_ack_out": 9012,
}


def send_payload(host: str, port: int, payload: Dict) -> None:
    line = json.dumps(payload, ensure_ascii=False, separators=(",", ":")) + "\n"
    with socket.create_connection((host, port), timeout=2.0) as sock:
        sock.sendall(line.encode("utf-8"))
    print(json.dumps(payload, ensure_ascii=False))


def wrap_vision_obs(base: Dict, kind: str, perception: Dict, *, mode: str, status: str = "RUNNING") -> Dict:
    stage = "RETURN" if kind == "home_tag_obs" else "SEARCH"
    payload = {
        **base,
        "type": "vision_obs",
        "stage": stage,
        "mode": mode,
        "status": status,
        "perception": {kind: dict(perception or {})},
    }
    return payload


def listen_tcp(host: str, port: int) -> None:
    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server.bind((host, port))
    server.listen(4)
    print(f"[listener] listening on {host}:{port}")

    def handle_client(conn: socket.socket, addr):
        with conn:
            fp = conn.makefile("r", encoding="utf-8", newline="\n")
            for line in fp:
                line = line.strip()
                if line:
                    print(f"[listener] {addr}: {line}")

    try:
        while True:
            conn, addr = server.accept()
            th = threading.Thread(target=handle_client, args=(conn, addr), daemon=True)
            th.start()
    finally:
        server.close()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Orchestrator 离线调试工具，复用真实 TCP JSONL 链路")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("listen", help="监听 orchestrator 输出通道")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int)
    p.add_argument("--channel", default="vision_req_out", choices=sorted(DEFAULT_PORTS))

    p = sub.add_parser("send-task", help="向 task_cmd_in 注入任务")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=DEFAULT_PORTS["task_cmd_in"])
    p.add_argument("--intent", default="FIND", choices=["FIND", "RETURN", "STOP"])
    p.add_argument("--target", default="apple")
    p.add_argument("--confidence", type=float, default=0.95)

    p = sub.add_parser("send-table", help="向 vision_obs_in 注入 table_edge_obs")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=DEFAULT_PORTS["vision_obs_in"])
    p.add_argument("--table-found", type=int, default=1)
    p.add_argument("--edge-found", type=int, default=1)
    p.add_argument("--yaw", type=float, default=0.05)
    p.add_argument("--dist", type=float, default=0.15)
    p.add_argument("--lat", type=float, default=0.0)
    p.add_argument("--edge-ready", type=int, default=0)
    p.add_argument("--table-cx", type=float, default=0.05)
    p.add_argument("--table-size", type=float, default=0.35)
    p.add_argument("--legacy", action="store_true", help="发送旧式裸 table_edge_obs，而不是新 vision_obs envelope")

    p = sub.add_parser("send-target", help="向 vision_obs_in 注入 target_obs")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=DEFAULT_PORTS["vision_obs_in"])
    p.add_argument("--target", default="apple")
    p.add_argument("--found", type=int, default=1)
    p.add_argument("--cx", type=float, default=0.0)
    p.add_argument("--size", type=float, default=0.12)
    p.add_argument("--vx", type=float)
    p.add_argument("--vy", type=float)
    p.add_argument("--wz", type=float)
    p.add_argument("--legacy", action="store_true", help="发送旧式裸 target_obs，而不是新 vision_obs envelope")

    p = sub.add_parser("scenario", help="发送一段完整的找桌-停靠-找目标序列")
    p.add_argument("--task-host", default="127.0.0.1")
    p.add_argument("--task-port", type=int, default=DEFAULT_PORTS["task_cmd_in"])
    p.add_argument("--vision-host", default="127.0.0.1")
    p.add_argument("--vision-port", type=int, default=DEFAULT_PORTS["vision_obs_in"])
    p.add_argument("--target", default="apple")
    p.add_argument("--period", type=float, default=0.35)
    p.add_argument("--legacy", action="store_true", help="场景中使用旧式裸感知消息")

    return parser


def main() -> None:
    args = build_parser().parse_args()

    if args.cmd == "listen":
        port = args.port if args.port is not None else DEFAULT_PORTS[args.channel]
        listen_tcp(args.host, port)
        return

    if args.cmd == "send-task":
        payload = {
            "ts": time.time(),
            "type": "task_cmd",
            "intent": args.intent,
            "confidence": args.confidence,
            "target": args.target if args.intent == "FIND" else None,
            "session_id": "sess_harness",
            "epoch": 1,
            "source": "offline_harness",
        }
        send_payload(args.host, args.port, payload)
        return

    if args.cmd == "send-table":
        base = {
            "ts": time.time(),
            "session_id": "sess_harness",
            "epoch": 1,
            "source": "offline_harness",
        }
        obs = {
            "table_found": bool(args.table_found),
            "edge_found": bool(args.edge_found),
            "confidence": 0.95,
            "yaw_err_rad": args.yaw,
            "dist_err_m": args.dist,
            "lateral_err_m": args.lat,
            "edge_ready": bool(args.edge_ready),
            "table_cx_norm": args.table_cx,
            "table_size_norm": args.table_size,
        }
        payload = {**base, "type": "table_edge_obs", **obs} if args.legacy else wrap_vision_obs(
            base,
            "table_edge_obs",
            obs,
            mode="FIND_EDGE",
        )
        send_payload(args.host, args.port, payload)
        return

    if args.cmd == "send-target":
        base = {
            "ts": time.time(),
            "session_id": "sess_harness",
            "epoch": 1,
            "source": "offline_harness",
        }
        obs = {
            "found": bool(args.found),
            "target": args.target if args.found else None,
            "confidence": 0.92 if args.found else None,
            "cx_norm": args.cx,
            "size_norm": args.size,
        }
        if args.vx is not None:
            obs["vx_mps"] = args.vx
        if args.vy is not None:
            obs["vy_mps"] = args.vy
        if args.wz is not None:
            obs["wz_radps"] = args.wz
        payload = {**base, "type": "target_obs", **obs} if args.legacy else wrap_vision_obs(
            base,
            "target_obs",
            obs,
            mode="FIND_OBJECT",
        )
        send_payload(args.host, args.port, payload)
        return

    if args.cmd == "scenario":
        send_payload(args.task_host, args.task_port, {
            "ts": time.time(),
            "type": "task_cmd",
            "intent": "FIND",
            "target": args.target,
            "confidence": 0.98,
            "session_id": "sess_harness",
            "epoch": 1,
            "source": "offline_harness",
        })
        time.sleep(args.period)
        for _ in range(2):
            base = {
                "ts": time.time(),
                "session_id": "sess_harness",
                "epoch": 1,
            }
            obs = {
                "table_found": True,
                "edge_found": True,
                "confidence": 0.94,
                "yaw_err_rad": 0.18,
                "dist_err_m": 0.30,
                "table_cx_norm": 0.18,
                "table_size_norm": 0.22,
            }
            payload = {**base, "type": "table_edge_obs", **obs} if args.legacy else wrap_vision_obs(
                base,
                "table_edge_obs",
                obs,
                mode="FIND_EDGE",
            )
            send_payload(args.vision_host, args.vision_port, payload)
            time.sleep(args.period)
        for _ in range(3):
            base = {
                "ts": time.time(),
                "session_id": "sess_harness",
                "epoch": 1,
            }
            obs = {
                "table_found": True,
                "edge_found": True,
                "confidence": 0.96,
                "yaw_err_rad": 0.02,
                "dist_err_m": 0.03,
                "edge_ready": True,
                "table_cx_norm": 0.01,
                "table_size_norm": 0.56,
            }
            payload = {**base, "type": "table_edge_obs", **obs} if args.legacy else wrap_vision_obs(
                base,
                "table_edge_obs",
                obs,
                mode="FIND_EDGE",
            )
            send_payload(args.vision_host, args.vision_port, payload)
            time.sleep(args.period)
        for vy in (0.14, 0.14, -0.14, -0.14):
            base = {
                "ts": time.time(),
                "session_id": "sess_harness",
                "epoch": 1,
            }
            obs = {
                "found": False,
                "vy_mps": vy,
            }
            payload = {**base, "type": "target_obs", **obs} if args.legacy else wrap_vision_obs(
                base,
                "target_obs",
                obs,
                mode="FIND_OBJECT",
            )
            send_payload(args.vision_host, args.vision_port, payload)
            time.sleep(args.period)
        for _ in range(3):
            base = {
                "ts": time.time(),
                "session_id": "sess_harness",
                "epoch": 1,
            }
            obs = {
                "found": True,
                "target": args.target,
                "confidence": 0.95,
                "cx_norm": 0.01,
                "size_norm": 0.16,
            }
            payload = {**base, "type": "target_obs", **obs} if args.legacy else wrap_vision_obs(
                base,
                "target_obs",
                obs,
                mode="FIND_OBJECT",
            )
            send_payload(args.vision_host, args.vision_port, payload)
            time.sleep(args.period)


if __name__ == "__main__":
    main()
