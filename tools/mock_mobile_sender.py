#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import json
import socket
import time
from typing import Any, Dict


def send_payload(host: str, port: int, payload: Dict[str, Any]) -> None:
    line = json.dumps(payload, ensure_ascii=False, separators=(",", ":")) + "\n"
    with socket.create_connection((host, port), timeout=2.0) as sock:
        sock.sendall(line.encode("utf-8"))
    print(json.dumps(payload, ensure_ascii=False))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Send mock mobile commands to the board-side mobile gateway.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=9101)
    parser.add_argument("--robot-id", default="sc171_v2")
    parser.add_argument("--session-id", default="")
    parser.add_argument("--epoch", type=int, default=0)
    parser.add_argument("--text", default="")
    parser.add_argument("cmd", choices=["fetch_object", "stop", "resume", "retry_search", "go_home", "query_status"])
    parser.add_argument("target", nargs="?", default="")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    payload: Dict[str, Any] = {
        "type": "mobile_cmd",
        "robot_id": args.robot_id,
        "session_id": args.session_id or None,
        "cmd": args.cmd,
        "text": args.text or None,
        "ts": time.time(),
        "epoch": args.epoch,
        "source": "mock_mobile_sender",
    }
    if args.cmd == "fetch_object":
        payload["target"] = args.target or "apple"
        if not payload.get("text"):
            payload["text"] = f"拿 {payload['target']}"
    send_payload(args.host, args.port, {k: v for k, v in payload.items() if v not in (None, "")})


if __name__ == "__main__":
    main()

