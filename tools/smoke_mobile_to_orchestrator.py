#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import json
import socket
import time
from typing import Any, Dict, List


def send_payload(host: str, port: int, payload: Dict[str, Any]) -> None:
    line = json.dumps(payload, ensure_ascii=False, separators=(",", ":")) + "\n"
    with socket.create_connection((host, port), timeout=2.0) as sock:
        sock.sendall(line.encode("utf-8"))
    print(json.dumps(payload, ensure_ascii=False))


def build_sequence(robot_id: str, session_id: str) -> List[Dict[str, Any]]:
    now = time.time()
    return [
        {"type": "mobile_cmd", "robot_id": robot_id, "session_id": session_id, "cmd": "query_status", "ts": now},
        {"type": "mobile_cmd", "robot_id": robot_id, "session_id": session_id, "cmd": "fetch_object", "target": "apple", "text": "拿苹果", "ts": now + 0.1},
        {"type": "mobile_cmd", "robot_id": robot_id, "session_id": session_id, "cmd": "stop", "ts": now + 0.2},
        {"type": "mobile_cmd", "robot_id": robot_id, "session_id": session_id, "cmd": "go_home", "ts": now + 0.3},
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description="Send a small mobile command sequence to the mobile gateway.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=9101)
    parser.add_argument("--robot-id", default="sc171_v2")
    parser.add_argument("--session-id", default="sess_smoke")
    parser.add_argument("--interval", type=float, default=0.35)
    args = parser.parse_args()

    for payload in build_sequence(args.robot_id, args.session_id):
        send_payload(args.host, args.port, payload)
        time.sleep(max(0.0, float(args.interval)))


if __name__ == "__main__":
    main()
