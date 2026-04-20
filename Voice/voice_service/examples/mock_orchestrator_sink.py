#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import socket
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["tcp", "uds"], default="tcp")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=9001)
    ap.add_argument("--uds", default="/tmp/robot_stack/task_cmd.sock")
    args = ap.parse_args()

    if args.mode == "tcp":
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        sock.bind((args.host, args.port))
    else:
        path = Path(args.uds)
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.exists():
            path.unlink()
        sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        sock.bind(str(path))

    sock.listen(1)
    print("mock orchestrator sink listening...")
    conn, _ = sock.accept()
    with conn:
        while True:
            data = conn.recv(4096)
            if not data:
                break
            print(data.decode("utf-8", errors="ignore"), end="")


if __name__ == "__main__":
    main()
