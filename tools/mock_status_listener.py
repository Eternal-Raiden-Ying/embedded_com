#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import json
import socket
import threading


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Listen for mobile gateway status JSONL messages.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=9102)
    return parser


def handle_client(conn: socket.socket, addr) -> None:
    with conn:
        fp = conn.makefile("r", encoding="utf-8", newline="\n")
        for line in fp:
            line = line.strip()
            if not line:
                continue
            try:
                payload = json.loads(line)
            except Exception:
                print(f"[status] {addr} raw={line}")
                continue
            print(json.dumps(payload, ensure_ascii=False, indent=2))


def main() -> None:
    args = build_parser().parse_args()
    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server.bind((args.host, args.port))
    server.listen(8)
    print(f"[mock_status_listener] listening on {args.host}:{args.port}")
    try:
        while True:
            conn, addr = server.accept()
            th = threading.Thread(target=handle_client, args=(conn, addr), daemon=True)
            th.start()
    finally:
        server.close()


if __name__ == "__main__":
    main()
