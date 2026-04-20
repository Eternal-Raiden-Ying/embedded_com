#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import json
import socket


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("text")
    ap.add_argument("--mode", choices=["tcp", "uds"], default="tcp")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=9011)
    ap.add_argument("--uds", default="/tmp/robot_stack/tts_event.sock")
    args = ap.parse_args()

    payload = {"type": "tts_event", "text": args.text, "source": "manual_test"}
    line = json.dumps(payload, ensure_ascii=False) + "\n"
    if args.mode == "tcp":
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.connect((args.host, args.port))
    else:
        sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        sock.connect(args.uds)
    with sock:
        sock.sendall(line.encode("utf-8"))


if __name__ == "__main__":
    main()
