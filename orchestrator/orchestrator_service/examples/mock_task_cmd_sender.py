#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json, socket, sys, time
HOST = "127.0.0.1"
PORT = 9001
if __name__ == "__main__":
    payload = {"ts": time.time(), "intent": sys.argv[1] if len(sys.argv) > 1 else "FIND", "confidence": 0.9}
    if payload["intent"].upper() == "FIND":
        payload["target"] = sys.argv[2] if len(sys.argv) > 2 else "bottle"
    line = json.dumps(payload, ensure_ascii=False, separators=(",", ":")) + "\n"
    with socket.create_connection((HOST, PORT), timeout=2.0) as sock:
        sock.sendall(line.encode("utf-8"))
    print("sent", payload)
