#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json, socket, sys, time
HOST = "127.0.0.1"
PORT = 9002
if __name__ == "__main__":
    kind = sys.argv[1] if len(sys.argv) > 1 else "target"
    if kind == "home":
        payload = {"ts": time.time(), "type": "home_tag_obs", "found": True, "yaw_err_rad": 0.05, "distance_m": 0.6}
    else:
        payload = {"ts": time.time(), "type": "target_obs", "found": True, "target": sys.argv[2] if len(sys.argv) > 2 else "cup", "confidence": 0.88, "cx_norm": 0.03, "size_norm": 0.30, "bbox": [100,80,180,200]}
    line = json.dumps(payload, ensure_ascii=False, separators=(",", ":")) + "\n"
    with socket.create_connection((HOST, PORT), timeout=2.0) as sock:
        sock.sendall(line.encode("utf-8"))
    print("sent", payload)
