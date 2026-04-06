#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import json
import socket
import time
import uuid


def build_payload(target: str, session_id: str, req_id: str, epoch: int):
    return {
        "ts": time.time(),
        "type": "vision_req",
        "mode": "FIND",
        "target": target,
        "session_id": session_id,
        "req_id": req_id,
        "epoch": int(epoch),
    }


def tcp_loop(host: str, port: int, interval_s: float, payload_builder):
    backoff = 0.5
    while True:
        sock = None
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(3.0)
            sock.connect((host, port))
            sock.settimeout(None)
            print(f"[INFO] connected tcp://{host}:{port}")
            backoff = 0.5
            while True:
                payload = payload_builder()
                sock.sendall((json.dumps(payload, ensure_ascii=False) + "\n").encode("utf-8"))
                print(f"[TX] {payload}")
                time.sleep(interval_s)
        except KeyboardInterrupt:
            raise
        except Exception as e:
            print(f"[WARN] tcp send loop error: {e}; reconnect in {backoff:.1f}s")
            time.sleep(backoff)
            backoff = min(backoff * 2.0, 5.0)
        finally:
            if sock is not None:
                try:
                    sock.close()
                except Exception:
                    pass


def uds_loop(path: str, interval_s: float, payload_builder):
    backoff = 0.5
    while True:
        sock = None
        try:
            sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
            sock.settimeout(3.0)
            sock.connect(path)
            sock.settimeout(None)
            print(f"[INFO] connected uds://{path}")
            backoff = 0.5
            while True:
                payload = payload_builder()
                sock.sendall((json.dumps(payload, ensure_ascii=False) + "\n").encode("utf-8"))
                print(f"[TX] {payload}")
                time.sleep(interval_s)
        except KeyboardInterrupt:
            raise
        except Exception as e:
            print(f"[WARN] uds send loop error: {e}; reconnect in {backoff:.1f}s")
            time.sleep(backoff)
            backoff = min(backoff * 2.0, 5.0)
        finally:
            if sock is not None:
                try:
                    sock.close()
                except Exception:
                    pass


def main():
    ap = argparse.ArgumentParser(description="持续向视觉服务发送固定 FIND 指令，便于脱离语音链路调试视觉")
    ap.add_argument("--transport", choices=["tcp", "uds"], default="tcp")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=9003)
    ap.add_argument("--uds-path", default="/tmp/robot_stack/vision_req.sock")
    ap.add_argument("--target", default="apple")
    ap.add_argument("--interval", type=float, default=0.8)
    ap.add_argument("--session-id", default=f"debug-{uuid.uuid4().hex[:8]}")
    ap.add_argument("--req-prefix", default="find")
    ap.add_argument("--epoch", type=int, default=1)
    args = ap.parse_args()

    counter = {"n": 0}

    def payload_builder():
        counter["n"] += 1
        req_id = f"{args.req_prefix}-{counter['n']:06d}"
        return build_payload(args.target, args.session_id, req_id, args.epoch)

    try:
        if args.transport == "tcp":
            tcp_loop(args.host, args.port, args.interval, payload_builder)
        else:
            uds_loop(args.uds_path, args.interval, payload_builder)
    except KeyboardInterrupt:
        print("\n[INFO] sender stopped by user")


if __name__ == "__main__":
    main()
