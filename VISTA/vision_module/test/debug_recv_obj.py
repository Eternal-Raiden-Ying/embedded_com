#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import json
import socket

try:
    from .debug_protocol_tools import remember_interaction, summarize_obs
except ImportError:
    from debug_protocol_tools import remember_interaction, summarize_obs

# 请修改为 App 尝试连接的端口 (obs_out 配置)
HOST = "127.0.0.1"
PORT = 9002


def print_payload(data: dict):
    msg_type = str(data.get("type", "unknown")).strip().upper()
    print(f"[RX {msg_type}] {summarize_obs(data)}")
    print(json.dumps(data, ensure_ascii=False, indent=2))
    state = remember_interaction(data)
    if state is not None:
        print(
            "[HINT] 已记录 interaction_id="
            f"{state['last_interaction_id']}，可在 debug_send_req.py 里直接发送 ACCEPT/REJECT"
        )


def main():
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as server:
        server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        server.bind((HOST, PORT))
        server.listen(1)
        print(f"[LISTEN] {HOST}:{PORT} 等待 VISTA 发送数据...")

        while True:
            try:
                conn, addr = server.accept()
                print(f"[CONNECTED] 来自 {addr}")

                with conn:
                    file_obj = conn.makefile("r", encoding="utf-8")
                    while True:
                        line = file_obj.readline()
                        if not line:
                            print("[DISCONNECTED] VISTA 断开连接")
                            break

                        line = line.strip()
                        if line:
                            try:
                                data = json.loads(line)
                                print_payload(data)
                            except json.JSONDecodeError:
                                print(f"[WARN] 无法解析: {line}")
            except KeyboardInterrupt:
                print("\n[STOP] 退出接收端")
                break
            except Exception as exc:
                print(f"[WARN] 发生错误: {exc}")

if __name__ == "__main__":
    main()
