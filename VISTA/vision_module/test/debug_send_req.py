#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
模拟业务端发送指令给 Vision App
"""
import socket
import json
import time

# 请修改为 App 监听的端口 (req_in 配置)
HOST = "127.0.0.1"
PORT = 9003 

def send_payload(payload: dict):
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.settimeout(2.0)
            s.connect((HOST, PORT))
            line = json.dumps(payload) + "\n"
            s.sendall(line.encode("utf-8"))
            print(f"[✅] 已发送 -> {line.strip()}")
    except ConnectionRefusedError:
        print(f"[❌] 无法连接到 {HOST}:{PORT}，请确认 App 是否已启动。")
    except Exception as e:
        print(f"[❌] 发送失败: {e}")

if __name__ == "__main__":
    print("VISTA Vision 调试发送器")
    print("1: 发送 FIND (寻找 bottle)")
    print("2: 发送 FIND (寻找 mouse)")
    print("3: 发送 STOP (进入 IDLE)")
    print("4: 发送 RETURN (返航模式)")
    
    while True:
        choice = input("\n请选择指令 (1-4, q 退出): ").strip()
        if choice == '1':
            send_payload({"type": "vision_req", "ts": time.time(), "mode": "FIND", "target": "bottle"})
        elif choice == '2':
            send_payload({"type": "vision_req", "ts": time.time(), "mode": "FIND", "target": "mouse"})
        elif choice == '3':
            send_payload({"type": "vision_req", "ts": time.time(), "mode": "STOP"})
        elif choice == '4':
            send_payload({"type": "home_tag_req", "ts": time.time(), "mode": "RETURN"})
        elif choice.lower() == 'q':
            break