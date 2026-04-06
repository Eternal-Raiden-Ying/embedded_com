#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
模拟业务端接收 Vision App 发送的识别结果
"""
import socket
import json

# 请修改为 App 尝试连接的端口 (obs_out 配置)
HOST = "127.0.0.1" 
PORT = 9002

def main():
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as server:
        server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        server.bind((HOST, PORT))
        server.listen(1)
        print(f"📡 [服务启动] 正在监听 {HOST}:{PORT} 等待 App 发送数据...")

        while True:
            try:
                conn, addr = server.accept()
                print(f"🔗 [客户端已连接] 来自 {addr}")
                
                with conn:
                    # 将 socket 包装成 file object 方便按行读取
                    file_obj = conn.makefile("r", encoding="utf-8")
                    while True:
                        line = file_obj.readline()
                        if not line:
                            print("❌ [连接断开] App 主动断开或崩溃。")
                            break
                        
                        line = line.strip()
                        if line:
                            try:
                                data = json.loads(line)
                                msg_type = data.get('type', 'unknown')
                                print(f"📥 [RX - {msg_type.upper()}] {json.dumps(data, ensure_ascii=False)}")
                            except json.JSONDecodeError:
                                print(f"⚠️ [脏数据] 无法解析: {line}")
            except KeyboardInterrupt:
                print("\n🛑 退出接收端。")
                break
            except Exception as e:
                print(f"⚠️ 发生错误: {e}")

if __name__ == "__main__":
    main()