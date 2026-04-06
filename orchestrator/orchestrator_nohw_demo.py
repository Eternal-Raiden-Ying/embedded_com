#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
无硬件 orchestrator 联调脚本。

作用：
1. 在进程内启动 orchestrator（UART dry-run，不打开 /dev/ttyHS1）
2. 自带 task_ack_out / vision_req_out / tts_event_out 监听器
3. 模拟 voice 发 task_cmd，模拟 vision 发 target_obs / home_tag_obs
4. 直接打印：观测字段 -> controller.py 的 CmdVel -> UART 分行协议
5. 适合阅读当前状态机与 controller.py 的真实作用，不需要外接任何硬件
"""

from __future__ import annotations

import argparse
import json
import os
import socket
import socketserver
import sys
import tempfile
import threading
import time
from pathlib import Path
from typing import Any, Dict, Optional


def add_project_to_syspath(project_root: str) -> None:
    root = str(Path(project_root).resolve())
    if root not in sys.path:
        sys.path.insert(0, root)


class JsonlTCPHandler(socketserver.StreamRequestHandler):
    def handle(self) -> None:
        callback = getattr(self.server, "callback", None)
        channel = getattr(self.server, "channel_name", "collector")
        while True:
            line = self.rfile.readline()
            if not line:
                break
            try:
                payload = json.loads(line.decode("utf-8").strip())
            except Exception as exc:
                print(f"[{channel}] bad json: {exc} | raw={line!r}")
                continue
            if callable(callback):
                callback(channel, payload)


class ReusableTCPServer(socketserver.ThreadingTCPServer):
    allow_reuse_address = True
    daemon_threads = True


class JsonlCollector:
    def __init__(self, host: str, port: int, channel_name: str, callback):
        self.host = host
        self.port = int(port)
        self.channel_name = channel_name
        self.callback = callback
        self.server: Optional[ReusableTCPServer] = None
        self.thread: Optional[threading.Thread] = None

    def start(self) -> None:
        self.server = ReusableTCPServer((self.host, self.port), JsonlTCPHandler)
        self.server.callback = self.callback
        self.server.channel_name = self.channel_name
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True, name=f"collector_{self.channel_name}")
        self.thread.start()

    def close(self) -> None:
        if self.server is not None:
            self.server.shutdown()
            self.server.server_close()
            self.server = None
        if self.thread is not None:
            self.thread.join(timeout=1.0)
            self.thread = None



def send_jsonl(host: str, port: int, payload: Dict[str, Any]) -> None:
    data = (json.dumps(payload, ensure_ascii=False) + "\n").encode("utf-8")
    with socket.create_connection((host, int(port)), timeout=2.0) as sock:
        sock.sendall(data)



def make_demo_config(project_root: str):
    from orchestrator_service.config.schema import OrchestratorConfig

    cfg = OrchestratorConfig()
    tmp_root = tempfile.mkdtemp(prefix="orch_nohw_demo_")
    cfg.runtime.project_root = project_root
    cfg.runtime.runs_dir = os.path.join(tmp_root, "runs")
    cfg.runtime.tick_hz = 10.0
    cfg.runtime.log_mode = "concise"
    cfg.runtime.debug = False
    cfg.runtime.state_block_period_s = 0.5

    cfg.serial.port = "/dev/null"
    cfg.serial.baudrate = 115200
    cfg.serial.timeout_s = 0.05
    cfg.serial.dry_run = True
    cfg.serial.readback_enabled = False

    # 使用独立端口，避免和真实服务冲突
    cfg.task_cmd_in.transport = "tcp"
    cfg.task_cmd_in.host = "127.0.0.1"
    cfg.task_cmd_in.port = 19001

    cfg.vision_obs_in.transport = "tcp"
    cfg.vision_obs_in.host = "127.0.0.1"
    cfg.vision_obs_in.port = 19002

    cfg.vision_req_out.transport = "tcp"
    cfg.vision_req_out.host = "127.0.0.1"
    cfg.vision_req_out.port = 19003
    cfg.vision_req_out.send_mode = "persistent"

    cfg.task_ack_out.transport = "tcp"
    cfg.task_ack_out.host = "127.0.0.1"
    cfg.task_ack_out.port = 19012
    cfg.task_ack_out.send_mode = "oneshot"

    # 这里显式打开 tts_event_out，便于观察状态机是否真的发了 TTS
    cfg.tts_event_out.transport = "tcp"
    cfg.tts_event_out.host = "127.0.0.1"
    cfg.tts_event_out.port = 19011
    cfg.tts_event_out.send_mode = "oneshot"

    return cfg



def patch_uart_verbose_print() -> None:
    from orchestrator_service.bridge.uart_bridge import UartBridge

    orig_write = UartBridge._write_line

    def _patched(self, line: str):
        line_show = line.rstrip("\n").replace("\n", " | ")
        print(f"[UART OUT] {line_show}")
        return orig_write(self, line)

    UartBridge._write_line = _patched



def print_json(channel: str, payload: Dict[str, Any]) -> None:
    if channel == "vision_req_out":
        print(f"[VISION_REQ_OUT] {json.dumps(payload, ensure_ascii=False)}")
    elif channel == "task_ack_out":
        print(f"[TASK_ACK_OUT ] {json.dumps(payload, ensure_ascii=False)}")
    elif channel == "tts_event_out":
        print(f"[TTS_EVENT_OUT] {json.dumps(payload, ensure_ascii=False)}")
    else:
        print(f"[{channel}] {json.dumps(payload, ensure_ascii=False)}")



def show_controller_chain(service, obs_payload: Dict[str, Any], kind: str) -> None:
    from orchestrator_service.ipc.protocol import HomeTagObs, TargetObs

    if kind == "target":
        obs = TargetObs.from_dict(obs_payload)
        decision = service.core.controller.search_cmd(obs)
        print(
            "[CONTROLLER ] target_obs"
            f" cx_norm={obs.cx_norm:.3f} size_norm={obs.size_norm:.3f}"
            f" -> CmdVel(mode={decision.cmd.mode}, vx_norm={decision.cmd.vx_norm:.3f}, wz_norm={decision.cmd.wz_norm:.3f})"
        )
    else:
        obs = HomeTagObs.from_dict(obs_payload)
        decision = service.core.controller.return_cmd(obs)
        dist = obs.distance_m if obs.distance_m is not None else 0.0
        print(
            "[CONTROLLER ] home_tag_obs"
            f" yaw_err_rad={obs.yaw_err_rad:.3f} distance_m={dist:.3f}"
            f" -> CmdVel(mode={decision.cmd.mode}, vx_norm={decision.cmd.vx_norm:.3f}, wz_norm={decision.cmd.wz_norm:.3f})"
        )
    car_cmd = service.mapper.from_cmd_vel(decision.cmd, decision.cx_norm_abs, decision.distance_ratio)
    print(f"[UART MAP   ] {car_cmd.raw_line.rstrip()}".replace("\n", " | "))



def wait_state(service, timeout_s: float = 2.0) -> None:
    time.sleep(timeout_s)
    block = service.core.export_state_block()
    print(f"[STATE      ] {json.dumps(block, ensure_ascii=False)}")



def run_demo(service, cfg) -> None:
    print("\n========== 场景 A：FIND 开始后，先没有目标 -> AUTOEXPLORE ==========")
    send_jsonl(cfg.task_cmd_in.host, cfg.task_cmd_in.port, {
        "ts": time.time(),
        "type": "task_cmd",
        "intent": "FIND",
        "target": "cup",
        "confidence": 0.98,
        "cmd_id": "demo_find_1",
        "session_id": "sess_demo_1",
        "epoch": 1,
        "source": "demo",
    })
    time.sleep(0.8)
    wait_state(service, 0.4)

    print("\n========== 场景 B：目标偏右，controller.py 先算转向，再发 SEARCH 的 V ==========")
    obs = {
        "ts": time.time(),
        "type": "target_obs",
        "found": True,
        "target": "cup",
        "confidence": 0.93,
        "cx_norm": 0.48,
        "size_norm": 0.10,
        "bbox": [500, 120, 120, 140],
        "score": 0.93,
        "session_id": "sess_demo_1",
    }
    show_controller_chain(service, obs, kind="target")
    for _ in range(2):
        obs["ts"] = time.time()
        send_jsonl(cfg.vision_obs_in.host, cfg.vision_obs_in.port, obs)
        time.sleep(0.35)
    wait_state(service, 0.3)

    print("\n========== 场景 C：基本对准但目标还远，controller.py 算前进速度 ==========")
    obs = {
        "ts": time.time(),
        "type": "target_obs",
        "found": True,
        "target": "cup",
        "confidence": 0.95,
        "cx_norm": 0.03,
        "size_norm": 0.18,
        "bbox": [300, 120, 180, 200],
        "score": 0.95,
        "session_id": "sess_demo_1",
    }
    show_controller_chain(service, obs, kind="target")
    for _ in range(3):
        obs["ts"] = time.time()
        send_jsonl(cfg.vision_obs_in.host, cfg.vision_obs_in.port, obs)
        time.sleep(0.35)
    wait_state(service, 0.3)

    print("\n========== 场景 D：目标很近，状态机进入 STOP ==========")
    obs = {
        "ts": time.time(),
        "type": "target_obs",
        "found": True,
        "target": "cup",
        "confidence": 0.97,
        "cx_norm": 0.01,
        "size_norm": 0.55,
        "bbox": [260, 110, 280, 280],
        "score": 0.97,
        "session_id": "sess_demo_1",
    }
    show_controller_chain(service, obs, kind="target")
    for _ in range(2):
        obs["ts"] = time.time()
        send_jsonl(cfg.vision_obs_in.host, cfg.vision_obs_in.port, obs)
        time.sleep(0.35)
    wait_state(service, 0.4)

    print("\n========== 场景 E：重新 FIND，先找到目标，再丢失 -> AUTOSEARCH ==========")
    send_jsonl(cfg.task_cmd_in.host, cfg.task_cmd_in.port, {
        "ts": time.time(),
        "type": "task_cmd",
        "intent": "FIND",
        "target": "cup",
        "confidence": 0.98,
        "cmd_id": "demo_find_2",
        "session_id": "sess_demo_2",
        "epoch": 2,
        "source": "demo",
    })
    time.sleep(0.7)
    found = {
        "ts": time.time(),
        "type": "target_obs",
        "found": True,
        "target": "cup",
        "confidence": 0.92,
        "cx_norm": -0.35,
        "size_norm": 0.12,
        "bbox": [120, 140, 110, 120],
        "score": 0.92,
        "session_id": "sess_demo_2",
    }
    for _ in range(2):
        found["ts"] = time.time()
        send_jsonl(cfg.vision_obs_in.host, cfg.vision_obs_in.port, found)
        time.sleep(0.30)
    lost = {
        "ts": time.time(),
        "type": "target_obs",
        "found": False,
        "target": "cup",
        "session_id": "sess_demo_2",
    }
    for _ in range(2):
        lost["ts"] = time.time()
        send_jsonl(cfg.vision_obs_in.host, cfg.vision_obs_in.port, lost)
        time.sleep(0.30)
    wait_state(service, 0.4)

    print("\n========== 场景 F：RETURN，controller.py 根据 yaw_err 和 distance 算 RETURN 的 V ==========")
    send_jsonl(cfg.task_cmd_in.host, cfg.task_cmd_in.port, {
        "ts": time.time(),
        "type": "task_cmd",
        "intent": "RETURN",
        "confidence": 0.99,
        "cmd_id": "demo_return_1",
        "session_id": "sess_ret_1",
        "epoch": 1,
        "source": "demo",
    })
    time.sleep(0.7)
    obs = {
        "ts": time.time(),
        "type": "home_tag_obs",
        "found": True,
        "yaw_err_rad": 0.42,
        "distance_m": 0.85,
        "session_id": "sess_ret_1",
    }
    show_controller_chain(service, obs, kind="home")
    for _ in range(2):
        obs["ts"] = time.time()
        send_jsonl(cfg.vision_obs_in.host, cfg.vision_obs_in.port, obs)
        time.sleep(0.35)
    obs2 = {
        "ts": time.time(),
        "type": "home_tag_obs",
        "found": True,
        "yaw_err_rad": 0.03,
        "distance_m": 0.28,
        "session_id": "sess_ret_1",
    }
    show_controller_chain(service, obs2, kind="home")
    for _ in range(2):
        obs2["ts"] = time.time()
        send_jsonl(cfg.vision_obs_in.host, cfg.vision_obs_in.port, obs2)
        time.sleep(0.35)
    wait_state(service, 0.5)

    print("\n========== 场景 G：显式 STOP，观察最高优先级 ==========")
    send_jsonl(cfg.task_cmd_in.host, cfg.task_cmd_in.port, {
        "ts": time.time(),
        "type": "task_cmd",
        "intent": "STOP",
        "confidence": 1.0,
        "cmd_id": "demo_stop_1",
        "session_id": "sess_stop_1",
        "epoch": 1,
        "source": "demo",
    })
    time.sleep(0.6)
    wait_state(service, 0.2)



def main() -> None:
    ap = argparse.ArgumentParser(description="无硬件 orchestrator 联调脚本")
    ap.add_argument("--project-root", default=".", help="orchestrator 工程根目录")
    args = ap.parse_args()

    add_project_to_syspath(args.project_root)
    patch_uart_verbose_print()

    from orchestrator_service.runtime.service import OrchestratorService

    cfg = make_demo_config(args.project_root)
    collectors = [
        JsonlCollector(cfg.task_ack_out.host, cfg.task_ack_out.port, "task_ack_out", print_json),
        JsonlCollector(cfg.vision_req_out.host, cfg.vision_req_out.port, "vision_req_out", print_json),
        JsonlCollector(cfg.tts_event_out.host, cfg.tts_event_out.port, "tts_event_out", print_json),
    ]
    for c in collectors:
        c.start()

    service = OrchestratorService(cfg)
    th = threading.Thread(target=service.run_forever, daemon=True, name="orchestrator_demo")
    th.start()
    time.sleep(0.8)
    print(f"[BOOT       ] run_dir = {service.run_logger.run_dir}")
    try:
        run_demo(service, cfg)
        print("\n========== 演示完成 ==========")
        print(f"日志目录：{service.run_logger.run_dir}")
        print("重点看这几个文件：task_cmd.jsonl / target_obs.jsonl / home_tag_obs.jsonl / cmd_vel.jsonl / car_cmd.jsonl / timeline.jsonl")
    finally:
        service._running = False
        th.join(timeout=3.0)
        for c in collectors:
            c.close()


if __name__ == "__main__":
    main()
