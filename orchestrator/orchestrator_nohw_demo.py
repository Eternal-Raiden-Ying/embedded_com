#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
无硬件 orchestrator 联调脚本。

作用：
1. 在进程内启动 orchestrator（UART dry-run，不打开 /dev/ttyHS1）
2. 自带 task_ack_out / vision_req_out / tts_event_out 监听器
3. 模拟 voice 发 task_cmd，模拟 vision 发桌边 / 目标 / 回家 tag 观测
4. 覆盖当前主状态链路，适合反复回归，不需要相机
5. 直接打印：观测字段 -> controller.py 的 CmdVel -> UART 分行协议
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
        self.thread = threading.Thread(
            target=self.server.serve_forever,
            daemon=True,
            name=f"collector_{self.channel_name}",
        )
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


def wrap_vision_obs(kind: str, base: Dict[str, Any], perception: Dict[str, Any], *, mode: str) -> Dict[str, Any]:
    stage = "RETURN" if kind == "home_tag_obs" else "SEARCH"
    return {
        **base,
        "type": "vision_obs",
        "stage": stage,
        "mode": mode,
        "status": "RUNNING",
        "perception": {kind: dict(perception or {})},
    }


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

    cfg.tts_event_out.transport = "tcp"
    cfg.tts_event_out.host = "127.0.0.1"
    cfg.tts_event_out.port = 19011
    cfg.tts_event_out.send_mode = "oneshot"
    return cfg


def patch_uart_verbose_print() -> None:
    from orchestrator_service.bridge.uart_bridge import UartBridge

    orig_write = UartBridge._write_line

    def _patched(self, line: str, tx_meta=None):
        line_show = line.rstrip("\n").replace("\n", " | ")
        print(f"[UART OUT] {line_show}")
        return orig_write(self, line, tx_meta=tx_meta)

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


def show_table_decision(service, obs_payload: Dict[str, Any], phase: str) -> None:
    from orchestrator_service.ipc.protocol import TableEdgeObs

    obs = TableEdgeObs.from_dict(obs_payload)
    if phase == "coarse":
        decision = service.core.controller.coarse_align_cmd(obs)
    elif phase == "approach":
        decision = service.core.controller.controlled_approach_cmd(obs)
    elif phase == "final":
        decision = service.core.controller.final_lock_cmd(obs)
    else:
        raise ValueError(f"unknown table phase: {phase}")
    print(
        "[CONTROLLER ] table_edge_obs"
        f" phase={phase}"
        f" yaw_err_rad={obs.yaw_err_rad if obs.yaw_err_rad is not None else 0.0:.3f}"
        f" dist_err_m={obs.dist_err_m if obs.dist_err_m is not None else 0.0:.3f}"
        f" -> CmdVel(mode={decision.cmd.mode}, vx_norm={decision.cmd.vx_norm:.3f},"
        f" vy_norm={decision.cmd.vy_norm:.3f}, wz_norm={decision.cmd.wz_norm:.3f})"
    )
    car_cmd = service.mapper.from_cmd_vel(decision.cmd, decision.cx_norm_abs, decision.distance_ratio)
    print(f"[UART MAP   ] {car_cmd.raw_line.rstrip()}".replace("\n", " | "))


def show_return_decision(service, obs_payload: Dict[str, Any]) -> None:
    from orchestrator_service.ipc.protocol import HomeTagObs

    obs = HomeTagObs.from_dict(obs_payload)
    decision = service.core.controller.return_cmd(obs)
    dist = obs.distance_m if obs.distance_m is not None else 0.0
    print(
        "[CONTROLLER ] home_tag_obs"
        f" yaw_err_rad={obs.yaw_err_rad:.3f} distance_m={dist:.3f}"
        f" -> CmdVel(mode={decision.cmd.mode}, vx_norm={decision.cmd.vx_norm:.3f},"
        f" vy_norm={decision.cmd.vy_norm:.3f}, wz_norm={decision.cmd.wz_norm:.3f})"
    )
    car_cmd = service.mapper.from_cmd_vel(decision.cmd, decision.cx_norm_abs, decision.distance_ratio)
    print(f"[UART MAP   ] {car_cmd.raw_line.rstrip()}".replace("\n", " | "))


def print_state(service, label: str = "STATE") -> None:
    block = service.core.export_state_block()
    print(f"[{label:<11}] {json.dumps(block, ensure_ascii=False)}")


def wait_until_state(service, expected: str, timeout_s: float = 3.0) -> None:
    deadline = time.time() + float(timeout_s)
    expected = str(expected).strip().upper()
    last_state = ""
    while time.time() < deadline:
        state = str(service.core.ctx.state.value).strip().upper()
        if state != last_state:
            print_state(service, label="STATE")
            last_state = state
        if state == expected:
            return
        time.sleep(0.05)
    raise RuntimeError(f"等待状态超时: expected={expected}, actual={service.core.ctx.state.value}")


def send_task(cfg, *, intent: str, session_id: str, epoch: int, target: Optional[str] = None) -> None:
    payload = {
        "ts": time.time(),
        "type": "task_cmd",
        "intent": intent,
        "confidence": 0.99,
        "cmd_id": f"demo_{intent.lower()}_{epoch}",
        "session_id": session_id,
        "epoch": int(epoch),
        "source": "demo",
    }
    if target:
        payload["target"] = target
    send_jsonl(cfg.task_cmd_in.host, cfg.task_cmd_in.port, payload)


def send_table_obs(cfg, *, session_id: str, epoch: int, yaw: float, dist: float, lat: float = 0.0, edge_ready: bool = False, table_cx: float = 0.0, table_size: float = 0.3) -> None:
    base = {"ts": time.time(), "session_id": session_id, "epoch": int(epoch), "source": "demo"}
    obs = {
        "table_found": True,
        "edge_found": True,
        "confidence": 0.95,
        "yaw_err_rad": float(yaw),
        "dist_err_m": float(dist),
        "lateral_err_m": float(lat),
        "edge_ready": bool(edge_ready),
        "table_cx_norm": float(table_cx),
        "table_size_norm": float(table_size),
    }
    send_jsonl(
        cfg.vision_obs_in.host,
        cfg.vision_obs_in.port,
        wrap_vision_obs("table_edge_obs", base, obs, mode="FIND_EDGE"),
    )


def send_target_obs(cfg, *, session_id: str, epoch: int, target: str, found: bool, cx: float = 0.0, size: float = 0.12) -> None:
    base = {"ts": time.time(), "session_id": session_id, "epoch": int(epoch), "source": "demo"}
    obs = {
        "found": bool(found),
        "target": target if found else None,
        "confidence": 0.93 if found else None,
        "cx_norm": float(cx),
        "size_norm": float(size),
    }
    send_jsonl(
        cfg.vision_obs_in.host,
        cfg.vision_obs_in.port,
        wrap_vision_obs("target_obs", base, obs, mode="FIND_OBJECT"),
    )


def send_home_obs(cfg, *, session_id: str, epoch: int, yaw: float, distance_m: float) -> None:
    base = {"ts": time.time(), "session_id": session_id, "epoch": int(epoch), "source": "demo"}
    obs = {
        "found": True,
        "yaw_err_rad": float(yaw),
        "distance_m": float(distance_m),
    }
    send_jsonl(
        cfg.vision_obs_in.host,
        cfg.vision_obs_in.port,
        wrap_vision_obs("home_tag_obs", base, obs, mode="FIND_OBJECT"),
    )


def run_demo(service, cfg) -> None:
    session_find = "sess_demo_find"
    target = "bottle"

    print("\n========== 场景 A：FIND -> SEARCH_TABLE ==========")
    send_task(cfg, intent="FIND", target=target, session_id=session_find, epoch=1)
    wait_until_state(service, "SEARCH_TABLE", timeout_s=2.0)

    print("\n========== 场景 B：桌边已发现但偏差较大 -> COARSE_ALIGN ==========")
    coarse_obs = {
        "ts": time.time(),
        "type": "table_edge_obs",
        "table_found": True,
        "edge_found": True,
        "confidence": 0.95,
        "yaw_err_rad": 0.18,
        "dist_err_m": 0.30,
        "lateral_err_m": 0.00,
        "table_cx_norm": 0.18,
        "table_size_norm": 0.22,
        "session_id": session_find,
        "epoch": 1,
    }
    show_table_decision(service, coarse_obs, phase="coarse")
    for _ in range(2):
        send_table_obs(cfg, session_id=session_find, epoch=1, yaw=0.18, dist=0.30, table_cx=0.18, table_size=0.22)
        time.sleep(0.20)
    wait_until_state(service, "COARSE_ALIGN", timeout_s=2.0)

    print("\n========== 场景 C：粗对齐完成 -> CONTROLLED_APPROACH ==========")
    approach_obs = {
        "ts": time.time(),
        "type": "table_edge_obs",
        "table_found": True,
        "edge_found": True,
        "confidence": 0.96,
        "yaw_err_rad": 0.03,
        "dist_err_m": 0.22,
        "lateral_err_m": 0.00,
        "table_cx_norm": 0.03,
        "table_size_norm": 0.30,
        "session_id": session_find,
        "epoch": 1,
    }
    show_table_decision(service, approach_obs, phase="approach")
    for _ in range(2):
        send_table_obs(cfg, session_id=session_find, epoch=1, yaw=0.03, dist=0.22, table_cx=0.03, table_size=0.30)
        time.sleep(0.20)
    wait_until_state(service, "CONTROLLED_APPROACH", timeout_s=2.0)

    print("\n========== 场景 D：边缘接近完成 -> FINAL_LOCK ==========")
    final_obs = {
        "ts": time.time(),
        "type": "table_edge_obs",
        "table_found": True,
        "edge_found": True,
        "confidence": 0.97,
        "yaw_err_rad": 0.02,
        "dist_err_m": 0.05,
        "lateral_err_m": 0.00,
        "table_cx_norm": 0.01,
        "table_size_norm": 0.50,
        "edge_ready": True,
        "session_id": session_find,
        "epoch": 1,
    }
    show_table_decision(service, final_obs, phase="final")
    send_table_obs(cfg, session_id=session_find, epoch=1, yaw=0.02, dist=0.05, table_cx=0.01, table_size=0.50, edge_ready=True)
    wait_until_state(service, "FINAL_LOCK", timeout_s=2.0)

    print("\n========== 场景 E：最终锁边成功 -> AT_TABLE_EDGE ==========")
    for _ in range(2):
        send_table_obs(cfg, session_id=session_find, epoch=1, yaw=0.01, dist=0.01, lat=0.0, table_cx=0.0, table_size=0.58, edge_ready=True)
        time.sleep(0.20)
    wait_until_state(service, "AT_TABLE_EDGE", timeout_s=2.0)

    print("\n========== 场景 F：自动进入 EDGE_SLIDE_SEARCH ==========")
    wait_until_state(service, "SEARCH_TARGET_INIT", timeout_s=2.0)
    wait_until_state(service, "EDGE_SLIDE_SEARCH", timeout_s=2.0)

    print("\n========== 场景 G：目标确认并完成任务 ==========")
    for _ in range(3):
        send_target_obs(cfg, session_id=session_find, epoch=1, target=target, found=True, cx=0.02, size=0.16)
        time.sleep(0.25)
    wait_until_state(service, "TARGET_LOCKED", timeout_s=2.0)
    wait_until_state(service, "DONE", timeout_s=3.0)
    wait_until_state(service, "IDLE", timeout_s=3.0)

    print("\n========== 场景 H：RETURN -> DONE ==========")
    session_return = "sess_demo_return"
    send_task(cfg, intent="RETURN", session_id=session_return, epoch=1)
    wait_until_state(service, "RETURN_HOME", timeout_s=2.0)
    far_home = {
        "ts": time.time(),
        "type": "home_tag_obs",
        "found": True,
        "yaw_err_rad": 0.42,
        "distance_m": 0.85,
        "session_id": session_return,
        "epoch": 1,
    }
    show_return_decision(service, far_home)
    send_home_obs(cfg, session_id=session_return, epoch=1, yaw=0.42, distance_m=0.85)
    time.sleep(0.30)
    near_home = {
        "ts": time.time(),
        "type": "home_tag_obs",
        "found": True,
        "yaw_err_rad": 0.03,
        "distance_m": 0.28,
        "session_id": session_return,
        "epoch": 1,
    }
    show_return_decision(service, near_home)
    for _ in range(2):
        send_home_obs(cfg, session_id=session_return, epoch=1, yaw=0.03, distance_m=0.28)
        time.sleep(0.25)
    wait_until_state(service, "DONE", timeout_s=3.0)
    wait_until_state(service, "IDLE", timeout_s=3.0)

    print("\n========== 演示完成 ==========")
    print(f"日志目录：{service.run_logger.run_dir}")
    print("重点看这几个文件：task_cmd.jsonl / vision_obs.jsonl / table_edge_obs.jsonl / target_obs.jsonl / home_tag_obs.jsonl / cmd_vel.jsonl / car_cmd.jsonl / timeline.jsonl")


def main() -> None:
    script_dir = Path(__file__).resolve().parent
    ap = argparse.ArgumentParser(description="无硬件 orchestrator 联调脚本")
    ap.add_argument("--project-root", default=str(script_dir), help="orchestrator 工程根目录")
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
    for collector in collectors:
        collector.start()

    service = OrchestratorService(cfg)
    thread = threading.Thread(target=service.run_forever, daemon=True, name="orchestrator_demo")
    thread.start()
    time.sleep(0.8)
    print(f"[BOOT       ] run_dir = {service.run_logger.run_dir}")
    try:
        run_demo(service, cfg)
    finally:
        service._running = False
        thread.join(timeout=3.0)
        for collector in collectors:
            collector.close()


if __name__ == "__main__":
    main()
