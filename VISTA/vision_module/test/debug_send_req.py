#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""调试发送器：向 VISTA 发送主线协议请求。"""

import json
import socket
import time

try:
    from .debug_protocol_tools import load_debug_state
except ImportError:
    from debug_protocol_tools import load_debug_state


HOST = "127.0.0.1"
PORT = 9003


def send_payload(payload: dict):
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.settimeout(2.0)
            sock.connect((HOST, PORT))
            line = json.dumps(payload, ensure_ascii=False) + "\n"
            sock.sendall(line.encode("utf-8"))
            print(f"[TX] {line.strip()}")
    except ConnectionRefusedError:
        print(f"[ERR] 无法连接到 {HOST}:{PORT}，请确认 VISTA 已启动。")
    except Exception as exc:
        print(f"[ERR] 发送失败: {exc}")


def build_req(op: str, stage: str, **kwargs) -> dict:
    payload = {
        "type": "vision_req",
        "ts": time.time(),
        "session_id": kwargs.pop("session_id", "debug_sess"),
        "req_id": kwargs.pop("req_id", f"req_{int(time.time() * 1000)}"),
        "epoch": kwargs.pop("epoch", 1),
        "op": op,
        "stage": stage,
    }
    payload.update(kwargs)
    return payload


def build_grasp_response(decision: str) -> dict:
    state = load_debug_state()
    interaction_id = state.get("last_interaction_id")
    if not interaction_id:
        raise RuntimeError("尚未从 debug_recv_obj.py 记录到 interaction_id")
    return build_req(
        "RESPOND",
        "GRASP",
        session_id=state.get("last_session_id", "debug_sess"),
        epoch=state.get("last_epoch", 1),
        interaction_id=interaction_id,
        response={"decision": decision},
        payload={
            "executed_motion": {
                "dx_m": 0.03,
                "dy_m": -0.01,
                "dyaw_rad": 0.08,
            }
        },
    )


def build_search_found_req(target: str) -> dict:
    """Test-only injection helper; not part of the normal external protocol flow."""
    return build_req(
        "START",
        "SEARCH",
        target=target,
        payload={
            "target_obs": {
                "found": True,
                "target": target,
                "confidence": 0.91,
                "cx_norm": 0.08,
                "size_norm": 0.22,
                "bbox": [160, 120, 360, 360],
            }
        },
    )


def build_grasp_start(mode_hint: str = "MICRO_ADJUST") -> dict:
    return build_req(
        "START",
        "GRASP",
        target="bottle",
        mode_hint=mode_hint,
        payload={"remote_grasp": True, "need_depth": True},
    )


def show_last_interaction():
    state = load_debug_state()
    interaction_id = state.get("last_interaction_id")
    if not interaction_id:
        print("[STATE] 当前没有已记录的 interaction_id")
        return
    print(
        "[STATE] "
        f"interaction_id={interaction_id} "
        f"stage={state.get('last_stage')} "
        f"mode={state.get('last_mode')} "
        f"status={state.get('last_status')}"
    )


def show_menu():
    print("VISTA Vision 调试发送器")
    print("1: 新协议 SEARCH bottle")
    print("2: 新协议 RETURN")
    print("3: 新协议 GRASP START")
    print("4: 新协议 GRASP START(remote)")
    print("5: 新协议 GRASP RESPOND ACCEPT")
    print("6: 新协议 GRASP RESPOND REJECT")
    print("7: 新协议 IDLE/STOP")
    print("8: 查看最近 interaction_id")
    print("")
    print("[NOTE] SEARCH bottle(found) 仅用于内部测试注入，不再作为主菜单协议入口。")


if __name__ == "__main__":
    show_menu()
    while True:
        choice = input("\n请选择指令 (1-8, q 退出): ").strip().lower()
        if choice == "1":
            send_payload(build_req("START", "SEARCH", target="bottle"))
        elif choice == "2":
            send_payload(build_req("START", "RETURN"))
        elif choice == "3":
            send_payload(build_grasp_start("MICRO_ADJUST"))
        elif choice == "4":
            send_payload(build_grasp_start("GRASP_REMOTE"))
        elif choice == "5":
            try:
                send_payload(build_grasp_response("ACCEPT"))
            except RuntimeError as exc:
                print(f"[ERR] {exc}")
        elif choice == "6":
            try:
                send_payload(build_grasp_response("REJECT"))
            except RuntimeError as exc:
                print(f"[ERR] {exc}")
        elif choice == "7":
            send_payload(build_req("STOP", "IDLE"))
        elif choice == "8":
            show_last_interaction()
        elif choice == "q":
            break
