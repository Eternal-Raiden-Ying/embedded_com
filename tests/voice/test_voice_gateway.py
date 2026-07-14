#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
import time
import socket
import pytest
from pathlib import Path

# Setup path priorities
repo_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(repo_root))
sys.path.insert(0, str(repo_root / "Voice"))
sys.path.insert(0, str(repo_root / "common"))
sys.path.insert(0, str(repo_root / "orchestrator"))

from voice_service.config.paths import resolve_path, REPO_ROOT
from voice_service.runtime.commands import CommandInterpreter
from voice_service.runtime.state import RuntimeState
from voice_service.app.main import check_models
from voice_service.config.schema import VoiceServiceConfig
from voice_service.ipc.protocol import build_task_cmd, normalize_task_ack
from voice_service.ipc.orchestrator_adapter import JsonlAckInbox

# Attempt to load orchestrator schemas
try:
    from orchestrator_service.ipc.protocol import TaskCmd
    from orchestrator_service.ipc.transport import JsonlInboundServer, JsonlClientSender
    HAS_ORCH = True
except ImportError:
    HAS_ORCH = False


def get_free_port() -> int:
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.bind(('127.0.0.1', 0))
    port = s.getsockname()[1]
    s.close()
    return port


def test_path_resolution():
    p = resolve_path("Voice/config/commands.json")
    assert p.is_absolute()
    assert p == REPO_ROOT / "Voice" / "config" / "commands.json"


def test_intent_matching():
    interpreter = CommandInterpreter()
    cases = [
        ("\u5e2e\u6211\u62ff\u82f9\u679c", "apple"), ("\u627e\u6c34\u74f6", "bottle"),
        ("\u627e\u836f\u74f6", "pill_bottle"), ("\u627e\u836f\u76d2", "pill_box"),
        ("\u627e\u70ed\u6c34\u5668", "water_dispenser"), ("\u627e\u996e\u6c34\u673a", "water_dispenser"),
        ("\u627e\u5f00\u6c34\u673a", "water_dispenser"),
    ]
    for text, expected in cases:
        intent, target, _ = interpreter.infer_intent_and_target(text)
        assert (intent, target) == ("FIND", expected)
    assert interpreter.infer_intent_and_target("\u627e\u4e00\u53f7\u684c\u9762")[0] == "REJECT"
    assert interpreter.infer_intent_and_target("\u627e\u6536\u7eb3\u7b50")[0] == "REJECT"


def test_stop_state_updates():
    rt = RuntimeState()
    assert rt.get_epoch() == 0
    rt.bump_epoch()
    assert rt.get_epoch() == 1
    
    # Trigger stop guard mutes
    rt.reset_after_stop(guard_secs=0.5, block_secs=0.8)
    assert rt.in_guard()
    assert not rt.can_trigger_stop()


def test_model_missing_checks():
    # In normal mode, missing model throws SystemExit
    cfg = VoiceServiceConfig(dry_run_text=False)
    cfg.kws.model_dir = "nonexistent_kws_model"
    cfg.kws.keywords_file = "nonexistent_keywords.txt"
    with pytest.raises(SystemExit):
        check_models(cfg)
        
    # In dry_run_text mode, missing model only prints a warning and does not exit
    cfg_dry = VoiceServiceConfig(dry_run_text=True)
    cfg_dry.kws.model_dir = "nonexistent_kws_model"
    cfg_dry.kws.keywords_file = "nonexistent_keywords.txt"
    check_models(cfg_dry)  # Should execute successfully without raising SystemExit


@pytest.mark.skipif(not HAS_ORCH, reason="orchestrator_service package not importable")
def test_ipc_protocol_validation():
    payload = {
        "intent": "FIND",
        "target": "banana",
        "cmd_id": "test_cmd_banana",
        "epoch": 5,
        "text": "帮我拿香蕉"
    }
    out = build_task_cmd(payload)
    
    # Validate structure compatibility against Orchestrator's TaskCmd schema
    cmd = TaskCmd.from_dict(out, {"apple", "banana", "bottle", "key", "mouse"})
    assert cmd.intent == "FIND"
    assert cmd.target == "banana"
    assert cmd.cmd_id == "test_cmd_banana"
    assert cmd.epoch == 5
    assert cmd.text == "帮我拿香蕉"


@pytest.mark.skipif(not HAS_ORCH, reason="orchestrator_service package not importable")
def test_tcp_ipc_roundtrips():
    cmd_port = get_free_port()
    ack_port = get_free_port()
    tts_port = get_free_port()
    
    # 1. Start Orchestrator's TaskCmd server
    orch_cmd_server = JsonlInboundServer(
        mode="tcp", tcp_host="127.0.0.1", tcp_port=cmd_port, name="orch_cmd_server"
    )
    orch_cmd_server.start()
    
    # 2. Start Voice Gateway's TaskAck and TTSEvent servers
    vg_ack_server = JsonlInboundServer(
        mode="tcp", tcp_host="127.0.0.1", tcp_port=ack_port, name="vg_ack_server"
    )
    vg_ack_server.start()
    
    vg_tts_server = JsonlInboundServer(
        mode="tcp", tcp_host="127.0.0.1", tcp_port=tts_port, name="vg_tts_server"
    )
    vg_tts_server.start()
    
    time.sleep(0.15)  # Wait for servers to bind
    
    # 3. Create Voice Gateway's TaskCmd sender client
    vg_cmd_sender = JsonlClientSender(
        mode="tcp", tcp_host="127.0.0.1", tcp_port=cmd_port, name="vg_cmd_sender", send_mode="persistent"
    )
    
    mg_cmd_sender = JsonlClientSender(
        mode="tcp", tcp_host="127.0.0.1", tcp_port=cmd_port, name="mg_cmd_sender", send_mode="persistent"
    )
    
    # 4. Create Orchestrator's TaskAck and TTSEvent sender clients
    orch_ack_sender = JsonlClientSender(
        mode="tcp", tcp_host="127.0.0.1", tcp_port=ack_port, name="orch_ack_sender", send_mode="persistent"
    )
    
    orch_tts_sender = JsonlClientSender(
        mode="tcp", tcp_host="127.0.0.1", tcp_port=tts_port, name="orch_tts_sender", send_mode="persistent"
    )
    
    try:
        # A. Send TaskCmd from Voice Gateway and Mobile Gateway to Orchestrator
        cmd_payload_vg = {"type": "task_cmd", "intent": "FIND", "target": "apple", "cmd_id": "c1", "epoch": 0, "source": "voice_gateway"}
        sent_vg = vg_cmd_sender.send(cmd_payload_vg)
        assert sent_vg
        
        cmd_payload_mg = {"type": "task_cmd", "intent": "FIND", "target": "bottle", "cmd_id": "c2", "epoch": 0, "source": "mobile_gateway"}
        sent_mg = mg_cmd_sender.send(cmd_payload_mg)
        assert sent_mg
        
        # Verify receipt of BOTH at Orchestrator
        time.sleep(0.2)
        received_cmds = orch_cmd_server.drain()
        assert len(received_cmds) == 2
        
        vg_received = [r for r in received_cmds if r["payload"]["source"] == "voice_gateway"][0]
        mg_received = [r for r in received_cmds if r["payload"]["source"] == "mobile_gateway"][0]
        
        assert vg_received["payload"]["cmd_id"] == "c1"
        assert vg_received["payload"]["target"] == "apple"
        assert mg_received["payload"]["cmd_id"] == "c2"
        assert mg_received["payload"]["target"] == "bottle"
        
        # B. Send TaskAck from Orchestrator to Voice Gateway
        ack_payload = {"type": "task_ack", "cmd_id": "c1", "accepted": True, "epoch": 0}
        sent = orch_ack_sender.send(ack_payload)
        assert sent
        
        # Verify receipt at Voice Gateway
        time.sleep(0.15)
        received_acks = vg_ack_server.drain()
        assert len(received_acks) == 1
        assert received_acks[0]["payload"]["cmd_id"] == "c1"
        assert received_acks[0]["payload"]["accepted"] is True
        
        # C. Send TTSEvent from Orchestrator to Voice Gateway
        tts_payload = {"type": "tts_event", "text": "Hello world", "interrupt": False}
        sent = orch_tts_sender.send(tts_payload)
        assert sent
        
        # Verify receipt at Voice Gateway
        time.sleep(0.15)
        received_tts = vg_tts_server.drain()
        assert len(received_tts) == 1
        assert received_tts[0]["payload"]["text"] == "Hello world"
        
    finally:
        vg_cmd_sender.close()
        mg_cmd_sender.close()
        orch_ack_sender.close()
        orch_tts_sender.close()
        orch_cmd_server.close()
        vg_ack_server.close()
        vg_tts_server.close()


def test_stop_protection():
    # 1. STOP triggers epoch bump
    rt = RuntimeState()
    assert rt.get_epoch() == 0
    rt.bump_epoch()
    assert rt.get_epoch() == 1
    
    # 2. Duplicate STOP is suppressed (using can_trigger_stop and block_stop_retrigger)
    assert rt.can_trigger_stop()
    rt.block_stop_retrigger(1.0)
    assert not rt.can_trigger_stop()  # Duplicate STOP suppressed
    
    # 3. STOP is always high_priority=True in build_task_cmd
    cmd = build_task_cmd({"intent": "STOP", "high_priority": False})
    assert cmd["high_priority"] is True


def test_ack_routing():
    inbox = JsonlAckInbox()
    
    # 1. Only processes pending cmd_id
    inbox.register_pending("pending_cmd_1")
    
    # Unknown cmd_id is ignored
    inbox.handle_message({"cmd_id": "unknown_cmd_1", "accepted": True, "type": "task_ack"})
    assert inbox.wait_ack("unknown_cmd_1", timeout=0.01) is None
    
    # Pending cmd_id is processed and returned
    inbox.handle_message({"cmd_id": "pending_cmd_1", "accepted": True, "type": "task_ack"})
    ack = inbox.wait_ack("pending_cmd_1", timeout=0.1)
    assert ack is not None
    assert ack["accepted"] is True


def test_target_catalog_rejection():
    from common.target_catalog import resolve_target, target_to_class_id
    assert target_to_class_id("apple") == 2
    assert target_to_class_id("bottle") == 5
    assert target_to_class_id("basket") == 4
    assert target_to_class_id("water_dispenser") == 14
    assert resolve_target("kiwi_fruit") is None


def test_voice_new_config_structures():
    from voice_service.config.schema import VoiceServiceConfig, VoiceConsoleConfig, VoiceLexiconConfig, VoiceInteractionConfig
    
    cfg = VoiceServiceConfig(
        wake_phrases="xiaoche",
        asr_dir="",
        vad_dir="",
        console=VoiceConsoleConfig(
            events=["WAKE_TRIGGERED", "ASR_FINAL"],
            fields=["session_id", "epoch"],
            periodic_health_s=5
        ),
        lexicon=VoiceLexiconConfig(
            intents=["FIND", "RETURN", "STOP"],
            targets=["apple", "key"],
            asr_hotwords=["停止", "返航"]
        ),
        interaction=VoiceInteractionConfig(
            post_command_cooldown_ms=1000,
            reject_commands_while_busy=True,
            allow_stop_while_busy=True
        )
    )
    
    assert cfg.console.events == ["WAKE_TRIGGERED", "ASR_FINAL"]
    assert cfg.lexicon.intents == ["FIND", "RETURN", "STOP"]
    assert cfg.lexicon.asr_hotwords == ["停止", "返航"]
    assert cfg.interaction.post_command_cooldown_ms == 1000
    assert cfg.interaction.reject_commands_while_busy is True
    assert cfg.interaction.allow_stop_while_busy is True
