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
    
    # 1. 帮我拿苹果 -> FIND/apple
    intent, target, score = interpreter.infer_intent_and_target("帮我拿苹果")
    assert intent == "FIND"
    assert target == "apple"
    
    # 2. 帮我拿水瓶 -> FIND/bottle
    intent, target, score = interpreter.infer_intent_and_target("帮我拿水瓶")
    assert intent == "FIND"
    assert target == "bottle"
    
    # 3. 帮我拿香蕉 -> FIND/banana
    intent, target, score = interpreter.infer_intent_and_target("帮我拿香蕉")
    assert intent == "FIND"
    assert target == "banana"
    
    # 4. 帮我拿钥匙 -> FIND/key
    intent, target, score = interpreter.infer_intent_and_target("帮我拿钥匙")
    assert intent == "FIND"
    assert target == "key"
    
    # 5. 帮我拿鼠标 -> FIND/mouse
    intent, target, score = interpreter.infer_intent_and_target("帮我拿鼠标")
    assert intent == "FIND"
    assert target == "mouse"
    
    # 6. 回来 -> RETURN
    intent, target, score = interpreter.infer_intent_and_target("回来")
    assert intent == "RETURN"
    assert target is None
    
    # 7. 停止 -> STOP
    intent, target, score = interpreter.infer_intent_and_target("停止")
    assert intent == "STOP"
    assert target is None
    
    # 8. 小车停止 -> STOP
    intent, target, score = interpreter.infer_intent_and_target("小车停止")
    assert intent == "STOP"
    assert target is None


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
    cfg = VoiceServiceConfig(
        dry_run_text=False,
        wake_tflite="nonexistent_wake_model.onnx"
    )
    with pytest.raises(SystemExit):
        check_models(cfg)
        
    # In dry_run_text mode, missing model only prints a warning and does not exit
    cfg_dry = VoiceServiceConfig(
        dry_run_text=True,
        wake_tflite="nonexistent_wake_model.onnx"
    )
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


@pytest.mark.skipif(not HAS_ORCH, reason="orchestrator_service package not importable")
def test_target_catalog_rejection():
    # 1. Verify target_catalog status lookup
    import yaml
    repo_root = Path(__file__).resolve().parents[2]
    catalog_path = repo_root / "configs" / "target_catalog.yaml"
    assert catalog_path.exists()
    
    with open(catalog_path, "r", encoding="utf-8") as f:
        cat = yaml.safe_load(f)
        targets = cat.get("targets", {})
        
    assert targets["apple"]["status"] == "executable"
    assert targets["key"]["status"] == "executable"
    assert targets["remote"]["status"] == "model_pending"
    assert targets["medicine_box"]["status"] == "model_pending"
    assert targets["eye_drops"]["status"] == "model_pending"
    assert targets["nail_clipper"]["status"] == "model_pending"
    assert targets["battery"]["status"] == "model_pending"

    # 2. Test mock orchestrator handle_task_cmd capability gating
    from orchestrator_service.ipc.protocol import TaskCmd
    from orchestrator_service.runtime.task_runtime import TaskRuntimeMixin
    from orchestrator_service.runtime.transitions import RuntimeContext
    
    class MockConfig:
        post_stop_ignore_s = 1.0
        cmd_confidence_th = 0.5
        
    class MockOrchestrator(TaskRuntimeMixin):
        def __init__(self):
            self.ctx = RuntimeContext()
            self.cfg = MockConfig()
            self._last_stop_mono = 0
            self.tts_calls = []
            self.find_calls = []
            
        def _log(self, level, msg):
            pass
            
        def _queue_tts(self, text):
            self.tts_calls.append(text)
            
        def _start_find_task(self, cmd):
            self.find_calls.append(cmd)

    orch = MockOrchestrator()
    
    # Test remote (model_pending)
    cmd_remote = TaskCmd.from_dict({
        "intent": "FIND",
        "target": "remote",
        "cmd_id": "cr",
        "confidence": 0.8,
        "ts": time.time(),
        "epoch": 0,
        "session_id": "s1"
    }, {"apple", "banana", "key", "mouse", "remote"})
    accepted, reason = orch.handle_task_cmd(cmd_remote)
    assert not accepted
    assert reason == "target_recognized_but_not_executable"
    assert "该物品模型尚在开发中，无法获取" in orch.tts_calls
    assert len(orch.find_calls) == 0
    
    # Test key (executable)
    orch.tts_calls.clear()
    cmd_key = TaskCmd.from_dict({
        "intent": "FIND",
        "target": "key",
        "cmd_id": "ck",
        "confidence": 0.8,
        "ts": time.time(),
        "epoch": 0,
        "session_id": "s1"
    }, {"apple", "banana", "key", "mouse", "remote"})
    accepted, reason = orch.handle_task_cmd(cmd_key)
    assert accepted
    assert reason == "accepted"
    assert len(orch.find_calls) == 1


def test_voice_new_config_structures():
    from voice_service.config.schema import VoiceServiceConfig, VoiceConsoleConfig, VoiceLexiconConfig, VoiceInteractionConfig
    
    cfg = VoiceServiceConfig(
        wake_key="xiaoche",
        stop_key="ting",
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


