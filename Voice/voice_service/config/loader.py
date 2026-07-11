#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
import argparse
import dataclasses
from pathlib import Path
from typing import Any, Dict, List, Optional, Type, Union

from .paths import REPO_ROOT, resolve_path
from .schema import VoiceServiceConfig

# Import common loader if possible, or fall back to simple parser
try:
    from common.config.loader import load_yaml_file
except ImportError:
    # Minimal fallback parser in case common package is not on pythonpath
    def load_yaml_file(path: Union[str, Path]) -> Dict[str, Any]:
        p = Path(path)
        if not p.is_file():
            return {}
        try:
            import yaml
            with open(p, "r", encoding="utf-8") as fp:
                return dict(yaml.safe_load(fp) or {})
        except Exception:
            # Fallback very simple line parsing
            root = {}
            for line in p.read_text(encoding="utf-8").splitlines():
                line = line.split("#", 1)[0].strip()
                if not line or ":" not in line:
                    continue
                k, v = line.split(":", 1)
                k = k.strip()
                v = v.strip()
                if v.lower() == "true":
                    v = True
                elif v.lower() == "false":
                    v = False
                elif v.isdigit():
                    v = int(v)
                root[k] = v
            return root

def map_nested_dict(nested: Dict[str, Any], flat: Dict[str, Any]) -> None:
    # 1. audio section
    if "audio" in nested and isinstance(nested["audio"], dict):
        audio = nested["audio"]
        if "input_device" in audio:
            flat["arecord_device"] = str(audio["input_device"])
        if "device" in audio:
            flat["arecord_device"] = str(audio["device"])
        if "backend" in audio:
            pass

    # 2. kws section
    if "kws" in nested and isinstance(nested["kws"], dict):
        kws = nested["kws"]
        if "model_path" in kws:
            flat["wake_tflite"] = str(kws["model_path"])
        if "threshold" in kws:
            flat["wake_th"] = float(kws["threshold"])
        if "wake_word" in kws:
            flat["wake_key"] = str(kws["wake_word"])
        if "frontend_backend" in kws:
            flat["frontend_backend"] = str(kws["frontend_backend"])
        if "classifier_backend" in kws:
            flat["classifier_backend"] = str(kws["classifier_backend"])

    # 3. stop_kws section
    if "stop_kws" in nested and isinstance(nested["stop_kws"], dict):
        stop_kws = nested["stop_kws"]
        if "model_path" in stop_kws:
            flat["stop_tflite"] = str(stop_kws["model_path"])
        if "threshold" in stop_kws:
            flat["stop_th"] = float(stop_kws["threshold"])

    # 4. vad section
    if "vad" in nested and isinstance(nested["vad"], dict):
        vad = nested["vad"]
        if "model_path" in vad:
            flat["vad_dir"] = str(vad["model_path"])
        if "energy_threshold" in vad:
            flat["energy_th"] = float(vad["energy_threshold"])
        for key in ("start_frames", "end_frames", "pre_frames", "max_frames"):
            if key in vad:
                flat[key] = int(vad[key])

    # 5. asr section
    if "asr" in nested and isinstance(nested["asr"], dict):
        asr = nested["asr"]
        if "model_path" in asr:
            flat["asr_dir"] = str(asr["model_path"])
        if "mode" in asr:
            flat["asr_mode"] = str(asr["mode"])
        if "chunk_frames" in asr:
            flat["asr_online_chunk_frames"] = int(asr["chunk_frames"])
        if "chunk_size" in asr:
            flat["asr_online_chunk_size"] = [int(i) for i in asr["chunk_size"]]
        if "quantized" in asr:
            flat["asr_quant"] = bool(asr["quantized"])

    # 6. commands section
    if "commands" in nested and isinstance(nested["commands"], dict):
        commands = nested["commands"]
        if "config_path" in commands:
            flat["commands_json"] = str(commands["config_path"])

    # 7. tts section
    if "tts" in nested and isinstance(nested["tts"], dict):
        tts = nested["tts"]
        if "model_path" in tts:
            flat["piper_model"] = str(tts["model_path"])
        if "enabled" in tts:
            flat["disable_tts"] = not bool(tts["enabled"])
        if "playback_device" in tts:
            dev = str(tts["playback_device"])
            if dev and dev != "default":
                flat["play_cmd"] = f"aplay -q -D {dev}"
            else:
                flat["play_cmd"] = "aplay -q"

    # 8. ipc section
    if "ipc" in nested and isinstance(nested["ipc"], dict):
        ipc = nested["ipc"]
        if "task_cmd_out" in ipc and isinstance(ipc["task_cmd_out"], dict):
            out = ipc["task_cmd_out"]
            if "transport" in out: flat["task_transport"] = str(out["transport"])
            if "host" in out: flat["task_tcp_host"] = str(out["host"])
            if "port" in out: flat["task_tcp_port"] = int(out["port"])
            if "uds_path" in out: flat["task_uds_path"] = str(out["uds_path"])
            if "ipc_socket_path" in out: flat["task_uds_path"] = str(out["ipc_socket_path"])
        if "task_ack_in" in ipc and isinstance(ipc["task_ack_in"], dict):
            ack = ipc["task_ack_in"]
            if "transport" in ack: flat["task_ack_transport"] = str(ack["transport"])
            if "host" in ack: flat["task_ack_tcp_host"] = str(ack["host"])
            if "port" in ack: flat["task_ack_tcp_port"] = int(ack["port"])
            if "uds_path" in ack: flat["task_ack_uds_path"] = str(ack["uds_path"])
            if "ipc_socket_path" in ack: flat["task_ack_uds_path"] = str(ack["ipc_socket_path"])
        if "tts_event_in" in ipc and isinstance(ipc["tts_event_in"], dict):
            tts_ev = ipc["tts_event_in"]
            if "transport" in tts_ev: flat["tts_event_transport"] = str(tts_ev["transport"])
            if "host" in tts_ev: flat["tts_event_host"] = str(tts_ev["host"])
            if "port" in tts_ev: flat["tts_event_port"] = int(tts_ev["port"])
            if "uds_path" in tts_ev: flat["tts_event_uds_path"] = str(tts_ev["uds_path"])
            if "ipc_socket_path" in tts_ev: flat["tts_event_uds_path"] = str(tts_ev["ipc_socket_path"])

    # 9. runtime section
    if "runtime" in nested and isinstance(nested["runtime"], dict):
        run = nested["runtime"]
        for k in ("input_mode", "dry_run_text", "debug_input_only", "runs_dir", "logs_dir"):
            if k in run:
                flat[k] = run[k]

    if "mobile_feedback" in nested and isinstance(nested["mobile_feedback"], dict):
        feedback = nested["mobile_feedback"]
        for source, target in (("transport", "mobile_feedback_transport"), ("ipc_socket_path", "mobile_tts_event_uds_path"), ("uds_path", "mobile_tts_event_uds_path")):
            if source in feedback:
                flat[target] = str(feedback[source])
    if "playback" in nested and isinstance(nested["playback"], dict):
        playback = nested["playback"]
        for source, target in (("transport", "playback_transport"), ("ipc_socket_path", "playback_uds_path"), ("uds_path", "playback_uds_path"), ("start_timeout_s", "playback_start_timeout_s"), ("finish_timeout_s", "playback_finish_timeout_s"), ("post_guard_s", "post_playback_guard_s")):
            if source in playback:
                flat[target] = playback[source]

    # Map direct flat values
    for k in list(flat.keys()):
        if k in nested:
            val = nested[k]
            if val is not None:
                flat[k] = val

def load_voice_config(argv: Optional[List[str]] = None) -> VoiceServiceConfig:
    # 1. Start with schema defaults
    default_config = VoiceServiceConfig()
    flat = dataclasses.asdict(default_config)

    # Resolve default paths relative to repo root
    flat["project_root"] = str(REPO_ROOT / "Voice")
    flat["runs_dir"] = str(REPO_ROOT / "Voice" / "runs")
    flat["logs_dir"] = str(REPO_ROOT / "Voice" / "logs")
    flat["tts_cache"] = str(REPO_ROOT / "Voice" / "tts_cache")
    flat["tts_out_dir"] = str(REPO_ROOT / "Voice" / "tts_out")
    flat["commands_json"] = str(REPO_ROOT / "Voice" / "config" / "commands.json")

    # 2. Parse command line for config path or profile path first
    parser = argparse.ArgumentParser(description="Voice Gateway Service", add_help=False)
    parser.add_argument("--config", "-c", type=str, default=None)
    parser.add_argument("--profile", "-p", type=str, default=None)
    parser.add_argument("--dry-run-text", action="store_true", default=None)

    # We parse known args, ignoring the rest for now (they will be handled as direct overrides later)
    cli_args, remaining = parser.parse_known_args(argv if argv is not None else sys.argv[1:])

    # 3. Load global voice_gateway.yaml if it exists
    global_yaml_path = cli_args.config
    if not global_yaml_path:
        # Default fallback
        cand = REPO_ROOT / "Voice" / "config" / "voice_gateway.yaml"
        if cand.exists():
            global_yaml_path = str(cand)

    if global_yaml_path:
        nested_global = load_yaml_file(global_yaml_path)
        map_nested_dict(nested_global, flat)

    # 3b. Load model_manifest.yaml if it exists
    manifest_path = REPO_ROOT / "Voice" / "config" / "model_manifest.yaml"
    if manifest_path.exists():
        manifest = load_yaml_file(manifest_path)
        if "models" in manifest and isinstance(manifest["models"], dict):
            models = manifest["models"]
            if "asr" in models and isinstance(models["asr"], dict):
                asr = models["asr"]
                if asr.get("enabled", True):
                    if "model_path" in asr: flat["asr_dir"] = str(asr["model_path"])
                    if "vad_model_path" in asr: flat["vad_dir"] = str(asr["vad_model_path"])
                    if "backend" in asr: flat["asr_mode"] = "online" if "online" in str(asr["backend"]) else "offline"
            if "wake_kws" in models and isinstance(models["wake_kws"], dict):
                wake = models["wake_kws"]
                if wake.get("enabled", True):
                    if "model_path" in wake: flat["wake_tflite"] = str(wake["model_path"])
                    if "threshold" in wake: flat["wake_th"] = float(wake["threshold"])
                    if "frontend_backend" in wake: flat["frontend_backend"] = str(wake["frontend_backend"])
                    if "classifier_backend" in wake: flat["classifier_backend"] = str(wake["classifier_backend"])
            if "stop_kws" in models and isinstance(models["stop_kws"], dict):
                stop = models["stop_kws"]
                if stop.get("enabled", True):
                    if "model_path" in stop: flat["stop_tflite"] = str(stop["model_path"])
                    if "threshold" in stop: flat["stop_th"] = float(stop["threshold"])
            if "tts" in models and isinstance(models["tts"], dict):
                tts = models["tts"]
                if tts.get("enabled", True):
                    if "model_path" in tts: flat["piper_model"] = str(tts["model_path"])
                    if "config_path" in tts: flat["piper_config"] = str(tts["config_path"])

    # 4. Load profile voice_gateway section
    profile_path = cli_args.profile
    if not profile_path:
        # Try getting from environment
        profile_env = os.getenv("VOICE_PROFILE") or os.getenv("ROBOT_PROFILE")
        if profile_env:
            profile_path = profile_env

    if profile_path:
        # Check if it's a relative path in configs/profiles/
        p = Path(profile_path)
        if not p.is_absolute():
            cand = REPO_ROOT / "configs" / "profiles" / p.name
            if cand.exists():
                p = cand
            else:
                cand2 = REPO_ROOT / "configs" / "profiles" / (p.name + ".yaml")
                if cand2.exists():
                    p = cand2
        if p.exists():
            profile_yaml = load_yaml_file(p)
            # Pull runtime_overrides -> voice_gateway, or top-level voice_gateway
            overrides = profile_yaml.get("runtime_overrides", {})
            voice_section = overrides.get("voice_gateway", profile_yaml.get("voice_gateway", {}))
            map_nested_dict(voice_section, flat)

    # 5. Overwrite with environment variables
    # Legacy env mapping
    env_mappings = {
        "VOICE_WAKE_MODEL": "wake_tflite",
        "VOICE_STOP_MODEL": "stop_tflite",
        "VOICE_PIPER_MODEL": "piper_model",
        "VOICE_RUNS_DIR": "runs_dir",
        "VOICE_ASR_DIR": "asr_dir",
        "VOICE_VAD_DIR": "vad_dir",
        "VOICE_COMMANDS_JSON": "commands_json",
        "VOICE_ARECORD_DEVICE": "arecord_device",
        "VOICE_ASR_MODE": "asr_mode",
        "VOICE_ASR_ONLINE_CHUNK_FRAMES": "asr_online_chunk_frames",
        "VOICE_DISABLE_TTS": "disable_tts",
        "VOICE_TTS_MODE": "tts_mode",
        "VOICE_PLAY_CMD": "play_cmd",
        "VOICE_TASK_TRANSPORT": "task_transport",
        "VOICE_TASK_TCP_HOST": "task_tcp_host",
        "VOICE_TASK_TCP_PORT": "task_tcp_port",
        "VOICE_TASK_UDS_PATH": "task_uds_path",
        "VOICE_TASK_SEND_MODE": "task_send_mode",
        "VOICE_TASK_ACK_TRANSPORT": "task_ack_transport",
        "VOICE_TASK_ACK_TCP_HOST": "task_ack_tcp_host",
        "VOICE_TASK_ACK_TCP_PORT": "task_ack_tcp_port",
        "VOICE_TASK_ACK_UDS_PATH": "task_ack_uds_path",
        "VOICE_TTS_EVENT_TRANSPORT": "tts_event_transport",
        "VOICE_TTS_EVENT_HOST": "tts_event_host",
        "VOICE_TTS_EVENT_PORT": "tts_event_port",
        "VOICE_TTS_EVENT_UDS_PATH": "tts_event_uds_path",
        "VOICE_WAKE_TH": "wake_th",
        "VOICE_STOP_TH": "stop_th",
        "VOICE_ARMED_SECS": "armed_secs",
        "VOICE_FOLLOWUP_SECS": "followup_secs",
        "VOICE_STOP_FOLLOWUP_SECS": "stop_followup_secs",
        "VOICE_MAX_FOLLOWUP_TURNS": "max_followup_turns",
        "VOICE_MAX_REJECT_STREAK": "max_reject_streak",
        "VOICE_DEBUG": "debug",
        "VOICE_INPUT_MODE": "input_mode",
        "ROBOT_INPUT_MODE": "input_mode",
        "VOICE_DRY_RUN_TEXT": "dry_run_text"
    }

    for env_name, attr_name in env_mappings.items():
        val = os.getenv(env_name)
        if val is not None:
            # Parse to correct type
            field_type = VoiceServiceConfig.__dataclass_fields__[attr_name].type
            if field_type is bool:
                flat[attr_name] = val.strip().lower() in ("1", "true", "yes", "on")
            elif field_type is int:
                flat[attr_name] = int(val)
            elif field_type is float:
                flat[attr_name] = float(val)
            elif attr_name == "asr_online_chunk_size" and isinstance(val, str):
                flat[attr_name] = [int(x.strip()) for x in val.split(",") if x.strip()]
            else:
                flat[attr_name] = str(val).strip()

    # Generic VOICE_<attr_name> env parser
    for k in flat.keys():
        env_name = "VOICE_" + k.upper()
        val = os.getenv(env_name)
        if val is not None:
            field_type = VoiceServiceConfig.__dataclass_fields__[k].type
            if field_type is bool:
                flat[k] = val.strip().lower() in ("1", "true", "yes", "on")
            elif field_type is int:
                flat[k] = int(val)
            elif field_type is float:
                flat[k] = float(val)
            else:
                flat[k] = str(val).strip()

    # 6. Apply direct CLI --key value overrides (highest priority)
    # Parse remaining flags
    i = 0
    while i < len(remaining):
        arg = remaining[i]
        if arg.startswith("--"):
            key = arg[2:].replace("-", "_")
            if key in flat:
                field_type = VoiceServiceConfig.__dataclass_fields__[key].type
                if field_type is bool:
                    # check if next element is true/false or if it is just a flag
                    if i + 1 < len(remaining) and remaining[i+1].lower() in ("true", "false", "1", "0"):
                        flat[key] = remaining[i+1].lower() in ("true", "1")
                        i += 2
                    else:
                        flat[key] = True
                        i += 1
                else:
                    if i + 1 < len(remaining):
                        val = remaining[i+1]
                        if field_type is int:
                            flat[key] = int(val)
                        elif field_type is float:
                            flat[key] = float(val)
                        elif key == "asr_online_chunk_size":
                            flat[key] = [int(x.strip()) for x in val.split(",") if x.strip()]
                        else:
                            flat[key] = str(val)
                        i += 2
                    else:
                        i += 1
            else:
                i += 1
        else:
            i += 1

    if cli_args.dry_run_text is not None:
        flat["dry_run_text"] = bool(cli_args.dry_run_text)

    # Resolve and print verification
    from .paths import resolve_and_verify_model

    p_asr, _ = resolve_and_verify_model("asr", flat.get("asr_dir"), "VOICE_ASR_MODEL_PATH", "funasr")
    if p_asr: flat["asr_dir"] = str(p_asr)

    p_vad, _ = resolve_and_verify_model("vad", flat.get("vad_dir"), "VOICE_VAD_MODEL_PATH", "fsmn_vad")
    if p_vad: flat["vad_dir"] = str(p_vad)

    p_wake, _ = resolve_and_verify_model("wake_kws", flat.get("wake_tflite"), "VOICE_WAKE_MODEL_PATH", "openwakeword")
    if p_wake: flat["wake_tflite"] = str(p_wake)

    p_stop, _ = resolve_and_verify_model("stop_kws", flat.get("stop_tflite"), "VOICE_STOP_MODEL_PATH", "openwakeword")
    if p_stop: flat["stop_tflite"] = str(p_stop)

    p_tts, _ = resolve_and_verify_model("tts", flat.get("piper_model"), "VOICE_TTS_MODEL_PATH", "piper")
    if p_tts: flat["piper_model"] = str(p_tts)

    p_tts_cfg, _ = resolve_and_verify_model("tts_config", flat.get("piper_config"), "VOICE_TTS_CONFIG_PATH", "piper")
    if p_tts_cfg: flat["piper_config"] = str(p_tts_cfg)

    # Optional Mel / Embedding overrides checks
    if os.getenv("VOICE_KWS_MEL_MODEL_PATH"):
        resolve_and_verify_model("kws_frontend_mel", os.environ["VOICE_KWS_MEL_MODEL_PATH"], "VOICE_KWS_MEL_MODEL_PATH", "openwakeword")
    if os.getenv("VOICE_KWS_EMBEDDING_MODEL_PATH"):
        resolve_and_verify_model("kws_frontend_embed", os.environ["VOICE_KWS_EMBEDDING_MODEL_PATH"], "VOICE_KWS_EMBEDDING_MODEL_PATH", "openwakeword")

    # Resolve regular config folders
    for path_field in ("commands_json", "runs_dir", "logs_dir", "tts_cache", "tts_out_dir"):
        if flat.get(path_field):
            flat[path_field] = str(resolve_path(flat[path_field]))

    return VoiceServiceConfig(**flat)
