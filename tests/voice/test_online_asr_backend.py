from types import SimpleNamespace
from pathlib import Path

import pytest

from voice_service.config.loader import load_voice_config
from voice_service.runtime.asr_engine import OfflineASREngine, OnlineASREngine, create_asr_backend
from voice_service.runtime.state import RuntimeState
from voice_service.runtime.workers import ASRDecisionWorker
from voice_service.app.main import check_models


def test_offline_phone_profile_remains_offline_and_online_profiles_are_explicit():
    offline = load_voice_config(["--profile", "configs/profiles/sc171_voice_phone_tts.yaml"])
    online = load_voice_config(["--profile", "configs/profiles/sc171_voice_phone_tts_online_asr.yaml"])
    dryrun = load_voice_config(["--profile", "configs/profiles/sc171_voice_phone_tts_online_asr_dryrun.yaml"])
    assert offline.asr_mode == "offline"
    assert online.asr_mode == dryrun.asr_mode == "online"
    assert online.asr_dir.endswith("vocab8404-online-onnx")
    assert online.asr_online_chunk_size == [5, 10, 5]
    assert online.asr_quant is True
    assert online.asr_online_encoder_chunk_look_back == 4
    assert online.asr_online_decoder_chunk_look_back == 1


def test_backend_factory_selects_one_backend_per_mode():
    offline_cfg = SimpleNamespace(asr_mode="offline", asr_dir="", asr_quant=False)
    online_cfg = SimpleNamespace(
        asr_mode="online", asr_dir="", asr_quant=True, asr_online_chunk_size=[5, 10, 5],
        asr_online_encoder_chunk_look_back=4, asr_online_decoder_chunk_look_back=1,
    )
    assert isinstance(create_asr_backend(offline_cfg, dry_run_text=True), OfflineASREngine)
    online = create_asr_backend(online_cfg, dry_run_text=True)
    assert isinstance(online, OnlineASREngine)
    assert online.step_samples == 9600


def test_online_sessions_have_independent_cache_and_abort_clears_it():
    cfg = SimpleNamespace(
        asr_mode="online", asr_dir="", asr_quant=True, asr_online_chunk_size=[5, 10, 5],
        asr_online_encoder_chunk_look_back=4, asr_online_decoder_chunk_look_back=1,
    )
    backend = create_asr_backend(cfg, dry_run_text=True)
    first, second = backend.create_session(), backend.create_session()
    first.cache["text"] = "苹果"
    assert second.cache == {}
    backend.close_session(first)
    assert first.cache == {}
    assert first.backend is None


class _Pipeline:
    def __init__(self):
        self.interpreter = SimpleNamespace(is_residual_text=lambda _text: False)
        self.aborted = 0

    def start_stream_session(self):
        return SimpleNamespace(finalized=False, cache={})

    def stream_feed(self, session, _audio, is_final=False):
        session.finalized = bool(is_final)
        return {"merged_text": "找苹果", "feed_latency_ms": 1.0}

    def finalize_stream_result(self, _session):
        return {"status": "REJECT", "text": "找苹果"}

    def abort_stream_session(self, _session):
        self.aborted += 1


def _online_worker():
    worker = object.__new__(ASRDecisionWorker)
    worker.cfg = SimpleNamespace(asr_emit_partial=True)
    worker.rt = RuntimeState()
    worker.pipeline = _Pipeline()
    worker.interpreter = worker.pipeline.interpreter
    worker.stream_session = None
    worker.stream_epoch = None
    worker.last_partial_text = ""
    worker.say_text = lambda _text: None
    worker._handle_result = lambda _result: {"keep_alive": False, "tts": ""}
    return worker


def test_start_chunk_final_only_final_reaches_decision_and_is_unique():
    worker = _online_worker()
    epoch = worker.rt.get_epoch()
    assert worker._handle_online_event({"kind": "START", "epoch": epoch, "session_id": "s"})["finalize_turn"] is False
    assert worker._handle_online_event({"kind": "CHUNK", "epoch": epoch, "audio": [], "seq": 0})["finalize_turn"] is False
    assert worker.last_partial_text == "找苹果"
    assert worker._handle_online_event({"kind": "FINAL", "epoch": epoch, "audio": []})["finalize_turn"] is True
    assert worker.stream_session is None
    assert worker._handle_online_event({"kind": "FINAL", "epoch": epoch, "audio": []})["finalize_turn"] is False


def test_stale_online_chunk_and_final_are_dropped_and_abort_clears_session():
    worker = _online_worker()
    epoch = worker.rt.get_epoch()
    worker._handle_online_event({"kind": "START", "epoch": epoch, "session_id": "s"})
    worker.rt.bump_epoch()
    assert worker._handle_online_event({"kind": "CHUNK", "epoch": epoch, "audio": []})["finalize_turn"] is False
    assert worker._handle_online_event({"kind": "FINAL", "epoch": epoch, "audio": []})["finalize_turn"] is False
    assert worker._handle_online_event({"kind": "ABORT", "epoch": worker.rt.get_epoch()})["finalize_turn"] is False
    assert worker.stream_session is None
    assert worker.pipeline.aborted >= 1


def test_online_missing_model_fails_explicitly_but_offline_is_independent(tmp_path: Path):
    wake = tmp_path / "wake.onnx"
    stop = tmp_path / "stop.onnx"
    wake.touch()
    stop.touch()
    base = dict(wake_tflite=str(wake), stop_tflite=str(stop), vad_dir=str(tmp_path), piper_model="",
                piper_config="", disable_tts=True, dry_run_text=False)
    with pytest.raises(SystemExit):
        check_models(SimpleNamespace(**base, asr_mode="online", asr_dir=str(tmp_path / "missing-online")))
    check_models(SimpleNamespace(**base, asr_mode="offline", asr_dir=str(tmp_path)))
