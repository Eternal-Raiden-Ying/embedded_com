from pathlib import Path
from types import SimpleNamespace
import sys

ROOT = Path(__file__).resolve().parents[2]
ORCH_ROOT = ROOT / "orchestrator"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ORCH_ROOT) not in sys.path:
    sys.path.insert(0, str(ORCH_ROOT))

from common.config.loader import load_global_config
from orchestrator_service.mobile_gateway.protocol import normalize_tts_event
from orchestrator_service.runtime.tts_feedback import EVENT_CATALOG, TtsFeedbackEmitter


def policy(**overrides):
    values = dict(enabled=True, verbosity="detailed", min_interval_s=0.0,
                  default_ttl_s=8.0, dedupe_window_s=30.0,
                  detailed_progress_enabled=True)
    values.update(overrides)
    return SimpleNamespace(**values)


def emit(emitter, pending, key, session="session-1", epoch=1, target="apple", **kwargs):
    return emitter.emit(pending, key, session_id=session, epoch=epoch, target=target, **kwargs)


def test_off_suppresses_business_but_allows_forced_stop():
    pending = []
    emitter = TtsFeedbackEmitter(policy(enabled=False, verbosity="off"))
    assert emit(emitter, pending, "TASK_ACCEPTED") is None
    stop = emit(emitter, pending, "TASK_STOPPED", force=True)
    assert stop["priority"] == "P0"
    assert stop["interrupt"] is True


def test_basic_filters_progress_and_detailed_emits_progress():
    basic_pending, detailed_pending = [], []
    assert emit(TtsFeedbackEmitter(policy(verbosity="basic")), basic_pending, "SEARCH_TABLE_STARTED") is None
    emitter = TtsFeedbackEmitter(policy(verbosity="detailed"))
    assert emit(emitter, detailed_pending, "SEARCH_TABLE_STARTED")["event_key"] == "SEARCH_TABLE_STARTED"
    assert emit(emitter, detailed_pending, "PICK_PREPARING")["event_key"] == "PICK_PREPARING"


def test_detailed_emits_each_catalogued_progress_stage_once():
    pending, emitter = [], TtsFeedbackEmitter(policy(min_interval_s=0.0))
    progress = (
        "SEARCH_TABLE_STARTED", "TABLE_FOUND", "TABLE_APPROACH_STARTED",
        "TABLE_EDGE_REACHED", "SEARCH_TARGET_STARTED", "TARGET_LOCKED",
        "PICK_PREPARING", "PICK_EXECUTING",
    )
    for key in progress:
        assert emit(emitter, pending, key)
    assert [item["event_key"] for item in pending] == list(progress)


def test_dedup_epoch_and_session_scope():
    pending, emitter = [], TtsFeedbackEmitter(policy())
    assert emit(emitter, pending, "SEARCH_TABLE_STARTED")
    assert emit(emitter, pending, "SEARCH_TABLE_STARTED") is None
    assert emit(emitter, pending, "SEARCH_TABLE_STARTED", epoch=2)
    assert emit(emitter, pending, "SEARCH_TABLE_STARTED", session="session-2")


def test_progress_min_interval_and_bounded_ttl():
    ticks = iter((10.0, 10.1, 12.0))
    emitter = TtsFeedbackEmitter(policy(min_interval_s=1.5), now=lambda: 100.0, monotonic=lambda: next(ticks))
    pending = []
    assert emit(emitter, pending, "SEARCH_TABLE_STARTED")
    assert emit(emitter, pending, "TABLE_FOUND") is None
    event = emit(emitter, pending, "TABLE_FOUND")
    assert event["ttl_s"] == 5.0
    assert event["expires_at"] == 105.0


def test_structured_fields_priority_target_and_switch_copy():
    pending, emitter = [], TtsFeedbackEmitter(policy())
    accepted = emit(emitter, pending, "TASK_ACCEPTED", target="apple")
    switched = emit(emitter, pending, "TASK_SWITCHED", session="session-2", target="bottle", old_target="apple")
    success = emit(emitter, pending, "PICK_SUCCEEDED", session="session-3", target="kiwi_fruit")
    basket = emit(emitter, pending, "TASK_ACCEPTED", session="session-4", target="basket")
    assert accepted["text"] == "\u5df2\u6536\u5230\uff0c\u51c6\u5907\u5bfb\u627e\u82f9\u679c\u3002"
    assert switched["text"] == "\u5df2\u53d6\u6d88\u82f9\u679c\u4efb\u52a1\uff0c\u6b63\u5728\u6539\u4e3a\u5bfb\u627e\u6c34\u74f6\u3002"
    assert success["text"] == "\u5df2\u7ecf\u62ff\u5230\u7315\u7334\u6843\u3002"
    assert basket["text"] == "\u5df2\u6536\u5230\uff0c\u51c6\u5907\u5bfb\u627e\u6536\u7eb3\u7b50\u3002"
    assert accepted["dedupe_key"] == "session-1:1:TASK_ACCEPTED:apple"
    assert accepted["created_at"] <= accepted["expires_at"]
    assert {accepted["priority"], success["priority"]} == {"P2", "P1"}


def test_catalog_has_no_remote_or_cloud_grasp_wording():
    wording = " ".join(template for template, _, _ in EVENT_CATALOG.values()).lower()
    assert "remote" not in wording
    assert "\\u4e91\\u7aef" not in wording
    assert "\\u89c4\\u5212\\u6293\\u53d6" not in wording


def test_gateway_preserves_structured_event_fields_without_rewriting_text():
    raw = emit(TtsFeedbackEmitter(policy()), [], "TASK_ACCEPTED")
    normalized = normalize_tts_event(raw)
    for field in ("event_id", "session_id", "text", "priority", "ts", "epoch", "event_key", "dedupe_key", "created_at", "expires_at"):
        assert normalized[field] == raw[field]


def test_phone_profiles_default_to_detailed_and_generic_dry_run_stays_disabled(monkeypatch):
    monkeypatch.setattr("common.config.validators.platform.system", lambda: "Linux")
    root = ROOT / "configs" / "system_config.yaml"
    monkeypatch.setenv("SYSTEM_CONFIG_PROFILE", "sc171_voice_phone_tts")
    phone = load_global_config(str(root))
    monkeypatch.setenv("SYSTEM_CONFIG_PROFILE", "sc171_voice_phone_tts_dryrun")
    phone_dry = load_global_config(str(root))
    monkeypatch.setenv("SYSTEM_CONFIG_PROFILE", "dry_run")
    generic = load_global_config(str(root))
    for config in (phone, phone_dry):
        assert config.orchestrator.tts_feedback.enabled is True
        assert config.orchestrator.tts_feedback.verbosity == "detailed"
    assert generic.orchestrator.tts_event_out.transport == "disabled"
