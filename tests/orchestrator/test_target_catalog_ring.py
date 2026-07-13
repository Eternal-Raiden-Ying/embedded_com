from pathlib import Path
from types import SimpleNamespace
import sys

ROOT = Path(__file__).resolve().parents[2]
ORCH_ROOT = ROOT / "orchestrator"
for path in (ROOT, ORCH_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from common.target_catalog import load_target_catalog, resolve_target, resolve_target_in_text, validate_target_catalog
from orchestrator_service.bridge.uart_bridge import UartBridge
from orchestrator_service.runtime.target_policy import route_locked_target
from orchestrator_service.config.schema import OrchestratorConfig
from orchestrator_service.runtime.context import State
from orchestrator_service.runtime.state_machine import OrchestratorCore
from orchestrator_service.runtime.tts_feedback import TtsFeedbackEmitter


def _policy(**values):
    base = dict(enabled=True, verbosity="detailed", min_interval_s=0.0, default_ttl_s=8.0, dedupe_window_s=30.0, detailed_progress_enabled=True)
    base.update(values)
    return SimpleNamespace(**base)


def test_trained_catalog_has_exact_unique_ids_and_safe_recipes():
    catalog = load_target_catalog()
    assert len(catalog) == 15
    assert {spec.class_id for spec in catalog.values()} == set(range(15))
    assert catalog["apple"].class_id == 2
    assert catalog["basket"].class_id == 4
    assert catalog["bottle"].class_id == 5
    assert catalog["water_dispenser"].class_id == 14
    assert catalog["apple"].grasp_recipe == "apple"
    assert catalog["bottle"].grasp_recipe == "bottle"
    assert all(spec.grasp_recipe is None for spec in catalog.values() if spec.action_policy in {"docking_only", "place_destination", "locate_and_ring"})
    assert resolve_target("kiwi_fruit") is None


def test_catalog_aliases_are_longest_and_non_fetch_targets_are_hidden():
    assert resolve_target_in_text("\u627e\u836f\u74f6").canonical_name == "pill_bottle"
    assert resolve_target_in_text("\u627e\u996e\u6c34\u673a").canonical_name == "water_dispenser"
    assert resolve_target_in_text("\u627e\u4e00\u53f7\u684c\u9762") is None
    assert resolve_target_in_text("\u627e\u6536\u7eb3\u7b50") is None


def test_validator_rejects_alias_collisions_and_missing_ready_recipe():
    raw = {"targets": {name: {**spec.__dict__, "aliases": list(spec.aliases)} for name, spec in load_target_catalog().items()}}
    raw["targets"]["apple"]["aliases"] = ["\u6c34\u74f6"]
    try:
        validate_target_catalog(raw)
        assert False, "collision must be rejected"
    except ValueError as exc:
        assert "collision" in str(exc)
    raw = {"targets": {name: {**spec.__dict__, "aliases": list(spec.aliases)} for name, spec in load_target_catalog().items()}}
    raw["targets"]["apple"]["grasp_recipe"] = None
    try:
        validate_target_catalog(raw)
        assert False, "ready recipe must be required"
    except ValueError as exc:
        assert "requires grasp_recipe" in str(exc)


def test_policy_routes_without_arm_fallbacks():
    assert route_locked_target(load_target_catalog()["apple"]).next_state == "GRASP"
    assert route_locked_target(load_target_catalog()["banana"]).tts_event == "RECIPE_NOT_READY"
    assert route_locked_target(load_target_catalog()["tea"]).tts_event == "UNSUPPORTED_GRASP_TARGET"
    assert route_locked_target(load_target_catalog()["water_dispenser"]).next_state == "LOCATE_GUIDANCE_ACTIVE"


def test_ring_commands_are_idempotent_and_dry_run_only():
    sent = []
    uart = UartBridge("DRY_RUN", 115200, 0.1, dry_run=True, tx_callback=lambda line, dry, meta: sent.append((line, dry, meta)))
    assert uart.ring_start("water_dispenser")
    assert uart.ring_start("water_dispenser")
    assert uart.ring_active and uart.ring_target == "water_dispenser"
    assert uart.ring_stop()
    assert uart.ring_stop()
    assert [line for line, _, _ in sent] == ["RING\r\n", "RING_STOP\r\n"]
    assert all(dry for _, dry, _ in sent)
    assert uart.handle_ring_ack("OK RING")
    assert uart.handle_ring_ack("OK RING_STOP")


def test_locate_tts_catalog_priority_ttl_and_dedupe():
    emitter = TtsFeedbackEmitter(_policy())
    out = []
    guidance = emitter.emit(out, "LOCATE_GUIDANCE_STARTED", session_id="s", epoch=1, target="water_dispenser")
    assert guidance["priority"] == "P1" and guidance["ttl_s"] == 15.0
    assert emitter.emit(out, "LOCATE_GUIDANCE_STARTED", session_id="s", epoch=1, target="water_dispenser") is None
    assert emitter.emit(out, "LOCATE_GUIDANCE_STARTED", session_id="s", epoch=2, target="water_dispenser")


def test_locate_waits_for_matching_tts_finished_before_waiting_for_user():
    cfg = OrchestratorConfig()
    core = OrchestratorCore(cfg.control, cfg.car, cfg.docking)
    core.ctx.active_target = "water_dispenser"
    core.ctx.canonical_target = "water_dispenser"
    core.ctx.active_session_id = "session-locate"
    core.ctx.active_epoch = 4
    core._transition(State.LOCATE_GUIDANCE_ACTIVE, "test")
    event_id = core.ctx.locate_guidance_event_id
    assert event_id
    assert not core.handle_tts_playback_state({"type": "tts_playback_state", "state": "finished", "event_id": "wrong", "session_id": "session-locate", "epoch": 4})
    assert core.ctx.state == State.LOCATE_GUIDANCE_ACTIVE
    assert core.handle_tts_playback_state({"type": "tts_playback_state", "state": "finished", "event_id": event_id, "session_id": "session-locate", "epoch": 4})
    core.tick()
    assert core.ctx.state == State.WAIT_USER_APPROACH


def test_vista_uses_catalog_class_ids_and_observation_metadata():
    from VISTA.vision_module.config.data import FINETUNE_YOLO26S_BGR15_CLASSES
    from VISTA.vision_module.utils.detect import compute_target_obs
    assert tuple(FINETUNE_YOLO26S_BGR15_CLASSES) == tuple(spec.canonical_name for spec in sorted(load_target_catalog().values(), key=lambda item: item.class_id))
    obs = compute_target_obs((100, 100, 3), "water_dispenser", [[1, 2, 50, 60, 0.9, 14]], class_names=FINETUNE_YOLO26S_BGR15_CLASSES)
    assert obs["canonical_target"] == "water_dispenser"
    assert obs["class_id"] == 14 and obs["model_class_name"] == "water_dispenser"

def test_phone_profiles_are_detailed_and_generic_dry_run_does_not_enable_phone_tts(monkeypatch):
    from common.config.loader import load_global_config
    monkeypatch.setenv("SYSTEM_CONFIG_PROFILE", "sc171_voice_phone_tts_dryrun")
    phone = load_global_config(str(ROOT / "configs" / "system_config.yaml"))
    assert phone.orchestrator.tts_feedback.verbosity == "detailed"
    assert phone.orchestrator.serial.dry_run is True
    assert phone.orchestrator.tts_playback_in.transport == "uds"
    assert phone.gateway.orchestrator_tts_playback_out.transport == "uds"
    monkeypatch.setenv("SYSTEM_CONFIG_PROFILE", "windows_dev")
    generic = load_global_config(str(ROOT / "configs" / "system_config.yaml"))
    assert generic.orchestrator.serial.dry_run is True
    assert generic.orchestrator.tts_event_out.transport == "disabled"
