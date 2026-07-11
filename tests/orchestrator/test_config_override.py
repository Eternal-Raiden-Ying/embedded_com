#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
from pathlib import Path
from unittest.mock import MagicMock


ROOT = Path(__file__).resolve().parents[2]
ORCH_ROOT = ROOT / "orchestrator"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ORCH_ROOT) not in sys.path:
    sys.path.insert(0, str(ORCH_ROOT))

sys.modules["serial"] = MagicMock()

from common.config.loader import load_global_config


def test_legacy_stage_and_car_files_are_disabled():
    config = load_global_config()

    assert not hasattr(config.orchestrator.runtime, "stage_params_file")
    assert not hasattr(config.orchestrator.runtime, "car_cmd_params_file")
    assert all("stage_params.yaml" not in item for item in config.orchestrator.runtime.loaded_config_files)
    assert all("car_cmd_params.yaml" not in item for item in config.orchestrator.runtime.loaded_config_files)


def test_migrated_motion_values_match_v2_baseline():
    config = load_global_config()
    car = config.orchestrator.car
    control = config.orchestrator.control

    assert car.search_table_wz_radps == 0.20
    assert car.edge_slide_vy_mps == 0.010
    assert car.edge_slide_max_vx_mps == 0.008
    assert car.edge_slide_max_wz_radps == 0.080
    assert car.send_period_ms == 100
    assert car.uart_keepalive_hz == 10.0
    assert car.min_uart_keepalive_hz == 7.0
    assert car.motion_hold_ms == 400
    assert car.hard_stale_stop_ms == 800
    assert control.table_target_dist_m == 0.30
    assert control.roi_final_stop_p10_m == 0.42
    assert control.depth_envelope_stop_p10_m == 0.35
