#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import platform
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
ORCH_ROOT = ROOT / "orchestrator"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ORCH_ROOT) not in sys.path:
    sys.path.insert(0, str(ORCH_ROOT))

from common.config.effective_dump import format_effective_config
from common.config.loader import load_global_config
from common.config.schema import SystemGlobalConfig
from common.config.validators import validate_config


class ConfigLoaderLayeringTest(unittest.TestCase):
    def _load_default(self):
        with patch.dict(os.environ, {"SYSTEM_CONFIG_PROFILE": "windows_dev"}, clear=False):
            return load_global_config(str(ROOT / "configs" / "system_config.yaml"))

    def test_default_load_uses_windows_profile_and_runtime_param_files(self):
        cfg = self._load_default()

        self.assertEqual(cfg.profile, "windows_dev")
        self.assertEqual(cfg.orchestrator.runtime.config_profile, "windows_dev")
        self.assertTrue(cfg.orchestrator.serial.dry_run)
        self.assertEqual(cfg.orchestrator.serial.port, "COM_DRY_RUN")
        self.assertEqual(cfg.orchestrator.runtime.stage_params_file, "orchestrator/configs/stage_params.yaml")
        self.assertEqual(cfg.orchestrator.runtime.car_cmd_params_file, "orchestrator/configs/car_cmd_params.yaml")
        self.assertAlmostEqual(cfg.orchestrator.car.edge_slide_vy_mps, 0.010)
        self.assertAlmostEqual(cfg.orchestrator.car.edge_slide_max_vx_mps, 0.008)

        loaded = "\n".join(cfg.orchestrator.runtime.loaded_config_files)
        self.assertIn("system_config.yaml", loaded)
        self.assertIn("windows_dev.yaml", loaded)
        self.assertIn("stage_params.yaml", loaded)
        self.assertIn("car_cmd_params.yaml", loaded)

    def test_effective_dump_contains_required_runtime_fields(self):
        cfg = self._load_default()
        dump = format_effective_config(cfg, effective_dry_run=True)

        for expected in (
            "config profile",
            "loaded config files",
            "loaded stage_params",
            "loaded car_cmd_params",
            "UART port / dry_run",
            "tick_hz",
            "near_stop_depth_m",
            "edge_slide_vy_mps",
            "edge_slide_max_vx_mps",
            "final_lock yaw/dist/lateral",
            "STOP/SSTOP policy",
            "vision control_obs interval",
        ):
            self.assertIn(expected, dump)
        self.assertIn("windows_dev", dump)
        self.assertIn("0.010", dump)

    def test_validate_rejects_windows_uds_for_real_runtime(self):
        cfg = SystemGlobalConfig()
        cfg.orchestrator.serial.dry_run = False
        cfg.orchestrator.car.edge_slide_vy_mps = 0.010

        with patch("platform.system", return_value="Windows"):
            with self.assertRaisesRegex(ValueError, "UDS"):
                validate_config(cfg)

    def test_validate_rejects_stop_sstop_policy_mixup(self):
        cfg = SystemGlobalConfig()
        cfg.orchestrator.serial.dry_run = True
        cfg.orchestrator.car.edge_slide_vy_mps = 0.010
        cfg.orchestrator.car.soft_stop_command = "STOP"

        with self.assertRaisesRegex(ValueError, "soft_stop_command"):
            validate_config(cfg)

    def test_sc171_board_profile_keeps_real_uart_on_non_windows(self):
        if platform.system().lower().startswith("win"):
            self.skipTest("sc171_board profile intentionally uses UDS and is board-only")

        env = {"SYSTEM_CONFIG_PROFILE": "sc171_board"}
        with patch.dict(os.environ, env, clear=False):
            cfg = load_global_config(str(ROOT / "configs" / "system_config.yaml"))

        self.assertEqual(cfg.profile, "sc171_board")
        self.assertFalse(cfg.orchestrator.serial.dry_run)
        self.assertEqual(cfg.orchestrator.serial.port, "/dev/ttyHS1")

    def test_dry_run_profile_disables_external_ack_and_vision_request_outputs(self):
        env = {"SYSTEM_CONFIG_PROFILE": "dry_run"}
        with patch.dict(os.environ, env, clear=False):
            cfg = load_global_config(str(ROOT / "configs" / "system_config.yaml"))

        self.assertEqual(cfg.profile, "dry_run")
        self.assertTrue(cfg.orchestrator.serial.dry_run)
        self.assertEqual(cfg.orchestrator.task_ack_out.transport, "disabled")
        self.assertEqual(cfg.orchestrator.vision_req_out.transport, "disabled")
        self.assertFalse(cfg.orchestrator.control.vision_req_fail_to_stop)


if __name__ == "__main__":
    unittest.main()
