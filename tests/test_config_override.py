#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import unittest
import sys
from pathlib import Path
from unittest.mock import patch, MagicMock

# Add workspace and orchestrator roots to sys.path
ROOT = Path(__file__).resolve().parents[1]
ORCH_ROOT = ROOT / "orchestrator"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ORCH_ROOT) not in sys.path:
    sys.path.insert(0, str(ORCH_ROOT))

# Mock serial module
sys.modules["serial"] = MagicMock()

from common.config_loader import load_global_config, validate_config, get_config
from common.schema import SystemGlobalConfig


class ConfigOverrideTest(unittest.TestCase):
    def test_default_schema_slide_vy_mps(self):
        """1. Default schema has edge_slide_vy_mps == 0.14"""
        config = SystemGlobalConfig()
        self.assertAlmostEqual(config.orchestrator.car.edge_slide_vy_mps, 0.14)

    def test_stage_params_override(self):
        """2. Loading stage_params.yaml overrides edge_slide_vy_mps to 0.010"""
        # We simulate loading by setting stage_params_file manually and loading it
        config = SystemGlobalConfig()
        config.orchestrator.runtime.stage_params_file = "orchestrator/configs/stage_params.yaml"
        
        # Call the private helper directly to verify mapping
        from common.config_loader import _load_and_merge_stage_params
        stage_path = ROOT / "orchestrator" / "configs" / "stage_params.yaml"
        self.assertTrue(stage_path.is_file())
        
        _load_and_merge_stage_params(config, stage_path)
        self.assertAlmostEqual(config.orchestrator.car.edge_slide_vy_mps, 0.010)

    def test_edge_slide_max_vx_wz_override(self):
        """3. edge_slide_max_vx_mps and edge_slide_max_wz_radps are overridden using actual YAML keys"""
        config = SystemGlobalConfig()
        from common.config_loader import _load_and_merge_stage_params
        stage_path = ROOT / "orchestrator" / "configs" / "stage_params.yaml"
        _load_and_merge_stage_params(config, stage_path)
        
        # In stage_params.yaml, edge_slide_search:
        #   max_vx_correction_mps: 0.008
        #   max_wz_correction_radps: 0.080
        self.assertAlmostEqual(config.orchestrator.car.edge_slide_max_vx_mps, 0.008)
        self.assertAlmostEqual(config.orchestrator.car.edge_slide_max_wz_radps, 0.080)

    def test_final_lock_dist_abs_th_m_mapping(self):
        """4. final_lock.dist_abs_th_m maps to final_lock_dist_tol_m"""
        config = SystemGlobalConfig()
        from common.config_loader import _load_and_merge_stage_params
        stage_path = ROOT / "orchestrator" / "configs" / "stage_params.yaml"
        _load_and_merge_stage_params(config, stage_path)
        
        # In stage_params.yaml, final_lock:
        #   dist_abs_th_m: 0.05
        self.assertAlmostEqual(config.orchestrator.control.final_lock_dist_tol_m, 0.05)

    def test_missing_config_raises_file_not_found(self):
        """5. Missing explicitly specified config file raises FileNotFoundError"""
        with patch.dict(os.environ, {"ORCH_STAGE_PARAMS_FILE": "nonexistent_stage_params.yaml"}):
            with self.assertRaises(FileNotFoundError):
                load_global_config()

    def test_validate_config_warning(self):
        """6. validate_config() warns when dangerous defaults remain"""
        config = SystemGlobalConfig()
        # Set dry_run to True so it triggers warning instead of exception
        config.orchestrator.serial.dry_run = True
        config.orchestrator.car.edge_slide_vy_mps = 0.14
        
        with patch('sys.stderr') as mock_stderr:
            validate_config(config)
            # Ensure validate_config logged some warning content to stderr
            self.assertTrue(mock_stderr.write.called)

        # In production mode (dry_run == False), it must raise ValueError
        config.orchestrator.serial.dry_run = False
        with self.assertRaises(ValueError):
            validate_config(config, force_production=True)


if __name__ == "__main__":
    unittest.main()
