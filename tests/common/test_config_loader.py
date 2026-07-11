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

    def test_default_load_uses_windows_profile_without_legacy_param_files(self):
        cfg = self._load_default()

        self.assertEqual(cfg.profile, "windows_dev")
        self.assertEqual(cfg.orchestrator.runtime.config_profile, "windows_dev")
        self.assertTrue(cfg.orchestrator.serial.dry_run)
        self.assertEqual(cfg.orchestrator.serial.port, "COM_DRY_RUN")
        self.assertFalse(hasattr(cfg.orchestrator.runtime, "stage_params_file"))
        self.assertFalse(hasattr(cfg.orchestrator.runtime, "car_cmd_params_file"))
        self.assertAlmostEqual(cfg.orchestrator.car.edge_slide_vy_mps, 0.010)
        self.assertAlmostEqual(cfg.orchestrator.car.edge_slide_max_vx_mps, 0.008)

        loaded = "\n".join(cfg.orchestrator.runtime.loaded_config_files)
        self.assertIn("system_config.yaml", loaded)
        self.assertIn("windows_dev.yaml", loaded)
        self.assertNotIn("stage_params.yaml", loaded)
        self.assertNotIn("car_cmd_params.yaml", loaded)

    def test_effective_dump_contains_required_runtime_fields(self):
        cfg = self._load_default()
        dump = format_effective_config(cfg, effective_dry_run=True)

        for expected in (
            "config profile",
            "loaded config files",
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

    def test_gateway_env_overrides_match_stack_launcher_socket_names(self):
        env = {
            "SYSTEM_CONFIG_PROFILE": "windows_dev",
            "MOBILE_GATEWAY_BACKEND": "orchestrator_uds",
            "MOBILE_GATEWAY_COMMAND_IN_TRANSPORT": "uds",
            "MOBILE_GATEWAY_COMMAND_IN_SOCKET_PATH": "/tmp/robot_stack/mobile_gateway_cmd.sock",
            "MOBILE_GATEWAY_ORCH_TASK_CMD_TRANSPORT": "uds",
            "MOBILE_GATEWAY_ORCH_TASK_CMD_SOCKET_PATH": "/tmp/robot_stack/task_cmd.sock",
            "MOBILE_GATEWAY_ORCH_TASK_ACK_TRANSPORT": "uds",
            "MOBILE_GATEWAY_ORCH_TASK_ACK_SOCKET_PATH": "/tmp/robot_stack/task_ack.sock",
            "MOBILE_GATEWAY_ORCH_STATE_BLOCKS_PATH": "/tmp/robot_stack/run/orchestrator/state_blocks.jsonl",
        }
        with patch.dict(os.environ, env, clear=False):
            cfg = load_global_config(str(ROOT / "configs" / "system_config.yaml"))

        self.assertEqual(cfg.gateway.backend.mode, "orchestrator_uds")
        self.assertEqual(cfg.gateway.command_in.transport, "uds")
        self.assertEqual(cfg.gateway.command_in.ipc_socket_path, "/tmp/robot_stack/mobile_gateway_cmd.sock")
        self.assertEqual(cfg.gateway.orchestrator_task_cmd_out.transport, "uds")
        self.assertEqual(cfg.gateway.orchestrator_task_cmd_out.ipc_socket_path, "/tmp/robot_stack/task_cmd.sock")
        self.assertEqual(cfg.gateway.orchestrator_task_ack_in.transport, "uds")
        self.assertEqual(cfg.gateway.orchestrator_task_ack_in.ipc_socket_path, "/tmp/robot_stack/task_ack.sock")
        self.assertEqual(cfg.gateway.backend.state_blocks_path, "/tmp/robot_stack/run/orchestrator/state_blocks.jsonl")

    def test_gateway_orchestrator_uds_uses_real_backend_and_ack_server(self):
        from common.config.schema import SystemGlobalConfig
        from orchestrator_service.mobile_gateway.runtime.service import MobileGatewayService, TcpTaskCmdBackend

        cfg = SystemGlobalConfig().gateway
        cfg.runtime.log_enabled = False
        cfg.backend.mode = "orchestrator_uds"
        cfg.orchestrator_task_cmd_out.transport = "uds"
        cfg.orchestrator_task_cmd_out.ipc_socket_path = "/tmp/robot_stack/task_cmd.sock"
        cfg.orchestrator_task_ack_in.transport = "uds"
        cfg.orchestrator_task_ack_in.ipc_socket_path = "/tmp/robot_stack/task_ack.sock"

        service = MobileGatewayService(cfg)

        self.assertEqual(service.backend_mode, "orchestrator_uds")
        self.assertIsInstance(service.backend, TcpTaskCmdBackend)
        self.assertIsNotNone(service.ack_server)


    def test_sc171_board_profile_keeps_real_uart_on_non_windows(self):
        if platform.system().lower().startswith("win"):
            self.skipTest("sc171_board profile intentionally uses UDS and is board-only")

        env = {"SYSTEM_CONFIG_PROFILE": "sc171_board"}
        with patch.dict(os.environ, env, clear=False):
            cfg = load_global_config(str(ROOT / "configs" / "system_config.yaml"))

        self.assertEqual(cfg.profile, "sc171_board")
        self.assertFalse(cfg.orchestrator.serial.dry_run)
        self.assertEqual(cfg.orchestrator.serial.port, "/dev/ttyHS1")
        self.assertTrue(cfg.orchestrator.control.yolo_table_control_enable)
        self.assertTrue(cfg.orchestrator.control.vision_req_fail_to_stop)
        self.assertAlmostEqual(cfg.orchestrator.control.no_table_bbox_timeout_s, 10.0)
        self.assertAlmostEqual(cfg.orchestrator.control.edge_geometry_timeout_s, 10.0)
        self.assertTrue(cfg.orchestrator.control.keep_vision_alive_after_task)
        self.assertFalse(cfg.orchestrator.control.task_done_shutdown_vision)
        self.assertTrue(cfg.vision.runtime.keep_vision_alive_after_task)
        self.assertTrue(cfg.vision.runtime.keep_preview_alive_after_task)
        self.assertFalse(cfg.vision.runtime.release_model_on_idle)

    def test_dry_run_profile_keeps_vision_link_enabled_but_disables_external_ack(self):
        env = {"SYSTEM_CONFIG_PROFILE": "dry_run"}
        with patch.dict(os.environ, env, clear=False):
            cfg = load_global_config(str(ROOT / "configs" / "system_config.yaml"))

        self.assertEqual(cfg.profile, "dry_run")
        self.assertTrue(cfg.orchestrator.serial.dry_run)
        self.assertTrue(cfg.vision.debug.preview)
        self.assertTrue(cfg.vision.preview.show_rgb)
        self.assertEqual(cfg.vision.req_in.transport, "uds")
        self.assertEqual(cfg.vision.req_in.ipc_socket_path, "/tmp/robot_stack/vision_req.sock")
        self.assertEqual(cfg.vision.obs_out.transport, "uds")
        self.assertEqual(cfg.vision.obs_out.ipc_socket_path, "/tmp/robot_stack/vision_obs.sock")
        self.assertEqual(cfg.orchestrator.task_ack_out.transport, "disabled")
        self.assertEqual(cfg.orchestrator.vision_obs_in.transport, "uds")
        self.assertEqual(cfg.orchestrator.vision_obs_in.ipc_socket_path, "/tmp/robot_stack/vision_obs.sock")
        self.assertEqual(cfg.orchestrator.vision_req_out.transport, "uds")
        self.assertEqual(cfg.orchestrator.vision_req_out.ipc_socket_path, "/tmp/robot_stack/vision_req.sock")
        self.assertFalse(cfg.orchestrator.control.vision_req_fail_to_stop)
        self.assertTrue(cfg.orchestrator.control.yolo_table_control_enable)
        self.assertAlmostEqual(cfg.orchestrator.control.no_table_bbox_timeout_s, 10.0)
        self.assertAlmostEqual(cfg.orchestrator.control.edge_geometry_timeout_s, 10.0)
        self.assertTrue(cfg.orchestrator.control.keep_vision_alive_after_task)
        self.assertFalse(cfg.orchestrator.control.task_done_shutdown_vision)
        self.assertTrue(cfg.vision.runtime.keep_vision_alive_after_task)
        self.assertTrue(cfg.vision.runtime.keep_preview_alive_after_task)
        self.assertFalse(cfg.vision.runtime.release_model_on_idle)


if __name__ == "__main__":
    unittest.main()
