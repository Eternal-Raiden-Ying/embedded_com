#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import importlib.util
import sys
from pathlib import Path

PKG_NAME = "orchestrator_service"
ROOT = Path(__file__).resolve().parents[1]
PARENT = ROOT.parent
WORKSPACE = ROOT.parent.parent
if str(PARENT) not in sys.path:
    sys.path.insert(0, str(PARENT))
if str(WORKSPACE) not in sys.path:
    sys.path.insert(0, str(WORKSPACE))

if PKG_NAME not in sys.modules:
    spec = importlib.util.spec_from_file_location(PKG_NAME, ROOT / "__init__.py")
    module = importlib.util.module_from_spec(spec)
    module.__path__ = [str(ROOT)]
    sys.modules[PKG_NAME] = module
    if spec and spec.loader:
        spec.loader.exec_module(module)

from orchestrator_service.bridge.simple_car_protocol import SimpleCarMapper, parse_car_state_line
from orchestrator_service.config.schema import CarMotionConfig
from orchestrator_service.ipc.protocol import CmdVel


def main():
    mapper = SimpleCarMapper(CarMotionConfig())
    cmd = CmdVel(ts=0.0, mode="CONTROLLED_APPROACH", vx_mps=0.12, vy_mps=-0.05, wz_radps=0.18, hold_ms=150)
    car_cmd = mapper.from_cmd_vel(cmd)
    raw = car_cmd.raw_line.strip().splitlines()
    if "MODE SEARCH" not in raw:
        raise AssertionError(f"expected MODE SEARCH line, got: {raw}")
    if not any(line.startswith("V ") for line in raw):
        raise AssertionError(f"expected V line, got: {raw}")
    if "0.12" not in car_cmd.raw_line or "-0.05" not in car_cmd.raw_line or "0.18" not in car_cmd.raw_line:
        raise AssertionError(f"unexpected V payload: {car_cmd.raw_line!r}")

    state = parse_car_state_line("STATE busy 0.10 -0.04 0.20 0")
    if state is None or state.state != "BUSY" or state.vx != 0.10 or state.vy != -0.04 or state.wz != 0.20:
        raise AssertionError(f"failed to parse new STATE line: {state}")

    estop = parse_car_state_line("ESTOP 1")
    if estop is None or not estop.estop:
        raise AssertionError(f"failed to parse ESTOP line: {estop}")

    print("PASS: control_module_smoke_test")
    print("  - mapper emits MODE SEARCH + V vx vy wz")
    print("  - parser accepts STATE vx vy wz fault_code")
    print("  - parser accepts ESTOP feedback")


if __name__ == "__main__":
    main()
