#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import sys
import os

# Ensure workspace root is in python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from common.config_loader import get_config

def main():
    cfg = get_config(reload=True)
    car = cfg.orchestrator.car
    print("================== CONFIG WZ LIMIT VERIFICATION ==================")
    print(f"table_controlled_wz_max_radps : {car.table_controlled_wz_max_radps:.3f} rad/s")
    print(f"table_wz_view_max_radps       : {car.table_wz_view_max_radps:.3f} rad/s")
    print("==================================================================")
    
    # Assertions to prevent silent errors during manual check
    assert car.table_controlled_wz_max_radps > 0.0, "Error: table_controlled_wz_max_radps is 0!"
    assert car.table_wz_view_max_radps > 0.0, "Error: table_wz_view_max_radps is 0!"
    assert car.table_controlled_wz_max_radps != car.table_wz_view_max_radps, "Error: parameters are not separated!"
    print("Success: WZ limits are correctly loaded and separated!")

if __name__ == "__main__":
    main()
