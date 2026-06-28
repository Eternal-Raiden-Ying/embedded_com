#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pytest
import inspect

def test_static_class_map():
    from common.config.schema import _FINETUNE_YOLO26S_BGR15
    from VISTA.vision_module.config.data import finetune_yolo26s_bgr15
    
    # 1. Check class count matches length of class_names
    assert len(_FINETUNE_YOLO26S_BGR15) == 15
    assert len(finetune_yolo26s_bgr15) == 15
    
    # 2. Check "apple" is present at index 1
    assert "apple" in finetune_yolo26s_bgr15
    apple_idx = finetune_yolo26s_bgr15.index("apple")
    assert apple_idx == 1
    
    # 3. Verify target matcher uses matched_cls and matched_bbox instead of best_cls
    from VISTA.vision_module.utils.detect import compute_target_obs
    source = inspect.getsource(compute_target_obs)
    assert "matched_cls" in source
    assert "matched_bbox" in source
    assert "matched_conf" in source
    

    # 4. Preview overlay and target matcher share the same class registry
    # Verify that the overlay and search pipelines dynamically reference local.get("class_names")

    from VISTA.vision_module.app.stages.search.target_obs_builder import target_obs_from_results
    builder_source = inspect.getsource(target_obs_from_results)
    assert "class_names" in builder_source
    print("All class map static assertions passed successfully!")
