#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from typing import Dict, Optional

from ....config.data import normalize_class_name
from ....utils.detect import compute_target_obs, resolve_target_classes


def target_obs_from_payload(payload: Optional[Dict[str, object]], target: Optional[str]) -> Dict[str, object]:
    base: Dict[str, object] = {
        "found": False,
        "target_found": False,
        "target": target,
        "raw_target": target,
        "canonical_target": target,
        "expected_class_name": target,
        "expected_class_id": None,
        "boxes_count": 0,
        "best_cls": "n/a",
        "best_conf": 0.0,
        "matched_cls": None,
        "matched_class_id": None,
        "matched_conf": None,
        "matched_bbox": None,
        "matched_center": None,
        "matched_center_full_norm": None,
        "matched_center_offset_norm": None,
        "matched_area": None,
        "matched_rank_in_all_boxes": None,
        "num_target_candidates": 0,
        "bbox_valid": None,
        "bbox_invalid_reason": None,
        "reason": "waiting_local_perception",
    }
    source = None
    if isinstance(payload, dict):
        source = payload.get("target_obs") or payload.get("mock_target_obs")
    if isinstance(source, dict):
        base.update(source)
    base.setdefault("target", target)
    base["target_found"] = bool(base.get("target_found", base.get("found", False)))
    base["found"] = bool(base["target_found"])
    return base


def payload_has_target_obs(payload: Optional[Dict[str, object]]) -> bool:
    return isinstance(payload, dict) and (
        isinstance(payload.get("target_obs"), dict) or isinstance(payload.get("mock_target_obs"), dict)
    )


def target_obs_from_results(results: Dict[str, object], target: Optional[str]) -> Optional[Dict[str, object]]:
    if "local_perception" not in (results or {}):
        return None
    local_raw = (results or {}).get("local_perception")
    if not isinstance(local_raw, dict):
        return {
            "found": False,
            "target": target,
            "boxes_count": 0,
            "best_cls": "n/a",
            "best_conf": 0.0,
            "reason": "invalid_local_perception",
        }
    local = dict(local_raw or {})
    contract_ok = bool(local.get("contract_ok", True))
    contract_error = str(local.get("contract_error") or "")
    contract_warnings = list(local.get("contract_warnings") or [])
    target_obs = local.get("target_obs")
    if isinstance(target_obs, dict):
        target_found = bool(target_obs.get("target_found", target_obs.get("found", True)))
        merged = {"found": target_found, "target_found": target_found, "target": target}
        merged.update(target_obs)
        merged.setdefault("target", target)
        merged.setdefault("obs_ts", local.get("obs_ts"))
        merged.setdefault("frame_id", local.get("frame_seq"))
        merged.setdefault("seq", local.get("frame_seq"))
        merged.setdefault("age_ms", local.get("age_ms"))
        if contract_error:
            merged.setdefault("contract_error", contract_error)
        if contract_warnings:
            merged.setdefault("contract_warnings", contract_warnings)
        return merged

    boxes = local.get("infer_boxes")
    class_names = local.get("class_names")
    rgb_shape = local.get("rgb_shape")
    weak_payload = {
        "found": False,
        "target_found": False,
        "target": target,
        "raw_target": target,
        "canonical_target": target,
        "expected_class_name": target,
        "expected_class_id": None,
        "obs_ts": local.get("obs_ts"),
        "frame_id": local.get("frame_seq"),
        "seq": local.get("frame_seq"),
        "age_ms": local.get("age_ms"),
        "boxes_count": int(local.get("box_count", 0) or 0),
        "best_cls": "n/a",
        "best_conf": 0.0,
        "matched_cls": None,
        "matched_class_id": None,
        "matched_conf": None,
        "matched_bbox": None,
        "matched_center": None,
        "matched_center_full_norm": None,
        "matched_center_offset_norm": None,
        "matched_area": None,
        "matched_rank_in_all_boxes": None,
        "num_target_candidates": 0,
        "bbox_valid": None,
        "bbox_invalid_reason": None,
    }
    if not local:
        weak_payload["reason"] = "no_local_perception"
    elif not local.get("rgb_shape"):
        weak_payload["reason"] = "rgb_unavailable"
    elif not bool(local.get("has_infer", False)):
        weak_payload["reason"] = "predictor_not_ready"
    valid_names = resolve_target_classes(target, class_names=class_names)
    available_names = [str(name) for name in class_names] if isinstance(class_names, (list, tuple)) else []
    if available_names:
        weak_payload["all_candidate_classes"] = available_names[:32]
    if not valid_names:
        weak_payload["target_unmapped"] = True
        weak_payload["reason"] = "target_unmapped"
        contract_warnings.append(f"target_unmapped target={target}")
    if valid_names and available_names:
        normalized_available = {normalize_class_name(name) for name in available_names}
        if not (valid_names & normalized_available):
            weak_payload["class_not_supported"] = True
            weak_payload["target_unmapped"] = True
            weak_payload["available_classes"] = available_names[:32]
            contract_warnings.append(
                f"class_not_supported target={target} available={','.join(available_names[:16])}"
            )
    if isinstance(boxes, list):
        weak_payload["boxes_count"] = int(local.get("box_count", len(boxes)) or len(boxes))
        best_cls = "n/a"
        best_conf = 0.0
        for row in boxes:
            try:
                conf = float(row[4])
                cls_id = int(float(row[5]))
                cls_name = str(row[6]).strip() if len(row) > 6 else ""
                if not cls_name and isinstance(class_names, (list, tuple)) and 0 <= cls_id < len(class_names):
                    cls_name = str(class_names[cls_id])
                if conf >= best_conf:
                    best_conf = conf
                    best_cls = cls_name or str(cls_id)
            except Exception:
                continue
        weak_payload["best_cls"] = best_cls
        weak_payload["best_conf"] = float(best_conf)
        if not boxes:
            weak_payload.setdefault("reason", "no_boxes")
        elif not weak_payload.get("reason"):
            weak_payload["reason"] = "no_target_candidate"
    if contract_error:
        weak_payload["contract_error"] = contract_error
    if contract_warnings:
        weak_payload["contract_warnings"] = contract_warnings
    if not isinstance(boxes, list) or not rgb_shape:
        return weak_payload
    # TODO: prefer target candidates inside table ROI/table bbox when the
    # search pipeline publishes a stable ROI gate for FIND_OBJECT.
    try:
        obs = compute_target_obs(tuple(rgb_shape), target, boxes, class_names=class_names)
    except Exception as exc:
        weak_payload["contract_error"] = weak_payload.get("contract_error") or f"invalid_local_perception:{exc}"
        return weak_payload
    if obs is None:
        return weak_payload if (isinstance(boxes, list) or not contract_ok or contract_error or contract_warnings) else None
    payload = {"found": True, "target_found": True, "target": target}
    payload.update(obs)
    payload.update({k: v for k, v in weak_payload.items() if k in {"boxes_count"}})
    for key in ("obs_ts", "frame_id", "seq", "age_ms"):
        if weak_payload.get(key) is not None:
            payload[key] = weak_payload.get(key)
    payload["found"] = bool(payload.get("target_found", payload.get("found", True)))
    try:
        if payload.get("bbox") and rgb_shape:
            h = float(rgb_shape[0])
            y1, y2 = float(payload["bbox"][1]), float(payload["bbox"][3])
            payload["cy_norm"] = max(0.0, min(1.0, ((y1 + y2) / 2.0) / max(1.0, h)))
    except Exception:
        pass
    payload.setdefault("target", target)
    if contract_error:
        payload["contract_error"] = contract_error
    if contract_warnings:
        payload["contract_warnings"] = contract_warnings
    return payload
