#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
from typing import Dict, Optional, Tuple

from ...config.data import normalize_class_name
from ...ipc.protocol import VisionReq
from ...utils.detect import compute_target_obs, resolve_target_classes
from .base import BaseStagePlan, StageContext, StageOutput, StageTickInput, normalize_upper, resolve_stage_summary




def _target_obs_from_payload(payload: Optional[Dict[str, object]], target: Optional[str]) -> Dict[str, object]:
    base: Dict[str, object] = {
        "found": False,
        "target_found": False,
        "target": target,
        "boxes_count": 0,
        "best_cls": "n/a",
        "best_conf": 0.0,
        "matched_cls": None,
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


def _payload_has_target_obs(payload: Optional[Dict[str, object]]) -> bool:
    return isinstance(payload, dict) and (
        isinstance(payload.get("target_obs"), dict) or isinstance(payload.get("mock_target_obs"), dict)
    )


def _payload_has_table_edge_obs(payload: Optional[Dict[str, object]]) -> bool:
    return isinstance(payload, dict) and (
        isinstance(payload.get("table_edge_obs"), dict) or isinstance(payload.get("mock_table_edge_obs"), dict)
    )

def _sync_edge_follow_payload(req: VisionReq, ctx: StageContext) -> None:
    payload = req.payload if isinstance(req.payload, dict) else {}
    for key in (
        "locked_edge_id",
        "locked_edge_line",
        "locked_roi",
        "locked_yaw_err",
        "locked_dist_err",
        "locked_edge_conf",
        "locked_obs_seq",
        "current_edge_id",
    ):
        if key in payload:
            ctx.stage_state[key] = payload.get(key)


def _default_table_edge_obs() -> Dict[str, object]:
    return {
        "table_found": False,
        "edge_found": False,
        "edge_valid": False,
        "confidence": 0.0,
        "edge_conf": 0.0,
        "yaw_err_rad": None,
        "yaw_err": None,
        "dist_err_m": None,
        "dist_err": None,
        "edge_k": None,
        "edge_b": None,
        "depth_valid": False,
        "edge_obs_unavailable": True,
        "point_count": 0,
        "table_point_count": 0,
        "obs_ts": None,
        "age_ms": None,
        "frame_id": None,
        "seq": None,
        "source_mode": "",
        "is_stale": True,
        "source": "vision_table_edge_manager",
        "type": "table_edge_obs",
    }


def _table_edge_obs_from_payload(payload: Optional[Dict[str, object]]) -> Dict[str, object]:
    base = _default_table_edge_obs()
    source = None
    if isinstance(payload, dict):
        source = payload.get("table_edge_obs") or payload.get("mock_table_edge_obs")
    if isinstance(source, dict):
        base.update(source)
    return base


def _target_obs_from_results(results: Dict[str, object], target: Optional[str]) -> Optional[Dict[str, object]]:
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
        "obs_ts": local.get("obs_ts"),
        "frame_id": local.get("frame_seq"),
        "seq": local.get("frame_seq"),
        "age_ms": local.get("age_ms"),
        "boxes_count": int(local.get("box_count", 0) or 0),
        "best_cls": "n/a",
        "best_conf": 0.0,
        "matched_cls": None,
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


def _table_edge_obs_from_results(results: Dict[str, object]) -> Optional[Dict[str, object]]:
    table_edge = (results or {}).get("table_edge_obs")
    if not isinstance(table_edge, dict):
        return None
    merged = _default_table_edge_obs()
    merged.update(table_edge)
    merged["type"] = "table_edge_obs"
    if "edge_obs_unavailable" not in table_edge:
        merged["edge_obs_unavailable"] = str(merged.get("reason") or "") in {
            "depth_unavailable",
            "depth_frame_missing",
            "depth_frame_not_2d",
            "detector_unavailable",
        }
    if "is_stale" not in table_edge:
        merged["is_stale"] = False
    return merged


def _table_edge_stale_ms() -> float:
    try:
        return max(0.0, float(os.getenv("VISTA_TABLE_EDGE_STALE_MS", "500") or 500.0))
    except Exception:
        return 500.0


def _annotate_table_edge_obs(
    obs: Dict[str, object],
    *,
    tick_ts: float,
    source: str,
    source_mode: str,
) -> Dict[str, object]:
    out = dict(obs or _default_table_edge_obs())
    out["type"] = "table_edge_obs"
    out["source_mode"] = str(source_mode or "").strip().upper()
    out["edge_conf"] = float(out.get("edge_conf", out.get("confidence", 0.0)) or 0.0)
    out["yaw_err"] = out.get("yaw_err", out.get("yaw_err_rad"))
    out["dist_err"] = out.get("dist_err", out.get("dist_err_m"))
    out.setdefault("seq", out.get("frame_seq", out.get("frame_id")))
    out.setdefault("frame_id", out.get("frame_seq", out.get("seq")))
    out["edge_obs_unavailable"] = bool(
        out.get("edge_obs_unavailable", False)
        or out.get("reason") in {"depth_unavailable", "depth_frame_missing", "depth_frame_not_2d", "detector_unavailable"}
    )
    out["edge_valid"] = bool(out.get("edge_found", False) and not out.get("edge_obs_unavailable", False))
    obs_ts = out.get("obs_ts", out.get("ts"))
    age_ms = None
    try:
        if obs_ts is not None:
            obs_ts_f = float(obs_ts)
            out["obs_ts"] = obs_ts_f
            age_ms = max(0.0, (float(tick_ts) - obs_ts_f) * 1000.0)
    except Exception:
        age_ms = None
    if age_ms is None:
        out.setdefault("obs_ts", None)
        out["age_ms"] = None
        stale = True
    else:
        upstream_age = out.get("age_ms")
        try:
            age_ms = max(float(age_ms), float(upstream_age))
        except Exception:
            pass
        out["age_ms"] = float(age_ms)
        stale = bool(age_ms > _table_edge_stale_ms())
    if source != "results":
        stale = True
        out.setdefault("reason", "no_new_table_edge_obs_result")
    out["is_stale"] = bool(out.get("is_stale", False) or stale or out.get("edge_obs_unavailable", False))
    return out


class SearchStagePlan(BaseStagePlan):
    """Stage plan for local target search and depth-based table-edge perception."""

    stage_name = "SEARCH"
    default_mode = "FIND_OBJECT"
    common_routes = ("frame_meta", "runtime_status")
    optional_routes = {
        "FIND_OBJECT": ("local_perception", "table_edge_obs"),
        "FIND_EDGE": ("local_perception", "table_edge_obs"),
        "FIND_TABLE": ("local_perception",),
    }

    @staticmethod
    def _mode_for_request(req: VisionReq, search_kind: str, default: str) -> str:
        explicit = normalize_upper(req.mode_hint, "")
        if explicit and not (
            explicit == "FIND_OBJECT"
            and search_kind in {"TABLE_EDGE", "EDGE_FOLLOW_TARGET", "TARGET_ON_EDGE"}
        ):
            return explicit
        if search_kind == "TABLE_EDGE":
            return "FIND_EDGE"
        if search_kind in {"EDGE_FOLLOW_TARGET", "TARGET_ON_EDGE"}:
            return "FIND_EDGE"
        return normalize_upper(default, "FIND_OBJECT")

    def on_enter(self, req: VisionReq, ctx: StageContext) -> Optional[StageOutput]:
        super().on_enter(req, ctx)
        ctx.target_name = req.target or ctx.target_name
        ctx.interaction_id = None
        payload = req.payload if isinstance(req.payload, dict) else {}
        search_kind = normalize_upper(payload.get("search_kind", ""), "")
        if search_kind not in {"TABLE_EDGE", "TARGET", "EDGE_FOLLOW_TARGET", "TARGET_ON_EDGE"}:
            return StageOutput(
                vision_obs=self.build_obs(
                    ctx,
                    status="FAILED",
                    perception={},
                    result={"reason": f"invalid search_kind: {search_kind}"}
                ),
                signals={"invalid_search_kind": True},
            )
        ctx.current_mode = self._mode_for_request(req, search_kind, self.default_mode)
        ctx.stage_state["search_kind"] = search_kind
        ctx.stage_state["target_obs"] = _target_obs_from_payload(payload, ctx.target_name)
        ctx.stage_state["table_edge_obs"] = _table_edge_obs_from_payload(payload)
        _sync_edge_follow_payload(req, ctx)
        return None

    def on_update(self, req: VisionReq, ctx: StageContext) -> Optional[StageOutput]:
        if req.target:
            ctx.target_name = req.target
        if isinstance(req.payload, dict):
            search_kind = normalize_upper(req.payload.get("search_kind", ""), "")
            if search_kind not in {"TABLE_EDGE", "TARGET", "EDGE_FOLLOW_TARGET", "TARGET_ON_EDGE"}:
                return StageOutput(
                    vision_obs=self.build_obs(
                        ctx,
                        status="FAILED",
                        perception={},
                        result={"reason": f"invalid search_kind: {search_kind}"}
                    ),
                    signals={"invalid_search_kind": True},
                )
            ctx.current_mode = self._mode_for_request(req, search_kind, self.default_mode)
            ctx.stage_state["search_kind"] = search_kind
            _sync_edge_follow_payload(req, ctx)
            if _payload_has_target_obs(req.payload):
                ctx.stage_state["target_obs"] = _target_obs_from_payload(req.payload, ctx.target_name)
            else:
                ctx.stage_state.setdefault("target_obs", _target_obs_from_payload(None, ctx.target_name))
            if _payload_has_table_edge_obs(req.payload):
                ctx.stage_state["table_edge_obs"] = _table_edge_obs_from_payload(req.payload)
            else:
                ctx.stage_state.setdefault("table_edge_obs", _table_edge_obs_from_payload(None))
        return StageOutput()

    def tick(self, tick_input: StageTickInput, ctx: StageContext) -> Optional[StageOutput]:
        results = dict(tick_input.results or {})
        mode = normalize_upper(ctx.current_mode, "")

        if mode == "FIND_EDGE":
            local_perception = results.get("local_perception")
            search_kind = normalize_upper(ctx.stage_state.get("search_kind", ""), "")
            target_obs = None
            target_source = ""
            if search_kind in {"EDGE_FOLLOW_TARGET", "TARGET_ON_EDGE"}:
                target_obs, target_source = resolve_stage_summary(
                    results=results,
                    stage_state=ctx.stage_state,
                    state_key="target_obs",
                    default_factory=lambda: _target_obs_from_payload(None, ctx.target_name),
                    result_factory=lambda payload: _target_obs_from_results(payload, ctx.target_name),
                )
                ctx.stage_state["target_obs"] = dict(target_obs)
            table_edge_obs, table_edge_source = resolve_stage_summary(
                results=results,
                stage_state=ctx.stage_state,
                state_key="table_edge_obs",
                default_factory=lambda: _table_edge_obs_from_payload(None),
                result_factory=_table_edge_obs_from_results,
                result_route="table_edge_obs",
            )
            table_edge_obs = _annotate_table_edge_obs(
                table_edge_obs,
                tick_ts=tick_input.ts,
                source=table_edge_source,
                source_mode=ctx.current_mode,
            )
            ctx.stage_state["table_edge_obs"] = dict(table_edge_obs)
            perception = {"table_edge_obs": table_edge_obs}
            if target_obs is not None:
                perception["target_obs"] = target_obs
            elif search_kind != "TABLE_EDGE":
                perception["local_perception"] = local_perception
            source_payload = {"table_edge_obs": table_edge_source}
            if target_source:
                source_payload["target_obs"] = target_source
            return StageOutput(
                vision_obs=self.build_obs(
                    ctx,
                    status="RUNNING",
                    perception=perception,
                ),
                snapshot={
                    "generation": int(tick_input.generation),
                    "result_keys": sorted(results.keys()),
                    "source": source_payload,
                    "search_kind": ctx.stage_state.get("search_kind"),
                },
            )

        if mode == "FIND_TABLE":
            target_obs, target_source = resolve_stage_summary(
                results=results,
                stage_state=ctx.stage_state,
                state_key="target_obs",
                default_factory=lambda: _target_obs_from_payload(None, ctx.target_name),
                result_factory=lambda payload: _target_obs_from_results(payload, ctx.target_name),
            )
            return StageOutput(
                vision_obs=self.build_obs(
                    ctx,
                    status="RUNNING",
                    perception={"target_obs": target_obs},
                ),
                snapshot={
                    "generation": int(tick_input.generation),
                    "result_keys": sorted(results.keys()),
                    "source": {"target_obs": target_source},
                    "search_kind": ctx.stage_state.get("search_kind"),
                },
            )

        if mode == "FIND_OBJECT":
            target_obs, source = resolve_stage_summary(
                results=results,
                stage_state=ctx.stage_state,
                state_key="target_obs",
                default_factory=lambda: _target_obs_from_payload(None, ctx.target_name),
                result_factory=lambda payload: _target_obs_from_results(payload, ctx.target_name),
            )
            table_edge_obs, table_edge_source = resolve_stage_summary(
                results=results,
                stage_state=ctx.stage_state,
                state_key="table_edge_obs",
                default_factory=lambda: _table_edge_obs_from_payload(None),
                result_factory=_table_edge_obs_from_results,
                result_route="table_edge_obs",
            )
            table_edge_obs = _annotate_table_edge_obs(
                table_edge_obs,
                tick_ts=tick_input.ts,
                source=table_edge_source,
                source_mode=ctx.current_mode,
            )
            ctx.stage_state["table_edge_obs"] = dict(table_edge_obs)
            return StageOutput(
                vision_obs=self.build_obs(
                    ctx,
                    status="RUNNING",
                    perception={
                        "target_obs": target_obs,
                        "table_edge_obs": table_edge_obs,
                    },
                ),
                snapshot={
                    "generation": int(tick_input.generation),
                    "result_keys": sorted(results.keys()),
                    "source": {"target_obs": source, "table_edge_obs": table_edge_source},
                    "search_kind": ctx.stage_state.get("search_kind"),
                },
            )

        return StageOutput(
            vision_obs=self.build_obs(ctx, status="RELAXING", perception={}),
            snapshot={
                "generation": int(tick_input.generation),
                "result_keys": sorted(results.keys()),
            },
        )
