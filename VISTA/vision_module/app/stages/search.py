#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
from typing import Dict, Optional

from ...config.data import ASR_VOCAB_MAP, normalize_class_name
from ...ipc.protocol import VisionReq
from ...utils.detect import compute_target_obs
from .base import BaseStagePlan, StageContext, StageOutput, StageTickInput, normalize_upper, resolve_stage_summary


def _search_kind(req: VisionReq, default_mode: str) -> str:
    payload = req.payload if isinstance(req.payload, dict) else {}
    explicit = normalize_upper(payload.get("search_kind"), "")
    if explicit in {"TABLE_EDGE", "TARGET", "EDGE_FOLLOW_TARGET", "TARGET_ON_EDGE"}:
        return explicit
    hinted_mode = normalize_upper(req.mode_hint, default_mode)
    if hinted_mode == "TABLE_EDGE_PERCEPTION":
        return "EDGE_FOLLOW_TARGET"
    if hinted_mode == "DEPTH_PERCEPTION":
        return "TABLE_EDGE"
    return "TARGET"


def _is_edge_follow_target(search_kind: str) -> bool:
    return normalize_upper(search_kind, "") in {"EDGE_FOLLOW_TARGET", "TARGET_ON_EDGE"}


def _target_obs_from_payload(payload: Optional[Dict[str, object]], target: Optional[str]) -> Dict[str, object]:
    base: Dict[str, object] = {
        "found": False,
        "target": target,
        "boxes_count": 0,
        "best_cls": "n/a",
        "best_conf": 0.0,
        "reason": "waiting_local_perception",
    }
    source = None
    if isinstance(payload, dict):
        source = payload.get("target_obs") or payload.get("mock_target_obs")
    if isinstance(source, dict):
        base.update(source)
    base.setdefault("target", target)
    base["found"] = bool(base.get("found", False))
    return base


def _payload_has_target_obs(payload: Optional[Dict[str, object]]) -> bool:
    return isinstance(payload, dict) and (
        isinstance(payload.get("target_obs"), dict) or isinstance(payload.get("mock_target_obs"), dict)
    )


def _payload_has_table_edge_obs(payload: Optional[Dict[str, object]]) -> bool:
    return isinstance(payload, dict) and (
        isinstance(payload.get("table_edge_obs"), dict) or isinstance(payload.get("mock_table_edge_obs"), dict)
    )


def _default_table_edge_obs() -> Dict[str, object]:
    return {
        "table_found": False,
        "edge_found": False,
        "confidence": 0.0,
        "yaw_err_rad": None,
        "dist_err_m": None,
        "edge_k": None,
        "edge_b": None,
        "depth_valid": False,
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
        merged = {"found": bool(target_obs.get("found", True)), "target": target}
        merged.update(target_obs)
        merged.setdefault("target", target)
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
        "target": target,
        "boxes_count": int(local.get("box_count", 0) or 0),
        "best_cls": "n/a",
        "best_conf": 0.0,
    }
    if not local:
        weak_payload["reason"] = "no_local_perception"
    elif not local.get("rgb_shape"):
        weak_payload["reason"] = "rgb_unavailable"
    elif not bool(local.get("has_infer", False)):
        weak_payload["reason"] = "predictor_not_ready"
    normalized_target = normalize_class_name(target)
    valid_names = set(ASR_VOCAB_MAP.get(normalized_target, set()))
    available_names = [str(name) for name in class_names] if isinstance(class_names, (list, tuple)) else []
    if valid_names and available_names:
        normalized_available = {normalize_class_name(name) for name in available_names}
        if not (valid_names & normalized_available):
            weak_payload["class_not_supported"] = True
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
    payload = {"found": True, "target": target}
    payload.update(obs)
    payload.update({k: v for k, v in weak_payload.items() if k in {"boxes_count", "best_cls", "best_conf"}})
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
    out.setdefault("edge_conf", out.get("confidence"))
    out.setdefault("seq", out.get("frame_seq", out.get("frame_id")))
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
    out["is_stale"] = bool(out.get("is_stale", False) or stale)
    return out


class SearchStagePlan(BaseStagePlan):
    """Stage plan for local target search and depth-based table-edge perception."""

    stage_name = "SEARCH"
    default_mode = "TRACK_LOCAL"

    def _resolve_mode(self, req: VisionReq) -> str:
        search_kind = _search_kind(req, self.default_mode)
        if search_kind == "TABLE_EDGE" or _is_edge_follow_target(search_kind):
            return "TABLE_EDGE_PERCEPTION"
        if req.mode_hint:
            return normalize_upper(req.mode_hint, self.default_mode)
        return self.default_mode

    def on_enter(self, req: VisionReq, ctx: StageContext) -> None:
        """Prepare target metadata and choose the initial local tracking mode."""
        super().on_enter(req, ctx)
        ctx.target_name = req.target or ctx.target_name
        ctx.current_mode = self._resolve_mode(req)
        ctx.interaction_id = None
        ctx.stage_state["search_kind"] = _search_kind(req, self.default_mode)
        ctx.stage_state["target_obs"] = _target_obs_from_payload(req.payload, ctx.target_name)
        ctx.stage_state["table_edge_obs"] = _table_edge_obs_from_payload(req.payload)

    def on_update(self, req: VisionReq, ctx: StageContext) -> Optional[StageOutput]:
        """Refresh target or search parameters without leaving SEARCH."""
        if req.target:
            ctx.target_name = req.target
        ctx.current_mode = self._resolve_mode(req)
        if isinstance(req.payload, dict):
            ctx.stage_state["search_kind"] = _search_kind(req, self.default_mode)
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
        """Produce SEARCH stage outputs for target or table-edge perception."""
        results = dict(tick_input.results or {})
        search_kind = normalize_upper(ctx.stage_state.get("search_kind"), "TARGET")

        if search_kind == "TABLE_EDGE" or _is_edge_follow_target(search_kind):
            table_edge_obs, source = resolve_stage_summary(
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
                source=source,
                source_mode=ctx.current_mode,
            )
            ctx.stage_state["table_edge_obs"] = dict(table_edge_obs)
            if _is_edge_follow_target(search_kind):
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
                        perception={
                            "table_edge_obs": table_edge_obs,
                            "target_obs": target_obs,
                        },
                    ),
                    snapshot={
                        "generation": int(tick_input.generation),
                        "result_keys": sorted(results.keys()),
                        "source": {"table_edge_obs": source, "target_obs": target_source},
                        "search_kind": search_kind,
                    },
                )
            return StageOutput(
                vision_obs=self.build_obs(
                    ctx,
                    status="RUNNING",
                    perception={"table_edge_obs": table_edge_obs},
                ),
                snapshot={
                    "generation": int(tick_input.generation),
                    "result_keys": sorted(results.keys()),
                    "source": source,
                    "search_kind": search_kind,
                },
            )

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
                "search_kind": search_kind,
            },
        )
