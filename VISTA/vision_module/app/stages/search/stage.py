#!/usr/bin/env python3
# -*- coding: utf-8 -*-

print("[VISTA_EDGE_MAPPING_FIX_ACTIVE] version=20260626_edge_payload_path_v2 file=" + __file__, flush=True)
print("[VISTA_EDGE_E2E_CHAIN_FIX_ACTIVE] version=20260626_edge_e2e_results_priority_v1 file=" + __file__, flush=True)

from collections.abc import Mapping
from functools import lru_cache
import time
from typing import Optional

from ....ipc.protocol import VisionReq
from ..base import BaseStagePlan, StageContext, StageOutput, StageTickInput, normalize_upper, resolve_stage_summary
from .request_mapping import invalid_search_kind_reason, is_valid_search_kind, mode_for_request
from .status_policy import FAILED, RELAXING, RUNNING, invalid_search_kind_result
from .table_edge_obs_builder import (
    annotate_table_edge_obs,
    merge_table_bbox_from_local_perception,
    local_explicit_negative,
    local_inference_completed,
    local_inference_identity,
    payload_has_table_edge_obs,
    table_edge_obs_from_payload,
    table_edge_obs_from_results,
    check_edge_current_enough,
)


@lru_cache(maxsize=1)
def _completed_table_bbox_stale_s() -> float:
    """Reuse the existing docking loss-hold TTL across the control boundary."""
    try:
        from common.config_loader import get_config

        cfg = get_config()
        return max(0.5, float(cfg.orchestrator.control.table_loss_hold_s))
    except Exception:
        return 1.2
from .target_obs_builder import (
    payload_has_target_obs,
    target_obs_from_payload,
    target_obs_from_results,
)


def _edge_flag(payload: object, key: str) -> bool:
    payload = _table_edge_result_dict(payload)
    return isinstance(payload, dict) and bool(payload.get(key, False))


def _edge_point_count(payload: object) -> int:
    payload = _table_edge_result_dict(payload)
    if not isinstance(payload, dict):
        return 0
    try:
        return int(payload.get("point_count", 0) or 0)
    except Exception:
        return 0


def _has_edge_candidate(payload: object) -> bool:
    payload = _table_edge_result_dict(payload)
    if not isinstance(payload, dict):
        return False
    return bool(
        payload.get("edge_found")
        or payload.get("edge_valid")
        or payload.get("edge_trusted")
        or payload.get("edge_detected")
        or payload.get("edge_geometry_valid")
        or payload.get("candidate_line_present")
        or payload.get("detector_candidate_line_present")
        or payload.get("edge_candidate_found")
        or _edge_point_count(payload) > 0
    )


def _table_edge_result_dict(payload: object) -> Optional[dict]:
    if isinstance(payload, dict):
        return dict(payload)
    if isinstance(payload, Mapping):
        try:
            return dict(payload)
        except Exception:
            pass
    converter = getattr(payload, "to_dict", None)
    if callable(converter):
        try:
            value = converter()
            if isinstance(value, dict):
                return dict(value)
            if isinstance(value, Mapping):
                return dict(value)
        except Exception:
            pass
    raw = getattr(payload, "__dict__", None)
    if isinstance(raw, dict):
        return dict(raw)
    if hasattr(payload, "get"):
        out = {}
        for key in (
            "frame_id",
            "frame_seq",
            "seq",
            "edge_found",
            "edge_valid",
            "edge_trusted",
            "edge_detected",
            "edge_geometry_valid",
            "point_count",
            "table_point_count",
            "valid_edge_points",
            "support_count",
            "inlier_count",
            "yaw_err_rad",
            "dist_err_m",
            "reason",
            "source",
            "obs_ts",
            "ts",
        ):
            try:
                value = payload.get(key)
            except Exception:
                continue
            if value is not None:
                out[key] = value
        if out:
            return out
    return None


def _local_perception_identity(local_perception: object) -> Optional[tuple]:
    """Return a stable, hashable identity for the current RGB inference."""
    if not isinstance(local_perception, dict):
        return None
    completed_identity = local_inference_identity(local_perception)
    if completed_identity is None:
        return None
    frame_id = next(
        (
            local_perception.get(key)
            for key in ("frame_id", "frame_seq", "camera_frame_seq", "yolo_frame_seq")
            if local_perception.get(key) is not None
        ),
        None,
    )
    timestamp = next(
        (
            local_perception.get(key)
            for key in ("capture_mono_ns", "obs_ts", "ts", "frame_capture_ts")
            if local_perception.get(key) is not None
        ),
        None,
    )
    bbox = next(
        (
            local_perception.get(key)
            for key in ("detected_table_bbox", "table_bbox", "table_bbox_xyxy", "yolo_table_bbox")
            if local_perception.get(key) is not None
        ),
        None,
    )
    bbox_identity = tuple(bbox[:4]) if isinstance(bbox, (list, tuple)) and len(bbox) >= 4 else None
    current_found = local_perception.get("table_bbox_current_found")
    if current_found is None:
        current_found = local_perception.get("table_bbox_detected", local_perception.get("table_found"))
    if current_found is None and bbox_identity is not None:
        current_found = True
    if current_found is None and bool(local_perception.get("has_infer", local_perception.get("yolo_has_infer", False))):
        current_found = False
    return (
        completed_identity,
        frame_id,
        timestamp,
        local_perception.get("trace_id"),
        bbox_identity,
        current_found,
        local_perception.get("yolo_table_fresh"),
    )


def _completed_local_perception_for_control(
    local_perception: object,
    stage_state: dict,
    *,
    tick_ts: float,
) -> Optional[dict]:
    """Return the latest completed inference, never an empty scheduler tick."""
    local = dict(local_perception) if isinstance(local_perception, dict) else None
    if local is not None and local_inference_completed(local):
        identity = local_inference_identity(local)
        previous_identity = stage_state.get("_last_completed_inference_identity")
        local["has_new_inference"] = bool(identity != previous_identity)
        local["explicit_negative_detection"] = bool(local_explicit_negative(local))
        local["explicit_negative"] = bool(local["explicit_negative_detection"])
        local["inference_age_ms"] = 0.0
        local["control_bbox_age_ms"] = 0.0
        local["bbox_hold_reason"] = "new_completed_inference"
        stage_state["_last_completed_local_perception"] = dict(local)
        stage_state["_last_completed_inference_identity"] = identity
        return local

    cached = stage_state.get("_last_completed_local_perception")
    if not isinstance(cached, dict):
        if local is not None:
            local["has_new_inference"] = False
            local["explicit_negative_detection"] = False
            local["explicit_negative"] = False
        return local

    completed_ts = cached.get("obs_ts", cached.get("frame_capture_ts"))
    age_s = None
    if completed_ts is not None:
        try:
            age_s = max(0.0, float(tick_ts) - float(completed_ts))
        except Exception:
            age_s = None
    if age_s is None:
        completed_mono_ns = cached.get("last_completed_inference_mono_ns") or cached.get("inference_done_mono_ns")
        if completed_mono_ns is not None:
            try:
                age_s = max(0.0, (time.monotonic_ns() - int(completed_mono_ns)) / 1_000_000_000.0)
            except Exception:
                age_s = None
    age_s = float(age_s if age_s is not None else 0.0)
    held = dict(cached)
    held["has_new_inference"] = False
    held["explicit_negative_detection"] = False
    held["explicit_negative"] = False
    held["inference_age_ms"] = age_s * 1000.0
    held["control_bbox_age_ms"] = age_s * 1000.0
    held["bbox_hold_reason"] = "inference_pending" if bool(local and local.get("inference_executed")) else "no_new_inference"
    if age_s <= _completed_table_bbox_stale_s():
        return held

    # Absolute timeout: invalidate the cached positive without fabricating a
    # completed negative or incrementing a detection-loss counter.
    held.update(
        {
            "detected_table_bbox": None,
            "table_bbox": None,
            "table_bbox_xyxy": None,
            "yolo_table_bbox": None,
            "table_bbox_current_found": False,
            "table_bbox_detected": False,
            "yolo_table_fresh": False,
            "explicit_negative_detection": False,
            "explicit_negative": False,
            "bbox_hold_reason": "completed_bbox_stale_timeout",
        }
    )
    return held

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
        "orchestrator_state",
        "final_phase_active",
        "final_roi_mode_latched",
        "final_distance_servo_active",
    ):
        if key in payload:
            ctx.stage_state[key] = payload.get(key)


def _sync_target_payload(req: VisionReq, ctx: StageContext) -> None:
    payload = req.payload if isinstance(req.payload, dict) else {}
    for key in ("task_id", "raw_target", "canonical_target", "class_name", "class_id", "session_id", "epoch"):
        if key in payload:
            ctx.stage_state[key] = payload.get(key)


def _annotate_target_obs(target_obs: dict, ctx: StageContext) -> dict:
    out = dict(target_obs or {})
    out.setdefault("raw_target", ctx.stage_state.get("raw_target") or ctx.target_name)
    out.setdefault("canonical_target", ctx.stage_state.get("canonical_target") or ctx.target_name)
    out.setdefault("expected_class_name", ctx.stage_state.get("class_name") or ctx.stage_state.get("canonical_target") or ctx.target_name)
    out.setdefault("expected_class_id", ctx.stage_state.get("class_id"))
    out.setdefault("task_id", ctx.stage_state.get("task_id"))
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
        return mode_for_request(req, search_kind, default)

    def on_enter(self, req: VisionReq, ctx: StageContext) -> Optional[StageOutput]:
        super().on_enter(req, ctx)
        ctx.target_name = req.target or ctx.target_name
        ctx.interaction_id = None
        payload = req.payload if isinstance(req.payload, dict) else {}
        search_kind = normalize_upper(payload.get("search_kind", ""), "")
        if not is_valid_search_kind(search_kind):
            reason = invalid_search_kind_reason(search_kind)
            return StageOutput(
                vision_obs=self.build_obs(
                    ctx,
                    status=FAILED,
                    perception={},
                    result=invalid_search_kind_result(reason),
                ),
                signals={"invalid_search_kind": True},
            )
        ctx.current_mode = self._mode_for_request(req, search_kind, self.default_mode)
        ctx.stage_state["search_kind"] = search_kind
        _sync_target_payload(req, ctx)
        ctx.stage_state["target_obs"] = target_obs_from_payload(payload, ctx.target_name)
        ctx.stage_state["table_edge_obs"] = table_edge_obs_from_payload(payload)
        _sync_edge_follow_payload(req, ctx)
        return None

    def on_update(self, req: VisionReq, ctx: StageContext) -> Optional[StageOutput]:
        if req.target:
            ctx.target_name = req.target
        if isinstance(req.payload, dict):
            search_kind = normalize_upper(req.payload.get("search_kind", ""), "")
            if not is_valid_search_kind(search_kind):
                reason = invalid_search_kind_reason(search_kind)
                return StageOutput(
                    vision_obs=self.build_obs(
                        ctx,
                        status=FAILED,
                        perception={},
                        result=invalid_search_kind_result(reason),
                    ),
                    signals={"invalid_search_kind": True},
                )
            ctx.current_mode = self._mode_for_request(req, search_kind, self.default_mode)
            ctx.stage_state["search_kind"] = search_kind
            _sync_target_payload(req, ctx)
            _sync_edge_follow_payload(req, ctx)
            if payload_has_target_obs(req.payload):
                ctx.stage_state["target_obs"] = target_obs_from_payload(req.payload, ctx.target_name)
            else:
                ctx.stage_state.setdefault("target_obs", target_obs_from_payload(None, ctx.target_name))
            if payload_has_table_edge_obs(req.payload):
                ctx.stage_state["table_edge_obs"] = table_edge_obs_from_payload(req.payload)
            else:
                ctx.stage_state.setdefault("table_edge_obs", table_edge_obs_from_payload(None))
        return StageOutput()

    def _process_table_edge_obs(
        self,
        results: dict,
        ctx: StageContext,
        tick_input: StageTickInput,
        local_perception: Optional[dict],
    ) -> tuple:
        import logging
        import time

        logger = logging.getLogger("vision.stage.search")
        raw_results_edge = (results or {}).get("table_edge_obs")
        local_perception = _completed_local_perception_for_control(
            local_perception,
            ctx.stage_state,
            tick_ts=tick_input.ts,
        )
        results_edge_dict = _table_edge_result_dict(raw_results_edge)
        results_present = isinstance(results_edge_dict, dict)
        results_summary = table_edge_obs_from_results({"table_edge_obs": results_edge_dict}) if results_present else None
        fallback_reason = ""
        if isinstance(results_summary, dict):
            table_edge_obs = dict(results_summary)
            table_edge_source = "results"
        else:
            cache_obs = ctx.stage_state.get("_table_edge_obs_cache")
            if isinstance(cache_obs, dict) and _has_edge_candidate(cache_obs):
                table_edge_obs = dict(cache_obs)
                table_edge_source = "cache"
                fallback_reason = "results_unavailable_using_cache"
            else:
                state_obs = ctx.stage_state.get("table_edge_obs")
                if isinstance(state_obs, dict):
                    table_edge_obs = dict(state_obs)
                    table_edge_source = "stage_state"
                    fallback_reason = "results_unavailable_using_stage_state"
                else:
                    table_edge_obs = table_edge_obs_from_payload(None)
                    table_edge_source = "default"
                    fallback_reason = "results_unavailable_using_default"

        table_edge_obs = annotate_table_edge_obs(
            table_edge_obs,
            tick_ts=tick_input.ts,
            source=table_edge_source,
            source_mode=ctx.current_mode,
        )
        table_edge_obs["selected_source"] = table_edge_source
        if table_edge_source == "results":
            table_edge_obs["source"] = "results"

        # Previous flags for force_send detection
        prev_obs = ctx.stage_state.get("table_edge_obs") or {}
        prev_found = bool(prev_obs.get("edge_found", False))
        prev_valid = bool(prev_obs.get("edge_valid", False))
        prev_trusted = bool(prev_obs.get("edge_trusted", False))
        prev_bbox_found = bool(prev_obs.get("table_bbox_current_found", False))
        prev_bbox_valid = bool(prev_obs.get("table_bbox_control_valid", False))
        prev_yolo_fresh = bool(prev_obs.get("yolo_table_fresh", False))

        local_frame_id = None
        if isinstance(local_perception, dict):
            local_frame_id = (
                local_perception.get("frame_seq")
                or local_perception.get("frame_id")
                or local_perception.get("camera_frame_seq")
                or local_perception.get("yolo_frame_seq")
            )
        
        edge_frame_id = table_edge_obs.get("frame_id") or table_edge_obs.get("frame_seq") or table_edge_obs.get("seq")
        
        edge_ts_val = None
        for k in ("obs_ts", "ts"):
            v = table_edge_obs.get(k)
            if v is not None:
                try:
                    edge_ts_val = float(v)
                    break
                except (ValueError, TypeError):
                    pass

        local_ts_val = None
        if isinstance(local_perception, dict):
            for k in ("obs_ts", "ts", "frame_capture_ts"):
                v = local_perception.get(k)
                if v is not None:
                    try:
                        local_ts_val = float(v)
                        break
                    except (ValueError, TypeError):
                        pass
        if local_ts_val is None:
            local_ts_val = tick_input.ts

        is_current_frame, _ = check_edge_current_enough(
            edge_frame_id=edge_frame_id,
            local_frame_id=local_frame_id,
            edge_ts=edge_ts_val,
            local_ts=local_ts_val,
        )

        logger.info(
            "[EDGE_SELECTION_TRACE] results_present=%s results_frame_id=%s results_edge_found=%s "
            "results_edge_valid=%s results_edge_trusted=%s results_point_count=%s selected_source=%s "
            "selected_frame_id=%s selected_edge_found=%s selected_edge_valid=%s selected_edge_trusted=%s "
            "selected_point_count=%s fallback_reason=%s",
            str(results_present).lower(),
            results_edge_dict.get("frame_id") if isinstance(results_edge_dict, dict) else None,
            str(_edge_flag(raw_results_edge, "edge_found")).lower(),
            str(_edge_flag(raw_results_edge, "edge_valid")).lower(),
            str(_edge_flag(raw_results_edge, "edge_trusted")).lower(),
            _edge_point_count(raw_results_edge),
            table_edge_source,
            edge_frame_id,
            str(bool(table_edge_obs.get("edge_found"))).lower(),
            str(bool(table_edge_obs.get("edge_valid"))).lower(),
            str(bool(table_edge_obs.get("edge_trusted"))).lower(),
            int(table_edge_obs.get("point_count", 0) or 0),
            fallback_reason,
        )
        if _edge_flag(raw_results_edge, "edge_found") and table_edge_source != "results":
            logger.warning(
                "[EDGE_SELECTION_BUG] results_edge_found=true selected_source=%s reason=%s",
                table_edge_source,
                fallback_reason,
            )

        # Before merge metrics:
        before_edge_found = bool(table_edge_obs.get("edge_found"))
        before_edge_valid = bool(table_edge_obs.get("edge_valid"))
        before_edge_trusted = bool(table_edge_obs.get("edge_trusted"))
        before_point_count = int(table_edge_obs.get("point_count", 0) or 0)
        before_table_point_count = int(table_edge_obs.get("table_point_count", 0) or 0)
        before_yaw = table_edge_obs.get("yaw_err")
        before_dist = table_edge_obs.get("dist_err")
        before_reason = table_edge_obs.get("reason")
        before_source = table_edge_obs.get("source")

        # Perform merge:
        table_edge_obs = merge_table_bbox_from_local_perception(
            table_edge_obs,
            local_perception,
            tick_ts=tick_input.ts,
        )
        # This is a newly assembled control observation even when its bbox is
        # held from the latest completed inference. Keep component timestamps
        # for diagnosis, while the aggregate timestamp follows this publish
        # cycle so Orchestrator does not confuse inference cadence with a dead
        # vision transport.
        table_edge_obs["edge_geometry_obs_ts"] = edge_ts_val
        table_edge_obs["edge_geometry_frame_id"] = edge_frame_id
        if isinstance(local_perception, dict) and local_perception.get("has_new_inference") is False:
            table_edge_obs["obs_ts"] = float(tick_input.ts)
            table_edge_obs["ts"] = float(tick_input.ts)
            table_edge_obs["age_ms"] = 0.0

        # After merge metrics:
        after_edge_found = bool(table_edge_obs.get("edge_found"))
        after_edge_valid = bool(table_edge_obs.get("edge_valid"))
        after_edge_trusted = bool(table_edge_obs.get("edge_trusted"))
        after_point_count = int(table_edge_obs.get("point_count", 0) or 0)
        after_table_point_count = int(table_edge_obs.get("table_point_count", 0) or 0)
        after_yaw = table_edge_obs.get("yaw_err")
        after_dist = table_edge_obs.get("dist_err")
        after_reason = table_edge_obs.get("reason")
        after_source = table_edge_obs.get("source")

        logger.info(
            "[EDGE_MERGE_TRACE] before_source=%s before_edge_found=%s before_edge_valid=%s "
            "before_edge_trusted=%s before_point_count=%s after_source=%s after_edge_found=%s "
            "after_edge_valid=%s after_edge_trusted=%s after_point_count=%s after_reason=%s",
            before_source,
            str(before_edge_found).lower(),
            str(before_edge_valid).lower(),
            str(before_edge_trusted).lower(),
            before_point_count,
            after_source,
            str(after_edge_found).lower(),
            str(after_edge_valid).lower(),
            str(after_edge_trusted).lower(),
            after_point_count,
            after_reason,
        )
        if before_edge_found and not after_edge_found:
            logger.warning(
                "[EDGE_MERGE_OVERWRITE_BUG] before_edge_found=true after_edge_found=false before_reason=%s after_reason=%s",
                before_reason,
                after_reason,
            )

        same_frame_mapping_ok = True
        mismatch_details = []
        if is_current_frame:
            if before_edge_found != after_edge_found:
                same_frame_mapping_ok = False
                mismatch_details.append(f"edge_found:{before_edge_found}->{after_edge_found}")
            if before_edge_valid != after_edge_valid:
                same_frame_mapping_ok = False
                mismatch_details.append(f"edge_valid:{before_edge_valid}->{after_edge_valid}")
            if before_edge_trusted != after_edge_trusted:
                same_frame_mapping_ok = False
                mismatch_details.append(f"edge_trusted:{before_edge_trusted}->{after_edge_trusted}")
            if before_point_count != after_point_count:
                same_frame_mapping_ok = False
                mismatch_details.append(f"point_count:{before_point_count}->{after_point_count}")
            if before_table_point_count != after_table_point_count:
                same_frame_mapping_ok = False
                mismatch_details.append(f"table_point_count:{before_table_point_count}->{after_table_point_count}")
            if before_yaw != after_yaw:
                same_frame_mapping_ok = False
                mismatch_details.append(f"yaw_err:{before_yaw}->{after_yaw}")
            if before_dist != after_dist:
                same_frame_mapping_ok = False
                mismatch_details.append(f"dist_err:{before_dist}->{after_dist}")
            if before_reason != after_reason:
                same_frame_mapping_ok = False
                mismatch_details.append(f"reason:{before_reason}->{after_reason}")

            if not same_frame_mapping_ok:
                logger.warning(
                    "[EDGE_MAPPING_MISMATCH] frame_id=%s %s",
                    edge_frame_id,
                    ", ".join(mismatch_details)
                )

        now = time.time()
        last_log = getattr(self, "_last_payload_log_ts", 0.0)
        if now - last_log >= 0.5:
            self._last_payload_log_ts = now
            logger.info(
                "[EDGE_OBS_PAYLOAD] frame_id=%s source=%s is_current=%s same_frame_mapping_ok=%s "
                "before=[found=%s valid=%s trusted=%s pts=%s table_pts=%s yaw=%s dist=%s reason=%s] "
                "after=[found=%s valid=%s trusted=%s pts=%s table_pts=%s yaw=%s dist=%s reason=%s]",
                edge_frame_id,
                table_edge_source,
                is_current_frame,
                same_frame_mapping_ok,
                before_edge_found, before_edge_valid, before_edge_trusted, before_point_count, before_table_point_count, before_yaw, before_dist, before_reason,
                after_edge_found, after_edge_valid, after_edge_trusted, after_point_count, after_table_point_count, after_yaw, after_dist, after_reason
            )

        ctx.stage_state["table_edge_obs"] = dict(table_edge_obs)
        if _has_edge_candidate(table_edge_obs):
            ctx.stage_state["_table_edge_obs_cache"] = dict(table_edge_obs)
        
        status_changed = bool(
            (after_edge_found != prev_found)
            or (after_edge_valid != prev_valid)
            or (after_edge_trusted != prev_trusted)
            or (bool(table_edge_obs.get("table_bbox_current_found", False)) != prev_bbox_found)
            or (bool(table_edge_obs.get("table_bbox_control_valid", False)) != prev_bbox_valid)
            or (bool(table_edge_obs.get("yolo_table_fresh", False)) != prev_yolo_fresh)
        )
        obs_seq = table_edge_obs.get("obs_seq")
        edge_identity = (obs_seq,) if obs_seq is not None else (edge_frame_id, table_edge_obs.get("trace_id"))
        local_identity = _local_perception_identity(local_perception)
        obs_identity = (edge_identity, local_identity)
        previous_identity = ctx.stage_state.get("last_seen_table_edge_identity")
        is_new_identity = obs_identity != previous_identity
        if is_new_identity:
            ctx.stage_state["last_seen_table_edge_identity"] = obs_identity
            ctx.stage_state["last_seen_table_edge_obs_seq"] = obs_seq
            ctx.stage_state["last_seen_table_edge_frame_id"] = edge_frame_id
        # A result is urgent only on first arrival; repeated scheduler reads are
        # retained as latest state but do not refresh observation freshness.
        local_updated = bool(local_identity is not None and is_new_identity)
        edge_updated = bool(table_edge_source == "results" and is_new_identity)
        force_send = bool(local_updated or edge_updated or status_changed)
        force_send_reasons = []
        if local_updated:
            force_send_reasons.append("local_perception_identity_changed")
        if edge_updated:
            force_send_reasons.append("current_edge_identity_changed")
        if status_changed:
            force_send_reasons.append("control_status_changed")
        table_edge_obs["force_send_reason"] = ",".join(force_send_reasons) if force_send_reasons else "deduplicated"

        return table_edge_obs, table_edge_source, force_send

    def tick(self, tick_input: StageTickInput, ctx: StageContext) -> Optional[StageOutput]:
        results = dict(tick_input.results or {})
        
        # DIAGNOSTIC LOG FOR ROUTING INVESTIGATION
        import logging
        logger = logging.getLogger("vision.stage.search")
        logger.info("[DIAG_STAGE_TICK] results_keys=%s", list(results.keys()))
        if "table_edge_obs" in results:
            obs = results["table_edge_obs"] or {}
            logger.info("[DIAG_STAGE_TICK] table_edge_obs present: frame_id=%s edge_found=%s", obs.get("frame_id"), obs.get("edge_found"))
        else:
            logger.info("[DIAG_STAGE_TICK] table_edge_obs is NOT in results keys")

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
                    default_factory=lambda: target_obs_from_payload(None, ctx.target_name),
                    result_factory=lambda payload: target_obs_from_results(payload, ctx.target_name),
                )
                target_obs = _annotate_target_obs(target_obs, ctx)
                target_obs["target_prewarm_active"] = True
                ctx.stage_state["target_obs"] = dict(target_obs)
            
            table_edge_obs, table_edge_source, force_send = self._process_table_edge_obs(
                results, ctx, tick_input, local_perception
            )
            
            perception = {"table_edge_obs": table_edge_obs}
            if target_obs is not None:
                perception["target_obs"] = target_obs
            elif search_kind != "TABLE_EDGE":
                perception["local_perception"] = local_perception
            source_payload = {"table_edge_obs": table_edge_source}
            if target_source:
                source_payload["target_obs"] = target_source
            
            signals = {}
            if force_send:
                signals["force_send"] = True
                
            return StageOutput(
                vision_obs=self.build_obs(
                    ctx,
                    status=RUNNING,
                    perception=perception,
                ),
                snapshot={
                    "generation": int(tick_input.generation),
                    "result_keys": sorted(results.keys()),
                    "source": source_payload,
                    "search_kind": ctx.stage_state.get("search_kind"),
                },
                signals=signals,
            )

        if mode == "FIND_TABLE":
            target_obs, target_source = resolve_stage_summary(
                results=results,
                stage_state=ctx.stage_state,
                state_key="target_obs",
                default_factory=lambda: target_obs_from_payload(None, ctx.target_name),
                result_factory=lambda payload: target_obs_from_results(payload, ctx.target_name),
            )
            target_obs = _annotate_target_obs(target_obs, ctx)
            return StageOutput(
                vision_obs=self.build_obs(
                    ctx,
                    status=RUNNING,
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
                default_factory=lambda: target_obs_from_payload(None, ctx.target_name),
                result_factory=lambda payload: target_obs_from_results(payload, ctx.target_name),
            )
            target_obs = _annotate_target_obs(target_obs, ctx)
            
            table_edge_obs, table_edge_source, force_send = self._process_table_edge_obs(
                results, ctx, tick_input, results.get("local_perception")
            )
            
            signals = {}
            if force_send:
                signals["force_send"] = True
                
            return StageOutput(
                vision_obs=self.build_obs(
                    ctx,
                    status=RUNNING,
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
                signals=signals,
            )

        return StageOutput(
            vision_obs=self.build_obs(ctx, status=RELAXING, perception={}),
            snapshot={
                "generation": int(tick_input.generation),
                "result_keys": sorted(results.keys()),
            },
        )
