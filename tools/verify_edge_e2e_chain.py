#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path
from statistics import median
from typing import Any, Dict, Iterable, List, Optional
from collections import Counter


def _read_texts(root: Path) -> str:
    chunks: List[str] = []
    if not root.exists():
        return ""
    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        if path.suffix not in {".log", ".out", ".txt", ".jsonl", ".json"}:
            continue
        try:
            chunks.append(path.read_text(errors="ignore"))
        except Exception:
            pass
    return "\n".join(chunks)


def _jsonl(path: Path) -> Iterable[Dict[str, Any]]:
    if not path.exists():
        return
    try:
        lines = path.read_text(errors="ignore").splitlines()
    except Exception:
        return
    for line in lines:
        try:
            obj = json.loads(line)
        except Exception:
            continue
        if isinstance(obj, dict):
            yield obj


def _latest_run() -> Path:
    runs = sorted(Path("logs/runs").glob("run_*"), key=lambda p: p.stat().st_mtime, reverse=True)
    if not runs:
        raise SystemExit("no logs/runs/run_* found")
    return runs[0]


def _count(pattern: str, text: str) -> int:
    return len(re.findall(pattern, text, flags=re.IGNORECASE | re.MULTILINE | re.DOTALL))


def _pct(values: List[float], pct: float) -> Optional[float]:
    vals = sorted(float(v) for v in values if isinstance(v, (int, float)) and math.isfinite(float(v)))
    if not vals:
        return None
    if len(vals) == 1:
        return vals[0]
    idx = (len(vals) - 1) * pct
    lo = int(math.floor(idx))
    hi = int(math.ceil(idx))
    if lo == hi:
        return vals[lo]
    return vals[lo] * (hi - idx) + vals[hi] * (idx - lo)


def _fmt(value: Optional[float], digits: int = 2) -> str:
    if value is None:
        return "n/a"
    return f"{value:.{digits}f}"


def _hz_from_intervals_ms(values: List[float]) -> Optional[float]:
    vals = [v for v in values if isinstance(v, (int, float)) and v > 0]
    if not vals:
        return None
    med = median(vals)
    if med <= 0:
        return None
    return 1000.0 / med


def _num(obj: Dict[str, Any], *keys: str) -> Optional[float]:
    for key in keys:
        value = obj.get(key)
        if value is None:
            continue
        try:
            return float(value)
        except Exception:
            pass
    return None


def _truthy(obj: Dict[str, Any], key: str) -> bool:
    return bool(obj.get(key))


def _ts(obj: Dict[str, Any]) -> Optional[float]:
    return _num(obj, "ts", "time")


def _vx_value(obj: Dict[str, Any]) -> float:
    return float(_num(obj, "vx", "vx_mps", "linear_x", "actual_vx_mps") or 0.0)


def _forward_stats(rows: List[Dict[str, Any]], gap_s: float = 0.35) -> Dict[str, Any]:
    ordered = sorted((obj for obj in rows if _ts(obj) is not None), key=lambda obj: float(_ts(obj) or 0.0))
    positive_ts = [float(_ts(obj) or 0.0) for obj in ordered if _vx_value(obj) > 1e-6]
    if not positive_ts:
        return {"count": 0, "first_ts": None, "max_consecutive": 0, "duration_s": 0.0}
    max_run = run = 1
    duration = 0.0
    run_start = positive_ts[0]
    prev = positive_ts[0]
    best_duration = 0.0
    for ts in positive_ts[1:]:
        if ts - prev <= gap_s:
            run += 1
            duration = ts - run_start
        else:
            max_run = max(max_run, run)
            best_duration = max(best_duration, duration)
            run = 1
            run_start = ts
            duration = 0.0
        prev = ts
    max_run = max(max_run, run)
    best_duration = max(best_duration, duration)
    return {"count": len(positive_ts), "first_ts": positive_ts[0], "max_consecutive": max_run, "duration_s": best_duration}


def _load_meta(run: Path) -> Dict[str, Any]:
    for rel in ("meta.json", "orchestrator/meta.json", "vision/meta.json", "orchestrator/run_summary.json"):
        p = run / rel
        if p.exists():
            try:
                data = json.loads(p.read_text(errors="ignore"))
            except Exception:
                continue
            if isinstance(data, dict):
                return data
    return {}


def analyze(run: Path) -> Dict[str, Any]:
    vision_text = _read_texts(run / "vision")
    orch_text = _read_texts(run / "orchestrator")
    table_rows = list(_jsonl(run / "orchestrator" / "table_edge_obs.jsonl"))
    vision_rows = list(_jsonl(run / "orchestrator" / "vision_obs.jsonl"))
    state_rows = list(_jsonl(run / "orchestrator" / "state_trace.jsonl"))
    cmd_rows = list(_jsonl(run / "orchestrator" / "cmd_vel.jsonl"))
    car_rows = list(_jsonl(run / "orchestrator" / "car_cmd.jsonl"))
    uart_rows = list(_jsonl(run / "orchestrator" / "uart_tx.jsonl"))
    gate_rows = list(_jsonl(run / "orchestrator" / "motion_gate_trace.jsonl"))
    control_rows = list(_jsonl(run / "orchestrator" / "control_summary.jsonl"))
    forward_bug_rows = list(_jsonl(run / "orchestrator" / "motion_forward_block_bug.jsonl"))
    sys_rows = list(_jsonl(run / "orchestrator" / "system_metrics.jsonl")) + list(_jsonl(run / "vision" / "system_metrics.jsonl"))
    perf_rows = list(_jsonl(run / "vision" / "perf_timing.jsonl")) + list(_jsonl(run / "vision" / "frame_timing.jsonl"))

    edge_publish_found = _count(r"\[EDGE_PUBLISH\][^\n]*edge_found=(?:1|true)", vision_text)
    edge_publish_valid = _count(r"\[EDGE_PUBLISH\][^\n]*edge_valid=(?:1|true)", vision_text)
    edge_publish_trusted = _count(r"\[EDGE_PUBLISH\][^\n]*edge_trusted=(?:1|true)", vision_text)
    scheduler_accepted = _count(r"\[PUBLISH_RESULT_TRACE\][^\n]*route=table_edge_obs[^\n]*accepted=true", vision_text)
    table_publish_success = _count(r"\[TABLE_EDGE_PUBLISH_GEN\][^\n]*route=table_edge_obs[^\n]*success=True", vision_text)
    stage_present = _count(r"\[DIAG_STAGE_TICK\][^\n]*table_edge_obs present[^\n]*edge_found=True", vision_text)
    selected_results = _count(r"\[EDGE_SELECTION_TRACE\][^\n]*selected_source=results", vision_text)
    selection_bug = _count(r"\[EDGE_SELECTION_BUG\]", vision_text)
    merge_bug = _count(r"\[EDGE_MERGE_OVERWRITE_BUG\]", vision_text)
    final_found = _count(r"\[EDGE_OBS_PAYLOAD_FINAL\].*?edge_found=true", vision_text)
    final_valid = _count(r"\[EDGE_OBS_PAYLOAD_FINAL\].*?edge_valid=true", vision_text)
    final_trusted = _count(r"\[EDGE_OBS_PAYLOAD_FINAL\].*?edge_trusted=true", vision_text)
    final_pts = _count(r"\[EDGE_OBS_PAYLOAD_FINAL\].*?point_count=[1-9]\d*", vision_text)
    final_tpts = _count(r"\[EDGE_OBS_PAYLOAD_FINAL\].*?table_point_count=[1-9]\d*", vision_text)

    orch_found = sum(1 for obj in table_rows if _truthy(obj, "edge_found"))
    orch_valid = sum(1 for obj in table_rows if _truthy(obj, "edge_valid"))
    orch_trusted = sum(1 for obj in table_rows if _truthy(obj, "edge_trusted"))
    orch_pts = sum(1 for obj in table_rows if int(obj.get("point_count") or 0) > 0)
    orch_tpts = sum(1 for obj in table_rows if int(obj.get("table_point_count") or 0) > 0)

    state_names = []
    for obj in state_rows:
        raw = obj.get("state") or obj.get("name") or obj.get("to_state") or obj.get("current_state")
        if raw is not None:
            state_names.append(str(raw))
    edge_control_states = {"CONTROLLED_APPROACH", "FINAL_LOCK", "EDGE_ADJUST", "AT_TABLE_EDGE"}
    edge_control_entered = any(any(name.upper() == s for s in edge_control_states) for name in state_names)
    if not edge_control_entered:
        edge_control_entered = bool(re.search(r"CONTROLLED_APPROACH|FINAL_LOCK|EDGE_ADJUST|edge_control", orch_text, re.IGNORECASE))

    intervals = [_num(obj, "table_edge_publish_interval_ms", "edge_update_interval_ms") for obj in table_rows]
    intervals = [v for v in intervals if v is not None]
    recv_intervals = [_num(obj, "table_edge_obs_recv_interval_ms", "orchestrator_recv_interval_ms") for obj in table_rows]
    recv_intervals = [v for v in recv_intervals if v is not None]
    yolo_ms = []
    for obj in perf_rows + vision_rows:
        value = _num(obj, "yolo_infer_ms", "infer_ms", "inference_ms")
        if value is not None:
            yolo_ms.append(value)
    obs_age = [_num(obj, "obs_total_age_ms", "table_edge_obs_age_ms", "age_ms") for obj in table_rows]
    obs_age = [v for v in obs_age if v is not None]
    tick_intervals = []
    last_ts = None
    for obj in state_rows:
        ts = _num(obj, "ts", "time")
        if ts is None:
            continue
        if last_ts is not None and ts >= last_ts:
            tick_intervals.append((ts - last_ts) * 1000.0)
        last_ts = ts
    cpu = [_num(obj, "cpu_percent", "process_cpu_percent", "vision_cpu_percent", "orch_cpu_percent") for obj in sys_rows]
    cpu = [v for v in cpu if v is not None]
    mem = [_num(obj, "rss_mb", "memory_mb", "process_rss_mb") for obj in sys_rows]
    mem = [v for v in mem if v is not None]
    vx = [_num(obj, "vx", "vx_mps", "linear_x") for obj in cmd_rows + car_rows]
    vx = [v for v in vx if v is not None]
    wz = [_num(obj, "wz", "wz_radps", "angular_z") for obj in cmd_rows + car_rows]
    wz = [v for v in wz if v is not None]
    cmd_vx_pos = sum(1 for obj in cmd_rows if (_num(obj, "vx", "vx_mps", "linear_x") or 0.0) > 1e-6)
    car_vx_pos = sum(1 for obj in car_rows if (_num(obj, "vx", "vx_mps", "linear_x", "actual_vx_mps") or 0.0) > 1e-6)
    uart_vx_pos = sum(1 for obj in uart_rows if (_num(obj, "vx", "vx_mps", "actual_vx_mps") or 0.0) > 1e-6)
    yolo_vx_pos = sum(
        1
        for obj in gate_rows + control_rows
        if bool(obj.get("yolo_table_visible") or obj.get("table_bbox_control_valid") or obj.get("yolo_table_control_valid"))
        and (_num(obj, "vx", "vx_mps", "actual_vx_mps") or 0.0) > 1e-6
    )
    edge_trusted_vx_pos = sum(
        1
        for obj in gate_rows + control_rows
        if bool(obj.get("edge_trusted") or obj.get("valid_for_control"))
        and (_num(obj, "vx", "vx_mps", "actual_vx_mps") or 0.0) > 1e-6
    )
    should_forward_rows = [
        obj
        for obj in gate_rows
        if bool(obj.get("yolo_table_visible") or obj.get("table_bbox_control_valid") or obj.get("edge_trusted"))
        and (_num(obj, "dist_err_m") or 0.0) > (_num(obj, "target_dist_m") or 0.0) + 0.02
        and abs(_num(obj, "yaw_err_rad") or 0.0) < (_num(obj, "hard_rotate_only_yaw_rad") or 0.45)
        and str(obj.get("forward_block_reason") or "").lower() not in {"soft_stale", "hard_stale", "dead", "vision_stale"}
        and str(obj.get("stale_level") or "").lower() not in {"soft_stale", "hard_stale", "dead"}
    ]
    should_forward_vx_pos = sum(1 for obj in should_forward_rows if (_num(obj, "vx_mps", "vx") or 0.0) > 1e-6)
    block_reasons = Counter(str(obj.get("forward_block_reason") or "") for obj in gate_rows + control_rows)
    state_sequence = []
    for obj in state_rows + gate_rows + control_rows:
        state = str(obj.get("state") or obj.get("to_state") or "").strip()
        if state and (not state_sequence or state_sequence[-1] != state):
            state_sequence.append(state)
    state_duration = Counter()
    last_state = None
    last_ts = None
    for obj in sorted(gate_rows + control_rows + state_rows, key=lambda x: _num(x, "ts", "time") or 0.0):
        ts = _num(obj, "ts", "time")
        state = str(obj.get("state") or obj.get("to_state") or "").strip()
        if ts is not None and last_state and last_ts is not None and ts >= last_ts:
            state_duration[last_state] += float(ts - last_ts)
        if state:
            last_state = state
            last_ts = ts
    detection_rows = [
        obj
        for obj in gate_rows + control_rows + table_rows
        if bool(
            obj.get("yolo_table_visible")
            or obj.get("table_bbox_control_valid")
            or obj.get("yolo_table_control_valid")
            or obj.get("edge_valid")
            or obj.get("edge_trusted")
        )
        and _ts(obj) is not None
    ]
    first_detection_ts = min((float(_ts(obj) or 0.0) for obj in detection_rows), default=None)
    cmd_forward = _forward_stats(cmd_rows)
    car_forward = _forward_stats(car_rows)
    uart_forward = _forward_stats(uart_rows)
    combined_forward = _forward_stats(cmd_rows + car_rows + gate_rows + control_rows)
    first_forward_ts = combined_forward["first_ts"]
    detection_to_forward_delay_s = (
        max(0.0, float(first_forward_ts) - float(first_detection_ts))
        if first_detection_ts is not None and first_forward_ts is not None
        else None
    )

    meta = _load_meta(run)
    dry_run = bool(
        meta.get("dry_run")
        or meta.get("ORCH_SERIAL_DRY_RUN")
        or re.search(r"ORCH_SERIAL_DRY_RUN=1|dry[-_ ]run\s*[:=]\s*(?:1|true)", vision_text + orch_text, re.IGNORECASE)
    )

    fail_stage = "pass"
    if edge_publish_found <= 0:
        fail_stage = "detector"
    elif scheduler_accepted <= 0 or table_publish_success <= 0:
        fail_stage = "scheduler"
    elif stage_present <= 0 or selected_results <= 0 or selection_bug > 0:
        fail_stage = "search_selection"
    elif merge_bug > 0:
        fail_stage = "merge_overwrite"
    elif final_found <= 0 or final_pts <= 0:
        fail_stage = "final_payload"
    elif orch_found <= 0 or orch_pts <= 0 or (edge_publish_trusted > 0 and orch_trusted <= 0):
        fail_stage = "orchestrator_parser"
    elif orch_trusted > 0 and not edge_control_entered:
        fail_stage = "orchestrator_control_gate"
    perception_verdict = "PASS" if fail_stage == "pass" else "FAIL"

    motion_fail_stage = "pass"
    has_forward_opportunity = bool(should_forward_rows or yolo_vx_pos or edge_trusted_vx_pos or orch_found > 0)
    non_stale_forward_bug_rows = [
        obj
        for obj in forward_bug_rows
        if str(obj.get("forward_block_reason") or "").lower() not in {"soft_stale", "hard_stale", "dead", "vision_stale"}
        and str(obj.get("stale_level") or "").lower() not in {"soft_stale", "hard_stale", "dead"}
    ]
    if has_forward_opportunity and cmd_vx_pos <= 0 and car_vx_pos <= 0:
        motion_fail_stage = "motion_cmd_no_forward"
    elif detection_to_forward_delay_s is None or detection_to_forward_delay_s > 2.0:
        motion_fail_stage = "motion_forward_delay"
    elif combined_forward["max_consecutive"] < 5 and combined_forward["duration_s"] < 0.8:
        motion_fail_stage = "motion_not_sustained"
    elif should_forward_rows and should_forward_vx_pos <= 0:
        motion_fail_stage = "motion_gate_forward_blocked"
    elif non_stale_forward_bug_rows:
        motion_fail_stage = "motion_forward_block_bug"
    elif not dry_run and (uart_forward["max_consecutive"] < 5 and uart_forward["duration_s"] < 0.8):
        motion_fail_stage = "uart_forward_not_sustained"
    motion_verdict = "PASS" if motion_fail_stage == "pass" else "FAIL"
    verdict = "PASS" if perception_verdict == "PASS" and motion_verdict == "PASS" else "FAIL"
    overall_fail_stage = fail_stage if perception_verdict == "FAIL" else motion_fail_stage

    return {
        "run": str(run),
        "edge_publish": (edge_publish_found, edge_publish_valid, edge_publish_trusted),
        "scheduler_accepted": scheduler_accepted,
        "table_publish_success": table_publish_success,
        "stage_present": stage_present,
        "selected_results": selected_results,
        "selection_bug": selection_bug,
        "merge_bug": merge_bug,
        "final": (final_found, final_valid, final_trusted, final_pts, final_tpts),
        "orchestrator": (len(table_rows), orch_found, orch_valid, orch_trusted, orch_pts, orch_tpts),
        "edge_control_entered": edge_control_entered,
        "verdict": verdict,
        "perception_chain": perception_verdict,
        "motion_chain": motion_verdict,
        "fail_stage": overall_fail_stage,
        "motion_fail_stage": motion_fail_stage,
        "motion": {
            "cmd_vel_vx_gt0": cmd_vx_pos,
            "car_cmd_vx_gt0": car_vx_pos,
            "uart_tx_vx_gt0": uart_vx_pos,
            "first_yolo_or_edge_ts": first_detection_ts,
            "first_forward_ts": first_forward_ts,
            "detection_to_forward_delay_s": detection_to_forward_delay_s,
            "cmd_vel_forward": cmd_forward,
            "car_cmd_forward": car_forward,
            "uart_tx_forward": uart_forward,
            "max_consecutive_forward_count": combined_forward["max_consecutive"],
            "forward_duration_s": combined_forward["duration_s"],
            "yolo_visible_while_vx_gt0": yolo_vx_pos,
            "edge_trusted_while_vx_gt0": edge_trusted_vx_pos,
            "should_forward_rows": len(should_forward_rows),
            "should_forward_vx_gt0": should_forward_vx_pos,
            "motion_forward_block_bug": len(non_stale_forward_bug_rows),
            "forward_block_reason_top": block_reasons.most_common(8),
            "state_duration_s": dict(state_duration.most_common(10)),
            "state_transition_sequence": state_sequence[:30],
        },
        "speed": {
            "edge_update_hz": _hz_from_intervals_ms(intervals),
            "table_edge_obs_recv_hz": _hz_from_intervals_ms(recv_intervals),
            "yolo_p50_ms": _pct(yolo_ms, 0.50),
            "yolo_p95_ms": _pct(yolo_ms, 0.95),
            "obs_age_p50_ms": _pct(obs_age, 0.50),
            "obs_age_p90_ms": _pct(obs_age, 0.90),
            "state_tick_p50_ms": _pct(tick_intervals, 0.50),
            "state_tick_p90_ms": _pct(tick_intervals, 0.90),
            "cpu_p50": _pct(cpu, 0.50),
            "cpu_p95": _pct(cpu, 0.95),
            "memory_p50_mb": _pct(mem, 0.50),
            "memory_p95_mb": _pct(mem, 0.95),
            "dry_run": dry_run,
            "planned_vx_range": (min(vx), max(vx)) if vx else None,
            "planned_wz_range": (min(wz), max(wz)) if wz else None,
        },
    }


def print_report(result: Dict[str, Any]) -> None:
    speed = result["speed"]
    print(f"Run: {result['run']}")
    print(f"EDGE_PUBLISH found/valid/trusted: {result['edge_publish'][0]} / {result['edge_publish'][1]} / {result['edge_publish'][2]}")
    print(f"Scheduler publish accepted: {result['scheduler_accepted']}")
    print(f"TableEdge publish success: {result['table_publish_success']}")
    print(f"SearchStage table_edge_obs present: {result['stage_present']}")
    print(f"SearchStage selected_source=results: {result['selected_results']}")
    print(f"EDGE_SELECTION_BUG: {result['selection_bug']}")
    print(f"EDGE_MERGE_OVERWRITE_BUG: {result['merge_bug']}")
    f = result["final"]
    print(f"Final payload edge_found/valid/trusted: {f[0]} / {f[1]} / {f[2]}")
    print(f"Final payload point_count/table_point_count >0: {f[3]} / {f[4]}")
    o = result["orchestrator"]
    print(f"Orchestrator table_edge_obs total: {o[0]}")
    print(f"Orchestrator edge_found/valid/trusted: {o[1]} / {o[2]} / {o[3]}")
    print(f"Orchestrator point_count/table_point_count >0: {o[4]} / {o[5]}")
    print(f"Orchestrator edge control entered: {'yes' if result['edge_control_entered'] else 'no'}")
    m = result["motion"]
    print("Motion summary:")
    print(f"  cmd_vel vx>0 count: {m['cmd_vel_vx_gt0']}")
    print(f"  car_cmd vx>0 count: {m['car_cmd_vx_gt0']}")
    print(f"  uart_tx vx>0 count: {m['uart_tx_vx_gt0']}")
    print(f"  first_yolo_or_edge_ts: {m['first_yolo_or_edge_ts']}")
    print(f"  first_forward_ts: {m['first_forward_ts']}")
    print(f"  detection_to_forward_delay_s: {_fmt(m['detection_to_forward_delay_s'])}")
    print(f"  max_consecutive_forward_count: {m['max_consecutive_forward_count']}")
    print(f"  forward_duration_s: {_fmt(m['forward_duration_s'])}")
    print(f"  YOLO visible while vx>0 count: {m['yolo_visible_while_vx_gt0']}")
    print(f"  edge_trusted while vx>0 count: {m['edge_trusted_while_vx_gt0']}")
    print(f"  should-forward rows / vx>0: {m['should_forward_rows']} / {m['should_forward_vx_gt0']}")
    print(f"  MOTION_FORWARD_BLOCK_BUG: {m['motion_forward_block_bug']}")
    print(f"  forward_block_reason top: {m['forward_block_reason_top']}")
    print(f"  state duration s top: {m['state_duration_s']}")
    print(f"  state transition sequence: {m['state_transition_sequence']}")
    print("Speed summary:")
    print(f"  Vision edge_update_hz: {_fmt(speed['edge_update_hz'])}")
    print(f"  orchestrator table_edge_obs_recv_hz: {_fmt(speed['table_edge_obs_recv_hz'])}")
    print(f"  YOLO infer p50 / p95 ms: {_fmt(speed['yolo_p50_ms'])} / {_fmt(speed['yolo_p95_ms'])}")
    print(f"  obs_age_at_consume p50 / p90 ms: {_fmt(speed['obs_age_p50_ms'])} / {_fmt(speed['obs_age_p90_ms'])}")
    print(f"  state_machine_tick_interval p50 / p90 ms: {_fmt(speed['state_tick_p50_ms'])} / {_fmt(speed['state_tick_p90_ms'])}")
    print(f"  CPU p50 / p95: {_fmt(speed['cpu_p50'])} / {_fmt(speed['cpu_p95'])}")
    print(f"  memory p50 / p95 MB: {_fmt(speed['memory_p50_mb'])} / {_fmt(speed['memory_p95_mb'])}")
    print(f"  dry_run status: {speed['dry_run']}")
    print(f"  planned vx range: {speed['planned_vx_range']}")
    print(f"  planned wz range: {speed['planned_wz_range']}")
    print(f"perception_chain: {result['perception_chain']}")
    print(f"motion_chain: {result['motion_chain']}")
    print(f"Verdict: {result['verdict']}")
    print(f"Fail stage: {result['fail_stage']}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Verify VISTA table-edge E2E chain for a run directory.")
    parser.add_argument("--run", type=Path, default=None, help="logs/runs/run_xxx directory. Defaults to latest run.")
    args = parser.parse_args()
    run = args.run or _latest_run()
    print_report(analyze(run))


if __name__ == "__main__":
    main()
