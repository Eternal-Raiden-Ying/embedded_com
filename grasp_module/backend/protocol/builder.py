"""
Protocol response builder.

Defines the frozen downstream output format.  The format uses semantic
versioning: MAJOR bumps on field removal / rename / semantic change; MINOR
bumps on backward-compatible additions.

Status semantics (v1.1)
------------------------
- ``success``             — at least one target passed all filters; ready to execute.
- ``reposition_required`` — object detected, grasps generated, but none passed the
  feasible-angle or score thresholds; the robot *may* succeed from another viewpoint.
- ``failure``             — hard failure that repositioning cannot fix: YOLO found
  nothing, or GraspNet produced zero grasps.
"""

from __future__ import annotations

FORMAT_VERSION = "1.1"

_REASON_NO_DETECTION = "no_detection"
_REASON_NO_GRASP = "no_grasp_detected"
_REASON_NO_FEASIBLE = "no_feasible_grasp"
_REASON_SCORE_LOW = "score_below_threshold"


def build_downstream_response(
    grasp_results,
    protocol_targets,
    predictor_cfgs,
    yolo_info=None,
    requested_class_id=None,
):
    """
    Build the downstream protocol response.

    Args:
        grasp_results: GraspGroup or None (raw grasps after collision filtering).
        protocol_targets: list[dict] — feasible targets sorted by confidence desc.
        predictor_cfgs: configuration object (protocol_min_score, response_max_targets,
                        protocol_feasible_angle_deg).
        yolo_info: optional dict — YOLO detection result from
                   ``RealSenseGraspPredictor.get_last_yolo_info()``.
        requested_class_id: optional int — the class_id originally requested by the
                            caller (may differ from the resolved class_id when
                            fallback is in use).

    Returns:
        dict — a JSON-serialisable response matching the frozen protocol schema.
    """
    raw_grasp_count = 0 if grasp_results is None else len(grasp_results)
    feasible_count = len(protocol_targets)
    detection = _build_detection(yolo_info, requested_class_id)

    # --- failure: YOLO found nothing ---
    if detection is not None and not detection["found"]:
        return _respond(
            status="failure",
            reason=_REASON_NO_DETECTION,
            message=_no_detection_message(detection),
            detection=detection,
            grasp_count=0,
            feasible_count=0,
            output_count=0,
        )

    # --- failure: YOLO found something but GraspNet returned nothing ---
    if raw_grasp_count == 0:
        return _respond(
            status="failure",
            reason=_REASON_NO_GRASP,
            message=_no_grasp_message(detection),
            detection=detection,
            grasp_count=0,
            feasible_count=0,
            output_count=0,
        )

    # --- reposition_required: grasps exist but no feasible approach ---
    if feasible_count == 0:
        angle = float(getattr(predictor_cfgs, "protocol_feasible_angle_deg", 5.0))
        return _respond(
            status="reposition_required",
            reason=_REASON_NO_FEASIBLE,
            message=_no_feasible_message(detection, raw_grasp_count, angle),
            detection=detection,
            grasp_count=raw_grasp_count,
            feasible_count=0,
            output_count=0,
        )

    # --- reposition_required: feasible grasps exist but all below score ---
    min_score = float(getattr(predictor_cfgs, "protocol_min_score", 0.0))
    max_targets = max(1, int(getattr(predictor_cfgs, "response_max_targets", 5)))
    output_targets = [
        target for target in protocol_targets if target["confidence"] >= min_score
    ][:max_targets]

    if not output_targets:
        best_conf = protocol_targets[0]["confidence"]
        return _respond(
            status="reposition_required",
            reason=_REASON_SCORE_LOW,
            message=_score_low_message(detection, feasible_count, min_score, best_conf),
            detection=detection,
            grasp_count=raw_grasp_count,
            feasible_count=feasible_count,
            output_count=0,
        )

    # --- success ---
    return _respond(
        status="success",
        reason=None,
        message=_success_message(detection, feasible_count, len(output_targets)),
        detection=detection,
        grasp_count=raw_grasp_count,
        feasible_count=feasible_count,
        output_count=len(output_targets),
        targets=output_targets,
    )


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _build_detection(yolo_info, requested_class_id):
    """Build the ``detection`` object from YOLO info and caller intent."""
    if yolo_info is None:
        return None

    resolved_class_id = yolo_info.get("class_id") or requested_class_id
    similar = (
        requested_class_id is not None
        and resolved_class_id is not None
        and int(resolved_class_id) != int(requested_class_id)
    )

    return {
        "requested_class_id": requested_class_id,
        "resolved_class_id": resolved_class_id,
        "found": bool(yolo_info.get("found", False)),
        "confidence": yolo_info.get("confidence"),
        "detection_count": int(yolo_info.get("count", 0) or 0),
        "multiple_detections": bool(yolo_info.get("multiple_detections", False)),
        "similar_detection_result": similar,
        "bbox": yolo_info.get("bbox"),
    }


def _respond(status, reason, message, detection, grasp_count, feasible_count,
             output_count, targets=None):
    resp = {
        "format_version": FORMAT_VERSION,
        "status": status,
        "reason": reason,
        "message": message,
        "detection": detection,
        "grasp_count": grasp_count,
        "feasible_count": feasible_count,
        "output_count": output_count,
        "targets": targets if targets is not None else [],
    }
    return resp


def _det_cid(det):
    """Safe accessor for the resolved class_id in a detection object."""
    if det is None:
        return "?"
    return det.get("resolved_class_id", "?")


def _no_detection_message(det):
    cid = det["requested_class_id"] if det else "?"
    return f"YOLO did not detect class_id={cid}"


def _no_grasp_message(det):
    cid = _det_cid(det)
    if det is None:
        return f"GraspNet produced no grasps (detection info unavailable)"
    conf = det.get("confidence")
    count = det.get("detection_count", 0)
    conf_str = f"{conf:.4f}" if conf is not None else "N/A"
    return (
        f"YOLO detected class_id={cid} (conf={conf_str}, "
        f"{count} instance(s)) but GraspNet produced no grasps"
    )


def _no_feasible_message(det, raw_count, angle):
    cid = _det_cid(det)
    return (
        f"YOLO detected class_id={cid}, {raw_count} grasps generated, "
        f"but all exceed the feasible angle threshold of {angle} deg"
    )


def _score_low_message(det, feasible_count, min_score, best_conf):
    cid = _det_cid(det)
    return (
        f"YOLO detected class_id={cid}, {feasible_count} feasible grasps, "
        f"but all below the minimum score threshold of {min_score} "
        f"(best: {best_conf:.4f})"
    )


def _success_message(det, feasible_count, output_count):
    cid = _det_cid(det)
    return (
        f"YOLO detected class_id={cid}, {feasible_count} feasible grasps, "
        f"{output_count} passed score filter"
    )
