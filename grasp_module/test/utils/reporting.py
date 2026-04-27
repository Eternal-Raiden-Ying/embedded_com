def build_downstream_response(grasp_results, protocol_targets, predictor_cfgs):
    raw_grasp_count = 0 if grasp_results is None else len(grasp_results)
    if raw_grasp_count == 0:
        return {
            "status": "reposition_required",
            "grasp_count": 0,
            "feasible_count": 0,
            "output_count": 0,
            "targets": [],
            "reason": "no_grasp_detected",
            "message": "placeholder",
        }

    feasible_count = len(protocol_targets)
    if feasible_count == 0:
        return {
            "status": "reposition_required",
            "grasp_count": raw_grasp_count,
            "feasible_count": 0,
            "output_count": 0,
            "targets": [],
            "reason": "no_feasible_grasp",
            "message": "placeholder",
        }

    min_score = float(getattr(predictor_cfgs, "protocol_min_score", 0.0))
    max_targets = max(1, int(getattr(predictor_cfgs, "response_max_targets", 5)))
    output_targets = [target for target in protocol_targets if target["confidence"] >= min_score][:max_targets]
    if not output_targets:
        return {
            "status": "reposition_required",
            "grasp_count": raw_grasp_count,
            "feasible_count": feasible_count,
            "output_count": 0,
            "targets": [],
            "reason": "score_below_threshold",
            "message": "placeholder",
        }

    return {
        "status": "success",
        "grasp_count": raw_grasp_count,
        "feasible_count": feasible_count,
        "output_count": len(output_targets),
        "targets": output_targets,
    }


def summarize_top_raw_grasp(grasp_results):
    if grasp_results is None or len(grasp_results) == 0:
        return None

    top_grasp = grasp_results[0]
    return {
        "score": float(top_grasp.score),
        "translation": [float(v) for v in top_grasp.translation],
        "rotation_matrix": [[float(v) for v in row] for row in top_grasp.rotation_matrix],
        "width": float(top_grasp.width),
        "depth": float(top_grasp.depth),
    }
