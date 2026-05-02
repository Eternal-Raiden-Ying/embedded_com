from grasp_module.backend.protocol import build_downstream_response


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


def summarize_response(grasp_results, protocol_targets, cfgs):
    response = build_downstream_response(grasp_results, protocol_targets, cfgs)
    response["top_raw_grasp"] = summarize_top_raw_grasp(grasp_results)
    return response
