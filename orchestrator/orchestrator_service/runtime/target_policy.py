"""Single capability dispatcher for post-lock target behavior."""
from __future__ import annotations
from dataclasses import dataclass
from common.target_catalog import TargetSpec

@dataclass(frozen=True)
class LockedTargetRoute:
    next_state: str
    tts_event: str

def route_locked_target(spec: TargetSpec) -> LockedTargetRoute:
    if spec.action_policy == "locate_and_ring":
        return LockedTargetRoute("LOCATE_GUIDANCE_ACTIVE", "LOCATE_TARGET_FOUND")
    if spec.action_policy == "fixed_grasp" and spec.support_status == "ready" and spec.grasp_recipe:
        return LockedTargetRoute("GRASP", "")
    if spec.action_policy == "fixed_grasp":
        return LockedTargetRoute("DONE", "RECIPE_NOT_READY")
    return LockedTargetRoute("DONE", "UNSUPPORTED_GRASP_TARGET")
