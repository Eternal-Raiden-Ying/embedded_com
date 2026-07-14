#!/usr/bin/env python3
"""Validated immutable POSE recipes loaded once at service startup."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Tuple

import yaml

from common.target_catalog import resolve_target


class _UniqueKeyLoader(yaml.SafeLoader):
    pass


def _construct_unique_mapping(loader, node, deep=False):
    mapping = {}
    for key_node, value_node in node.value:
        key = loader.construct_object(key_node, deep=deep)
        if key in mapping:
            raise ValueError(f"duplicate YAML key: {key!r}")
        mapping[key] = loader.construct_object(value_node, deep=deep)
    return mapping


_UniqueKeyLoader.add_constructor(
    yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG,
    _construct_unique_mapping,
)


@dataclass(frozen=True)
class PoseStep:
    name: str
    x: float
    y: float
    z: float
    pitch: float
    roll: float
    claw: float
    time_ms: int
    timeout_s: float
    settle_after_s: float = 0.0
    success_ack: str = "OK POSE"


@dataclass(frozen=True)
class PoseRecipe:
    name: str
    steps: Tuple[PoseStep, ...]
    stop_on_failure: bool = True


class GraspRecipeRegistry:
    def __init__(self, recipes: Dict[str, PoseRecipe], errors: Tuple[str, ...] = ()):
        self._recipes = dict(recipes)
        self.errors = tuple(errors)

    @property
    def enabled_names(self) -> Tuple[str, ...]:
        return tuple(sorted(self._recipes))

    def get(self, canonical_target: str) -> Optional[PoseRecipe]:
        return self._recipes.get(str(canonical_target or "").strip())

    @classmethod
    def load(cls, path: Path) -> "GraspRecipeRegistry":
        errors = []
        recipes: Dict[str, PoseRecipe] = {}
        try:
            with Path(path).open("r", encoding="utf-8") as handle:
                payload = yaml.load(handle, Loader=_UniqueKeyLoader) or {}
        except Exception as exc:
            return cls({}, (f"config_load_failed:{exc}",))
        if int(payload.get("version", 0) or 0) != 1:
            return cls({}, (f"unsupported_version:{payload.get('version')!r}",))
        if str(payload.get("protocol") or "").strip() != "pose_v1":
            return cls({}, (f"unsupported_protocol:{payload.get('protocol')!r}",))
        defaults = payload.get("defaults") if isinstance(payload.get("defaults"), dict) else {}
        default_ack = str(defaults.get("ack") or "OK POSE").strip()
        default_timeout = float(defaults.get("timeout_s", 6.0) or 6.0)
        default_stop = bool(defaults.get("stop_on_failure", True))
        raw_targets = payload.get("targets") if isinstance(payload.get("targets"), dict) else {}
        for raw_name, raw_recipe in raw_targets.items():
            name = str(raw_name or "").strip()
            try:
                spec = resolve_target(name)
                if spec is None or spec.canonical_target != name:
                    raise ValueError("recipe key is not a canonical target")
                if not isinstance(raw_recipe, dict):
                    raise ValueError("recipe must be a mapping")
                if not bool(raw_recipe.get("enabled", False)):
                    continue
                raw_steps = raw_recipe.get("steps")
                if not isinstance(raw_steps, list) or not raw_steps:
                    raise ValueError("enabled recipe requires at least one complete POSE step")
                steps = []
                for index, raw_step in enumerate(raw_steps):
                    if not isinstance(raw_step, dict):
                        raise ValueError(f"step {index} must be a mapping")
                    step_name = str(raw_step.get("name") or "").strip()
                    if not step_name:
                        raise ValueError(f"step {index} requires name")
                    values = {}
                    for field_name in ("x", "y", "z", "pitch", "roll", "claw"):
                        value = raw_step.get(field_name)
                        if isinstance(value, bool) or not isinstance(value, (int, float)):
                            raise ValueError(f"step {index} requires numeric {field_name}")
                        values[field_name] = float(value)
                    time_ms = raw_step.get("time_ms")
                    if isinstance(time_ms, bool) or not isinstance(time_ms, int) or time_ms <= 0:
                        raise ValueError(f"step {index} time_ms must be a positive integer")
                    timeout_s = float(raw_step.get("timeout_s", default_timeout) or 0.0)
                    if timeout_s <= 0.0:
                        raise ValueError(f"step {index} timeout_s must be positive")
                    settle_s = float(raw_step.get("settle_after_s", 0.0) or 0.0)
                    if settle_s < 0.0:
                        raise ValueError(f"step {index} settle_after_s must be non-negative")
                    ack = str(raw_step.get("ack") or default_ack).strip()
                    if not ack:
                        raise ValueError(f"step {index} requires ack")
                    steps.append(
                        PoseStep(
                            name=step_name,
                            time_ms=time_ms,
                            timeout_s=timeout_s,
                            settle_after_s=settle_s,
                            success_ack=ack,
                            **values,
                        )
                    )
                recipes[name] = PoseRecipe(
                    name=name,
                    steps=tuple(steps),
                    stop_on_failure=bool(raw_recipe.get("stop_on_failure", default_stop)),
                )
            except Exception as exc:
                errors.append(f"recipe_disabled:{name or '<empty>'}:{exc}")
        return cls(recipes, tuple(errors))
