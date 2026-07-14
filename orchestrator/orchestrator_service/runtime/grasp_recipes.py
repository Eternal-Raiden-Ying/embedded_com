#!/usr/bin/env python3
"""Validated, immutable fixed-grasp recipes loaded once at service startup."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re
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
class GraspStep:
    command: str
    expect_ack: bool
    timeout_s: float
    settle_after_s: float
    success_ack: str


@dataclass(frozen=True)
class GraspRecipe:
    name: str
    recipe_type: str
    description: str
    steps: Tuple[GraspStep, ...]
    stop_on_failure: bool = True


class GraspRecipeRegistry:
    def __init__(self, recipes: Dict[str, GraspRecipe], errors: Tuple[str, ...] = ()):
        self._recipes = dict(recipes)
        self.errors = tuple(errors)

    @property
    def enabled_names(self) -> Tuple[str, ...]:
        return tuple(sorted(self._recipes))

    def get(self, canonical_target: str) -> Optional[GraspRecipe]:
        return self._recipes.get(str(canonical_target or "").strip())

    @classmethod
    def load(cls, path: Path) -> "GraspRecipeRegistry":
        errors = []
        recipes: Dict[str, GraspRecipe] = {}
        try:
            with Path(path).open("r", encoding="utf-8") as handle:
                payload = yaml.load(handle, Loader=_UniqueKeyLoader) or {}
        except Exception as exc:
            return cls({}, (f"config_load_failed:{exc}",))
        if int(payload.get("version", 0) or 0) != 1:
            return cls({}, (f"unsupported_version:{payload.get('version')!r}",))
        defaults = payload.get("defaults") if isinstance(payload.get("defaults"), dict) else {}
        default_timeout = float(defaults.get("command_timeout_s", 8.0) or 8.0)
        default_settle = float(defaults.get("settle_after_step_s", 0.0) or 0.0)
        default_stop = bool(defaults.get("stop_on_failure", True))
        raw_recipes = payload.get("recipes") if isinstance(payload.get("recipes"), dict) else {}
        for raw_name, raw_recipe in raw_recipes.items():
            name = str(raw_name or "").strip()
            try:
                spec = resolve_target(name)
                if spec is None or spec.canonical_target != name:
                    raise ValueError("recipe key is not a canonical target")
                if not isinstance(raw_recipe, dict):
                    raise ValueError("recipe must be a mapping")
                if not bool(raw_recipe.get("enabled", False)):
                    continue
                if str(raw_recipe.get("recipe_type") or "") != "fixed_sequence":
                    raise ValueError("recipe_type must be fixed_sequence")
                raw_steps = raw_recipe.get("steps")
                if not isinstance(raw_steps, list) or not raw_steps:
                    raise ValueError("enabled recipe requires at least one step")
                steps = []
                for index, raw_step in enumerate(raw_steps):
                    if not isinstance(raw_step, dict):
                        raise ValueError(f"step {index} must be a mapping")
                    command = str(raw_step.get("command") or "").strip()
                    if not command or re.fullmatch(r"[A-Za-z][A-Za-z0-9_]*(?: [^\r\n]+)?", command) is None:
                        raise ValueError(f"step {index} has invalid command")
                    timeout_s = float(raw_step.get("timeout_s", default_timeout) or 0.0)
                    if timeout_s <= 0.0:
                        raise ValueError(f"step {index} timeout_s must be positive")
                    expect_ack = bool(raw_step.get("expect_ack", True))
                    success_ack = str(raw_step.get("success_ack") or "").strip()
                    if expect_ack and not success_ack:
                        raise ValueError(f"step {index} requires success_ack")
                    settle_s = float(raw_step.get("settle_after_s", default_settle) or 0.0)
                    if settle_s < 0.0:
                        raise ValueError(f"step {index} settle_after_s must be non-negative")
                    steps.append(GraspStep(command, expect_ack, timeout_s, settle_s, success_ack))
                recipes[name] = GraspRecipe(
                    name=name,
                    recipe_type="fixed_sequence",
                    description=str(raw_recipe.get("description") or ""),
                    steps=tuple(steps),
                    stop_on_failure=bool(raw_recipe.get("stop_on_failure", default_stop)),
                )
            except Exception as exc:
                errors.append(f"recipe_disabled:{name or '<empty>'}:{exc}")
        return cls(recipes, tuple(errors))
