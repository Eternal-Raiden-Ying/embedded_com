"""Canonical trained-YOLO target catalog shared by Voice, VISTA and Orchestrator."""
from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Optional, Tuple

import yaml


CATALOG_PATH = Path(__file__).resolve().parents[1] / "configs" / "target_catalog.yaml"
MODEL_CLASS_IDS = tuple(range(15))


@dataclass(frozen=True)
class TargetSpec:
    class_id: int
    canonical_name: str
    display_name: str
    aliases: Tuple[str, ...]
    role: str
    action_policy: str
    support_status: str
    grasp_recipe: Optional[str]
    selectable: bool

    # Compatibility names used by the existing orchestration protocol.
    @property
    def canonical_target(self) -> str:
        return self.canonical_name

    @property
    def class_name(self) -> str:
        return self.canonical_name


def _normalise(value: object) -> str:
    return "".join(str(value or "").strip().lower().split())


def validate_target_catalog(raw: Mapping[str, Any]) -> None:
    targets = raw.get("targets")
    if not isinstance(targets, Mapping):
        raise ValueError("target_catalog.targets must be a mapping")
    ids = []
    names = []
    aliases: Dict[str, str] = {}
    for key, value in targets.items():
        if not isinstance(value, Mapping):
            raise ValueError(f"target {key!r} must be a mapping")
        canonical = str(value.get("canonical_name") or key).strip()
        class_id = value.get("class_id")
        if not isinstance(class_id, int) or class_id not in MODEL_CLASS_IDS:
            raise ValueError(f"target {canonical!r} has invalid model class_id {class_id!r}")
        ids.append(class_id)
        names.append(canonical)
        policy = str(value.get("action_policy") or "")
        status = str(value.get("support_status") or "")
        recipe = value.get("grasp_recipe")
        if policy == "fixed_grasp" and status == "ready" and not recipe:
            raise ValueError(f"ready fixed_grasp target {canonical!r} requires grasp_recipe")
        if policy in {"place_destination", "docking_only", "locate_and_ring"} and recipe:
            raise ValueError(f"non-grasp target {canonical!r} must not configure grasp_recipe")
        for alias in (canonical, *(value.get("aliases") or ())):
            normalized = _normalise(alias)
            if not normalized:
                continue
            owner = aliases.setdefault(normalized, canonical)
            if owner != canonical:
                raise ValueError(f"dangerous target alias collision: {alias!r}: {owner!r} vs {canonical!r}")
    if len(ids) != len(set(ids)):
        raise ValueError("target model class_id values must be unique")
    if len(names) != len(set(names)):
        raise ValueError("target canonical_name values must be unique")
    if set(ids) != set(MODEL_CLASS_IDS):
        raise ValueError(f"target catalog must contain exactly trained IDs 0-14, got {sorted(ids)}")


@lru_cache(maxsize=1)
def load_target_catalog() -> Dict[str, TargetSpec]:
    with CATALOG_PATH.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle) or {}
    validate_target_catalog(raw)
    catalog: Dict[str, TargetSpec] = {}
    for key, value in raw["targets"].items():
        canonical = str(value.get("canonical_name") or key).strip()
        catalog[canonical] = TargetSpec(
            class_id=int(value["class_id"]), canonical_name=canonical,
            display_name=str(value["display_name"]),
            aliases=tuple(str(item) for item in (value.get("aliases") or ())),
            role=str(value["role"]), action_policy=str(value["action_policy"]),
            support_status=str(value["support_status"]),
            grasp_recipe=(str(value["grasp_recipe"]) if value.get("grasp_recipe") else None),
            selectable=bool(value.get("selectable", False)),
        )
    return catalog


def all_targets() -> Dict[str, TargetSpec]:
    return dict(load_target_catalog())


def resolve_target(value: object) -> Optional[TargetSpec]:
    needle = _normalise(value)
    if not needle:
        return None
    for spec in load_target_catalog().values():
        if needle == _normalise(spec.canonical_name) or needle in {_normalise(item) for item in spec.aliases}:
            return spec
    return None


def resolve_target_in_text(text: object, *, selectable_only: bool = True) -> Optional[TargetSpec]:
    source = _normalise(text)
    candidates = []
    for spec in load_target_catalog().values():
        if selectable_only and not spec.selectable:
            continue
        # navigation anchors and placement destinations are not voice find targets.
        if spec.action_policy in {"docking_only", "place_destination"}:
            continue
        for alias in (spec.canonical_name, *spec.aliases):
            normalized = _normalise(alias)
            if normalized and normalized in source:
                candidates.append((len(normalized), normalized, spec))
    return max(candidates, key=lambda item: (item[0], item[1]))[2] if candidates else None


def target_to_class_id(value: object) -> Optional[int]:
    spec = resolve_target(value)
    return spec.class_id if spec else None


def target_display_name(value: object) -> str:
    spec = value if isinstance(value, TargetSpec) else resolve_target(value)
    return spec.display_name if spec else str(value or "")


def supported_targets(*, selectable_only: bool = False) -> list[str]:
    specs: Iterable[TargetSpec] = load_target_catalog().values()
    if selectable_only:
        specs = (item for item in specs if item.selectable and item.action_policy not in {"docking_only", "place_destination"})
    return [item.canonical_name for item in specs]


def model_class_names() -> Tuple[str, ...]:
    return tuple(spec.canonical_name for spec in sorted(load_target_catalog().values(), key=lambda item: item.class_id))
