"""Compatibility facade for the repository-wide target catalog.

Do not add target mappings here. Voice, VISTA and Orchestrator all resolve
targets through :mod:`common.target_catalog`.
"""
from common.target_catalog import (
    TargetSpec,
    all_targets,
    resolve_target,
    supported_targets,
    target_display_name,
)

OBJECT_REGISTRY = all_targets()
TARGET_ALIAS_TO_CANONICAL = {
    alias: spec.canonical_name
    for spec in OBJECT_REGISTRY.values()
    for alias in (spec.canonical_name, *spec.aliases)
}
TARGET_NAME_TO_CLASS_ID = {
    alias: OBJECT_REGISTRY[canonical].class_id
    for alias, canonical in TARGET_ALIAS_TO_CANONICAL.items()
}
DISPLAY_NAMES = {name: spec.display_name for name, spec in OBJECT_REGISTRY.items()}

def target_to_canonical(target: str) -> str:
    spec = resolve_target(target)
    if spec is None:
        raise KeyError(f"Unknown target {target!r}. Known targets: {supported_targets()}")
    return spec.canonical_name

def target_to_class_name(target: str) -> str:
    return target_to_canonical(target)

def target_to_class_id(target: str) -> int:
    spec = resolve_target(target)
    if spec is None:
        raise KeyError(f"Unknown target {target!r}. Known targets: {supported_targets()}")
    return int(spec.class_id)
