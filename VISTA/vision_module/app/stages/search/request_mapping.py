#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from typing import Any

from ..base import normalize_upper


VALID_SEARCH_KINDS = {"TABLE_EDGE", "TARGET", "EDGE_FOLLOW_TARGET", "TARGET_ON_EDGE"}
EDGE_SEARCH_KINDS = {"TABLE_EDGE", "EDGE_FOLLOW_TARGET", "TARGET_ON_EDGE"}

LEGACY_MODE_ALIASES = {
    "TRACK_LOCAL": "FIND_OBJECT",
    "DEPTH_PERCEPTION": "FIND_EDGE",
    "TABLE_EDGE_PERCEPTION": "FIND_EDGE",
}


def canonical_search_mode(mode: Any, default: str = "FIND_OBJECT") -> str:
    token = normalize_upper(mode, default)
    return LEGACY_MODE_ALIASES.get(token, token)


def canonical_search_kind(search_kind: Any) -> str:
    return normalize_upper(search_kind, "")


def is_valid_search_kind(search_kind: Any) -> bool:
    return canonical_search_kind(search_kind) in VALID_SEARCH_KINDS


def invalid_search_kind_reason(search_kind: Any) -> str:
    return f"invalid search_kind: {canonical_search_kind(search_kind)}"


def mode_for_request(req, search_kind: str, default: str) -> str:
    kind = canonical_search_kind(search_kind)
    explicit = canonical_search_mode(getattr(req, "mode_hint", None), "")
    if explicit and not (explicit == "FIND_OBJECT" and kind in EDGE_SEARCH_KINDS):
        return explicit
    if kind in EDGE_SEARCH_KINDS:
        return "FIND_EDGE"
    return canonical_search_mode(default, "FIND_OBJECT")
