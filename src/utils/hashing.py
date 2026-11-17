"""Stable hashing for cache keys."""

import hashlib
import json
from typing import Any


def stable_hash(payload: Any) -> str:
    """SHA-256 of the canonical JSON form: same value → same key on every run and platform."""
    canonical = json.dumps(payload, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()
