"""Simple file-based data cache with TTL support.

Provides lightweight JSON-backed caching keyed by SHA-256 hashed
identifiers. Cached entries carry a timestamp and are considered
expired once the configured time-to-live window has elapsed.
Cache files are stored under the project-level ``.cache`` directory.
"""

import json
import hashlib
from pathlib import Path
from datetime import datetime, timedelta

_CACHE_DIR = Path(__file__).resolve().parent.parent / ".cache"


def _ensure_dir():
    """Create the cache directory if it does not already exist.

    Returns:
        Path object pointing to the cache directory, guaranteed to
        exist after this call.
    """
    _CACHE_DIR.mkdir(parents=True, exist_ok=True)
    return _CACHE_DIR


def cache_key(prefix, *args):
    """Build a short hexadecimal cache key from a prefix and variable arguments.

    Concatenates the prefix and all positional arguments with ``"|"``
    separators, hashes the result with SHA-256, and returns the first
    16 hexadecimal characters as a compact, collision-resistant key.

    Args:
        prefix: String namespace or category label for the cache entry
            (e.g. ``"eia_prices"``).
        *args: Additional string components that uniquely identify the
            cached payload within the prefix namespace (e.g. ticker,
            date range).

    Returns:
        16-character lowercase hexadecimal string derived from the
        SHA-256 digest of the joined inputs.
    """
    raw = f"{prefix}|{'|'.join(args)}"
    return hashlib.sha256(raw.encode()).hexdigest()[:16]


def get_cached(key, ttl_hours=24.0):
    """Retrieve a cached payload by key if it exists and has not expired.

    Reads the JSON cache file for ``key``, checks the embedded timestamp
    against the current time, and returns the stored payload only if it
    falls within the TTL window. Returns ``None`` on any read or
    deserialisation error so callers can proceed to re-fetch the data.

    Args:
        key: 16-character hex cache key as returned by ``cache_key``.
        ttl_hours: Maximum age of the cached entry in hours before it
            is considered expired. Defaults to 24.0.

    Returns:
        The cached payload object (any JSON-serialisable type) if the
        entry exists and is within the TTL, otherwise ``None``.
    """
    path = _ensure_dir() / f"{key}.json"
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text())
        ts = datetime.fromisoformat(data["_ts"])
        if datetime.now() - ts > timedelta(hours=ttl_hours):
            return None
        return data.get("payload")
    except Exception:
        return None


def set_cached(key, payload):
    """Persist a payload to the cache under the given key.

    Writes a JSON file containing the payload and the current ISO-format
    timestamp to the cache directory. Any serialisation or I/O errors
    are silently suppressed so that a cache write failure never disrupts
    the calling code.

    Args:
        key: 16-character hex cache key as returned by ``cache_key``.
        payload: JSON-serialisable object to store. Non-serialisable
            values are coerced to strings via ``default=str``.
    """
    path = _ensure_dir() / f"{key}.json"
    try:
        data = {"_ts": datetime.now().isoformat(), "payload": payload}
        path.write_text(json.dumps(data, default=str))
    except Exception:
        pass
