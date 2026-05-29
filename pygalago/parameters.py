# BSD License (http://www.galagosearch.org/license)
"""Python port of org.lemurproject.galago.utility.Parameters.

Parameters is a typed JSON-object wrapper that supports:
  - Typed getters/checkers (string, long/int, double, bool, list, map)
  - A backoff chain: if a key is missing here, look in the backoff Parameters
  - JSON serialization matching the Java format (sorted keys, custom spacing)
  - Parsing from strings, files, and byte arrays
"""

from __future__ import annotations

import json
import math
import os
from typing import Any, Iterator, Optional, Union

# ── Sentinel objects ──────────────────────────────────────────────────────────

class _NullMarker:
    """Represents a JSON null literal, distinct from a missing key."""
    _instance: Optional["_NullMarker"] = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __repr__(self) -> str:
        return "null"

    def __eq__(self, other: object) -> bool:
        return isinstance(other, _NullMarker)

    def __hash__(self) -> int:
        return hash("_NullMarker")


_NULL = _NullMarker()
_MISSING = object()  # sentinel for "key not present"


# ── JSON codec ────────────────────────────────────────────────────────────────

def _object_hook(d: dict) -> "Parameters":
    """Convert nested dicts to Parameters during JSON parsing."""
    p = Parameters()
    for k, v in d.items():
        if v is None:
            p._data[k] = _NULL
        else:
            p._data[k] = v
    return p


def _encode_value(v: Any) -> str:
    """Recursively encode a value to the Java-compatible JSON format."""
    if v is None or isinstance(v, _NullMarker):
        return "null"
    if isinstance(v, bool):
        return "true" if v else "false"
    if isinstance(v, int):
        return str(v)
    if isinstance(v, float):
        if math.isnan(v) or math.isinf(v):
            return f'"{v}"'
        # Use repr to avoid 0.1 → "0.1000000000000001" but keep Java parity
        return repr(v)
    if isinstance(v, str):
        return json.dumps(v)  # handles escaping
    if isinstance(v, list):
        parts = " , ".join(_encode_value(item) for item in v)
        return f"[ {parts} ]"
    if isinstance(v, Parameters):
        return str(v)
    raise TypeError(f"Cannot encode value of type {type(v)}: {v!r}")


# ── Parameters class ──────────────────────────────────────────────────────────

class Parameters:
    """Typed JSON-object wrapper with optional backoff chain.

    Mirrors the Java ``org.lemurproject.galago.utility.Parameters`` API while
    following Python conventions (snake_case methods, Pythonic iteration, etc.).
    CamelCase aliases are provided for callers that are closer to the Java code.
    """

    def __init__(self) -> None:
        self._data: dict[str, Any] = {}
        self._backoff: Optional[Parameters] = None

    # ── Factory methods ───────────────────────────────────────────────────────

    @classmethod
    def create(cls) -> "Parameters":
        return cls()

    @classmethod
    def parse_string(cls, text: str) -> "Parameters":
        return json.loads(text, object_hook=_object_hook)

    @classmethod
    def parse_file(cls, path: Union[str, os.PathLike]) -> "Parameters":
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f, object_hook=_object_hook)

    @classmethod
    def parse_bytes(cls, data: bytes) -> "Parameters":
        return json.loads(data.decode("utf-8"), object_hook=_object_hook)

    @classmethod
    def parse_args(cls, args: list[str]) -> "Parameters":
        """Parse command-line arguments of the form --key=value or --key value."""
        p = cls.create()
        i = 0
        while i < len(args):
            arg = args[i]
            if arg.startswith("--"):
                arg = arg[2:]
                if "=" in arg:
                    key, val = arg.split("=", 1)
                else:
                    key = arg
                    i += 1
                    val = args[i] if i < len(args) else "true"
                p.put(key, cls._coerce(val))
            i += 1
        return p

    @classmethod
    def _coerce(cls, s: str) -> Any:
        """Parse a string token into the most specific JSON type."""
        if s.lower() == "true":
            return True
        if s.lower() == "false":
            return False
        try:
            return int(s)
        except ValueError:
            pass
        try:
            return float(s)
        except ValueError:
            pass
        return s

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _get_raw(self, key: str) -> Any:
        """Return raw stored value or _MISSING (checks backoff)."""
        if key in self._data:
            return self._data[key]
        if self._backoff is not None:
            return self._backoff._get_raw(key)
        return _MISSING

    def _get_or_throw(self, key: str) -> Any:
        val = self._get_raw(key)
        if val is _MISSING:
            raise KeyError(f"No key '{key}' present in Parameters object.")
        return val

    # ── Getters ───────────────────────────────────────────────────────────────

    def get(self, key: str, default: Any = None) -> Any:
        """Return the value for *key*, or *default* if missing."""
        val = self._get_raw(key)
        if val is _MISSING:
            return default
        if isinstance(val, _NullMarker):
            return None
        return val

    def get_string(self, key: str) -> Optional[str]:
        val = self._get_or_throw(key)
        if isinstance(val, _NullMarker):
            return None
        if isinstance(val, str):
            return val
        raise TypeError(f"Key '{key}' is not a string, found {type(val).__name__}")

    def get_long(self, key: str) -> int:
        val = self._get_or_throw(key)
        if isinstance(val, bool):
            raise TypeError(f"Key '{key}' is boolean, not long")
        if isinstance(val, int):
            return val
        raise TypeError(f"Key '{key}' is not a long/int, found {type(val).__name__}")

    # Java alias
    def get_int(self, key: str) -> int:
        v = self.get_long(key)
        if v > 2**31 - 1 or v < -(2**31):
            raise OverflowError(f"Key '{key}'={v} is too large for a 32-bit int")
        return v

    def get_double(self, key: str) -> float:
        val = self._get_or_throw(key)
        if isinstance(val, bool):
            raise TypeError(f"Key '{key}' is boolean, not double")
        if isinstance(val, (int, float)):
            return float(val)
        raise TypeError(f"Key '{key}' is not a double, found {type(val).__name__}")

    def get_as_double(self, key: str) -> float:
        """Like get_double but also accepts string representations of floats."""
        val = self._get_or_throw(key)
        if isinstance(val, bool):
            raise TypeError(f"Key '{key}' is boolean, not double")
        if isinstance(val, (int, float)):
            return float(val)
        if isinstance(val, str):
            return float(val)
        raise TypeError(f"Key '{key}' is not convertible to double")

    def get_bool(self, key: str) -> bool:
        val = self._get_or_throw(key)
        if isinstance(val, bool):
            return val
        raise TypeError(f"Key '{key}' is not a boolean, found {type(val).__name__}")

    def get_list(self, key: str) -> list:
        val = self._get_or_throw(key)
        if isinstance(val, list):
            return val
        raise TypeError(f"Key '{key}' is not a list, found {type(val).__name__}")

    def get_as_list(self, key: str) -> list:
        """Always returns a list; scalars are wrapped, missing returns []."""
        val = self._get_raw(key)
        if val is _MISSING or isinstance(val, _NullMarker):
            return []
        if isinstance(val, list):
            return val
        return [val]

    def get_map(self, key: str) -> "Parameters":
        val = self._get_or_throw(key)
        if isinstance(val, Parameters):
            return val
        raise TypeError(f"Key '{key}' is not a Parameters map, found {type(val).__name__}")

    def get_as_string(self, key: str) -> str:
        """Return any primitive value as a string."""
        val = self._get_or_throw(key)
        if isinstance(val, bool):
            return str(val).lower()
        if isinstance(val, (int, float, str)):
            return str(val)
        raise TypeError(f"Key '{key}' is not a primitive")

    # ── Typed checkers ────────────────────────────────────────────────────────

    def is_string(self, key: str) -> bool:
        val = self._get_raw(key)
        return val is not _MISSING and isinstance(val, (str, _NullMarker))

    def is_long(self, key: str) -> bool:
        val = self._get_raw(key)
        return val is not _MISSING and not isinstance(val, bool) and isinstance(val, int)

    def is_double(self, key: str) -> bool:
        val = self._get_raw(key)
        return val is not _MISSING and not isinstance(val, bool) and isinstance(val, float)

    def is_bool(self, key: str) -> bool:
        val = self._get_raw(key)
        return val is not _MISSING and isinstance(val, bool)

    def is_list(self, key: str) -> bool:
        val = self._get_raw(key)
        return val is not _MISSING and isinstance(val, list)

    def is_map(self, key: str) -> bool:
        val = self._get_raw(key)
        return val is not _MISSING and isinstance(val, Parameters)

    def contains_key(self, key: str) -> bool:
        return self._get_raw(key) is not _MISSING

    # ── Setters ───────────────────────────────────────────────────────────────

    def put(self, key: str, value: Any) -> None:
        if value is None:
            self._data[key] = _NULL
        elif isinstance(value, os.PathLike):
            self._data[key] = os.fspath(value)
        else:
            self._data[key] = value

    def set(self, key: str, value: Any) -> None:
        self.put(key, value)

    def set_if_missing(self, key: str, value: Any) -> None:
        if not self.contains_key(key):
            self.put(key, value)

    def put_if_not_none(self, key: str, value: Any) -> None:
        if value is not None and key is not None:
            self.put(key, value)

    def extend_list(self, key: str, items: Any) -> None:
        """Append item(s) to a list stored at *key*, creating it if needed."""
        existing = self._data.get(key)
        if existing is None:
            existing = []
            self._data[key] = existing
        if not isinstance(existing, list):
            raise TypeError(f"Key '{key}' is not a list")
        if hasattr(items, "__iter__") and not isinstance(items, (str, Parameters)):
            existing.extend(items)
        else:
            existing.append(items)

    def remove(self, key: str) -> bool:
        if key in self._data:
            del self._data[key]
            return True
        return False

    # ── Backoff ───────────────────────────────────────────────────────────────

    @property
    def backoff(self) -> Optional["Parameters"]:
        return self._backoff

    def set_backoff(self, backoff: Optional["Parameters"]) -> None:
        assert backoff is not self, "Recursive backoff"
        self._backoff = backoff

    # ── Keys / iteration ──────────────────────────────────────────────────────

    def keys(self) -> set[str]:
        if self._backoff is not None:
            return self._backoff.keys() | set(self._data.keys())
        return set(self._data.keys())

    def get_keys(self) -> set[str]:
        return self.keys()

    def __iter__(self) -> Iterator[str]:
        return iter(self.keys())

    def __len__(self) -> int:
        return len(self._data)

    def __contains__(self, key: object) -> bool:
        return isinstance(key, str) and self.contains_key(key)

    def __getitem__(self, key: str) -> Any:
        val = self._get_raw(key)
        if val is _MISSING:
            raise KeyError(key)
        if isinstance(val, _NullMarker):
            return None
        return val

    def __setitem__(self, key: str, value: Any) -> None:
        self.put(key, value)

    def __delitem__(self, key: str) -> None:
        if not self.remove(key):
            raise KeyError(key)

    # ── Copying ───────────────────────────────────────────────────────────────

    def copy_from(self, other: "Parameters") -> None:
        """Deep-copy all keys from *other* into self."""
        p = other.clone()
        for k in p._data:
            self._data[k] = p._data[k]

    def copy_to(self, other: "Parameters") -> None:
        """Deep-copy all keys from self into *other*."""
        p = self.clone()
        for k in p._data:
            other._data[k] = p._data[k]

    def clone(self) -> "Parameters":
        """Return a deep copy (backoff is reference-copied, not deep-copied)."""
        copy = Parameters()
        copy._data = _deep_copy_dict(self._data)
        copy._backoff = self._backoff
        return copy

    def __copy__(self) -> "Parameters":
        return self.clone()

    def __deepcopy__(self, memo: dict) -> "Parameters":
        import copy
        copy_obj = Parameters()
        copy_obj._data = copy.deepcopy(self._data, memo)
        if self._backoff is not None:
            copy_obj._backoff = copy.deepcopy(self._backoff, memo)
        return copy_obj

    # ── Equality ──────────────────────────────────────────────────────────────

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Parameters):
            return False
        if other is self:
            return True
        # Symmetric: both must have the same key set with matching values
        if self.keys() != other.keys():
            return False
        for key in self.keys():
            if self[key] != other[key]:
                return False
        return True

    def __hash__(self) -> int:
        h = 0xDEADBEEF
        for k in self._data:
            h ^= hash(k)
            v = self._data[k]
            try:
                h ^= hash(v)
            except TypeError:
                pass
        return h

    # ── Serialization ─────────────────────────────────────────────────────────

    def __str__(self) -> str:
        """Java-compatible JSON format: sorted keys, custom spacing."""
        parts: list[str] = []
        for key in sorted(self.keys()):
            val = self._get_raw(key)
            if val is _MISSING:
                continue
            parts.append(f'"{_json_escape(key)}" : {_encode_value(val)}')
        return "{ " + " , ".join(parts) + " }"

    def __repr__(self) -> str:
        return f"Parameters({str(self)})"

    def to_pretty_string(self, indent: int = 2) -> str:
        """Human-readable indented JSON."""
        return json.dumps(_to_plain(self), indent=indent, sort_keys=True,
                         ensure_ascii=False, default=_json_default)

    def write(self, path: Union[str, os.PathLike]) -> None:
        """Write JSON to *path*."""
        with open(path, "w", encoding="utf-8") as f:
            f.write(str(self))


# ── Helpers ───────────────────────────────────────────────────────────────────

def _json_escape(s: str) -> str:
    """Escape a string for JSON output (delegates to json.dumps, strips quotes)."""
    return json.dumps(s)[1:-1]


def _json_default(obj: Any) -> Any:
    if isinstance(obj, _NullMarker):
        return None
    raise TypeError(f"Not JSON serializable: {obj!r}")


def _to_plain(val: Any) -> Any:
    """Recursively convert Parameters to plain dict for json.dumps."""
    if isinstance(val, Parameters):
        return {k: _to_plain(val._get_raw(k)) for k in sorted(val.keys())}
    if isinstance(val, list):
        return [_to_plain(item) for item in val]
    if isinstance(val, _NullMarker):
        return None
    return val


def _deep_copy_value(v: Any) -> Any:
    if isinstance(v, _NullMarker):
        return _NULL
    if isinstance(v, Parameters):
        return v.clone()
    if isinstance(v, list):
        return [_deep_copy_value(item) for item in v]
    return v  # bool, int, float, str are immutable


def _deep_copy_dict(d: dict) -> dict:
    return {k: _deep_copy_value(v) for k, v in d.items()}


# ── CamelCase aliases (Java API compatibility) ─────────────────────────────────

Parameters.parseString = staticmethod(Parameters.parse_string)
Parameters.parseFile = staticmethod(Parameters.parse_file)
Parameters.parseBytes = staticmethod(Parameters.parse_bytes)
Parameters.parseArgs = staticmethod(Parameters.parse_args)
Parameters.getString = Parameters.get_string
Parameters.getLong = Parameters.get_long
Parameters.getInt = Parameters.get_int
Parameters.getDouble = Parameters.get_double
Parameters.getAsDouble = Parameters.get_as_double
Parameters.getBoolean = Parameters.get_bool
Parameters.getList = Parameters.get_list
Parameters.getAsList = Parameters.get_as_list
Parameters.getMap = Parameters.get_map
Parameters.getAsString = Parameters.get_as_string
Parameters.isString = Parameters.is_string
Parameters.isLong = Parameters.is_long
Parameters.isDouble = Parameters.is_double
Parameters.isBoolean = Parameters.is_bool
Parameters.isList = Parameters.is_list
Parameters.isMap = Parameters.is_map
Parameters.containsKey = Parameters.contains_key
Parameters.getKeys = Parameters.get_keys
Parameters.setBackoff = Parameters.set_backoff
Parameters.getBackoff = lambda self: self.backoff
Parameters.copyFrom = Parameters.copy_from
Parameters.copyTo = Parameters.copy_to
Parameters.setIfMissing = Parameters.set_if_missing
Parameters.putIfNotNull = Parameters.put_if_not_none
Parameters.extendList = Parameters.extend_list
Parameters.toPrettyString = Parameters.to_pretty_string
