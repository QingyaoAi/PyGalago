"""Tests for pygalago.parameters.Parameters — mirrors Java ParametersTest."""

import json
import math
import os
import tempfile

import pytest

from pygalago.parameters import Parameters, _NULL, _NullMarker


# ── Construction ──────────────────────────────────────────────────────────────

class TestConstruction:
    def test_create_empty(self):
        p = Parameters.create()
        assert len(p) == 0
        assert list(p.keys()) == []

    def test_parse_string_flat(self):
        p = Parameters.parse_string('{"a": 1, "b": "hello", "c": true}')
        assert p.get_long("a") == 1
        assert p.get_string("b") == "hello"
        assert p.get_bool("c") is True

    def test_parse_string_nested(self):
        p = Parameters.parse_string('{"outer": {"inner": 42}}')
        inner = p.get_map("outer")
        assert inner.get_long("inner") == 42

    def test_parse_bytes(self):
        data = b'{"x": 3.14}'
        p = Parameters.parse_bytes(data)
        assert abs(p.get_double("x") - 3.14) < 1e-10

    def test_parse_file(self, tmp_path):
        f = tmp_path / "params.json"
        f.write_text('{"key": "value"}')
        p = Parameters.parse_file(str(f))
        assert p.get_string("key") == "value"

    def test_parse_args_equals(self):
        p = Parameters.parse_args(["--index=/tmp/idx", "--n=100", "--verbose=true"])
        assert p.get_string("index") == "/tmp/idx"
        assert p.get_long("n") == 100
        assert p.get_bool("verbose") is True

    def test_parse_args_space_separated(self):
        p = Parameters.parse_args(["--query", "information retrieval"])
        assert p.get_string("query") == "information retrieval"


# ── Typed getters / checkers ──────────────────────────────────────────────────

class TestTypedGetters:
    def setup_method(self):
        self.p = Parameters.parse_string(
            '{"s":"hello","n":42,"f":3.14,"b":true,"lst":[1,2,3],"m":{"k":1},"null_val":null}'
        )

    def test_string(self):
        assert self.p.is_string("s")
        assert self.p.get_string("s") == "hello"

    def test_long(self):
        assert self.p.is_long("n")
        assert self.p.get_long("n") == 42
        assert self.p.get_int("n") == 42

    def test_double(self):
        assert self.p.is_double("f")
        assert abs(self.p.get_double("f") - 3.14) < 1e-10

    def test_bool(self):
        assert self.p.is_bool("b")
        assert self.p.get_bool("b") is True

    def test_list(self):
        assert self.p.is_list("lst")
        assert self.p.get_list("lst") == [1, 2, 3]

    def test_map(self):
        assert self.p.is_map("m")
        m = self.p.get_map("m")
        assert m.get_long("k") == 1

    def test_null_value(self):
        assert self.p.is_string("null_val")  # null is treated as nullable string
        assert self.p.get_string("null_val") is None

    def test_type_errors(self):
        with pytest.raises((TypeError, KeyError)):
            self.p.get_long("s")  # string is not long
        with pytest.raises((TypeError, KeyError)):
            self.p.get_string("n")  # long is not string

    def test_missing_throws(self):
        with pytest.raises(KeyError):
            self.p.get_long("nonexistent")

    def test_get_with_default(self):
        assert self.p.get("nonexistent", 99) == 99
        assert self.p.get("n", 0) == 42

    def test_contains_key(self):
        assert self.p.contains_key("s")
        assert not self.p.contains_key("missing")

    def test_bool_not_long(self):
        assert not self.p.is_long("b")
        assert not self.p.is_double("b")

    def test_get_as_list_scalar(self):
        assert self.p.get_as_list("n") == [42]

    def test_get_as_list_missing(self):
        assert self.p.get_as_list("nonexistent") == []

    def test_int_overflow(self):
        p = Parameters.create()
        p.set("big", 2**32)
        with pytest.raises(OverflowError):
            p.get_int("big")


# ── Setters ───────────────────────────────────────────────────────────────────

class TestSetters:
    def test_put_and_get(self):
        p = Parameters.create()
        p.put("x", 10)
        assert p.get_long("x") == 10

    def test_set_various_types(self):
        p = Parameters.create()
        p.set("s", "hello")
        p.set("n", 42)
        p.set("f", 1.5)
        p.set("b", True)
        assert p.get_string("s") == "hello"
        assert p.get_long("n") == 42
        assert p.get_double("f") == 1.5
        assert p.get_bool("b") is True

    def test_put_none_becomes_null(self):
        p = Parameters.create()
        p.put("k", None)
        assert p.is_string("k")
        assert p.get_string("k") is None

    def test_set_if_missing(self):
        p = Parameters.create()
        p.set("x", 1)
        p.set_if_missing("x", 99)
        p.set_if_missing("y", 99)
        assert p.get_long("x") == 1
        assert p.get_long("y") == 99

    def test_remove(self):
        p = Parameters.create()
        p.set("x", 1)
        assert p.remove("x")
        assert not p.contains_key("x")
        assert not p.remove("nonexistent")

    def test_extend_list(self):
        p = Parameters.create()
        p.extend_list("nums", [1, 2])
        p.extend_list("nums", 3)
        assert p.get_list("nums") == [1, 2, 3]

    def test_extend_list_creates(self):
        p = Parameters.create()
        p.extend_list("x", "hello")
        assert p.get_list("x") == ["hello"]

    def test_put_if_not_none(self):
        p = Parameters.create()
        p.put_if_not_none("a", "val")
        p.put_if_not_none("b", None)
        assert p.contains_key("a")
        assert not p.contains_key("b")


# ── Backoff ───────────────────────────────────────────────────────────────────

class TestBackoff:
    def test_falls_back(self):
        fallback = Parameters.parse_string('{"a": 1, "b": 2}')
        p = Parameters.parse_string('{"a": 10}')
        p.set_backoff(fallback)
        assert p.get_long("a") == 10  # own value wins
        assert p.get_long("b") == 2   # from backoff

    def test_keys_include_backoff(self):
        fallback = Parameters.parse_string('{"x": 1}')
        p = Parameters.parse_string('{"y": 2}')
        p.set_backoff(fallback)
        assert "x" in p.keys()
        assert "y" in p.keys()

    def test_recursive_backoff(self):
        b1 = Parameters.parse_string('{"deep": 99}')
        b2 = Parameters.parse_string('{"mid": 50}')
        b2.set_backoff(b1)
        p = Parameters.parse_string('{"top": 1}')
        p.set_backoff(b2)
        assert p.get_long("deep") == 99
        assert p.get_long("mid") == 50

    def test_self_backoff_raises(self):
        p = Parameters.create()
        with pytest.raises(AssertionError):
            p.set_backoff(p)


# ── Clone & copy ──────────────────────────────────────────────────────────────

class TestCloneAndCopy:
    def test_clone_independence(self):
        p = Parameters.parse_string('{"a": [1, 2, 3]}')
        c = p.clone()
        c.get_list("a").append(4)
        assert p.get_list("a") == [1, 2, 3]
        assert c.get_list("a") == [1, 2, 3, 4]

    def test_clone_nested(self):
        p = Parameters.parse_string('{"m": {"x": 1}}')
        c = p.clone()
        c.get_map("m").set("x", 99)
        assert p.get_map("m").get_long("x") == 1

    def test_copy_from(self):
        src = Parameters.parse_string('{"a": 1, "b": 2}')
        dst = Parameters.parse_string('{"a": 99}')
        dst.copy_from(src)
        assert dst.get_long("a") == 1
        assert dst.get_long("b") == 2

    def test_copy_to(self):
        src = Parameters.parse_string('{"a": 1}')
        dst = Parameters.create()
        src.copy_to(dst)
        assert dst.get_long("a") == 1


# ── Equality ──────────────────────────────────────────────────────────────────

class TestEquality:
    def test_equal(self):
        a = Parameters.parse_string('{"x": 1, "y": "hello"}')
        b = Parameters.parse_string('{"y": "hello", "x": 1}')
        assert a == b

    def test_not_equal_missing_key(self):
        a = Parameters.parse_string('{"x": 1}')
        b = Parameters.parse_string('{"x": 1, "y": 2}')
        assert a != b

    def test_not_equal_different_value(self):
        a = Parameters.parse_string('{"x": 1}')
        b = Parameters.parse_string('{"x": 2}')
        assert a != b

    def test_not_equal_non_params(self):
        p = Parameters.parse_string('{"x": 1}')
        assert p != {"x": 1}


# ── Serialization ─────────────────────────────────────────────────────────────

class TestSerialization:
    def test_roundtrip(self):
        original = '{"a": 1, "b": "hello", "c": true, "d": [1, 2]}'
        p = Parameters.parse_string(original)
        # Re-parse the serialized form; must equal original
        q = Parameters.parse_string(str(p))
        assert p == q

    def test_sorted_keys(self):
        p = Parameters.parse_string('{"z": 1, "a": 2, "m": 3}')
        s = str(p)
        pos_a = s.index('"a"')
        pos_m = s.index('"m"')
        pos_z = s.index('"z"')
        assert pos_a < pos_m < pos_z

    def test_nested_roundtrip(self):
        p = Parameters.parse_string('{"a": {"b": {"c": 42}}}')
        q = Parameters.parse_string(str(p))
        assert q.get_map("a").get_map("b").get_long("c") == 42

    def test_nan_serialized_as_string(self):
        p = Parameters.create()
        p.set("v", float("nan"))
        s = str(p)
        assert '"nan"' in s.lower() or '"NaN"' in s

    def test_write_and_read_file(self, tmp_path):
        p = Parameters.parse_string('{"hello": "world", "n": 7}')
        out = tmp_path / "out.json"
        p.write(str(out))
        q = Parameters.parse_file(str(out))
        assert p == q

    def test_list_roundtrip(self):
        p = Parameters.parse_string('{"items": [1, "two", true, null]}')
        q = Parameters.parse_string(str(p))
        lst = q.get_list("items")
        assert lst[0] == 1
        assert lst[1] == "two"
        assert lst[2] is True
        assert lst[3] is None

    def test_pretty_string_is_valid_json(self):
        p = Parameters.parse_string('{"a": 1, "b": [1, 2]}')
        pretty = p.to_pretty_string()
        parsed = json.loads(pretty)
        assert parsed["a"] == 1
        assert parsed["b"] == [1, 2]


# ── Iteration & dict-like interface ───────────────────────────────────────────

class TestDictInterface:
    def test_getitem(self):
        p = Parameters.parse_string('{"x": 5}')
        assert p["x"] == 5

    def test_setitem(self):
        p = Parameters.create()
        p["k"] = "val"
        assert p.get_string("k") == "val"

    def test_delitem(self):
        p = Parameters.parse_string('{"x": 1}')
        del p["x"]
        assert not p.contains_key("x")

    def test_iter(self):
        p = Parameters.parse_string('{"a": 1, "b": 2}')
        assert set(p) == {"a", "b"}

    def test_contains(self):
        p = Parameters.parse_string('{"a": 1}')
        assert "a" in p
        assert "b" not in p

    def test_len(self):
        p = Parameters.parse_string('{"a": 1, "b": 2}')
        assert len(p) == 2


# ── CamelCase aliases ─────────────────────────────────────────────────────────

class TestCamelCaseAliases:
    def test_camelcase_parse(self):
        p = Parameters.parseString('{"x": 1}')
        assert p.getLong("x") == 1

    def test_camelcase_getters(self):
        p = Parameters.parseString('{"s": "hi", "b": false, "f": 2.5}')
        assert p.getString("s") == "hi"
        assert p.getBoolean("b") is False
        assert p.getDouble("f") == 2.5

    def test_camelcase_checkers(self):
        p = Parameters.parseString('{"s": "hi"}')
        assert p.isString("s")
        assert not p.isLong("s")
