"""Tests for the pbrt tokenizer (task 2.1)."""

from __future__ import annotations

import pytest

from skinny.pbrt.errors import PbrtParseError
from skinny.pbrt.tokenizer import tokenize


def _kinds(text):
    return [(t.kind, t.value) for t in tokenize(text)]


def test_comments_are_stripped():
    toks = tokenize("# leading comment\nIdentity # trailing\n")
    assert [t.value for t in toks] == ["Identity"]


def test_quoted_strings_keep_inner_text():
    toks = tokenize('"float roughness" [0.1]')
    assert toks[0].kind == "string"
    assert toks[0].value == "float roughness"
    assert [t.kind for t in toks[1:]] == ["lbracket", "number", "rbracket"]


def test_numbers_vs_words():
    toks = tokenize("Translate 1 -2.5 3e-2")
    assert [(t.kind, t.value) for t in toks] == [
        ("word", "Translate"),
        ("number", "1"),
        ("number", "-2.5"),
        ("number", "3e-2"),
    ]
    assert toks[3].number == pytest.approx(0.03)


def test_rgb_array():
    assert _kinds('"rgb reflectance" [0.2 0.3 0.4]') == [
        ("string", "rgb reflectance"),
        ("lbracket", "["),
        ("number", "0.2"),
        ("number", "0.3"),
        ("number", "0.4"),
        ("rbracket", "]"),
    ]


def test_brackets_need_no_surrounding_space():
    assert [t.kind for t in tokenize("[1 2]")] == [
        "lbracket",
        "number",
        "number",
        "rbracket",
    ]


def test_line_tracking():
    toks = tokenize("Identity\n\nTranslate 0 0 1")
    by_value = {t.value: t.line for t in toks}
    assert by_value["Identity"] == 1
    assert by_value["Translate"] == 3


def test_unterminated_string_raises_with_location():
    with pytest.raises(PbrtParseError) as exc:
        tokenize('Shape "trianglemesh\n', file="scene.pbrt")
    assert "scene.pbrt" in str(exc.value)
    assert "unterminated" in str(exc.value)
