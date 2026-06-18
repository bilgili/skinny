"""Directive parser for the pbrt v4 text format.

Turns a token stream into a list of :class:`Directive` (a name, positional
args, and a typed :class:`ParamSet`). Handles recursive ``Include``/``Import``
and splitting the scene into the pre-``WorldBegin`` options block and the
post-``WorldBegin`` world block.

Tokenization rule that makes parsing unambiguous: every *bare word* token
starts a new directive. All directive values are numbers, quoted strings, or
bracketed arrays, and every parameter is declared as a quoted ``"type name"``
string. The lone exception ``ActiveTransform`` (whose arg is a bare word) is
special-cased.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field

from .errors import PbrtParseError
from .tokenizer import Token, tokenize

# pbrt parameter type keywords. The first word of a `"type name"` declaration.
TYPE_KEYWORDS = frozenset(
    {
        "float",
        "integer",
        "int",
        "bool",
        "string",
        "point",
        "point2",
        "point3",
        "vector",
        "vector2",
        "vector3",
        "normal",
        "normal3",
        "rgb",
        "color",
        "spectrum",
        "blackbody",
        "texture",
    }
)

# Directives whose first positional argument is a bare word, not a string.
_BARE_WORD_ARG = frozenset({"ActiveTransform"})


@dataclass(frozen=True)
class Param:
    """A single typed pbrt parameter, e.g. ``"float roughness" [0.1]``."""

    type: str
    name: str
    values: tuple  # mix of float | str, as tokenized

    @property
    def floats(self) -> list[float]:
        return [float(v) for v in self.values]

    @property
    def ints(self) -> list[int]:
        # parse from the raw token text so large indices stay exact
        return [int(round(float(v))) if not isinstance(v, str) else int(v) for v in self.values]

    @property
    def strings(self) -> list[str]:
        return [str(v) for v in self.values]

    @property
    def float(self) -> float:
        return float(self.values[0])

    @property
    def int(self) -> int:
        v = self.values[0]
        return int(v) if isinstance(v, str) else int(round(float(v)))

    @property
    def string(self) -> str:
        return str(self.values[0])

    @property
    def bool(self) -> bool:
        v = self.values[0]
        if isinstance(v, str):
            return v.lower() == "true"
        return bool(v)


@dataclass
class ParamSet:
    """Typed parameter lookup with defaults."""

    params: dict[str, Param] = field(default_factory=dict)

    def __contains__(self, name: str) -> bool:
        return name in self.params

    def get(self, name: str) -> Param | None:
        return self.params.get(name)

    def floats(self, name: str, default=None):
        p = self.params.get(name)
        return p.floats if p is not None else default

    def float(self, name: str, default=None):
        p = self.params.get(name)
        return p.float if p is not None else default

    def int(self, name: str, default=None):
        p = self.params.get(name)
        return p.int if p is not None else default

    def ints(self, name: str, default=None):
        p = self.params.get(name)
        return p.ints if p is not None else default

    def string(self, name: str, default=None):
        p = self.params.get(name)
        return p.string if p is not None else default

    def bool(self, name: str, default=None):
        p = self.params.get(name)
        return p.bool if p is not None else default

    def rgb(self, name: str, default=None):
        """Return a 3-float color, broadcasting a scalar to grey."""
        p = self.params.get(name)
        if p is None:
            return default
        vals = p.floats
        if len(vals) == 1:
            return [vals[0], vals[0], vals[0]]
        return vals[:3]


@dataclass
class Directive:
    name: str
    args: list  # positional values: float | str | list (bracketed array)
    params: ParamSet
    file: str | None = None
    line: int = 0

    def type_arg(self, default: str | None = None) -> str | None:
        """First string positional arg (pbrt 'implementation type', e.g. 'perspective')."""
        for a in self.args:
            if isinstance(a, str):
                return a
        return default


def _read_array(tokens: list[Token], idx: int, file) -> tuple[list, int]:
    """Consume ``[ v v v ]`` starting at the lbracket; return (values, next_idx)."""
    assert tokens[idx].kind == "lbracket"
    idx += 1
    out: list = []
    n = len(tokens)
    while idx < n and tokens[idx].kind != "rbracket":
        t = tokens[idx]
        if t.kind == "number":
            out.append(t.number)
        elif t.kind == "string":
            out.append(t.value)
        elif t.kind == "word":
            out.append(t.value)  # e.g. bare true/false inside an array
        else:
            raise PbrtParseError(
                f"unexpected {t.kind} inside array", file=file, line=t.line
            )
        idx += 1
    if idx >= n:
        raise PbrtParseError("unterminated array", file=file, line=tokens[-1].line)
    return out, idx + 1  # skip rbracket


def _read_param_values(tokens: list[Token], idx: int, file) -> tuple[tuple, int]:
    """Read a parameter's value(s): either a bracketed array or one scalar."""
    t = tokens[idx]
    if t.kind == "lbracket":
        vals, idx = _read_array(tokens, idx, file)
        return tuple(vals), idx
    if t.kind == "number":
        return (t.number,), idx + 1
    if t.kind == "string":
        return (t.value,), idx + 1
    if t.kind == "word":
        return (t.value,), idx + 1
    raise PbrtParseError(f"expected parameter value, got {t.kind}", file=file, line=t.line)


def parse_directives(tokens: list[Token], *, file: str | None = None) -> list[Directive]:
    """Parse a flat token stream into directives (no Include resolution)."""
    out: list[Directive] = []
    idx = 0
    n = len(tokens)
    while idx < n:
        t = tokens[idx]
        if t.kind != "word":
            raise PbrtParseError(
                f"expected a directive name, got {t.kind} {t.value!r}", file=file, line=t.line
            )
        name = t.value
        line = t.line
        idx += 1
        args: list = []
        params: dict[str, Param] = {}

        if name in _BARE_WORD_ARG and idx < n and tokens[idx].kind == "word":
            args.append(tokens[idx].value)
            idx += 1

        while idx < n:
            tk = tokens[idx]
            if tk.kind == "word":
                break  # next directive
            if tk.kind == "number":
                args.append(tk.number)
                idx += 1
                continue
            if tk.kind == "lbracket":
                arr, idx = _read_array(tokens, idx, file)
                args.append(arr)
                continue
            if tk.kind == "string":
                parts = tk.value.split()
                if len(parts) == 2 and parts[0] in TYPE_KEYWORDS:
                    ptype, pname = parts
                    idx += 1
                    if idx >= n:
                        raise PbrtParseError(
                            f"parameter {tk.value!r} missing a value", file=file, line=tk.line
                        )
                    vals, idx = _read_param_values(tokens, idx, file)
                    params[pname] = Param(ptype, pname, vals)
                else:
                    args.append(tk.value)
                    idx += 1
                continue
            raise PbrtParseError(f"unexpected {tk.kind} token", file=file, line=tk.line)

        out.append(Directive(name, args, ParamSet(params), file=file, line=line))
    return out


def parse_file(path: str, *, _seen: set | None = None) -> list[Directive]:
    """Parse a pbrt file, resolving Include/Import recursively and inline.

    Paths in Include/Import are resolved relative to the including file. Import
    is treated like Include (inlined); its graphics-state isolation is not
    modeled (documented limitation).
    """
    path = os.path.abspath(path)
    _seen = _seen if _seen is not None else set()
    if path in _seen:
        raise PbrtParseError(f"circular Include of {path!r}")
    _seen = _seen | {path}
    try:
        with open(path, encoding="utf-8") as fh:
            text = fh.read()
    except OSError as exc:
        raise PbrtParseError(f"cannot read {path!r}: {exc}") from exc

    directives = parse_directives(tokenize(text, file=path), file=path)
    base = os.path.dirname(path)
    resolved: list[Directive] = []
    for d in directives:
        if d.name in ("Include", "Import"):
            ref = d.type_arg()
            if ref is None:
                raise PbrtParseError(
                    f"{d.name} without a filename", file=d.file, line=d.line
                )
            inc_path = ref if os.path.isabs(ref) else os.path.join(base, ref)
            resolved.extend(parse_file(inc_path, _seen=_seen))
        else:
            resolved.append(d)
    return resolved


def split_options_world(directives: list[Directive]) -> tuple[list[Directive], list[Directive]]:
    """Split a directive list at ``WorldBegin`` into (options, world).

    The options block excludes the ``WorldBegin`` marker; the world block is
    everything after it. Raises if ``WorldBegin`` is missing.
    """
    for i, d in enumerate(directives):
        if d.name == "WorldBegin":
            return directives[:i], directives[i + 1 :]
    raise PbrtParseError("scene has no WorldBegin")
