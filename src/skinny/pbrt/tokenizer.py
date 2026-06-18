"""Tokenizer for the pbrt v4 text scene format.

The grammar is small. A scene file is a flat stream of:

* directive names and bare values  -> ``word`` / ``number`` tokens
* quoted strings (parameter declarations, string/bool values, filenames)
  -> ``string`` tokens (quotes stripped)
* array delimiters ``[`` ``]``      -> ``lbracket`` / ``rbracket`` tokens

``#`` starts a comment that runs to end of line. Bare tokens that parse as a
float are classified as ``number`` (keeping the raw text so integer parameters
can be read losslessly); everything else is a ``word``.
"""

from __future__ import annotations

from dataclasses import dataclass

from .errors import PbrtParseError

_WHITESPACE = " \t\r\n"
_SPECIAL = '"[]#' + _WHITESPACE


@dataclass(frozen=True)
class Token:
    kind: str  # 'word' | 'number' | 'string' | 'lbracket' | 'rbracket'
    value: str  # textual value (quotes stripped for strings)
    line: int  # 1-based source line of the token start

    @property
    def number(self) -> float:
        return float(self.value)


def _is_number(text: str) -> bool:
    try:
        float(text)
    except ValueError:
        return False
    return True


def tokenize(text: str, *, file: str | None = None) -> list[Token]:
    """Split *text* into a list of :class:`Token`.

    Raises :class:`PbrtParseError` (with file/line) on an unterminated string.
    """
    tokens: list[Token] = []
    i = 0
    n = len(text)
    line = 1
    while i < n:
        c = text[i]
        if c == "\n":
            line += 1
            i += 1
            continue
        if c in _WHITESPACE:
            i += 1
            continue
        if c == "#":  # comment to end of line
            j = text.find("\n", i)
            i = n if j < 0 else j
            continue
        if c == '"':
            j = text.find('"', i + 1)
            if j < 0:
                raise PbrtParseError("unterminated string", file=file, line=line)
            # pbrt strings do not span lines, but guard the line counter anyway
            inner = text[i + 1 : j]
            tokens.append(Token("string", inner, line))
            line += inner.count("\n")
            i = j + 1
            continue
        if c == "[":
            tokens.append(Token("lbracket", "[", line))
            i += 1
            continue
        if c == "]":
            tokens.append(Token("rbracket", "]", line))
            i += 1
            continue
        # bare token: read until whitespace or a special character
        j = i
        while j < n and text[j] not in _SPECIAL:
            j += 1
        word = text[i:j]
        kind = "number" if _is_number(word) else "word"
        tokens.append(Token(kind, word, line))
        i = j
    return tokens
