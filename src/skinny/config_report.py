"""Pure builders for the startup configuration matrix (change
online-training-observability).

The renderer collects one :class:`ConfigRow` per render-selection axis —
backend, execution mode, integrator, proposals, neural trainer, neural handoff,
train precision, online training — each carrying the *requested* value
(CLI/env/persisted), the *resolved* value actually in use, and a *status*
token. :func:`build_config_matrix` renders an aligned console table;
:func:`matrix_signature` produces a stable string the renderer dedups on so the
matrix reprints only when a status actually flips (e.g. online training going
``WAITING`` → ``APPROVED`` when a neural proposal is selected at runtime).

This module is intentionally dependency-free (no renderer / Qt imports) so it is
trivially unit-testable.
"""

from __future__ import annotations

from dataclasses import dataclass

# ── Status vocabulary ───────────────────────────────────────────────────────
ON = "ON"
OFF = "OFF"
NA = "n/a"
APPROVED = "APPROVED"


def refused(reason: str) -> str:
    """Status for a permanently-refused axis, naming the missing prerequisite."""
    return f"REFUSED ({reason})"


def waiting(reason: str) -> str:
    """Status for an armed-but-waiting axis (a runtime-selectable prerequisite)."""
    return f"WAITING ({reason})"


@dataclass(frozen=True)
class ConfigRow:
    """One render-selection axis: what was asked for, what resolved, its state."""

    axis: str
    requested: str
    resolved: str
    status: str


_HEADERS = ("axis", "requested", "resolved", "status")


def build_config_matrix(rows: list[ConfigRow], *,
                        title: str = "configuration") -> str:
    """Render ``rows`` as an aligned, `[skinny]`-prefixed console table."""
    cells = [_HEADERS] + [(r.axis, r.requested, r.resolved, r.status) for r in rows]
    widths = [max(len(c[i]) for c in cells) for i in range(4)]

    def fmt(c: tuple[str, str, str, str]) -> str:
        return "  ".join(c[i].ljust(widths[i]) for i in range(4))

    rule = "-" * (sum(widths) + 6)
    out = [f"[skinny] {title}"]
    out.append("  " + fmt(_HEADERS))
    out.append("  " + rule)
    out.extend("  " + fmt((r.axis, r.requested, r.resolved, r.status)) for r in rows)
    return "\n".join(out)


def matrix_signature(rows: list[ConfigRow]) -> str:
    """Stable signature over the *resolved* + *status* of every row. The
    requested column is static for a session, so resolved+status is what flips
    at runtime; the renderer reprints the matrix only when this string changes."""
    return "|".join(f"{r.axis}={r.resolved}/{r.status}" for r in rows)
