"""Translation report: what mapped exactly, what was approximated, what was skipped."""

from __future__ import annotations

from dataclasses import dataclass, field

EXACT = "exact"
APPROX = "approx"
SKIPPED = "skipped"


@dataclass
class ReportEntry:
    construct: str
    status: str  # EXACT | APPROX | SKIPPED
    detail: str = ""


@dataclass
class Report:
    entries: list[ReportEntry] = field(default_factory=list)

    def add(self, construct: str, status: str, detail: str = "") -> None:
        self.entries.append(ReportEntry(construct, status, detail))

    def exact(self, construct: str, detail: str = "") -> None:
        self.add(construct, EXACT, detail)

    def approx(self, construct: str, detail: str = "") -> None:
        self.add(construct, APPROX, detail)

    def skipped(self, construct: str, detail: str = "") -> None:
        self.add(construct, SKIPPED, detail)

    def count(self, status: str) -> int:
        return sum(1 for e in self.entries if e.status == status)

    def has_unsupported(self) -> bool:
        return self.count(SKIPPED) > 0

    def __str__(self) -> str:
        lines = ["pbrt import report:"]
        lines.append(
            f"  {self.count(EXACT)} exact, {self.count(APPROX)} approx, "
            f"{self.count(SKIPPED)} skipped"
        )
        for e in self.entries:
            tag = {EXACT: "[ok ]", APPROX: "[~  ]", SKIPPED: "[!  ]"}.get(e.status, "[?  ]")
            detail = f" — {e.detail}" if e.detail else ""
            lines.append(f"  {tag} {e.construct}{detail}")
        return "\n".join(lines)
