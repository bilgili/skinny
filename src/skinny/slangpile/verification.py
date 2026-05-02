from __future__ import annotations

import os
import re
import shutil
import subprocess
from dataclasses import dataclass, field
from pathlib import Path

from .diagnostics import Diagnostic


@dataclass
class VerificationOptions:
    slangc: str | None = None
    include_dirs: list[Path] = field(default_factory=list)
    target: str | None = None
    profile: str | None = None
    entry_point: str | None = None


@dataclass
class VerificationResult:
    ok: bool
    command: list[str]
    stdout: str
    stderr: str
    diagnostics: list[Diagnostic] = field(default_factory=list)


def resolve_slangc(explicit: str | None = None) -> str | None:
    if explicit:
        return explicit
    env = os.environ.get("SLANGPILE_SLANGC")
    if env:
        return env
    return shutil.which("slangc")


def build_command(path: Path, options: VerificationOptions) -> list[str]:
    slangc = resolve_slangc(options.slangc)
    if slangc is None:
        raise FileNotFoundError("could not find slangc; set --slangc or SLANGPILE_SLANGC")
    command = [slangc, str(path)]
    for include_dir in options.include_dirs:
        command.extend(["-I", str(include_dir)])
    if options.entry_point:
        command.extend(["-entry", options.entry_point])
    if options.profile:
        command.extend(["-profile", options.profile])
    if options.target:
        command.extend(["-target", options.target])
    return command


def verify_file(path: str | Path, options: VerificationOptions | None = None) -> VerificationResult:
    path = Path(path)
    options = options or VerificationOptions()
    command = build_command(path, options)
    completed = subprocess.run(command, check=False, text=True, capture_output=True)
    diagnostics = parse_slangc_diagnostics(completed.stderr)
    return VerificationResult(
        ok=completed.returncode == 0,
        command=command,
        stdout=completed.stdout,
        stderr=completed.stderr,
        diagnostics=diagnostics,
    )


_DIAGNOSTIC_RE = re.compile(
    r"^(?P<file>.*?)(?:\((?P<line1>\d+)(?::(?P<col1>\d+))?\)|:(?P<line2>\d+)(?::(?P<col2>\d+))?):\s*(?P<severity>error|warning|note)\s*:?\s*(?P<message>.*)$",
    re.IGNORECASE,
)


def parse_slangc_diagnostics(text: str) -> list[Diagnostic]:
    diagnostics: list[Diagnostic] = []
    for line in text.splitlines():
        match = _DIAGNOSTIC_RE.match(line.strip())
        if not match:
            continue
        line_number = match.group("line1") or match.group("line2")
        column = match.group("col1") or match.group("col2")
        diagnostics.append(
            Diagnostic(
                severity=match.group("severity").lower(),
                source="slangc",
                file=Path(match.group("file")),
                line=int(line_number) if line_number else None,
                column=int(column) if column else None,
                message=match.group("message"),
            )
        )
    return diagnostics

