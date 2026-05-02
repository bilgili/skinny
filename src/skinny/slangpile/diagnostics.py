from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal


@dataclass(frozen=True)
class Diagnostic:
    severity: Literal["error", "warning", "note"]
    message: str
    source: Literal["python", "slangpile", "slangc"] = "slangpile"
    file: Path | None = None
    line: int | None = None
    column: int | None = None

    def format(self) -> str:
        location = ""
        if self.file is not None:
            location = str(self.file)
            if self.line is not None:
                location += f":{self.line}"
                if self.column is not None:
                    location += f":{self.column}"
            location += ": "
        return f"{location}{self.severity}[{self.source}]: {self.message}"


class SlangPileError(Exception):
    def __init__(self, diagnostic: Diagnostic):
        super().__init__(diagnostic.format())
        self.diagnostic = diagnostic

