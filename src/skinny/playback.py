"""USD animation playback clock.

Pure-logic time model: maps wall-clock delta time onto a USD time-code range.
No pxr or GPU dependencies so it can be unit-tested in isolation. The renderer
owns one instance; `usd_loader` builds it from stage metadata.
"""

from __future__ import annotations

from dataclasses import dataclass

DEFAULT_FPS = 24.0


@dataclass
class PlaybackClock:
    """Drives USD time during playback.

    `playback_fps` is the playback rate in time codes per second, so
    `current_time_code += dt * playback_fps`. Defaulting it to the stage's
    `timeCodesPerSecond` yields real-time playback.
    """

    start_time_code: float = 0.0
    end_time_code: float = 0.0
    time_codes_per_second: float = DEFAULT_FPS
    playback_fps: float = DEFAULT_FPS
    playing: bool = False
    loop: bool = True
    has_animation: bool = False
    current_time_code: float = 0.0

    @property
    def range(self) -> float:
        return self.end_time_code - self.start_time_code

    def advance(self, dt: float) -> None:
        """Move the time code by `dt` seconds when playing."""
        if not (self.playing and self.has_animation):
            return
        t = self.current_time_code + dt * self.playback_fps
        span = self.range
        if span <= 0.0:
            self.current_time_code = self.start_time_code
            return
        if t > self.end_time_code:
            if self.loop:
                t = self.start_time_code + (t - self.start_time_code) % span
            else:
                t = self.end_time_code
        elif t < self.start_time_code:
            if self.loop:
                t = self.start_time_code + (t - self.start_time_code) % span
            else:
                t = self.start_time_code
        self.current_time_code = t

    @property
    def normalized(self) -> float:
        """Current time code as a 0-1 fraction of the range."""
        span = self.range
        if span <= 0.0:
            return 0.0
        return (self.current_time_code - self.start_time_code) / span

    @normalized.setter
    def normalized(self, t: float) -> None:
        self.set_normalized(t)

    def set_normalized(self, t: float) -> None:
        """Set the current time code from a 0-1 normalized scrubber value."""
        self.current_time_code = self.start_time_code + t * self.range
