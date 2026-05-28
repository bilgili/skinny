"""Unit tests for the USD playback clock (pure logic, no GPU/pxr)."""

from __future__ import annotations

import pytest

from skinny.playback import PlaybackClock


def make_clock(**kw) -> PlaybackClock:
    defaults = dict(
        start_time_code=0.0,
        end_time_code=48.0,
        time_codes_per_second=24.0,
        playback_fps=24.0,
        has_animation=True,
        loop=True,
    )
    defaults.update(kw)
    return PlaybackClock(**defaults)


class TestAdvance:
    def test_advance_while_playing_moves_time_code(self):
        clock = make_clock()
        clock.playing = True
        clock.current_time_code = 0.0
        clock.advance(0.5)  # 0.5 s * 24 fps = 12 time codes
        assert clock.current_time_code == pytest.approx(12.0)

    def test_paused_does_not_advance(self):
        clock = make_clock()
        clock.playing = False
        clock.current_time_code = 7.0
        clock.advance(1.0)
        assert clock.current_time_code == pytest.approx(7.0)

    def test_loop_wraps_at_end(self):
        clock = make_clock(start_time_code=0.0, end_time_code=48.0)
        clock.playing = True
        clock.current_time_code = 46.0
        clock.advance(0.5)  # +12 -> 58 -> wraps into [0, 48] -> 10
        assert clock.current_time_code == pytest.approx(10.0)

    def test_loop_wraps_with_nonzero_start(self):
        clock = make_clock(start_time_code=10.0, end_time_code=20.0)
        clock.playing = True
        clock.current_time_code = 19.0
        clock.advance(0.25)  # +6 -> 25 -> range 10, wrap -> 10 + (25-10)%10 = 15
        assert clock.current_time_code == pytest.approx(15.0)

    def test_no_loop_clamps_to_end(self):
        clock = make_clock(start_time_code=0.0, end_time_code=48.0, loop=False)
        clock.playing = True
        clock.current_time_code = 46.0
        clock.advance(1.0)  # +24 -> 70 -> clamp to 48
        assert clock.current_time_code == pytest.approx(48.0)


class TestNoAnimation:
    def test_inert_when_no_animation(self):
        clock = make_clock(has_animation=False)
        clock.playing = True
        clock.current_time_code = 0.0
        clock.advance(1.0)
        assert clock.current_time_code == pytest.approx(0.0)


class TestNormalized:
    def test_set_normalized_maps_to_time_code(self):
        clock = make_clock(start_time_code=10.0, end_time_code=20.0)
        clock.set_normalized(0.25)
        assert clock.current_time_code == pytest.approx(12.5)

    def test_normalized_reads_back_fraction(self):
        clock = make_clock(start_time_code=10.0, end_time_code=20.0)
        clock.current_time_code = 15.0
        assert clock.normalized == pytest.approx(0.5)

    def test_normalized_zero_range_is_zero(self):
        clock = make_clock(start_time_code=5.0, end_time_code=5.0)
        clock.current_time_code = 5.0
        assert clock.normalized == pytest.approx(0.0)
