"""A deterministic cursor over a simulation run, driven only by explicit actions."""

from dataclasses import dataclass
from math import floor, isfinite


@dataclass
class Playback:
    dt: float
    samples: int
    index: int = 0
    playing: bool = False
    clock_anchor: float | None = None
    fractional_steps: float = 0.0

    @property
    def time_s(self):
        return self.index * self.dt

    def play(self, now: float):
        if not self.playing and self.index < self.samples - 1:
            self.playing = True
            self.clock_anchor = now

    def pause(self):
        self.playing = False
        self.clock_anchor = None

    def stop(self):
        self.pause()
        self.index = 0
        self.fractional_steps = 0.0

    def restart(self, now: float):
        self.stop()
        self.play(now)

    def step(self):
        if not self.playing:
            self.index = min(self.index + 1, self.samples - 1)
            self.fractional_steps = 0.0

    def seek(self, time_s: float):
        if not isfinite(time_s):
            raise ValueError("이동할 시각은 유한한 값이어야 합니다.")
        self.pause()
        self.index = min(max(0, floor(time_s / self.dt + 0.5)), self.samples - 1)
        self.fractional_steps = 0.0

    def tick(self, now: float, speed: float = 1.0):
        if not self.playing or self.clock_anchor is None:
            return
        elapsed = max(0.0, now - self.clock_anchor)
        self.clock_anchor = now
        steps = self.fractional_steps + elapsed * speed / self.dt
        whole = floor(steps + 1e-10)
        self.index = min(self.index + whole, self.samples - 1)
        self.fractional_steps = max(0.0, steps - whole)
        if self.index == self.samples - 1:
            self.pause()
            self.fractional_steps = 0.0
