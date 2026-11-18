"""Audio alerts. Tones are synthesised, so no sound files are needed.

Tries pygame first. If pygame cannot open an audio device, falls back to the built-in
Windows beeper (winsound). The backend console says which one is in use and why.
"""
from __future__ import annotations

import threading
import time

import numpy as np

RATE = 22050
COOLDOWN = {1: None, 2: 4.0, 3: 1.2}  # seconds between repeats; caution beeps once when it begins
BEEPS = {1: [(660, 250)], 2: [(880, 180), (880, 180)], 3: [(1000, 250), (700, 250), (1000, 250), (700, 250)]}


def _tone(freq: float, dur: float, vol: float = 0.4) -> np.ndarray:
    t = np.linspace(0, dur, int(RATE * dur), endpoint=False)
    wave = np.sin(2 * np.pi * freq * t) * vol
    fade = np.minimum(1.0, np.minimum(t, dur - t) / 0.01)
    return wave * fade


class AudioAlerter:
    def __init__(self, enabled: bool = True) -> None:
        self.ok = False
        self.backend = "none"
        self._sounds: dict = {}
        self._last = 0.0
        self._prev_level = 0
        self._busy = threading.Event()
        if not enabled:
            print("[DRIVIO] Audio alerts are switched off (DRIVIO_AUDIO=0).", flush=True)
            return
        if self._init_pygame():
            self.backend = "pygame"
        elif self._init_winsound():
            self.backend = "winsound"
        self.ok = self.backend != "none"
        print(f"[DRIVIO] Audio alerts: {self.backend}", flush=True)

    # -- backends -------------------------------------------------------
    def _init_pygame(self) -> bool:
        try:
            import pygame

            pygame.mixer.init(frequency=RATE, size=-16, channels=2)
            channels = pygame.mixer.get_init()[2]  # the device decides how many channels it really opened
            gap = np.zeros(int(RATE * 0.1))
            clips = {
                1: _tone(660, 0.25),
                2: np.concatenate([_tone(880, 0.18), gap, _tone(880, 0.18)]),
                3: np.concatenate([_tone(1000, 0.25), _tone(700, 0.25)] * 2),
            }

            def build(wave: np.ndarray):
                samples = (wave * 32767).astype(np.int16)
                if channels >= 2:
                    samples = np.ascontiguousarray(np.column_stack([samples] * channels))
                return pygame.sndarray.make_sound(samples)

            self._pg = pygame
            self._sounds = {k: build(v) for k, v in clips.items()}
            return True
        except Exception as exc:
            print(f"[DRIVIO] pygame audio unavailable: {exc!r}", flush=True)
            return False

    def _init_winsound(self) -> bool:
        try:
            import winsound  # Windows only

            self._ws = winsound
            return True
        except ImportError:
            return False

    def _beep(self, level: int) -> None:
        try:
            for freq, ms in BEEPS[level]:
                self._ws.Beep(freq, ms)
        finally:
            self._busy.clear()

    # -- public ---------------------------------------------------------
    def play(self, level: int) -> None:
        if not self.ok:
            return
        previous, self._prev_level = self._prev_level, level
        if level == 0:
            if self.backend == "pygame":
                self._pg.mixer.stop()
            return
        now, cooldown = time.monotonic(), COOLDOWN[level]
        if not (level > previous or (cooldown is not None and now - self._last >= cooldown)):
            return
        self._last = now
        if self.backend == "pygame":
            self._sounds[level].play()
        elif not self._busy.is_set():  # Beep() blocks, so it runs in its own thread
            self._busy.set()
            threading.Thread(target=self._beep, args=(level,), daemon=True).start()