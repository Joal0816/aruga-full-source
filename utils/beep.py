"""Cross-platform audible alarm + acknowledge state.

Windows: winsound.Beep (no asset needed). Elsewhere: play a generated 880 Hz
wav with the first available CLI player; terminal bell as last resort.

ponytail: CLI players instead of an audio dependency — one beep does not
justify sounddevice/pyaudio in requirements.

AlarmAck: silences *repeating* alarm beeps for N seconds after the operator
presses Acknowledge; re-arms automatically when the alarm clears so the next
incident sounds again.
"""
import os
import shutil
import struct
import subprocess
import tempfile
import wave

_ACK_SECONDS = 600.0
_WAV = None
_last_proc: "subprocess.Popen | None" = None


def _beep_wav() -> str:
    global _WAV
    if _WAV is None:
        fd, path = tempfile.mkstemp(suffix="_aruga_beep.wav")
        os.close(fd)
        rate, freq, ms, amp = 8000, 880, 300, 12000
        half = max(1, rate // (2 * freq))
        n = rate * ms // 1000
        data = b"".join(struct.pack("<h", amp if (i // half) % 2 else -amp)
                        for i in range(n))
        with wave.open(path, "wb") as w:
            w.setnchannels(1)
            w.setsampwidth(2)
            w.setframerate(rate)
            w.writeframes(data)
        _WAV = path
    return _WAV


def beep():
    """Non-fatal audible alert; never raises."""
    global _last_proc
    try:
        if os.name == "nt":
            try:
                import winsound
                winsound.Beep(880, 300)
                return
            except Exception:
                pass
        # Skip if the previous beep is still playing (avoids zombies: poll() reaps).
        if _last_proc is not None and _last_proc.poll() is None:
            return
        wav = _beep_wav()
        for player, args in (("paplay", ()), ("pw-play", ()), ("aplay", ("-q",)),
                             ("afplay", ()),  # macOS built-in
                             ("ffplay", ("-nodisp", "-autoexit", "-loglevel", "quiet"))):
            if shutil.which(player):
                try:
                    _last_proc = subprocess.Popen(
                        [player, *args, wav],
                        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                    return
                except Exception:
                    continue
        print("\a", end="", flush=True)  # terminal bell fallback
    except Exception:
        pass


class AlarmAck:
    """Acknowledge window for repeating alarm beeps (see module docstring)."""

    def __init__(self, seconds: float = _ACK_SECONDS):
        self.seconds = seconds
        self.until = 0.0
        self._was_active = False

    def ack(self, now: float):
        self.until = now + self.seconds

    def rearm_if_cleared(self, alarm_active: bool, now: float):
        # Falling edge (alarm cleared) => next incident beeps again.
        if self._was_active and not alarm_active:
            self.until = 0.0
        self._was_active = alarm_active

    def muted(self, now: float) -> bool:
        return now < self.until
