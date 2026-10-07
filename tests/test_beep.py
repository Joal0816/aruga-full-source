"""Tests for utils.beep: wav generation, player fallback, AlarmAck state machine."""
import os
import sys
import wave

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from utils.beep import beep, _beep_wav, AlarmAck

ok = True


def check(name, cond, detail=""):
    global ok
    print(f"  [{'PASS' if cond else 'FAIL'}] {name} {detail}")
    ok = ok and bool(cond)


def main():
    # 1. Generated wav is a valid mono16 8kHz file with audible-length frames
    path = _beep_wav()
    check("wav file exists", os.path.exists(path), f"({path})")
    with wave.open(path, "rb") as w:
        check("wav mono 8kHz", w.getnchannels() == 1 and w.getframerate() == 8000)
        check("wav ~300ms", 2000 <= w.getnframes() <= 3200, f"({w.getnframes()} frames)")
        check("wav non-silent", any(w.readframes(200)), "")

    # 2. beep() never raises (players may or may not exist here)
    try:
        beep()
        check("beep() no exception", True)
    except Exception as e:
        check("beep() no exception", False, f"({e})")

    # 3. AlarmAck: ack mutes, expiry unmutes, clear re-arms, active keeps muted
    a = AlarmAck(seconds=600)
    t = 1000.0
    check("initially unmuted", not a.muted(t))
    a.rearm_if_cleared(True, t)
    a.ack(t)
    check("ack mutes", a.muted(t + 1))
    check("still muted at +599", a.muted(t + 599))
    check("unmuted after window", not a.muted(t + 601))
    # ack again while alarm still active -> no re-arm on the clear-edge that hasn't happened
    a.ack(t + 700)
    a.rearm_if_cleared(True, t + 701)
    check("active alarm does not re-arm", a.muted(t + 701))
    # alarm clears -> re-armed immediately even though window had time left
    a.rearm_if_cleared(False, t + 702)
    check("clear re-arms instantly", not a.muted(t + 702))

    print("\n" + ("ALL BEEP TESTS PASS" if ok else "SOME TESTS FAILED"))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
