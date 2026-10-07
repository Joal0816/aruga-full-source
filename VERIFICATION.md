# ARUGA — Real-World Verification Guide (Linux)

Everything below is verified on this box (Linux, Python 3.12/3.14, i5-3230M,
no GPU, HD WebCam on `/dev/video0`). All 7 test suites pass; the 70-clip UR
benchmark results are in `eval_report.csv`.

## Automated verification (run this first)

```bash
# Self-test: full pipeline on the synthetic fall clip — no camera needed
.venv/bin/python tools/live_verify.py --file assets/synthetic_fall_demo.mp4

# Find the Tapo on the LAN, then run live (Camera Account creds, not cloud login)
.venv/bin/python tools/live_verify.py --discover
.venv/bin/python tools/live_verify.py --url rtsp://USER:PASS@192.168.1.x:554/stream1 --seconds 30
```

Prints a PASS/FAIL checklist for every automatable item, saves a snapshot to
`/tmp/aruga_live_verify_snapshot.jpg`, and ends with the explicit
**[HUMAN]** items (physical drill, audible beep, zone approval).

## Environments

```bash
cd ~/aruga-full-source

.venv312/bin/python   # py3.12 — native MediaPipe (~17 fps single-person) — preferred for live
.venv/bin/python      # py3.14 — YOLO-ONNX fallback (~4 fps) — equally valid for hallway mode
```

Both venvs can run all three apps. Tests: `.venv/bin/python tests/simulate.py` (plus `tests/test_*.py`).

## 1. Live hallway monitor (flagship: multi-person + zones)

```bash
DISPLAY=:1 .venv312/bin/python hallway_app.py
```

- ⟳ scan picks the HD WebCam → **▶ Start Monitoring**
- Status bar must show rising FPS, `CPU` backend, and person count
- **First session:** *Capture Background* on the empty hallway →
  *Calibrate Zones* (draw `bench` / `floor` / `ignore` areas; the editor has
  inline help, Cancel, and a discard prompt)
- Alarm beeps must sound (uses `paplay`/`pw-play`/`aplay` — present here)
- **🔕 Acknowledge (10 min)** silences repeating beeps; visual alerts and
  logs keep running; re-arms automatically when the alarm clears

## 2. Browser dashboard (review / demo)

```bash
.venv312/bin/python -m streamlit run app.py    # → http://localhost:8501
```

- Heartbeat line shows `● LIVE — N.N fps` (or `○ Idle` / `○ SOURCE LOST`)
- Slider changes mid-run must NOT reset detection state (fixed in Phase 4)
- **Acknowledge** button sits next to Start/Stop

## 3. Desktop app

```bash
.venv312/bin/python desktop_app.py
```

Same checks as hallway (minus zones).

## Acceptance checklist

- [ ] Person tracked with correct risk colors — **red = EMERGENCY only**,
      deep orange = CONCERNING, amber = UNUSUAL, orange = SLUMP
- [ ] Fall → CONCERNING within ~1 s → EMERGENCY after 6 s of stillness
- [ ] Incident log + snapshots accumulate; **Export CSV** works
- [ ] `SIGNAL_LOST → VERIFY IN PERSON` banner when an alarmed person leaves frame
- [ ] Acknowledge silences repeats for 10 min, next incident beeps again
- [ ] Web heartbeat stays `● LIVE` while running; `SOURCE LOST` on camera unplug

## Calibration truth-check (what SHOULD happen)

| Action | Expected |
|---|---|
| Bend over to pick something up | brief CONCERNING, clears on standing — known limitation, **no** escalation |
| Lie still on the floor ≥6 s | FALL_DETECTED + EMERGENCY |
| Recline still on a **zoned bench** | RESTING (no alarm) |
| Fast collapse onto a zoned bench | SLUMP — "check on person" |
| Stand still in a doorway (`ignore` zone) | capped at UNUSUAL |

## Performance expectations (measured, idle CPU)

| Path | fps |
|---|---|
| Single-person, MediaPipe (py3.12) | **17.0** |
| Hallway full chain, YOLO CPU | **3.85** (thread auto-tune picks best count at startup) |
| Benchmark: `python tools/bench_pipeline.py` | per-stage ms/fps |

For real-time multi-person on this box, expect ~4 fps analysis (adequate for
fall timescales, which are ≥0.5 s). GPU upgrade path: `onnxruntime-gpu` in
`requirements.txt` (code auto-selects CUDA).

## Benchmark baseline (70-clip UR Fall Dataset, headless)

- Recall **90.0%** (27/30 falls; 25 escalated to EMERGENCY)
- Precision **58.7%** (19/40 ADLs false-alarm; UR includes ~10 deliberately
  fall-like ADLs; no zones calibrated headless; EMERGENCY is never
  zone-suppressed by design)
- F1 **71.1%** — full per-clip data in `eval_report.csv`

## Before any distribution

YOLOv8n-pose weights are **AGPL-3.0**; this repo has **no LICENSE file**;
UR dataset is **CC BY-NC-SA 4.0 (non-commercial)**. Internal evaluation is
fine — commercial deployment needs legal review (and an Ultralytics
Enterprise License unless the whole derivative work is open-sourced).
