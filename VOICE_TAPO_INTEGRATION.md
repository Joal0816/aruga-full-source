# Two-Way Voice: Tapo C200 Integration Research

> Source: @librarian research lane, 2026-10-03. Companion to the `/voice` WebSocket relay in
> `stream_bridge.py`. Captures the confirmed protocol facts and the phased plan for
> nurse ⇄ patient two-way talk.

## TL;DR

- **Camera mic → app (patient speaks):** plain RTSP. Audio codec is **G.711 A-law / PCMA 8 kHz**,
  not AAC. ffmpeg extraction (verified command):

  ```bash
  ffmpeg -rtsp_transport tcp -i "rtsp://user:pass@IP:554/stream1" \
    -vn -acodec pcm_s16le -ar 48000 -ac 1 -f s16le pipe:1
  ```

  `-f s16le` is mandatory (ffmpeg errors without an output format for `pipe:1`).
  Native rate is 8 kHz; `-ar 48000` is a lossless upsample for transport.

- **App → camera speaker (nurse speaks):** no official API, **ONVIF is a dead end**
  (Tapo = Profile S only; audio output needs Profile T). The one mature unofficial route is
  **go2rtc's `tapo://` source** (MIT, ~14.3k★): local HTTP-multipart protocol on **TCP 8800**,
  Digest auth with `admin` + Tapo **cloud** password (plain or UPPERCASE MD5/SHA256),
  speaker backchannel = MPEG-TS carrying **G.711 A-law 8 kHz**. Same-LAN only.

## Direction A — camera mic → app (works today over RTSP)

- Codec per live SDP probe of C200 (go2rtc issue #1494): `a=rtpmap:8 PCMA/8000`, audio on
  both `stream1` and `stream2` (verify with ffprobe on the actual unit; prefer `stream1`).
- Requirements / quirks:
  - **Camera Account** must be enabled in Tapo app (Device Settings → Advanced Settings →
    Camera Account) — RTSP uses this user/pass, NOT the Tapo cloud login.
  - **Mic unmuted** in Tapo app (Tapo FAQ 724: RTSP/ONVIF sound issues → adjust mic settings).
  - **Max 2 concurrent streams per camera** — the Tapo app itself counts as one; close it if
    RTSP pull fails.
  - Use `-rtsp_transport tcp` (UDP flaky through tunnels/NAT).

## Direction B — app → camera speaker (ranked)

| # | Option | Notes | Risk |
|---|--------|-------|------|
| 1 | **go2rtc `tapo://` sidecar (RECOMMENDED)** | github.com/AlexxIT/go2rtc, MIT, active. LAN TCP 8800, digest auth (cloud password, UPPERCASE SHA256/MD5 hash OK). Speaker backchannel implemented in `pkg/tapo/backchannel.go` | Medium — unofficial, firmware-sensitive |
| 2 | Direct Python re-implementation of the 8800 backchannel | Protocol documented at drmnsamoliu.github.io/video.html + go2rtc source as reference | High — you own breakage |
| 3 | Scrypted `@scrypted/tapo` | One user confirmed C200 fw v1.1.15 works; C225 failed | Medium-high |
| 4 | pytapo / python-kasa / Rust tapo | Active libs, **no talk/backchannel** | n/a |
| 5 | ONVIF (onvif-zeep etc.) | **Dead end** — Profile S only, no AudioOutput | ruled out |
| 6 | Camera web page / official TP-Link API | No talk endpoint; cloud API is plugs/lights only | ruled out |

### go2rtc integration paths (our bridge holds PCM16LE/48k)

- **B1 — "Stream to camera" API (one POST):**
  ```bash
  curl -X POST "http://127.0.0.1:1984/api/streams?dst=tapo_cam&src=ffmpeg:http://127.0.0.1:8080/nurse-feed#audio=pcma"
  # stop:  curl -X POST "http://127.0.0.1:1984/api/streams?dst=tapo_cam&src="
  ```
- **B2 — ffmpeg RTSP push (pipe PCM straight from Python — cleanest for our `tapo_speaker_push` hook):**
  ```bash
  ffmpeg -re -f s16le -ar 48000 -ac 1 -i pipe:0 \
    -vn -c:a pcma -ar 8000 -ac 1 -rtsp_transport tcp \
    -f rtsp rtsp://127.0.0.1:8554/tapo_cam
  ```
- **B3 — browser mic → go2rtc WebRTC directly:** `https://<tunnel>/webrtc.html?src=tapo_cam&media=video+audio+microphone` (needs HTTPS — we have it).

go2rtc config:
```yaml
streams:
  tapo_cam:
    - tapo://admin:UPPERCASE-SHA256-OF-CLOUD-PASSWORD@192.168.1.100
    - rtsp://rtspuser:rtspass@192.168.1.100:554/stream1
```
Hash: `echo -n "cloud password" | sha256sum | awk '{print toupper($0)}'`

The `tapo://` downlink also carries the camera mic → Direction A can optionally come from
go2rtc too (one dependency for both directions).

## Firmware gotcha

Firmware **build 230921+** broke cloud-password auth for tapo:// (Invalid cloud password,
HomeAssistant-Tapo-Control#551). Official fix: Tapo app → **Tapo Lab → Third-Party
Compatibility** toggle. Fallback: factory reset + block camera internet access
(`n-device-api.tplinkcloud.com`, `security.iot.i.tplinknbu.com`).

## Fallbacks if camera-speaker proves unstable

1. **(a) PC speakers next to camera** — bridge plays nurse PCM locally. **Default; already
   implemented** as `--voice-sink pulse/file` in `stream_bridge.py`.
2. **(b) Old phone/tablet as "room speaker"** — runs the PWA in a room-end mode (WS audio +
   alerts + own mic return). Strong long-term substitute; doubles as room mic if camera path fails.
3. (c) Smart-speaker APIs — weak fit (cloud binding, latency). Only if already deployed.

## Phased plan

**TODAY (no camera):** synthetic patient-audio demo + PC-speaker sink (done in voice relay);
WS audio channel at PCM16LE/48k (done); keep this doc as the camera-arrival checklist.

**WHEN CAMERA ARRIVES:**
1. Note firmware build. In app: enable Camera Account, unmute mic, enable
   Tapo Lab → Third-Party Compatibility.
2. Probe: `ffprobe -rtsp_transport tcp rtsp://user:pass@IP:554/stream1` (expect PCMA/8000);
   test `tapo://admin:HASH@IP` in go2rtc WebUI (http://IP:1984). Auth fails → factory reset +
   block camera internet.
3. Vendor go2rtc (binary/Docker, same LAN). Wire `tapo_speaker_push()` hook in
   `stream_bridge.py` → Path **B2** (ffmpeg pipe from the nurse PCM queue).
4. Watch the 2-concurrent-stream limit (close RTSP producer while talking).
5. Validate sound at camera speaker; keep fallback (a) as always-works path.

**Key sources:** go2rtc `pkg/tapo/{client,backchannel}.go`, `internal/tapo/README.md`,
`internal/streams/README.md`; issues #1494, #1464, #1954; Tapo FAQ 724; TP-Link FAQ 2680;
drmnsamoliu.github.io/video.html; JurajNyiri/HomeAssistant-Tapo-Control#551.
