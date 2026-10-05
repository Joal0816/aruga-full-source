"""
ARUGA Voice Relay — two-way WebSocket audio bridge for the stream bridge.

Adds a `GET /voice` WebSocket endpoint (RFC 6455, hand-rolled, zero deps) on the
same port as the HTTP bridge:

  - Server -> client (broadcast, all clients): PCM signed 16-bit LE, mono, 48 kHz
    patient audio, plus JSON control frames (hello / VAD speaking transitions).
  - Client -> server: binary frames of raw PCM (nurse speaking) routed to the
    nurse audio sink (PulseAudio `pacat` or a local file for testing).

Patient audio sources:  synthetic (demo speech-like signal) | mic (pulse) | rtsp | silence
Nurse audio sinks:      auto (pulse -> file fallback) | pulse | file

All voice code paths are isolated: an exception kills only that client/thread,
never the video endpoints or the main HTTP loop.
"""

import base64
import hashlib
import json
import logging
import math
import os
import queue
import select
import socket
import struct
import subprocess
import sys
import threading
import time

logger = logging.getLogger("aruga_bridge")

# ---------------------------------------------------------------------------
# Audio format (both directions): PCM signed 16-bit little-endian, mono, 48 kHz
# ---------------------------------------------------------------------------
SAMPLE_RATE = 48000
SAMPLE_WIDTH = 2          # bytes per sample (s16le)
CHANNELS = 1
CHUNK_MS = 100            # source read/generation granularity
CHUNK_BYTES = SAMPLE_RATE * SAMPLE_WIDTH * CHUNK_MS // 1000   # 9600 bytes
CHUNK_SAMPLES = CHUNK_BYTES // SAMPLE_WIDTH                   # 4800

# ---------------------------------------------------------------------------
# VAD (patient stream): windowed RMS + hysteresis
# ---------------------------------------------------------------------------
VAD_WINDOW_MS = 50
VAD_WINDOW_SAMPLES = SAMPLE_RATE * VAD_WINDOW_MS // 1000      # 2400
VAD_ON_RMS = 0.02       # RMS above this -> "loud"  (tunable)
VAD_OFF_RMS = 0.012     # RMS below this -> "quiet" (tunable hysteresis band)
VAD_ATTACK_MS = 150     # loud for this long before speaking=True
VAD_RELEASE_MS = 400    # quiet for this long before speaking=False

# ---------------------------------------------------------------------------
# WebSocket (RFC 6455) constants
# ---------------------------------------------------------------------------
WS_GUID = "258EAFA5-E914-47DA-95CA-C5AB0DC85B11"
MAX_PAYLOAD = 1024 * 1024          # 1 MiB per frame — close on violation
PING_INTERVAL = 30.0               # server -> client ping every 30 s
READ_TIMEOUT = 60.0                # stall detection for a single client

# Nurse sink file (test/demo mode)
NURSE_SINK_FILE = "/tmp/aruga_nurse_speak.pcm"

# opcodes
OP_CONT = 0x0
OP_TEXT = 0x1
OP_BINARY = 0x2
OP_CLOSE = 0x8
OP_PING = 0x9
OP_PONG = 0xA


# ---------------------------------------------------------------------------
# Tapo speaker hook — wired by a later lane (P2P speaker output)
# ---------------------------------------------------------------------------
def tapo_speaker_push(pcm_bytes):
    """
    Hook for pushing nurse PCM to a Tapo speaker via P2P streaming.

    NOT IMPLEMENTED YET — a later lane will wire the real Tapo P2P speaker
    output here. Kept as a stub so the relay can arm the hook without
    breaking; callers must treat NotImplementedError as "no speaker yet".
    """
    logger.info("🔈 tapo_speaker_push() called — Tapo speaker output not wired yet (NotImplemented)")
    raise NotImplementedError("Tapo P2P speaker output is not implemented yet")


# ---------------------------------------------------------------------------
# WebSocket framing helpers (zero-dependency RFC 6455)
# ---------------------------------------------------------------------------
def ws_make_frame(opcode, payload=b"", fin=True):
    """Build a server->client frame (never masked)."""
    b1 = (0x80 if fin else 0x00) | (opcode & 0x0F)
    n = len(payload)
    if n < 126:
        header = bytes((b1, n))
    elif n <= 0xFFFF:
        header = bytes((b1, 126)) + struct.pack("!H", n)
    else:
        header = bytes((b1, 127)) + struct.pack("!Q", n)
    return header + payload


def _unmask(payload, mask):
    """XOR a masked payload with its 4-byte masking key (fast int-based)."""
    n = len(payload)
    if n == 0:
        return b""
    key = int.from_bytes((mask * (n // 4 + 1))[:n], "big")
    return (int.from_bytes(payload, "big") ^ key).to_bytes(n, "big")


def ws_read_frame(sock):
    """
    Read one WebSocket frame from a connected socket.
    Returns (fin, opcode, payload). Raises ValueError on protocol violations
    (oversized payload, reserved opcode, unexpected continuation).
    """
    def recv_exact(n):
        buf = bytearray()
        while len(buf) < n:
            chunk = sock.recv(n - len(buf))
            if not chunk:
                raise ConnectionError("WebSocket peer closed mid-frame")
            buf.extend(chunk)
        return bytes(buf)

    header = recv_exact(2)
    b1, b2 = header[0], header[1]
    fin = bool(b1 & 0x80)
    opcode = b1 & 0x0F
    if opcode in (0x3, 0x4, 0x5, 0x6, 0x7, 0xB, 0xC, 0xD, 0xE, 0xF):
        raise ValueError(f"reserved opcode 0x{opcode:x}")
    masked = bool(b2 & 0x80)
    length = b2 & 0x7F
    if length == 126:
        length = struct.unpack("!H", recv_exact(2))[0]
    elif length == 127:
        length = struct.unpack("!Q", recv_exact(8))[0]
    if length > MAX_PAYLOAD:
        raise ValueError(f"frame payload {length} exceeds {MAX_PAYLOAD} limit")
    mask = recv_exact(4) if masked else None
    payload = recv_exact(length) if length else b""
    if mask:
        payload = _unmask(payload, mask)
    return fin, opcode, payload


def ws_compute_accept(key):
    digest = hashlib.sha1((key + WS_GUID).encode("utf-8")).digest()
    return base64.b64encode(digest).decode("ascii")


def pcm_rms(pcm):
    """RMS of an s16le mono PCM buffer, normalised to [-1.0, 1.0]."""
    if not pcm:
        return 0.0
    import array
    samples = array.array("h")
    samples.frombytes(pcm)
    if sys.byteorder == "big":
        samples.byteswap()
    if not samples:
        return 0.0
    return math.sqrt(sum(v * v for v in samples) / len(samples)) / 32768.0


# ---------------------------------------------------------------------------
# Synthetic patient audio — speech-like demo signal
# ---------------------------------------------------------------------------
class SyntheticPatientAudio:
    """
    Generates a speech-like signal: continuous low noise floor (~0.015) with
    voiced bursts (~2 s) every ~8 s. Bursts are a multi-tone modulated signal
    at ~0.35 amplitude with a 5 Hz syllabic envelope, so the VAD triggers
    clearly and repeatedly.
    """

    NOISE_AMPLITUDE = 0.015
    BURST_AMPLITUDE = 0.35
    BURST_SECONDS = 2.0
    BURST_PERIOD = 8.0
    SYLLABLE_HZ = 5.0
    TONES = ((175.0, 0.50), (260.0, 0.30), (340.0, 0.20))

    def __init__(self):
        self.t = 0.0
        try:
            import random
            self._random = random.Random()
        except Exception:
            self._random = None

    def next_chunk(self, n_samples):
        out = bytearray()
        t = self.t
        dt = 1.0 / SAMPLE_RATE
        rand = self._random.random if self._random else None
        for _ in range(n_samples):
            cycle = t % self.BURST_PERIOD
            in_burst = cycle < self.BURST_SECONDS
            value = (rand() * 2.0 - 1.0) * self.NOISE_AMPLITUDE
            if in_burst:
                env = 0.5 - 0.5 * math.cos(2.0 * math.pi * self.SYLLABLE_HZ * t)
                voice = 0.0
                for freq, gain in self.TONES:
                    voice += gain * math.sin(2.0 * math.pi * freq * t + freq)
                value += voice * env * self.BURST_AMPLITUDE
            if value > 1.0:
                value = 1.0
            elif value < -1.0:
                value = -1.0
            out += struct.pack("<h", int(value * 32767.0))
            t += dt
        self.t = t
        return bytes(out)


# ---------------------------------------------------------------------------
# VAD — windowed RMS with attack/release hysteresis
# ---------------------------------------------------------------------------
class VoiceActivityDetector:
    def __init__(self):
        self.speaking = False
        self._loud_since = None    # monotonic timestamp of first loud window
        self._quiet_since = None   # monotonic timestamp of first quiet window

    def update(self, pcm):
        """Feed a PCM chunk; returns a list of transition values (True/False)."""
        transitions = []
        for i in range(0, len(pcm) - VAD_WINDOW_SAMPLES * SAMPLE_WIDTH + 1,
                       VAD_WINDOW_SAMPLES * SAMPLE_WIDTH):
            window = pcm[i:i + VAD_WINDOW_SAMPLES * SAMPLE_WIDTH]
            rms = pcm_rms(window)
            now = time.monotonic()
            if rms > VAD_ON_RMS:
                if self._loud_since is None:
                    self._loud_since = now
                self._quiet_since = None
                if (not self.speaking
                        and (now - self._loud_since) * 1000.0 >= VAD_ATTACK_MS):
                    self.speaking = True
                    transitions.append(True)
            elif rms < VAD_OFF_RMS:
                if self._quiet_since is None:
                    self._quiet_since = now
                self._loud_since = None
                if (self.speaking
                        and (now - self._quiet_since) * 1000.0 >= VAD_RELEASE_MS):
                    self.speaking = False
                    transitions.append(False)
            # else: inside the hysteresis band — freeze both timers
        return transitions


# ---------------------------------------------------------------------------
# Nurse audio sink — pulse (pacat) or file, with automatic fallback
# ---------------------------------------------------------------------------
class NurseSink:
    """
    Consumes nurse PCM from a bounded queue (so WS handler threads never block)
    and plays it via `pacat`, or appends it to a file in test/demo mode.

    pulse: pacat --rate=48000 --channels=1 --format=s16le -d default
           if the process dies -> restart once, then fall back to file.
    auto : pulse if a PulseAudio server socket is detected, else file (warning).
    """

    PACAT_RESTARTS = 1

    def __init__(self, mode="auto"):
        self.requested_mode = mode
        self.effective_mode = None
        self._queue = queue.Queue(maxsize=64)     # ~6.4 s of PCM at 48 kHz
        self._thread = None
        self._stopped = False
        self._file_bytes = 0
        self._last_size_log = time.monotonic()

    # -- lifecycle ---------------------------------------------------------
    def start(self):
        mode = self.requested_mode
        if mode == "auto":
            if self._pulse_available():
                mode = "pulse"
            else:
                logger.warning("⚠️  PulseAudio sink unavailable — falling back to file nurse audio")
                mode = "file"
        self.effective_mode = mode

        # Arm the future Tapo speaker hook (safe no-op until wired).
        try:
            tapo_speaker_push(b"")
        except NotImplementedError:
            pass

        if mode == "pulse" and self._spawn_pacat() is None:
            logger.warning("⚠️  pacat unavailable — falling back to file nurse audio")
            mode = "file"
            self.effective_mode = mode

        self._thread = threading.Thread(target=self._loop, name="voice-nurse-sink", daemon=True)
        self._thread.start()
        logger.info(f"🔊 Nurse audio sink ready: {mode}")

    def stop(self):
        self._stopped = True

    def write(self, pcm):
        """Queue nurse PCM. Never blocks the calling WS handler thread."""
        try:
            self._queue.put_nowait(pcm)
        except queue.Full:
            pass  # real-time audio: drop the chunk rather than block a client

    # -- internals ---------------------------------------------------------
    def _pulse_available(self):
        candidates = []
        pulse_server = os.environ.get("PULSE_SERVER", "")
        if pulse_server:
            candidates.append(pulse_server)
        candidates.append(f"/run/user/{os.getuid()}/pulse/native")
        xdg = os.environ.get("XDG_RUNTIME_DIR")
        if xdg:
            candidates.append(os.path.join(xdg, "pulse", "native"))
        for cand in candidates:
            path = cand[len("unix:"):] if cand.startswith("unix:") else cand
            if path and os.path.exists(path):
                return True
        return False

    def _spawn_pacat(self):
        try:
            return subprocess.Popen(
                ["pacat", "--rate=48000", "--channels=1", "--format=s16le",
                 "-d", "default"],
                stdin=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
            )
        except OSError as exc:
            logger.warning(f"⚠️  pacat launch failed ({exc})")
            return None

    def _loop(self):
        pacat = self._spawn_pacat() if self.effective_mode == "pulse" else None
        restarts_left = self.PACAT_RESTARTS
        while not self._stopped:
            try:
                chunk = self._queue.get(timeout=0.5)
            except queue.Empty:
                if pacat is not None and pacat.poll() is not None:
                    pacat = self._handle_pacat_death(pacat, restarts_left)
                    restarts_left = max(0, restarts_left - 1) if pacat is not None else restarts_left
                continue

            if pacat is not None:
                try:
                    pacat.stdin.write(chunk)
                    pacat.stdin.flush()
                    continue
                except (BrokenPipeError, OSError, ValueError):
                    pacat = self._handle_pacat_death(pacat, restarts_left)
                    if pacat is None:
                        restarts_left = 0
                    else:
                        restarts_left -= 1
                    # Requeue the chunk for the restarted process / fallback.
                    self._queue.put(chunk)
                    continue
            else:
                self._write_file(chunk)

    def _handle_pacat_death(self, pacat, restarts_left):
        logger.warning("⚠️  pacat process died")
        try:
            if pacat.stdin:
                pacat.stdin.close()
        except Exception:
            pass
        try:
            pacat.wait(timeout=2)
        except Exception:
            pacat.kill()
        if restarts_left > 0:
            logger.warning("⚠️  Restarting pacat once...")
            return self._spawn_pacat()
        logger.warning("⚠️  pacat restart failed — falling back to file nurse audio")
        self.effective_mode = "file"
        return None

    def _write_file(self, chunk):
        try:
            with open(NURSE_SINK_FILE, "ab") as fh:
                fh.write(chunk)
            self._file_bytes += len(chunk)
            now = time.monotonic()
            if now - self._last_size_log > 30.0:
                self._last_size_log = now
                logger.info(f"💾 Nurse audio file: {self._file_bytes / 1024:.0f} KiB appended at {NURSE_SINK_FILE}")
        except OSError as exc:
            logger.warning(f"⚠️  Nurse audio file write failed ({exc})")


# ---------------------------------------------------------------------------
# Per-client WebSocket connection
# ---------------------------------------------------------------------------
class VoiceClient:
    """One connected /voice WebSocket client (reader in handler thread,
    writer + pings in a dedicated thread)."""

    def __init__(self, sock, relay):
        self.sock = sock
        self.relay = relay
        self.closed = False
        self.outgoing = queue.Queue()
        self._frag_buf = None
        self._frag_opcode = 0
        self.writer_thread = threading.Thread(target=self._writer_loop,
                                              name="voice-client-writer", daemon=True)
        self.writer_thread.start()

    # -- sending -----------------------------------------------------------
    def send_binary(self, data):
        if not self.closed:
            self.outgoing.put(ws_make_frame(OP_BINARY, data))

    def send_json(self, obj):
        if not self.closed:
            try:
                self.outgoing.put(ws_make_frame(OP_TEXT, json.dumps(obj).encode("utf-8")))
            except (TypeError, ValueError):
                pass

    def _writer_loop(self):
        last_ping = time.monotonic()
        while not self.closed:
            try:
                item = self.outgoing.get(timeout=0.5)
            except queue.Empty:
                item = None
            try:
                if item is not None:
                    self.sock.sendall(item)
                now = time.monotonic()
                if now - last_ping >= PING_INTERVAL:
                    self.sock.sendall(ws_make_frame(OP_PING, b""))
                    last_ping = now
            except OSError:
                self.close()
                return

    # -- reading (runs in the HTTP handler thread) --------------------------
    def run_reader(self):
        try:
            self.sock.settimeout(READ_TIMEOUT)
            while not self.closed:
                ready, _, _ = select.select([self.sock], [], [], 1.0)
                if not ready:
                    continue
                fin, opcode, payload = ws_read_frame(self.sock)
                self._handle_frame(fin, opcode, payload)
        except (ConnectionError, ConnectionResetError, BrokenPipeError,
                socket.timeout, OSError, ValueError) as exc:
            logger.info(f"🎙 Voice client disconnected ({exc})")
        finally:
            self.close()

    def _handle_frame(self, fin, opcode, payload):
        if opcode == OP_CLOSE:
            # Echo the close handshake, then tear down.
            try:
                self.sock.sendall(ws_make_frame(OP_CLOSE, payload[:125]))
            except OSError:
                pass
            self.close()
            return
        if opcode == OP_PING:
            self.outgoing.put(ws_make_frame(OP_PONG, payload))
            return
        if opcode == OP_PONG:
            return  # liveness acknowledgement — nothing to do

        if opcode in (OP_TEXT, OP_BINARY):
            if not fin:
                self._frag_buf = bytearray(payload)
                self._frag_opcode = opcode
                return
            self._dispatch(opcode, payload)
            return
        if opcode == OP_CONT:
            if self._frag_buf is None:
                raise ValueError("continuation frame without a start frame")
            self._frag_buf.extend(payload)
            if len(self._frag_buf) > MAX_PAYLOAD:
                raise ValueError("fragmented message exceeds payload limit")
            if fin:
                self._dispatch(self._frag_opcode, bytes(self._frag_buf))
                self._frag_buf = None
            return
        raise ValueError(f"unexpected opcode 0x{opcode:x}")

    def _dispatch(self, opcode, payload):
        try:
            if opcode == OP_BINARY:
                # Nurse speaking — raw PCM16LE/48 kHz/mono -> sink.
                if self.relay.nurse_sink is not None:
                    self.relay.nurse_sink.write(payload)
            elif opcode == OP_TEXT:
                try:
                    msg = json.loads(payload.decode("utf-8"))
                    logger.debug(f"🎙 Voice control text frame: {msg}")
                except (UnicodeDecodeError, json.JSONDecodeError):
                    logger.debug("🎙 Voice text frame (non-JSON) ignored")
        except Exception as exc:  # never let a client kill the relay
            logger.warning(f"⚠️  Voice frame dispatch error ({exc}) — closing client")
            self.close()

    def close(self):
        if self.closed:
            return
        self.closed = True
        try:
            self.sock.shutdown(socket.SHUT_RDWR)
        except OSError:
            pass
        try:
            self.sock.close()
        except OSError:
            pass


# ---------------------------------------------------------------------------
# Voice relay — client registry + patient audio source + broadcast
# ---------------------------------------------------------------------------
class VoiceRelay:
    def __init__(self, patient_mode="synthetic", sink_mode="auto", rtsp_url=""):
        self.patient_source_mode = patient_mode
        self.rtsp_url = rtsp_url
        self.nurse_sink = NurseSink(sink_mode)
        self.vad = VoiceActivityDetector()
        self.clients = []
        self.clients_lock = threading.Lock()
        self.stopped = False
        self._source_thread = None

    # -- lifecycle ---------------------------------------------------------
    def start(self):
        self.nurse_sink.start()
        self._source_thread = threading.Thread(target=self._source_loop,
                                               name="voice-patient-source", daemon=True)
        self._source_thread.start()

    def stop(self):
        self.stopped = True
        self.nurse_sink.stop()
        with self.clients_lock:
            for client in list(self.clients):
                client.close()
            self.clients.clear()

    # -- client registry ----------------------------------------------------
    def register(self, client):
        with self.clients_lock:
            self.clients.append(client)
        logger.info(f"🎙 Voice client connected ({len(self.clients)} connected)")

    def unregister(self, client):
        with self.clients_lock:
            if client in self.clients:
                self.clients.remove(client)
        logger.info(f"🎙 Voice client disconnected ({len(self.clients)} connected)")

    def broadcast_binary(self, pcm):
        with self.clients_lock:
            targets = list(self.clients)
        for client in targets:
            try:
                client.send_binary(pcm)
            except Exception as exc:
                logger.warning(f"⚠️  Broadcast error ({exc}) — dropping client")
                client.close()

    def broadcast_json(self, obj):
        with self.clients_lock:
            targets = list(self.clients)
        for client in targets:
            try:
                client.send_json(obj)
            except Exception as exc:
                logger.warning(f"⚠️  Broadcast error ({exc}) — dropping client")
                client.close()

    # -- patient audio source ----------------------------------------------
    def _source_loop(self):
        try:
            mode = self.patient_source_mode
            if mode == "rtsp":
                if not self._ffmpeg_loop(self._rtsp_command(), "RTSP patient audio"):
                    self._fallback_to_synthetic()
            elif mode == "mic":
                if not self._ffmpeg_loop(self._mic_command(), "Mic patient audio"):
                    self._fallback_to_synthetic()
            elif mode == "silence":
                logger.info("🔇 Patient audio source: silence")
                while not self.stopped:
                    self._process_chunk(bytes(CHUNK_BYTES))
                    time.sleep(0.1)
            else:
                self._synthetic_loop()
        except Exception as exc:  # never kill the process — only this thread
            logger.warning(f"⚠️  Patient audio source failed ({exc}) — switching to synthetic")
            self._fallback_to_synthetic()

    def _fallback_to_synthetic(self):
        if self.patient_source_mode == "synthetic":
            return
        logger.warning("⚠️  Falling back to SYNTHETIC patient audio (no camera) — demo mode")
        self.patient_source_mode = "synthetic"
        self._synthetic_loop()

    def _synthetic_loop(self):
        logger.info("🧪 SYNTHETIC patient audio (no camera) — demo mode")
        generator = SyntheticPatientAudio()
        next_tick = time.monotonic()
        while not self.stopped:
            chunk = generator.next_chunk(CHUNK_SAMPLES)
            self._process_chunk(chunk)
            next_tick += CHUNK_MS / 1000.0
            delay = next_tick - time.monotonic()
            if delay > 0:
                time.sleep(delay)
            else:
                next_tick = time.monotonic()

    def _rtsp_command(self):
        return ["ffmpeg", "-nostdin", "-rtsp_transport", "tcp",
                "-i", self.rtsp_url, "-vn",
                "-acodec", "pcm_s16le", "-ar", str(SAMPLE_RATE),
                "-ac", str(CHANNELS), "-f", "s16le", "pipe:1"]

    def _mic_command(self):
        return ["ffmpeg", "-nostdin", "-f", "pulse", "-i", "default",
                "-acodec", "pcm_s16le", "-ar", str(SAMPLE_RATE),
                "-ac", str(CHANNELS), "-f", "s16le", "pipe:1"]

    def _ffmpeg_loop(self, command, label):
        """
        Spawn ffmpeg and read CHUNK_BYTES (100 ms) per read in this daemon
        thread. Returns True while the stream is healthy; False when ffmpeg
        exits or produces no audio (caller falls back to synthetic).
        """
        try:
            proc = subprocess.Popen(command, stdout=subprocess.PIPE,
                                    stderr=subprocess.DEVNULL, bufsize=CHUNK_BYTES)
        except OSError as exc:
            logger.warning(f"⚠️  {label}: ffmpeg launch failed ({exc})")
            return False
        logger.info(f"🎚 Patient audio source: {label}")
        try:
            while not self.stopped:
                try:
                    chunk = proc.stdout.read(CHUNK_BYTES)
                except OSError:
                    chunk = b""
                if not chunk:
                    logger.warning(f"⚠️  {label}: stream ended — falling back to synthetic")
                    return False
                self._process_chunk(chunk)
                # The pipe fills at real-time rate, pacing the loop naturally.
        finally:
            try:
                proc.terminate()
                proc.wait(timeout=2)
            except Exception:
                try:
                    proc.kill()
                except Exception:
                    pass
        return True

    def _process_chunk(self, pcm):
        # Broadcast PCM to all clients regardless of VAD state so playback
        # stays in sync, then run the VAD for speaking transitions.
        self.broadcast_binary(pcm)
        for speaking in self.vad.update(pcm):
            self.broadcast_json({"type": "speaking", "on": speaking})


# ---------------------------------------------------------------------------
# HTTP integration entry point
# ---------------------------------------------------------------------------
def handle_voice_websocket(handler, relay):
    """
    Take over a BaseHTTPRequestHandler connection as a WebSocket.

    Called from BridgeRequestHandler.do_GET when path == "/voice" and an
    `Upgrade: websocket` header is present. Performs the RFC 6455 handshake,
    then runs the client's WS read loop in this handler thread (one thread per
    client — safe for long-lived connections).
    """
    sock = handler.connection
    try:
        key = handler.headers.get("Sec-WebSocket-Key")
        if not key:
            handler.send_response(400)
            handler.send_header("Content-Type", "text/plain")
            handler.end_headers_with_cors()
            handler.wfile.write(b"Missing Sec-WebSocket-Key header")
            return

        handler.send_response(101, "Switching Protocols")
        handler.send_header("Upgrade", "websocket")
        handler.send_header("Connection", "Upgrade")
        handler.send_header("Sec-WebSocket-Accept", ws_compute_accept(key))
        handler.end_headers()
        handler.wfile.flush()

        client = VoiceClient(sock, relay)
        relay.register(client)
        try:
            client.send_json({
                "type": "hello",
                "patientSource": relay.patient_source_mode,
                "sampleRate": SAMPLE_RATE,
            })
            client.run_reader()
        finally:
            relay.unregister(client)
            client.close()
    except Exception as exc:  # never let a voice client kill the bridge
        logger.warning(f"⚠️  Voice WebSocket handler error ({exc})")
    finally:
        handler.close_connection = True
