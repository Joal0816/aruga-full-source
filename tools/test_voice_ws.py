#!/usr/bin/env python3
"""
Raw-socket (zero-dependency) test for the ARUGA voice relay at
ws://<host>:<port>/voice.

Checks:
  1. RFC 6455 handshake (101 + correct Sec-WebSocket-Accept)
  2. >= 100 KB of binary PCM16LE/48 kHz audio within 5 s (synthetic mode)
  3. {"type":"hello",...} on connect, then {"type":"speaking","on":true}
     within ~10 s (synthetic bursts every ~8 s)
  4. Sends a 0.5 s 440 Hz PCM16LE tone to the server as masked binary frames
  5. With --voice-sink file, /tmp/aruga_nurse_speak.pcm grows by ~48 KB +/-20%

Usage:
  python3 tools/test_voice_ws.py [--host 127.0.0.1] [--port 8080]
                                 [--sink-file /tmp/aruga_nurse_speak.pcm]
"""

import argparse
import base64
import hashlib
import json
import math
import os
import socket
import struct
import sys
import time

WS_GUID = "258EAFA5-E914-47DA-95CA-C5AB0DC85B11"

RESULTS = []


def check(name, ok, detail=""):
    RESULTS.append(ok)
    print(f"[{'PASS' if ok else 'FAIL'}] {name}" + (f"  ({detail})" if detail else ""))


class WSClient:
    """Minimal RFC 6455 client over a raw socket (client frames masked)."""

    def __init__(self, host, port, path):
        self.sock = socket.create_connection((host, port), timeout=10)
        self.buf = b""
        self._handshake(path)

    def _handshake(self, path):
        key = base64.b64encode(os.urandom(16)).decode("ascii")
        request = (
            f"GET {path} HTTP/1.1\r\n"
            f"Host: {self.sock.getpeername()[0]}:{self.sock.getpeername()[1]}\r\n"
            "Upgrade: websocket\r\n"
            "Connection: Upgrade\r\n"
            f"Sec-WebSocket-Key: {key}\r\n"
            "Sec-WebSocket-Version: 13\r\n\r\n"
        )
        self.sock.sendall(request.encode("ascii"))

        # Read the handshake response (may already carry early frames).
        deadline = time.monotonic() + 10
        while b"\r\n\r\n" not in self.buf and time.monotonic() < deadline:
            self.sock.settimeout(max(0.1, deadline - time.monotonic()))
            data = self.sock.recv(4096)
            if not data:
                raise ConnectionError("server closed before handshake completed")
            self.buf += data

        header_blob, _, remainder = self.buf.partition(b"\r\n\r\n")
        self.buf = remainder
        header_text = header_blob.decode("latin-1")
        status_line = header_text.split("\r\n")[0] if header_text else ""

        if "101" not in status_line:
            raise RuntimeError(f"handshake rejected: {status_line}")

        expected = base64.b64encode(
            hashlib.sha1((key + WS_GUID).encode("ascii")).digest()
        ).decode("ascii")
        if f"Sec-WebSocket-Accept: {expected}" not in header_text.replace(" ", " ").replace("\r", ""):
            # tolerate header casing / spacing
            accept_line = ""
            for line in header_text.split("\r\n"):
                if line.lower().startswith("sec-websocket-accept:"):
                    accept_line = line.split(":", 1)[1].strip()
                    break
            if accept_line != expected:
                raise RuntimeError(
                    f"bad Sec-WebSocket-Accept: {accept_line!r} != {expected!r}")
        return status_line

    # ---- frame reading ------------------------------------------------
    def _try_parse(self):
        buf = self.buf
        if len(buf) < 2:
            return None
        b1, b2 = buf[0], buf[1]
        fin = bool(b1 & 0x80)
        opcode = b1 & 0x0F
        masked = bool(b2 & 0x80)
        length = b2 & 0x7F
        offset = 2
        if length == 126:
            if len(buf) < offset + 2:
                return None
            length = struct.unpack("!H", buf[offset:offset + 2])[0]
            offset += 2
        elif length == 127:
            if len(buf) < offset + 8:
                return None
            length = struct.unpack("!Q", buf[offset:offset + 8])[0]
            offset += 8
        mask = b""
        if masked:
            if len(buf) < offset + 4:
                return None
            mask = buf[offset:offset + 4]
            offset += 4
        if len(buf) < offset + length:
            return None
        payload = buf[offset:offset + length]
        self.buf = buf[offset + length:]
        if masked:
            payload = bytes(b ^ mask[i % 4] for i, b in enumerate(payload))
        return fin, opcode, payload

    def read_frame(self, timeout=10.0):
        deadline = time.monotonic() + timeout
        while True:
            frame = self._try_parse()
            if frame is not None:
                return frame
            if time.monotonic() >= deadline:
                raise TimeoutError(f"no frame within {timeout}s")
            self.sock.settimeout(max(0.1, deadline - time.monotonic()))
            data = self.sock.recv(65536)
            if not data:
                raise ConnectionError("server closed the connection")
            self.buf += data

    def read_json(self, timeout=10.0):
        """Read frames until a text (JSON) frame arrives."""
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            remaining = deadline - time.monotonic()
            fin, opcode, payload = self.read_frame(timeout=max(0.1, remaining))
            if opcode == 0x1:
                return json.loads(payload.decode("utf-8"))
        raise TimeoutError("no JSON control frame received")

    # ---- frame writing ---------------------------------------------
    def send_frame(self, opcode, payload=b""):
        header = bytes((0x80 | opcode,))
        n = len(payload)
        if n < 126:
            header += bytes((0x80 | n,))
        elif n <= 0xFFFF:
            header += bytes((0x80 | 126,)) + struct.pack("!H", n)
        else:
            header += bytes((0x80 | 127,)) + struct.pack("!Q", n)
        mask = os.urandom(4)
        masked = bytes(b ^ mask[i % 4] for i, b in enumerate(payload))
        self.sock.sendall(header + mask + masked)

    def send_binary(self, data):
        self.send_frame(0x2, data)

    def close(self):
        try:
            self.send_frame(0x8, b"")
        except OSError:
            pass
        try:
            self.sock.close()
        except OSError:
            pass


def tone_440hz(seconds=0.5, sample_rate=48000, amplitude=0.3):
    samples = int(seconds * sample_rate)
    out = bytearray()
    for i in range(samples):
        value = amplitude * math.sin(2.0 * math.pi * 440.0 * i / sample_rate)
        out += struct.pack("<h", int(value * 32767.0))
    return bytes(out)


def main():
    parser = argparse.ArgumentParser(description="Test ARUGA voice relay WebSocket")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--sink-file", default="/tmp/aruga_nurse_speak.pcm")
    args = parser.parse_args()

    print(f"Testing voice relay at ws://{args.host}:{args.port}/voice")
    print("=" * 70)

    # ---- Check 1: handshake ------------------------------------------
    try:
        ws = WSClient(args.host, args.port, "/voice")
        check("1. WebSocket handshake (101 Switching Protocols + Accept)", True)
    except Exception as exc:
        check("1. WebSocket handshake (101 Switching Protocols + Accept)", False, str(exc))
        print(f"\n{sum(1 for r in RESULTS if r)}/{len(RESULTS)} checks passed")
        return 1

    # ---- Checks 2 & 3a: 5 s of binary PCM + hello --------------------
    binary_bytes = 0
    hello = None
    t0 = time.monotonic()
    while time.monotonic() - t0 < 5.0:
        try:
            remaining = 5.0 - (time.monotonic() - t0)
            fin, opcode, payload = ws.read_frame(timeout=max(0.1, remaining))
        except TimeoutError:
            break
        except (ConnectionError, OSError) as exc:
            check("2. >=100 KB binary PCM within 5 s", False, f"connection error: {exc}")
            break
        if opcode == 0x2:
            binary_bytes += len(payload)
        elif opcode == 0x1:
            try:
                msg = json.loads(payload.decode("utf-8"))
                if msg.get("type") == "hello":
                    hello = msg
            except (UnicodeDecodeError, json.JSONDecodeError):
                pass

    check("2. >=100 KB binary PCM within 5 s (synthetic mode)",
          binary_bytes >= 100 * 1024,
          f"received {binary_bytes / 1024:.1f} KiB in 5 s")
    check("3a. hello JSON on connect",
          (hello is not None
           and hello.get("type") == "hello"
           and hello.get("sampleRate") == 48000
           and hello.get("patientSource") in ("synthetic", "mic", "rtsp", "silence")),
          str(hello))

    # ---- Check 3b: speaking:true within ~10 s total -------------------
    speaking_seen = None
    while time.monotonic() - t0 < 12.0:
        try:
            remaining = 12.0 - (time.monotonic() - t0)
            fin, opcode, payload = ws.read_frame(timeout=max(0.1, remaining))
        except (TimeoutError, ConnectionError, OSError):
            break
        if opcode == 0x1:
            try:
                msg = json.loads(payload.decode("utf-8"))
                if msg.get("type") == "speaking":
                    speaking_seen = msg.get("on")
                    if speaking_seen is True:
                        break
            except (UnicodeDecodeError, json.JSONDecodeError):
                pass
    check("3b. speaking:true JSON within ~10 s", speaking_seen is True,
          f"last speaking state seen: {speaking_seen}")

    # ---- Checks 4 & 5: send 0.5 s 440 Hz tone, watch sink file grow ---
    try:
        before = os.path.getsize(args.sink_file) if os.path.exists(args.sink_file) else 0
    except OSError:
        before = 0

    tone = tone_440hz(0.5)
    frames_sent = 0
    try:
        chunk = 4800  # 100 ms per frame
        for i in range(0, len(tone), chunk):
            ws.send_binary(tone[i:i + chunk])
            frames_sent += 1
            time.sleep(0.02)
        check("4. Sent 0.5 s 440 Hz PCM16LE tone (masked binary frames)",
              len(tone) == 48000 and frames_sent == 10,
              f"{len(tone)} bytes in {frames_sent} masked frames")
    except OSError as exc:
        check("4. Sent 0.5 s 440 Hz PCM16LE tone (masked binary frames)",
              False, str(exc))

    time.sleep(1.5)  # let the sink drain the queued PCM
    try:
        after = os.path.getsize(args.sink_file) if os.path.exists(args.sink_file) else 0
    except OSError:
        after = 0
    delta = after - before
    expected = 48000  # 0.5 s * 48000 Hz * 2 bytes
    check("5. Nurse sink file grew by ~48 KB +/-20% (--voice-sink file)",
          abs(delta - expected) <= 0.2 * expected,
          f"grew {delta} bytes (expected ~{expected})")

    ws.close()

    print("=" * 70)
    passed = sum(1 for r in RESULTS if r)
    print(f"{passed}/{len(RESULTS)} checks passed")
    return 0 if passed == len(RESULTS) else 1


if __name__ == "__main__":
    sys.exit(main())
