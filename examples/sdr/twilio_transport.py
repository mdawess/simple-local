import asyncio
import audioop
import base64
import json
import os
import queue
import subprocess
import tempfile
import threading
import time
import wave

import websockets
from twilio.rest import Client

import stt
from settings import Settings
from telephony import Transport

# Twilio Media Streams audio is 8 kHz, mono, 8-bit mu-law, in 20 ms frames.
SAMPLE_RATE = 8000
FRAME_BYTES = 160  # 20 ms of mu-law at 8 kHz

# Energy-based endpointing (crude but works for a demo; tune per line quality).
SPEECH_RMS = 500       # 16-bit RMS above this counts as speech
ENDPOINT_MS = 800      # trailing silence that ends a turn
MAX_WAIT_START_MS = 8000
MAX_UTTERANCE_MS = 15000


def _text_to_mulaw(text: str) -> bytes:
    wav_path = tempfile.NamedTemporaryFile(suffix=".wav", delete=False).name
    try:
        subprocess.run(
            ["say", "--file-format=WAVE", "--data-format=LEI16@8000", "-o", wav_path, text],
            check=True, capture_output=True,
        )
        with wave.open(wav_path) as w:
            pcm = w.readframes(w.getnframes())
        return audioop.lin2ulaw(pcm, 2)
    finally:
        if os.path.exists(wav_path):
            os.unlink(wav_path)


def _mulaw_to_wav16k(mulaw: bytes, path: str) -> None:
    pcm8k = audioop.ulaw2lin(mulaw, 2)
    pcm16k, _ = audioop.ratecv(pcm8k, 2, 1, SAMPLE_RATE, 16000, None)
    with wave.open(path, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(16000)
        w.writeframes(pcm16k)


def _frame_rms(mulaw_frame: bytes) -> int:
    return audioop.rms(audioop.ulaw2lin(mulaw_frame, 2), 2)


class TwilioTransport(Transport):
    """Places a real outbound call and bridges its audio to local models.

    Half-duplex: the agent speaks a full turn (say -> mu-law -> Twilio), then
    listens until the caller goes silent (Twilio -> mu-law -> whisper.cpp).
    A public wss:// endpoint (e.g. ngrok) must forward to this ws server.
    """

    def __init__(self, settings: Settings):
        self.settings = settings
        try:
            self.sid = os.environ["TWILIO_ACCOUNT_SID"]
            self.token = os.environ["TWILIO_AUTH_TOKEN"]
        except KeyError as e:
            raise RuntimeError("missing Twilio credentials in the root .env") from e
        if not settings.twilio_from or not settings.twilio_stream_url:
            raise RuntimeError("set telephony.twilio.from_number and stream_url in config")

        if stt.backend() is None:
            raise RuntimeError("no whisper backend on PATH (install whisper.cpp or openai-whisper)")

        self._client = Client(self.sid, self.token)
        self._call = None
        self._loop = None
        self._server = None
        self._ws = None
        self._stream_sid = None
        self._inbound: queue.Queue = queue.Queue()
        self._marks: dict[str, threading.Event] = {}
        self._stream_started = threading.Event()
        self._call_ended = threading.Event()

    def start(self, phone_number: str) -> None:
        threading.Thread(target=self._run_server, daemon=True).start()
        while self._server is None:
            time.sleep(0.05)

        twiml = f'<Response><Connect><Stream url="{self.settings.twilio_stream_url}"/></Connect></Response>'
        self._call = self._client.calls.create(to=phone_number, from_=self.settings.twilio_from, twiml=twiml)
        print(f"[twilio] dialing {phone_number} (call {self._call.sid}) ...")
        if not self._stream_started.wait(timeout=60):
            raise TimeoutError("Twilio media stream never connected — check ngrok URL and from_number")
        print("[twilio] stream connected")

    def say(self, text: str) -> None:
        print(f"AGENT>    {text}")
        mulaw = _text_to_mulaw(text)
        name = f"mark-{int(time.time() * 1000)}"
        done = threading.Event()
        self._marks[name] = done

        async def send():
            for i in range(0, len(mulaw), FRAME_BYTES):
                payload = base64.b64encode(mulaw[i:i + FRAME_BYTES]).decode()
                await self._ws.send(json.dumps(
                    {"event": "media", "streamSid": self._stream_sid, "media": {"payload": payload}}
                ))
            await self._ws.send(json.dumps(
                {"event": "mark", "streamSid": self._stream_sid, "mark": {"name": name}}
            ))

        asyncio.run_coroutine_threadsafe(send(), self._loop).result()
        done.wait(timeout=len(mulaw) / SAMPLE_RATE + 10)  # mark returns when playback finishes
        self._drain_inbound()

    def listen(self) -> str:
        if self._call_ended.is_set():
            return ""
        collected = bytearray()
        speaking = False
        silence_ms = 0
        waited_ms = 0
        while True:
            if self._call_ended.is_set():
                break
            try:
                frame = self._inbound.get(timeout=0.1)
            except queue.Empty:
                if speaking:
                    silence_ms += 100
                    if silence_ms >= ENDPOINT_MS:
                        break
                elif (waited_ms := waited_ms + 100) >= MAX_WAIT_START_MS:
                    break
                continue
            if _frame_rms(frame) >= SPEECH_RMS:
                speaking, silence_ms = True, 0
                collected += frame
            elif speaking:
                silence_ms += 20
                collected += frame
                if silence_ms >= ENDPOINT_MS:
                    break
            if speaking and len(collected) >= MAX_UTTERANCE_MS * 8:
                break

        if not collected:
            return ""
        wav = tempfile.NamedTemporaryFile(suffix=".wav", delete=False).name
        try:
            _mulaw_to_wav16k(bytes(collected), wav)
            text = stt.transcribe(wav, self.settings)
        finally:
            if os.path.exists(wav):
                os.unlink(wav)
        print(f"PROSPECT> {text}")
        return text

    def hangup(self) -> None:
        try:
            if self._call:
                self._client.calls(self._call.sid).update(status="completed")
        except Exception:
            pass
        if self._loop:
            self._loop.call_soon_threadsafe(self._loop.stop)
        print("[twilio] call ended")

    def _drain_inbound(self) -> None:
        try:
            while True:
                self._inbound.get_nowait()
        except queue.Empty:
            pass

    def _run_server(self) -> None:
        loop = asyncio.new_event_loop()
        self._loop = loop
        asyncio.set_event_loop(loop)

        async def serve() -> None:
            # serve() must be awaited inside the running loop (websockets v14 API).
            self._server = await websockets.serve(
                self._handler, self.settings.ws_host, self.settings.ws_port
            )
            await asyncio.Future()  # keep serving until the loop is stopped

        try:
            loop.run_until_complete(serve())
        except RuntimeError:
            pass  # loop stopped by hangup()

    async def _handler(self, websocket, *_) -> None:
        self._ws = websocket
        async for message in websocket:
            data = json.loads(message)
            event = data.get("event")
            if event == "start":
                self._stream_sid = data["start"]["streamSid"]
                self._stream_started.set()
            elif event == "media":
                self._inbound.put(base64.b64decode(data["media"]["payload"]))
            elif event == "mark":
                mark = self._marks.get(data["mark"]["name"])
                if mark:
                    mark.set()
            elif event == "stop":
                self._call_ended.set()
