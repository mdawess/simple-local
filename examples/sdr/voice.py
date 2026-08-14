import os
import shutil
import subprocess
import sys
import tempfile

from settings import Settings

# Local voice stack, matching the llama.cpp ethos:
#   TTS -> piper           (https://github.com/rhasspy/piper)
#   STT -> whisper.cpp     (whisper-cli)
# Both are optional; if a binary is missing we fall back to text so the demo
# still runs.


def speak(text: str, settings: Settings) -> None:
    piper = shutil.which("piper")
    if not piper:
        print("(voice: piper not installed — printing instead of speaking)", file=sys.stderr)
        return
    wav = tempfile.NamedTemporaryFile(suffix=".wav", delete=False).name
    try:
        subprocess.run(
            [piper, "--model", settings.tts_model, "--output_file", wav],
            input=text.encode(), check=True, capture_output=True,
        )
        player = shutil.which("afplay") or shutil.which("aplay")
        if player:
            subprocess.run([player, wav], check=False)
    finally:
        if os.path.exists(wav):
            os.unlink(wav)


def listen(settings: Settings, seconds: int = 6) -> str:
    ffmpeg = shutil.which("ffmpeg")
    whisper = shutil.which("whisper-cli") or shutil.which("whisper")
    if not ffmpeg or not whisper:
        print("(voice: ffmpeg/whisper-cli not found — type the reply instead)", file=sys.stderr)
        return input("PROSPECT> ").strip()

    wav = tempfile.NamedTemporaryFile(suffix=".wav", delete=False).name
    try:
        # macOS microphone capture; on Linux swap avfoundation for alsa/pulse.
        subprocess.run(
            [ffmpeg, "-y", "-f", "avfoundation", "-i", ":0", "-t", str(seconds), "-ar", "16000", wav],
            check=True, capture_output=True,
        )
        out = subprocess.run(
            [whisper, "-m", settings.stt_model, "-f", wav, "-nt", "-otxt", "-of", wav],
            check=True, capture_output=True, text=True,
        )
        transcript_file = wav + ".txt"
        if os.path.exists(transcript_file):
            with open(transcript_file) as f:
                return f.read().strip()
        return out.stdout.strip()
    finally:
        for p in (wav, wav + ".txt"):
            if os.path.exists(p):
                os.unlink(p)
