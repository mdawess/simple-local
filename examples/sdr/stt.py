import os
import shutil
import subprocess

from settings import Settings

# Two supported backends, whichever is on PATH:
#   whisper.cpp   -> `whisper-cli` / `whisper-cpp`, ggml model, ~0.5s/turn
#   openai-whisper-> `whisper`, downloads its own model, ~1.8s/turn (warm)
_CPP = shutil.which("whisper-cli") or shutil.which("whisper-cpp")
_OPENAI = shutil.which("whisper")


def backend() -> str | None:
    if _CPP:
        return "whisper.cpp"
    if _OPENAI:
        return "openai-whisper"
    return None


def transcribe(wav_path: str, settings: Settings) -> str:
    if _CPP:
        subprocess.run(
            [_CPP, "-m", settings.stt_model, "-f", wav_path, "-nt", "-otxt", "-of", wav_path],
            check=False, capture_output=True,
        )
        return _read(wav_path + ".txt")
    if _OPENAI:
        out_dir = os.path.dirname(wav_path) or "."
        base = os.path.splitext(os.path.basename(wav_path))[0]
        subprocess.run(
            [_OPENAI, wav_path, "--model", settings.whisper_name, "--language", "en",
             "--output_format", "txt", "--output_dir", out_dir, "--fp16", "False", "--verbose", "False"],
            check=False, capture_output=True,
        )
        return _read(os.path.join(out_dir, base + ".txt"))
    raise RuntimeError("no whisper backend found (install whisper.cpp or openai-whisper)")


def _read(path: str) -> str:
    if os.path.exists(path):
        with open(path) as f:
            text = f.read().strip()
        os.unlink(path)
        return text
    return ""
