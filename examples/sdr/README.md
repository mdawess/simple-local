# AI SDR example

A local-model "sales development rep" that runs a phone conversation from a
script and returns a call log:

```python
call_prospect(phone_number: str, script: str) -> dict[str, str]
```

The brains (LLM) and the voice (TTS/STT) run locally. **Placing a real call is
the only non-local part** — that needs a telephony provider (Twilio).
By default nothing is dialed: it runs in **simulation mode** so you can judge how
good the local model is.

## Read before going live

Automated calls with an artificial/AI voice are heavily regulated. In the US the
**TCPA** and the **FCC's Feb-2024 ruling** treat AI-voice robocalls as requiring
**prior express consent**; there are also Do-Not-Call rules, state laws, and
calling-hour limits. You are responsible for compliance.

This example is built safe-by-default:

- A spoken **AI disclosure** opens every call (`require_disclosure: true`).
- Live dialing is **off** (`allow_live_calls: false`) and gated behind a
  **consent allowlist** — pre-set to your own cell (`+12899716341`) only.
- The real providers are **stubs** (`telephony.py`) — they raise rather than dial.

Keep it in simulation unless you have genuine consent, and only ever test against
a number you own.

## Setup

Start the `simple-local` LLM server first (from the repo root):

```bash
export SIMPLE_LOCAL_API_KEY=my-secret-key
uv run simple-local serve -c config.yml     # serving on :8081 in your setup
```

The SDR reads `SIMPLE_LOCAL_API_KEY` from the environment and points at
`http://localhost:8081/...` (see `settings.py` / `config.example.yml`).

## Run

`run.py` places a **real call by default**; pass `--dry-run` to simulate via text.

```bash
cd examples/sdr
export SIMPLE_LOCAL_API_KEY=my-secret-key

# simulation — you play the prospect by typing (no carrier needed)
uv run --with openai --with pyyaml python run.py --dry-run

# real call (needs the Twilio setup below); --to overrides the destination
uv run --with openai --with pyyaml --with twilio --with websockets python run.py
```

In `--dry-run` you type the prospect's replies; press Enter on an empty line to
hang up. Either way you get the call log at the end (status, duration, full
transcript, outcome, LLM summary).

Use it directly too:

```python
from call import call_prospect
log = call_prospect("+12899716341", "Objective: book a demo of ...")
print(log["outcome"], "-", log["summary"])
```

## Voice mode (local mic + speakers)

Set `voice.enabled: true` (or `Settings(voice=True)`) to hear/speak instead of
type. Requires local binaries:

- **TTS** — [piper](https://github.com/rhasspy/piper) on `PATH` + a voice model (`tts_model`).
- **STT** — whisper.cpp `whisper-cli` on `PATH` + a model (`stt_model`), and `ffmpeg` for mic capture.

Missing binaries fall back to text automatically, so the demo never hard-fails.

## Going live (Twilio)

`twilio_transport.py` implements real outbound calling over **Twilio Media
Streams**, with fully local voice: macOS `say` for TTS (μ-law → Twilio) and
whisper for STT (Twilio μ-law → text). It's half-duplex — the agent speaks a
turn, then listens until you go quiet.

**Requirements**

- Twilio account + a purchased number; `TWILIO_ACCOUNT_SID` / `TWILIO_AUTH_TOKEN`
  in the root `.env` (loaded automatically by `run.py`). On a trial account, the
  destination must be a *verified* number (your cell).
- A whisper backend on `PATH`: `openai-whisper` (`whisper`, ~1.8s/turn) or
  whisper.cpp (`whisper-cli`, ~0.5s) — `stt.py` auto-detects.
- `ngrok` to expose the media WebSocket publicly.

**Run**

```bash
# 1. LLM server (terminal 1)
export SIMPLE_LOCAL_API_KEY=my-secret-key
uv run simple-local serve -c config.yml            # port 8081

# 2. Tunnel the media WebSocket (terminal 2) — ws_port defaults to 8090
ngrok http 8090
#   copy the https host, e.g. https://ab12.ngrok.app

# 3. Configure config.yml
cp config.example.yml config.yml
#   telephony.twilio.from_number: your Twilio number
#   telephony.twilio.stream_url:  wss://ab12.ngrok.app/media

# 4. Place the call to your cell (terminal 3) — live is the default
cd examples/sdr
uv run --with openai --with pyyaml --with twilio --with websockets python run.py
```

Your phone rings; the agent opens with the AI disclosure and works the script.
The call log prints at the end. Tune `SPEECH_RMS` / `ENDPOINT_MS` in
`twilio_transport.py` if turn-taking feels too eager or too slow.

> Uses Python's `audioop` for μ-law/resampling — present in 3.12 (this repo is
> pinned there), removed in 3.13.

## Verify numbers (no calls)

`verify_numbers.py` checks phone numbers with **Twilio Lookup** — a data query
that validates a number and normalizes it to E.164. It does **not** place a call
or contact anyone; use it for list hygiene, not outreach. Reads the `C1/C2/C3`
contact layout and adds a missing `+` (`15626537308` → `+15626537308`).

```bash
# basic validity (free); writes inputs/testing.verified.csv
uv run --with twilio python verify_numbers.py inputs/testing.csv

uv run --with twilio python verify_numbers.py inputs/testing.csv --limit 20   # cap for big lists
uv run --with twilio python verify_numbers.py inputs/testing.csv --carrier    # + line type/carrier (billed add-on)
```

Output columns: `name, email, country, phone_raw, e164, valid, line_type,
carrier, tag`. The **`tag`** is a plain-English read of the line —
`mobile` / `voip` / `google-voice` / `landline` / `toll-free` / `unknown` — so
you can prioritize real mobiles and drop VoIP/company lines. It's derived from
`line_type` + `carrier`, so it only fills in with `--carrier`.

Basic lookup is free; `--carrier` uses Twilio's billed Line Type Intelligence,
so `--limit`, `--prefix +1`, and `--offset` (batching) are handy on large lists.
Inputs and outputs under `inputs/` are gitignored (they hold real contact PII).

## Files

| File | Role |
|---|---|
| `call.py` | `call_prospect()` — orchestrates gate → disclosure → conversation → log |
| `agent.py` | LLM policy (script-following replies) + call summarizer |
| `telephony.py` | `Transport` interface + `SimulatedTransport` |
| `twilio_transport.py` | real Twilio Media Streams call (say TTS + whisper STT) |
| `voice.py` | local piper/whisper for the simulation's voice mode |
| `compliance.py` | disclosure line + consent/DNC/hours gate |
| `settings.py` | config with defaults; `Settings.load()` for YAML |
| `stt.py` | whisper backend (openai-whisper or whisper.cpp) |
| `run.py` | entry point — real call by default, `--dry-run` to simulate |
| `verify_numbers.py` | validate numbers via Twilio Lookup (no calls) |
| `transcripts/` | call logs saved here as JSON (gitignored — contain PII) |
| `inputs/` | contact CSVs + verification outputs (gitignored — PII) |
