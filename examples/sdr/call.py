import datetime
import json
from pathlib import Path

from openai import OpenAI

from agent import Agent, enrich, summarize
from compliance import ComplianceError, ConsentGate, disclosure_line
from settings import DEFAULT_NUMBER, Settings
from telephony import build_transport

_HANGUP_WORDS = {"", "bye", "goodbye", "hang up", "hangup", "quit"}
TRANSCRIPTS_DIR = Path(__file__).resolve().parent / "transcripts"


def call_prospect(phone_number: str = DEFAULT_NUMBER, script: str = "", settings: Settings | None = None) -> dict[str, str]:
    settings = settings or Settings()
    started = datetime.datetime.now()
    base = {
        "phone_number": phone_number,
        "provider": settings.telephony_provider,
        "started_at": started.isoformat(timespec="seconds"),
    }

    try:
        ConsentGate(settings).check(phone_number)
    except ComplianceError as error:
        return _finish(base, started, status="blocked", transcript="", outcome="blocked", summary=str(error))

    client = OpenAI(base_url=settings.llm_base_url, api_key=settings.llm_api_key)
    agent = Agent(client, settings.model, settings.company, settings.agent_name, script)
    transport = build_transport(settings)

    turns: list[str] = []

    def record(speaker: str, text: str) -> None:
        turns.append(f"{speaker}: {text}")

    transport.start(phone_number)
    if settings.require_disclosure:
        line = disclosure_line(settings.company, settings.agent_name)
        transport.say(line)
        record("AGENT", line)
    else:
        record("SYSTEM", "WARNING: AI disclosure was disabled — this is unlawful for real calls.")

    opener = agent.opener()
    transport.say(opener)
    record("AGENT", opener)

    status = "completed"
    for _ in range(settings.max_turns):
        prospect = transport.listen()
        if prospect.lower() in _HANGUP_WORDS:
            record("PROSPECT", prospect or "[hung up]")
            status = "completed" if prospect else "hung_up"
            break
        record("PROSPECT", prospect)

        reply = agent.reply(prospect)
        ending = "[END]" in reply
        reply = reply.replace("[END]", "").strip()
        transport.say(reply)
        record("AGENT", reply)
        if ending:
            break

    transport.hangup()

    transcript = "\n".join(turns)
    outcome, summary = summarize(client, settings.model, transcript)
    details = enrich(client, settings.model, transcript)
    return _finish(base, started, status=status, transcript=transcript, outcome=outcome, summary=summary, details=details)


def _finish(
    base: dict,
    started: datetime.datetime,
    *,
    status: str,
    transcript: str,
    outcome: str,
    summary: str,
    details: dict[str, str] | None = None,
) -> dict[str, str]:
    ended = datetime.datetime.now()
    log = {
        **base,
        "status": status,
        "ended_at": ended.isoformat(timespec="seconds"),
        "duration_s": str(int((ended - started).total_seconds())),
        "transcript": transcript,
        "outcome": outcome,
        "summary": summary,
        **(details or {}),
    }
    _save_log(log)
    return log


def _save_log(log: dict[str, str]) -> None:
    TRANSCRIPTS_DIR.mkdir(exist_ok=True)
    stamp = log["started_at"].replace(":", "-")
    number = log["phone_number"].replace("+", "")
    path = TRANSCRIPTS_DIR / f"{stamp}_{number}.json"
    path.write_text(json.dumps(log, indent=2))
    print(f"[log] saved {path}")
