import argparse
import os
import sys
from pathlib import Path

from call import call_prospect
from settings import DEFAULT_NUMBER, Settings

SCRIPT = """Objective: book a 20-minute demo of Acme Analytics, a tool that turns a
company's raw product data into plain-English weekly insights.
Keep it brief and friendly. If they're interested, offer Tuesday or Thursday.
If they're busy, offer to send a one-pager and call back next week.
"""


def _load_root_env() -> None:
    env = Path(__file__).resolve().parents[2] / ".env"
    if not env.exists():
        return
    for line in env.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            key, value = line.split("=", 1)
            os.environ.setdefault(key.strip(), value.strip())


def main() -> None:
    parser = argparse.ArgumentParser(description="AI SDR — places a real call by default")
    parser.add_argument("--dry-run", action="store_true", help="simulate via text instead of a real call")
    parser.add_argument("--to", default=DEFAULT_NUMBER, help="destination number, E.164 (default: your cell)")
    parser.add_argument("-c", "--config", default="config.yml", help="path to config file")
    args = parser.parse_args()

    _load_root_env()
    settings = Settings.load(args.config) if os.path.exists(args.config) else Settings()

    if args.dry_run:
        settings.telephony_provider = "simulated"
    else:
        settings.telephony_provider = "twilio"
        settings.allow_live_calls = True  # you are calling your own consented number
        if not settings.twilio_from or not settings.twilio_stream_url:
            sys.exit("Live call needs telephony.twilio.from_number and stream_url in config.yml (see README).")

    log = call_prospect(args.to, SCRIPT, settings)
    print("\n=== CALL LOG ===")
    for key, value in log.items():
        print(f"{key}: {value}" if key != "transcript" else f"\ntranscript:\n{value}\n")


if __name__ == "__main__":
    main()
