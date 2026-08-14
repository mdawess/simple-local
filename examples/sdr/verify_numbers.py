import argparse
import csv
import os
import sys
from pathlib import Path

from twilio.rest import Client

# Verifies phone numbers with Twilio Lookup — a data/validation query that does
# NOT place a call or contact anyone. Basic lookup (validity + E.164) is free;
# --carrier adds line type + carrier via Twilio's billed add-on.


def _load_root_env() -> None:
    env = Path(__file__).resolve().parents[2] / ".env"
    if not env.exists():
        return
    for line in env.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            key, value = line.split("=", 1)
            os.environ.setdefault(key.strip(), value.strip())


def to_e164(raw: str) -> str | None:
    digits = "".join(ch for ch in raw if ch.isdigit())
    if not digits:
        return None
    if raw.strip().startswith("+"):
        return "+" + digits
    if digits.startswith("00"):  # international dialing prefix
        return "+" + digits[2:]
    return "+" + digits


def extract(csv_path: Path) -> list[dict]:
    contacts, seen = [], set()
    with open(csv_path, newline="", encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            for c in ("C1", "C2", "C3"):
                e164 = to_e164((row.get(f"{c} phone") or "").strip())
                if not e164 or e164 in seen:
                    continue
                seen.add(e164)
                contacts.append({
                    "name": (row.get(f"{c} name") or "").strip(),
                    "email": (row.get(f"{c} email") or "").strip(),
                    "country": (row.get(f"{c} country") or "").strip(),
                    "phone_raw": (row.get(f"{c} phone") or "").strip(),
                    "e164": e164,
                })
    return contacts


def classify(line_type: str, carrier: str) -> str:
    """A plain-English tag for what the line appears to be, from Twilio's line
    type + carrier. VoIP/toll-free/landline are less likely to reach a person's
    personal phone than a mobile."""
    carrier_l = (carrier or "").lower()
    if line_type == "mobile":
        return "mobile"
    if line_type == "landline":
        return "landline"
    if line_type == "tollFree":
        return "toll-free"
    if line_type in ("fixedVoip", "nonFixedVoip"):
        if "google" in carrier_l:
            return "google-voice"
        return "voip"
    return "unknown"


def lookup(client: Client, e164: str, with_carrier: bool) -> dict:
    try:
        numbers = client.lookups.v2.phone_numbers(e164)
        info = numbers.fetch(fields="line_type_intelligence") if with_carrier else numbers.fetch()
    except Exception as e:
        return {"valid": "error", "e164": e164, "line_type": "", "carrier": "", "tag": "error", "note": str(e)[:120]}
    lti = getattr(info, "line_type_intelligence", None)
    lti = lti if isinstance(lti, dict) else {}
    line_type = lti.get("type") or ""
    carrier = lti.get("carrier_name") or ""
    return {
        "valid": str(bool(info.valid)).lower(),
        "e164": info.phone_number or e164,
        "line_type": line_type,
        "carrier": carrier,
        "tag": classify(line_type, carrier),
        "note": "",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Verify phone numbers from a CSV via Twilio Lookup (no calls placed).")
    parser.add_argument("csv", help="input CSV with C1/C2/C3 contact columns")
    parser.add_argument("-o", "--out", help="output CSV (default: <input>.verified.csv)")
    parser.add_argument("--carrier", action="store_true", help="also fetch line type + carrier (billed Twilio add-on)")
    parser.add_argument("--prefix", default="", help="only numbers whose E.164 starts with this (e.g. +1)")
    parser.add_argument("--exclude-prefix", default="", help="skip numbers starting with these (comma-separated, e.g. +91,+86,+972)")
    parser.add_argument("--offset", type=int, default=0, help="skip the first N matching numbers (for batching)")
    parser.add_argument("--limit", type=int, default=0, help="only verify the first N numbers (0 = all)")
    args = parser.parse_args()

    _load_root_env()
    sid, token = os.environ.get("TWILIO_ACCOUNT_SID", ""), os.environ.get("TWILIO_AUTH_TOKEN", "")
    if not sid or not token:
        sys.exit("missing Twilio credentials in the root .env")
    client = Client(sid, token)

    src = Path(args.csv)
    contacts = extract(src)
    if not contacts:
        sys.exit(f"no phone numbers found in {src}")
    if args.prefix:
        contacts = [c for c in contacts if c["e164"].startswith(args.prefix)]
    if args.exclude_prefix:
        skip = tuple(p.strip() for p in args.exclude_prefix.split(",") if p.strip())
        contacts = [c for c in contacts if not c["e164"].startswith(skip)]
    if args.offset:
        contacts = contacts[args.offset:]
    if args.limit:
        contacts = contacts[:args.limit]

    out = Path(args.out) if args.out else src.with_suffix(".verified.csv")
    fields = ["name", "email", "country", "phone_raw", "e164", "valid", "line_type", "carrier", "tag", "note"]
    counts = {"true": 0, "false": 0, "error": 0}

    print(f"Looking up {len(contacts)} unique numbers (no calls placed)...")
    with open(out, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for c in contacts:
            result = lookup(client, c["e164"], args.carrier)
            writer.writerow({k: {**c, **result}.get(k, "") for k in fields})
            counts[result["valid"]] = counts.get(result["valid"], 0) + 1
            label = {"true": "valid", "false": "INVALID", "error": "error"}.get(result["valid"], result["valid"])
            print(f"  {result['e164']:<16} {label:<8} {result.get('tag', ''):<13} {result['carrier']:<20} {c['name']}")

    print(f"\n{len(contacts)} numbers -> {counts['true']} valid, {counts['false']} invalid, {counts['error']} errors")
    print(f"written to {out}")


if __name__ == "__main__":
    main()
