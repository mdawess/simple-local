import json
import random
from pathlib import Path

HERE = Path(__file__).resolve().parent
TRANSCRIPTS = HERE.parent / "sdr" / "transcripts"
OUT = HERE / "data"

SYSTEM = (
    "You are Alex, an outbound sales development rep for Acme Analytics, a tool "
    "that turns a company's raw product data into plain-English weekly insights. "
    "Speak in one or two short, natural sentences. Handle objections briefly and "
    "honestly, and aim to book a 20-minute demo."
)

# Seed examples so the dataset is large enough to actually train even before you
# have many real transcripts. These teach the house style for common objections.
SEED_PAIRS = [
    ("Who is this?",
     "Hi, it's Alex from Acme Analytics — we turn your product data into a plain-English weekly report. Do you have a quick minute?"),
    ("We already have a BI tool.",
     "Totally, most teams do. We sit on top of it and write the weekly summary for you, so nobody spends Friday building slides."),
    ("I'm really busy right now.",
     "No problem — can I send a one-pager and grab 20 minutes next week instead?"),
    ("How much does it cost?",
     "It scales with your data volume, but most teams land around a few hundred a month. Happy to get you exact numbers on a quick demo."),
    ("We don't have a data team.",
     "That's exactly who this helps — it writes the analysis so you don't need one. Want to see it on your own data?"),
    ("Send me an email.",
     "Will do — what's the best address? I'll include a one-pager and a couple of example reports."),
    ("What makes you different from a dashboard?",
     "Dashboards make you go look; we push you the two or three things that actually changed each week, in words."),
    ("Is it hard to set up?",
     "Under an hour — it connects to your existing warehouse, no migration needed."),
    ("We're not looking to buy anything.",
     "Fair enough — this is more about saving your team the weekly reporting slog. Worth a 20-minute look if that's a pain?"),
    ("Can it work with our data warehouse?",
     "Yes — Snowflake, BigQuery, Postgres, and Redshift all work out of the box."),
    ("I've never heard of Acme Analytics.",
     "Fair — we're focused on product and data teams who are tired of hand-writing the metrics review every week."),
    ("Just tell me what it does.",
     "It reads your product data and writes a short weekly note on what moved and why — the report a data analyst would."),
    ("How long is the demo?",
     "Twenty minutes, and I'll use a sample close to your business so it's concrete. Does Thursday afternoon work?"),
    ("We tried something like this before.",
     "Understood — what fell short? A lot of those tools stopped at charts; we actually write the takeaways."),
    ("Not interested, thanks.",
     "No worries — thanks for the minute, and I'll send a one-pager in case timing changes. Have a good one."),
    ("Do you store our data?",
     "We only read what's needed to write the summary, and it's encrypted — happy to walk your security team through it."),
]


def _from_seeds() -> list[dict]:
    return [
        {"messages": [
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": prospect},
            {"role": "assistant", "content": agent},
        ]}
        for prospect, agent in SEED_PAIRS
    ]


def _from_transcripts() -> list[dict]:
    examples = []
    for path in sorted(TRANSCRIPTS.glob("*.json")):
        transcript = json.loads(path.read_text()).get("transcript", "")
        messages = [{"role": "system", "content": SYSTEM}]
        for line in transcript.splitlines():
            if line.startswith("AGENT:"):
                role, text = "assistant", line[len("AGENT:"):].strip()
            elif line.startswith("PROSPECT:"):
                role, text = "user", line[len("PROSPECT:"):].strip()
            else:
                continue
            if not text or text.startswith("["):
                continue
            if len(messages) > 1 and messages[-1]["role"] == role:
                messages[-1]["content"] += " " + text
            else:
                messages.append({"role": role, "content": text})
        # Chat training wants a user turn first; outbound calls open with the agent.
        if len(messages) > 1 and messages[1]["role"] == "assistant":
            messages.insert(1, {"role": "user", "content": "(The prospect just answered the phone.)"})
        roles = {m["role"] for m in messages}
        if {"user", "assistant"} <= roles:
            examples.append({"messages": messages})
    return examples


def main() -> None:
    data = _from_seeds() + _from_transcripts()
    random.seed(0)
    random.shuffle(data)

    OUT.mkdir(exist_ok=True)
    n_valid = max(1, len(data) // 10)
    valid, train = data[:n_valid], data[n_valid:]
    (OUT / "train.jsonl").write_text("\n".join(json.dumps(x) for x in train) + "\n")
    (OUT / "valid.jsonl").write_text("\n".join(json.dumps(x) for x in valid) + "\n")
    print(f"wrote {len(train)} train / {len(valid)} valid examples to {OUT}")
    print(f"(seeds: {len(SEED_PAIRS)}, real transcripts used: {len(_from_transcripts())})")


if __name__ == "__main__":
    main()
