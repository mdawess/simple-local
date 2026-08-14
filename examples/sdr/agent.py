from openai import OpenAI

SYSTEM_PROMPT = """You are {agent_name}, an outbound sales development representative for {company}.
You are on a LIVE phone call and have already disclosed that you are an AI.

Speak the way a person speaks on the phone:
- One or two short sentences per turn. No lists, no markdown, no stage directions.
- Respond to what the prospect actually says. Handle objections briefly and honestly.
- Never pretend to be human. Never invent facts about the company or product.
- When the objective is met, the prospect clearly declines, or it is time to wrap up,
  give a brief polite closing and append the token [END] to that final message.

Your campaign script and objective:
{script}
"""


class Agent:
    def __init__(self, client: OpenAI, model: str, company: str, agent_name: str, script: str):
        self.client = client
        self.model = model
        self.messages = [
            {"role": "system", "content": SYSTEM_PROMPT.format(agent_name=agent_name, company=company, script=script)}
        ]

    def opener(self) -> str:
        self.messages.append({
            "role": "user",
            "content": "[The prospect just answered and said hello. Give your one-sentence opening.]",
        })
        return self._complete(max_tokens=80)

    def reply(self, prospect_text: str) -> str:
        self.messages.append({"role": "user", "content": prospect_text})
        return self._complete(max_tokens=120)

    def _complete(self, max_tokens: int) -> str:
        response = self.client.chat.completions.create(
            model=self.model,
            messages=self.messages,
            max_tokens=max_tokens,
            temperature=0.7,
        )
        text = (response.choices[0].message.content or "").strip()
        self.messages.append({"role": "assistant", "content": text})
        return text


def summarize(client: OpenAI, model: str, transcript: str) -> tuple[str, str]:
    prompt = (
        "Below is a transcript of a phone call between an AI SDR and a prospect.\n\n"
        f"{transcript}\n\n"
        "Reply with exactly two lines and nothing else:\n"
        "OUTCOME: one of [interested, not_interested, callback, voicemail, no_answer]\n"
        "SUMMARY: a one or two sentence summary including any agreed next step."
    )
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=160,
        temperature=0.2,
    )
    text = response.choices[0].message.content or ""
    outcome, summary = "unknown", text.strip()
    for line in text.splitlines():
        upper = line.upper()
        if upper.startswith("OUTCOME:"):
            outcome = line.split(":", 1)[1].strip()
        elif upper.startswith("SUMMARY:"):
            summary = line.split(":", 1)[1].strip()
    return outcome, summary


# Post-call enrichment: pull structured contact details and an exec summary out
# of the transcript. Labelled lines parse more reliably from a small model than
# free-form JSON.
_ENRICH_LABELS = {
    "NAME": "name",
    "COMPANY": "company",
    "EMAIL": "email",
    "ROLE": "role",
    "INTEREST": "interest",
    "NEXT_STEP": "next_step",
    "EXEC_SUMMARY": "exec_summary",
}


def enrich(client: OpenAI, model: str, transcript: str) -> dict[str, str]:
    prompt = (
        "Extract details from this sales call transcript. Use 'unknown' for any "
        "field the transcript does not mention — do not guess.\n\n"
        f"{transcript}\n\n"
        "Reply with exactly these lines and nothing else:\n"
        "NAME: <prospect's name>\n"
        "COMPANY: <their company>\n"
        "EMAIL: <their email>\n"
        "ROLE: <their job title>\n"
        "INTEREST: <high | medium | low | none>\n"
        "NEXT_STEP: <agreed next step, if any>\n"
        "EXEC_SUMMARY: <2-3 sentence executive summary of the call>"
    )
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=300,
        temperature=0.2,
    )
    text = response.choices[0].message.content or ""
    fields = {key: "unknown" for key in _ENRICH_LABELS.values()}
    for line in text.splitlines():
        for label, key in _ENRICH_LABELS.items():
            if line.upper().startswith(label + ":"):
                fields[key] = line.split(":", 1)[1].strip()
    return fields
