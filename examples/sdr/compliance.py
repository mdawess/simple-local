import datetime

from settings import Settings


class ComplianceError(Exception):
    pass


def disclosure_line(company: str, agent_name: str) -> str:
    return (
        f"Hi, this is {agent_name}, an AI virtual assistant calling on behalf of {company}. "
        "I'm using an artificial voice, and this call may be recorded. "
        "Do you have a quick moment?"
    )


def _within_hours(window: str) -> bool:
    start_s, end_s = window.split("-")
    now = datetime.datetime.now().time()
    start = datetime.time.fromisoformat(start_s.strip())
    end = datetime.time.fromisoformat(end_s.strip())
    return start <= now <= end


class ConsentGate:
    """Blocks a live call unless every legal precondition is satisfied.

    Simulation never places a real call, so it only warns; live calls (twilio)
    must clear consent, do-not-call, allowlist, and calling-hours checks.
    """

    def __init__(self, settings: Settings):
        self.settings = settings

    def check(self, phone_number: str) -> None:
        s = self.settings
        if s.telephony_provider == "simulated":
            return
        if not s.allow_live_calls:
            raise ComplianceError(
                "live calls are disabled — set compliance.allow_live_calls=true only "
                "for numbers you have consent to call"
            )
        if phone_number in s.dnc:
            raise ComplianceError(f"{phone_number} is on the do-not-call list")
        if s.consented_numbers and phone_number not in s.consented_numbers:
            raise ComplianceError(f"{phone_number} is not in the consented-contacts allowlist")
        if not _within_hours(s.calling_hours):
            raise ComplianceError(f"outside permitted calling hours ({s.calling_hours})")
