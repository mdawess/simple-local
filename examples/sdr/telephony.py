from abc import ABC, abstractmethod

import voice
from settings import Settings


class Transport(ABC):
    @abstractmethod
    def start(self, phone_number: str) -> None: ...

    @abstractmethod
    def say(self, text: str) -> None: ...

    @abstractmethod
    def listen(self) -> str: ...

    @abstractmethod
    def hangup(self) -> None: ...


class SimulatedTransport(Transport):
    """No real call. You play the prospect via text, or via mic/speakers when
    settings.voice is on. This is where you judge the local model's quality."""

    def __init__(self, settings: Settings):
        self.settings = settings

    def start(self, phone_number: str) -> None:
        print(f"\n[sim] dialing {phone_number} ... connected")
        print("[sim] (press Enter on an empty line to hang up)\n")

    def say(self, text: str) -> None:
        print(f"AGENT>    {text}")
        if self.settings.voice:
            voice.speak(text, self.settings)

    def listen(self) -> str:
        if self.settings.voice:
            return voice.listen(self.settings)
        return input("PROSPECT> ").strip()

    def hangup(self) -> None:
        print("\n[sim] call ended\n")


def build_transport(settings: Settings) -> Transport:
    if settings.telephony_provider == "twilio":
        # Imported lazily so the simulation demo needs no twilio/websockets deps.
        from twilio_transport import TwilioTransport

        return TwilioTransport(settings)
    return SimulatedTransport(settings)
