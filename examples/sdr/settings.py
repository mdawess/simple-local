import os
from dataclasses import dataclass, field
from pathlib import Path

import yaml

DEFAULT_NUMBER = "+12899716341"


@dataclass
class Settings:
    llm_base_url: str = "http://localhost:8081/environments/development/sync/v1"
    llm_api_key: str = field(
        default_factory=lambda: os.environ.get("SIMPLE_LOCAL_API_KEY", "local")
    )
    model: str = "Qwen-2.5-3B"

    company: str = "Acme Analytics"
    agent_name: str = "Alex"

    voice: bool = False  # True = speak/listen via local piper + whisper.cpp
    tts_model: str = "en_US-lessac-medium.onnx"
    stt_model: str = "ggml-base.en.bin"

    # Compliance — do not weaken these for real calls.
    require_disclosure: bool = True
    allow_live_calls: bool = False
    calling_hours: str = "09:00-20:00"
    consented_numbers: list = field(default_factory=lambda: [DEFAULT_NUMBER])
    dnc: list = field(default_factory=list)

    telephony_provider: str = "simulated"  # simulated | twilio
    max_turns: int = 12

    @classmethod
    def load(cls, path: str) -> "Settings":
        data = yaml.safe_load(os.path.expandvars(Path(path).read_text())) or {}
        llm = data.get("llm", {})
        ident = data.get("identity", {})
        voice = data.get("voice", {})
        comp = data.get("compliance", {})
        tel = data.get("telephony", {})
        s = cls()
        return cls(
            llm_base_url=llm.get("base_url", s.llm_base_url),
            llm_api_key=llm.get("api_key") or s.llm_api_key,
            model=llm.get("model", s.model),
            company=ident.get("company", s.company),
            agent_name=ident.get("agent_name", s.agent_name),
            voice=voice.get("enabled", s.voice),
            tts_model=voice.get("tts_model", s.tts_model),
            stt_model=voice.get("stt_model", s.stt_model),
            require_disclosure=comp.get("require_disclosure", s.require_disclosure),
            allow_live_calls=comp.get("allow_live_calls", s.allow_live_calls),
            calling_hours=comp.get("calling_hours", s.calling_hours),
            consented_numbers=comp.get("consented_numbers", s.consented_numbers),
            dnc=comp.get("dnc", s.dnc),
            telephony_provider=tel.get("provider", s.telephony_provider),
            max_turns=data.get("max_turns", s.max_turns),
        )
