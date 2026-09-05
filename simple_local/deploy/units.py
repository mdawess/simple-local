import re

from .config import MEMORY_PATTERN

_MEMORY = re.compile(MEMORY_PATTERN, re.IGNORECASE)


def parse_memory_gi(value) -> float:
    """Accepts what a config writes ('16.0Gi'), what the console spec writes
    (16), and what Azure returns ('16Gi')."""
    if isinstance(value, (int, float)):
        return float(value)
    match = _MEMORY.match(str(value))
    if not match:
        raise ValueError(f"memory must look like '16Gi', got {value!r}")
    return float(match.group(1))
