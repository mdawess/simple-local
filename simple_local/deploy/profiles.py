"""Container Apps workload profiles, read from Azure.

There is no table of profiles in this file on purpose. Azure knows what it
offers, it differs by region, and it changes — a snapshot compiled into the
package is a belief about Azure that drifts silently, and this one did within
minutes of being written. Fetched results are cached on disk so offline use is
still possible, and a catalog always says where its numbers came from, so a
stale answer is visible rather than assumed.
"""

from typing import Literal

from pydantic import BaseModel, Field

from . import azure, cache
from .azure import AzureError
from .config import CATALOG_STALE_AFTER_DAYS, DEFAULT_WORKLOAD_PROFILE


class Profile(BaseModel):
    name: str  # also the --workload-profile-type Azure expects
    category: str = "Consumption"
    cores: float
    memory_gi: float
    gpus: int = 0
    quota: int | None = None  # replicas per subscription; no API exposes this

    @property
    def is_gpu(self) -> bool:
        return self.gpus > 0


class Catalog(BaseModel):
    location: str
    profiles: dict[str, Profile] = Field(default_factory=dict)
    source: Literal["azure", "cache", "config", "none"] = "none"
    fetched_at: float | None = None

    @property
    def age_days(self) -> float | None:
        return cache.age_days(self.fetched_at)

    @property
    def stale(self) -> bool:
        age = self.age_days
        return self.source == "cache" and age is not None and age > CATALOG_STALE_AFTER_DAYS

    @property
    def provenance(self) -> str:
        if self.source == "azure":
            return f"live from Azure ({self.location})"
        if self.source == "cache":
            age = self.age_days
            return f"cached {age:.0f}d ago ({self.location})" if age is not None else "cached"
        if self.source == "config":
            return "deploy.profiles only"
        return "unavailable"

    def gpus(self) -> list[Profile]:
        return sorted(
            (p for p in self.profiles.values() if p.is_gpu), key=lambda p: (p.gpus, p.cores)
        )

    def sorted(self) -> list[Profile]:
        return sorted(self.profiles.values(), key=lambda p: (p.gpus, p.cores, p.name))

    def resolve(self, name: str | None) -> tuple[Profile | None, str | None]:
        """A config may use Azure's name (`Consumption-GPU-NC8as-T4`) or the
        short form (`NC8as-T4`)."""
        if name is None:
            return self.profiles.get(DEFAULT_WORKLOAD_PROFILE), DEFAULT_WORKLOAD_PROFILE
        if name in self.profiles:
            return self.profiles[name], name
        for candidate, profile in self.profiles.items():
            if candidate.endswith(f"-{name}") or candidate == f"Consumption-GPU-{name}":
                return profile, candidate
        return None, name


def _parse(rows) -> dict[str, Profile]:
    found = {}
    for row in rows:
        properties = row.get("properties") or {}
        name = row.get("name")
        if not name or properties.get("cores") is None:
            continue
        found[name] = Profile(
            name=name,
            category=properties.get("category") or "Consumption",
            cores=float(properties["cores"]),
            memory_gi=float(properties.get("memoryGiB") or 0),
            gpus=int(properties.get("gpus") or 0),
        )
    return found


def fetch(location: str) -> dict[str, Profile]:
    """What this region offers today, straight from az. Writes the cache."""
    rows = azure.json_out(
        ["containerapp", "env", "workload-profile", "list-supported", "-l", location]
    )
    profiles = _parse(rows)
    cache.write(
        "profiles",
        location,
        {"location": location, "profiles": {n: p.model_dump() for n, p in profiles.items()}},
    )
    return profiles


def read_cache(location: str) -> tuple[dict[str, Profile], float] | None:
    cached = cache.read("profiles", location)
    if cached is None:
        return None
    data, fetched_at = cached
    try:
        return {n: Profile.model_validate(p) for n, p in data["profiles"].items()}, fetched_at
    except (ValueError, KeyError, TypeError):
        return None


def load(
    location: str,
    overrides: dict[str, Profile] | None = None,
    quotas: dict[str, int] | None = None,
    live: bool = True,
) -> Catalog:
    """Azure first, then the cache, then whatever the config declares.

    A catalog that could not be loaded at all is reported as such rather than
    silently treated as empty — `check` turns that into a finding, so "sizing
    was not verified" never reads as "sizing is fine".
    """
    profiles: dict[str, Profile] = {}
    source: Literal["azure", "cache", "config", "none"] = "none"
    fetched_at: float | None = None

    if live:
        try:
            profiles = fetch(location)
            source, fetched_at = "azure", cache.read("profiles", location)[1]
        except AzureError:
            pass

    if not profiles:
        cached = read_cache(location)
        if cached:
            profiles, fetched_at = cached
            source = "cache"

    if overrides:
        profiles = {**profiles, **overrides}
        if source == "none":
            source = "config"

    # Sizes come from Azure; quota does not, so it is overlaid separately rather
    # than forcing a config to restate cores and memory to record one number.
    for name, quota in (quotas or {}).items():
        profile = profiles.get(name)
        if profile is None:
            match = Catalog(location=location, profiles=profiles).resolve(name)[1]
            profile = profiles.get(match) if match else None
        if profile is not None:
            profiles[profile.name] = profile.model_copy(update={"quota": quota})

    return Catalog(location=location, profiles=profiles, source=source, fetched_at=fetched_at)
