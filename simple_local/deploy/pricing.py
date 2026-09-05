"""Monthly cost estimates from Azure's public retail price list.

Prices come from the Retail Prices API — no auth, no hardcoded rates, same
reasoning as the workload profile catalog. Everything is list price in the
billing currency the API returns: no reservations, savings plans, AHB or
enterprise discount is applied, so treat these as an upper bound and as a way to
compare two options rather than a forecast of the invoice.
"""

from typing import Literal

import httpx
from pydantic import BaseModel, Field

from . import cache
from .config import (
    CONSUMPTION_GPU_METERS,
    DEDICATED_GPU,
    DEDICATED_MEMORY,
    DEDICATED_PLAN,
    DEDICATED_VCPU,
    HOURS_PER_MONTH,
    MEMORY_ACTIVE,
    MEMORY_IDLE,
    PRICES_ENDPOINT,
    PRICES_SERVICE,
    PRICES_TIMEOUT_SECONDS,
    REQUESTS,
    VCPU_ACTIVE,
    VCPU_IDLE,
)
from .profiles import Profile


class Prices(BaseModel):
    location: str
    currency: str = "USD"
    meters: dict[str, float] = Field(default_factory=dict)
    source: Literal["azure", "cache", "none"] = "none"
    fetched_at: float | None = None

    @property
    def provenance(self) -> str:
        if self.source == "azure":
            return f"list price, live from Azure ({self.location})"
        if self.source == "cache":
            age = cache.age_days(self.fetched_at)
            return f"list price, cached {age:.0f}d ago ({self.location})"
        return "prices unavailable"

    def get(self, meter: str) -> float | None:
        return self.meters.get(meter)


class CostLine(BaseModel):
    label: str
    monthly: float


class Estimate(BaseModel):
    currency: str = "USD"
    floor: float = 0.0  # min_replicas, at rest
    ceiling: float = 0.0  # max_replicas, fully active
    lines: list[CostLine] = Field(default_factory=list)
    assumptions: list[str] = Field(default_factory=list)
    provenance: str = ""
    available: bool = True

    def summary(self) -> str:
        if not self.available:
            return "cost unknown"
        if abs(self.ceiling - self.floor) < 0.01:
            return f"~{self.currency} {self.floor:,.0f}/month"
        return f"{self.currency} {self.floor:,.0f}–{self.ceiling:,.0f}/month"


def fetch(location: str) -> Prices:
    query = f"serviceName eq '{PRICES_SERVICE}' and armRegionName eq '{location}'"
    response = httpx.get(
        PRICES_ENDPOINT, params={"$filter": query}, timeout=PRICES_TIMEOUT_SECONDS
    )
    response.raise_for_status()
    body = response.json()
    meters = {item["meterName"]: float(item["unitPrice"]) for item in body.get("Items", [])}
    currency = body.get("BillingCurrency", "USD")
    cache.write("prices", location, {"location": location, "currency": currency, "meters": meters})
    return Prices(location=location, currency=currency, meters=meters, source="azure")


def load(location: str, live: bool = True) -> Prices:
    if live:
        try:
            return fetch(location)
        except (httpx.HTTPError, ValueError, KeyError):
            pass
    cached = cache.read("prices", location)
    if cached:
        data, fetched_at = cached
        return Prices(
            location=location,
            currency=data.get("currency", "USD"),
            meters=data.get("meters", {}),
            source="cache",
            fetched_at=fetched_at,
        )
    return Prices(location=location, source="none")


def _consumption_gpu_meter(profile: Profile) -> str | None:
    for fragment, meter in CONSUMPTION_GPU_METERS.items():
        if fragment in profile.category or fragment in profile.name:
            return meter
    return None


def _is_dedicated(profile: Profile) -> bool:
    return not profile.category.startswith("Consumption")


def estimate(
    cpu: float,
    memory_gi: float,
    min_replicas: int,
    max_replicas: int,
    profile: Profile,
    prices: Prices,
) -> Estimate:
    """Floor is what min_replicas costs sitting there; ceiling is max_replicas
    fully active for the whole month. Real usage lands between, and for a
    scale-to-zero app with no traffic the floor is genuinely zero."""
    if not prices.meters:
        return Estimate(available=False, provenance=prices.provenance)

    lines: list[CostLine] = []
    assumptions = [f"{HOURS_PER_MONTH} hours/month, list price, no reservations or discounts"]

    if _is_dedicated(profile):
        # The node bills whole, regardless of what the replica requests.
        vcpu = (prices.get(DEDICATED_VCPU) or 0) * profile.cores
        memory = (prices.get(DEDICATED_MEMORY) or 0) * profile.memory_gi
        gpu = (prices.get(DEDICATED_GPU) or 0) * profile.gpus
        per_hour = vcpu + memory + gpu
        plan = (prices.get(DEDICATED_PLAN) or 0) * HOURS_PER_MONTH

        lines.append(CostLine(label=f"{profile.name} node", monthly=per_hour * HOURS_PER_MONTH))
        lines.append(CostLine(label="dedicated plan management", monthly=plan))
        assumptions.append(
            f"dedicated profiles bill the whole {profile.cores:g}-core node, "
            "not the replica's cpu/memory request"
        )
        floor = per_hour * HOURS_PER_MONTH * max(min_replicas, 0) + plan
        ceiling = per_hour * HOURS_PER_MONTH * max(max_replicas, 1) + plan
        return Estimate(
            currency=prices.currency,
            floor=floor,
            ceiling=ceiling,
            lines=lines,
            assumptions=assumptions,
            provenance=prices.provenance,
        )

    seconds = HOURS_PER_MONTH * 3600
    active = ((prices.get(VCPU_ACTIVE) or 0) * cpu + (prices.get(MEMORY_ACTIVE) or 0) * memory_gi)
    idle = ((prices.get(VCPU_IDLE) or 0) * cpu + (prices.get(MEMORY_IDLE) or 0) * memory_gi)

    gpu_meter = _consumption_gpu_meter(profile) if profile.is_gpu else None
    gpu_rate = (prices.get(gpu_meter) or 0) * profile.gpus if gpu_meter else 0.0
    if profile.is_gpu and gpu_meter is None:
        assumptions.append(f"no GPU meter matched {profile.name}; GPU cost not included")

    lines.append(CostLine(label="vCPU + memory, active", monthly=active * seconds))
    lines.append(CostLine(label="vCPU + memory, idle", monthly=idle * seconds))
    if gpu_rate:
        lines.append(CostLine(label=f"{profile.gpus}x GPU", monthly=gpu_rate * seconds))

    request_rate = prices.get(REQUESTS)
    if request_rate:
        assumptions.append(f"requests billed separately at {prices.currency} {request_rate}/1M")
    if min_replicas == 0:
        assumptions.append("min_replicas is 0, so an idle app with no traffic costs nothing")

    per_replica_idle = (idle + gpu_rate) * seconds
    per_replica_active = (active + gpu_rate) * seconds
    return Estimate(
        currency=prices.currency,
        floor=per_replica_idle * max(min_replicas, 0),
        ceiling=per_replica_active * max(max_replicas, 1),
        lines=lines,
        assumptions=assumptions,
        provenance=prices.provenance,
    )


def estimate_for(deploy, catalog, prices: Prices) -> Estimate:
    profile, _ = catalog.resolve(deploy.gpu)
    if profile is None:
        return Estimate(available=False, provenance="no workload profile to price")
    return estimate(
        deploy.cpu, deploy.memory_gi, deploy.min_replicas, deploy.max_replicas, profile, prices
    )
