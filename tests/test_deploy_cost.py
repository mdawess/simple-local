import pytest

from simple_local.deploy import pricing
from simple_local.deploy.profiles import Catalog, Profile
from simple_local.deploy.pricing import Prices, estimate

# Real meter names and rates from the Azure Retail Prices API for canadacentral.
METERS = {
    "Standard vCPU Active Usage": 3.4e-05,
    "Standard vCPU Idle Usage": 4e-06,
    "Standard Memory Active Usage": 4e-06,
    "Standard Memory Idle Usage": 4e-06,
    "Standard Requests": 0.4,
    "Standard NC T4 v3 GPU Usage": 8.8e-05,
    "Standard NC A100 v4 GPU Usage": 0.000635,
    "Dedicated vCPU Usage": 0.080859,
    "Dedicated Memory Usage": 0.006638,
    "Dedicated GPU Usage": 5.28912,
    "Dedicated Plan Management": 0.1,
}

PRICES = Prices(location="test", currency="USD", meters=METERS, source="azure")

CONSUMPTION = Profile(name="Consumption", cores=4, memory_gi=8)
T4 = Profile(
    name="Consumption-GPU-NC8as-T4", category="Consumption-GPU-T4", cores=8, memory_gi=56, gpus=1
)
NC96 = Profile(name="NC96-A100", category="GPU-NC-A100", cores=96, memory_gi=880, gpus=4)


def test_scale_to_zero_costs_nothing_at_rest():
    e = estimate(4, 8, min_replicas=0, max_replicas=3, profile=CONSUMPTION, prices=PRICES)
    assert e.floor == 0
    assert e.ceiling > 0
    assert any("costs nothing" in a for a in e.assumptions)


def test_consumption_bills_the_replica_request_not_the_node():
    # 4 vCPU + 8GiB active: (3.4e-5*4 + 4e-6*8) * 730*3600
    e = estimate(4, 8, min_replicas=1, max_replicas=1, profile=CONSUMPTION, prices=PRICES)
    assert e.ceiling == pytest.approx(441.50, abs=1)
    assert e.floor == pytest.approx(126.14, abs=1)


def test_a_gpu_replica_adds_the_gpu_meter():
    without = estimate(4, 16, 1, 1, CONSUMPTION, PRICES).ceiling
    with_gpu = estimate(4, 16, 1, 1, T4, PRICES).ceiling
    assert with_gpu - without == pytest.approx(8.8e-05 * 730 * 3600, abs=1)


def test_replicas_scale_the_ceiling():
    one = estimate(4, 8, 1, 1, CONSUMPTION, PRICES).ceiling
    four = estimate(4, 8, 1, 4, CONSUMPTION, PRICES).ceiling
    assert four == pytest.approx(one * 4, abs=1)


def test_a_dedicated_node_bills_whole_regardless_of_the_request():
    # A 4-GPU A100 node: 96 vCPU + 880GiB + 4 GPUs, hourly, plus plan management.
    small_request = estimate(4, 8, 1, 1, NC96, PRICES)
    full_request = estimate(96, 880, 1, 1, NC96, PRICES)
    assert small_request.ceiling == full_request.ceiling
    assert small_request.ceiling == pytest.approx(25_449, rel=0.01)
    assert any("bill the whole" in a for a in small_request.assumptions)


def test_dedicated_is_orders_of_magnitude_above_consumption():
    assert estimate(4, 8, 1, 1, NC96, PRICES).ceiling > 20 * estimate(
        4, 8, 1, 1, T4, PRICES
    ).ceiling


def test_no_prices_is_reported_not_guessed():
    e = estimate(4, 8, 1, 1, CONSUMPTION, Prices(location="test", source="none"))
    assert e.available is False
    assert e.summary() == "cost unknown"


def test_prices_round_trip_through_the_cache(tmp_path, monkeypatch):
    monkeypatch.setenv("SIMPLE_LOCAL_CACHE", str(tmp_path))

    class FakeResponse:
        def raise_for_status(self):
            pass

        def json(self):
            return {
                "BillingCurrency": "USD",
                "Items": [
                    {"meterName": "Standard vCPU Active Usage", "unitPrice": 3.4e-05},
                    {"meterName": "Standard Requests", "unitPrice": 0.4},
                ],
            }

    monkeypatch.setattr(pricing.httpx, "get", lambda *a, **k: FakeResponse())
    live = pricing.load("canadacentral", live=True)
    assert live.source == "azure"
    assert live.get("Standard Requests") == 0.4

    offline = pricing.load("canadacentral", live=False)
    assert offline.source == "cache"
    assert offline.meters == live.meters


def test_estimate_for_prices_what_the_config_asks_for():
    catalog = Catalog(location="test", source="azure", profiles={T4.name: T4})

    class FakeDeploy:
        gpu, cpu, memory_gi, min_replicas, max_replicas = "NC8as-T4", 4.0, 16.0, 1, 2

    e = pricing.estimate_for(FakeDeploy(), catalog, PRICES)
    assert e.available and e.ceiling > e.floor > 0
