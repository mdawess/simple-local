"""A route is the contract; which revision and model sit behind it is not.

The app name is the public hostname, so these cover the two things that keeps
honest: names resolving to stable routes, and a new image landing as a candidate
rather than silently taking over the route.
"""

import pytest

from simple_local.deploy.config import CANDIDATE_LABEL, LIVE_LABEL, TIER_ORDER, TIER_SECONDS
from simple_local.deploy.spec import Deploy


def deploy(**overrides) -> Deploy:
    data = {
        "name": "image-embed",
        "directory": ".",
        "container_config": "config.yml",
        "domain": {"zone": "model.aeriumhq.com", "zone_resource_group": "aerium-dns"},
    }
    data.update(overrides)
    return Deploy.model_validate(data)


def test_the_route_comes_from_the_app_name():
    assert deploy().route == "image-embed.model.aeriumhq.com"


def test_a_hostname_can_differ_from_the_app_name():
    target = deploy(domain={"zone": "model.aeriumhq.com", "hostname": "vision"})
    assert target.route == "vision.model.aeriumhq.com"


def test_no_domain_means_no_route():
    assert deploy(domain=None).route is None


def test_label_routes_stay_a_single_dns_label():
    # A wildcard cert for *.model.aeriumhq.com only covers one label, so the
    # candidate URL has to keep the --- inside the leftmost label.
    candidate = deploy().label_route(CANDIDATE_LABEL)
    assert candidate == "image-embed---candidate.model.aeriumhq.com"
    assert candidate.count(".") == deploy().route.count(".")


def test_the_zone_resource_group_falls_back_to_the_app(tmp_path):
    assert deploy().dns_zone_group == "aerium-dns"
    assert deploy(domain={"zone": "model.aeriumhq.com"}).dns_zone_group == "simple-local-ca"


def test_multiple_revisions_is_the_default():
    assert deploy().revisions.multiple is True
    assert deploy().revisions.live_label == LIVE_LABEL
    assert deploy(revisions={"mode": "Single"}).revisions.multiple is False


def test_promote_is_the_cheapest_action_there_is():
    assert TIER_ORDER.index("promote") < TIER_ORDER.index("scale")
    assert TIER_SECONDS["promote"] < TIER_SECONDS["scale"]


class FakeOps:
    """Records what apply would do to labels without touching Azure."""

    def __init__(self, latest="image-embed--rev2", live=None):
        self._latest, self._live = latest, live
        self.calls = []

    def latest_revision(self, deploy):
        return self._latest

    def labelled(self, deploy, label):
        return self._live if label == LIVE_LABEL else None

    def set_label(self, deploy, label, revision, weight, emit):
        self.calls.append((label, revision, weight))


@pytest.fixture
def label_new_revision(monkeypatch):
    from simple_local.deploy import apply as apply_mod

    def run(target, ops):
        monkeypatch.setattr(apply_mod, "operations", ops)
        apply_mod._label_new_revision(target, lambda payload: None)
        return ops.calls

    return run


def test_a_first_deploy_takes_the_route_immediately(label_new_revision):
    calls = label_new_revision(deploy(), FakeOps(live=None))
    assert calls == [(LIVE_LABEL, "image-embed--rev2", 100)]


def test_a_later_deploy_lands_as_a_candidate_at_zero_traffic(label_new_revision):
    calls = label_new_revision(deploy(), FakeOps(live="image-embed--rev1"))
    assert calls == [(CANDIDATE_LABEL, "image-embed--rev2", 0)]


def test_redeploying_the_live_revision_changes_nothing(label_new_revision):
    calls = label_new_revision(deploy(), FakeOps(latest="r1", live="r1"))
    assert calls == []


def test_single_revision_mode_skips_labelling_entirely(label_new_revision):
    calls = label_new_revision(deploy(revisions={"mode": "Single"}), FakeOps(live="r1"))
    assert calls == []


def test_a_managed_certificate_needs_only_a_cname_and_a_txt():
    # These are per app and can live in any zone — no NS delegation, and only
    # record types every registrar offers. Hostinger has no NS support at all.
    from simple_local.deploy import operations

    records = operations.required_dns_records(
        deploy(), "VERIFY123", None, default_domain="redstone-1.canadacentral.azurecontainerapps.io"
    )
    # Names are relative to the zone actually being edited. Without delegation
    # the registrar holds aeriumhq.com, so the host is image-embed.model — not
    # image-embed, which would resolve at the wrong name and never work.
    assert [(r["type"], r["name"], r["zone"]) for r in records] == [
        ("CNAME", "image-embed.model", "aeriumhq.com"),
        ("TXT", "asuid.image-embed.model", "aeriumhq.com"),
    ]
    assert records[0]["fqdn"] == "image-embed.model.aeriumhq.com"
    assert records[0]["value"] == "image-embed.redstone-1.canadacentral.azurecontainerapps.io"
    assert records[1]["value"] == "VERIFY123"
    assert not any(r["type"] == "NS" for r in records)


def test_a_delegated_zone_shortens_the_record_name():
    from simple_local.deploy import operations

    target = deploy(domain={"zone": "model.aeriumhq.com", "registrar_zone": "model.aeriumhq.com"})
    records = operations.required_dns_records(target, "V", None, default_domain="d.io")
    assert records[0]["name"] == "image-embed"


def test_managed_is_the_default_because_azure_renews_it():
    assert deploy().domain.managed is True
    assert deploy(domain={"zone": "model.aeriumhq.com", "certificate": "wildcard"}).domain.managed is False


def test_a_wildcard_covers_the_whole_environment_with_one_pair():
    from simple_local.deploy import operations

    target = deploy(domain={"zone": "model.aeriumhq.com", "certificate": "wildcard"})
    records = operations.required_dns_records(target, "VERIFY123", "20.1.2.3")
    assert [(r["type"], r["fqdn"]) for r in records] == [
        ("A", "*.model.aeriumhq.com"),
        ("TXT", "asuid.model.aeriumhq.com"),
    ]
