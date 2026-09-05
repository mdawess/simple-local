"""What another repo gets when it depends on this package."""

import subprocess
import sys
from pathlib import Path

import pytest

import simple_local

REPO = Path(__file__).resolve().parent.parent


def test_the_config_layer_is_importable_without_the_heavy_stack():
    # A caller that only wants to read a YAML file should not pay for
    # scikit-learn, fastapi and boto3 to be imported.
    probe = (
        "import sys, simple_local;"
        "assert simple_local.load;"
        "heavy = [m for m in ('sklearn', 'fastapi', 'boto3', 'torch') if m in sys.modules];"
        "print(','.join(heavy))"
    )
    out = subprocess.run(
        [sys.executable, "-c", probe], capture_output=True, text=True, cwd=REPO, check=True
    )
    assert out.stdout.strip() == ""


def test_lazy_names_still_resolve():
    assert simple_local.Models.__name__ == "Models"
    assert simple_local.create_app.__name__ == "create_app"
    assert simple_local.build_registry.__name__ == "build_registry"


def test_everything_advertised_actually_exists():
    for name in simple_local.__all__:
        assert getattr(simple_local, name) is not None, name


def test_an_unknown_attribute_says_so_rather_than_importing_nothing():
    with pytest.raises(AttributeError, match="no attribute 'nope'"):
        simple_local.nope


def test_dir_lists_the_public_api():
    assert set(dir(simple_local)) == set(simple_local.__all__)


def test_the_package_reports_a_version():
    assert simple_local.__version__ != "0.0.0.dev0"


def test_type_annotations_ship_with_the_package():
    assert (REPO / "simple_local" / "py.typed").exists()


CUSTOM_CONFIG = REPO / "examples/custom/config.yml"


def test_models_loads_and_calls_a_runtime_in_process():
    from simple_local import Models

    with Models.from_config(CUSTOM_CONFIG) as models:
        assert models.names == ["cost-ensemble"]
        assert "cost-ensemble" in models
        assert len(models) == 1
        result = models.predict("cost-ensemble", {"kva": 100, "phase": "1ph"})
        assert result["predicted_cost"] > 0


def test_an_unknown_model_names_the_ones_that_exist():
    from simple_local import Models

    with Models.from_config(CUSTOM_CONFIG) as models:
        with pytest.raises(KeyError, match="cost-ensemble"):
            models["nope"]


def test_asking_an_in_process_runtime_for_a_socket_explains_why_there_is_none():
    from simple_local import Models

    with Models.from_config(CUSTOM_CONFIG) as models:
        with pytest.raises(ValueError, match="runs in this process"):
            models.endpoint("cost-ensemble")


def test_close_stops_the_runtimes():
    from simple_local import Models

    models = Models.from_config(CUSTOM_CONFIG)
    models.close()
    assert models.registry.entries  # entries remain readable; runtimes are stopped


def test_serve_is_what_reads_host_and_port(monkeypatch):
    # create_app returns an ASGI app, which cannot bind — so uvicorn.run(app)
    # silently uses uvicorn's own defaults instead of the config's.
    import simple_local.server as server

    captured = {}
    monkeypatch.setattr(server.uvicorn if hasattr(server, "uvicorn") else server,
                        "uvicorn", None, raising=False)

    import uvicorn

    monkeypatch.setattr(uvicorn, "run", lambda app, **kw: captured.update(kw))
    server.serve(REPO / "examples/custom/config.yml")
    assert (captured["host"], captured["port"]) == ("localhost", 8081)


def test_serve_arguments_override_the_config(monkeypatch):
    import uvicorn

    import simple_local.server as server

    captured = {}
    monkeypatch.setattr(uvicorn, "run", lambda app, **kw: captured.update(kw))
    server.serve(REPO / "examples/custom/config.yml", host="0.0.0.0", port=9999)
    assert (captured["host"], captured["port"]) == ("0.0.0.0", 9999)


def test_watching_needs_a_path_not_a_loaded_config():
    import simple_local.server as server

    config = simple_local.load(str(REPO / "examples/custom/config.yml"))
    with pytest.raises(ValueError, match="needs a config path"):
        server.serve(config, watch=True)


def test_a_blank_env_var_warns_before_falling_back(caplog):
    # Binding somewhere other than the config says is a bad way to discover an
    # unset variable.
    import logging

    with caplog.at_level(logging.WARNING, logger="simple_local.config"):
        config = simple_local.Config.model_validate(
            {"models": [{"name": "m", "kind": "remote", "remote": {"url": "http://x"}}],
             "server": {"host": "", "port": ""}}
        )
    assert config.server.host == "localhost" and config.server.port == 8081
    assert "expanded to nothing" in caplog.text
