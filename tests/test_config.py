import pytest
from pydantic import ValidationError

from simple_local import config as config_mod


def write(tmp_path, text):
    path = tmp_path / "config.yml"
    path.write_text(text)
    return str(path)


def test_multi_model_with_adapters_and_draft(tmp_path):
    cfg = config_mod.load(
        write(
            tmp_path,
            """
models:
  - name: base
    source: { provider: huggingface, repo: org/repo, file: m.gguf }
    adapters:
      - name: tuned
        source: { provider: local, file: a.gguf }
        scale: 0.7
    inference:
      parallel: 4
      draft:
        source: { provider: huggingface, repo: org/tiny, file: d.gguf }
        max: 24
  - name: churn
    kind: predictor
    source: { provider: local, file: m.joblib }
server:
  port: 9000
""",
        )
    )
    assert cfg.served_names() == ["base", "tuned", "churn"]
    assert cfg.models[0].adapters[0].scale == 0.7
    assert cfg.models[0].inference.draft.max_tokens == 24
    assert cfg.models[0].inference.draft.min_tokens == 1
    assert cfg.server.port == 9000


def test_legacy_single_model_upgrades(tmp_path):
    cfg = config_mod.load(
        write(
            tmp_path,
            """
kind: llm
model_name: Qwen
source: { provider: huggingface, repo: org/repo, file: m.gguf }
inference: { context_length: 2048 }
server: { port: 8082 }
""",
        )
    )
    assert [m.name for m in cfg.models] == ["Qwen"]
    assert cfg.models[0].inference.context_length == 2048
    assert cfg.server.port == 8082


def test_duplicate_names_rejected(tmp_path):
    with pytest.raises(ValidationError, match="duplicate"):
        config_mod.load(
            write(
                tmp_path,
                """
models:
  - name: base
    source: { provider: local, file: m.gguf }
    adapters:
      - name: base
        source: { provider: local, file: a.gguf }
""",
            )
        )


def test_both_chat_templates_rejected(tmp_path):
    with pytest.raises(ValidationError, match="not both"):
        config_mod.load(
            write(
                tmp_path,
                """
models:
  - name: base
    source: { provider: local, file: m.gguf }
    chat_template: chatml
    chat_template_file: t.jinja
""",
            )
        )


def test_predictor_adapters_rejected(tmp_path):
    with pytest.raises(ValidationError, match="kind: llm"):
        config_mod.load(
            write(
                tmp_path,
                """
models:
  - name: churn
    kind: predictor
    source: { provider: local, file: m.joblib }
    adapters:
      - name: a
        source: { provider: local, file: a.gguf }
""",
            )
        )


def test_predictor_embeddings_rejected(tmp_path):
    with pytest.raises(ValidationError, match="kind: llm"):
        config_mod.load(
            write(
                tmp_path,
                """
models:
  - name: churn
    kind: predictor
    embeddings: true
    source: { provider: local, file: m.joblib }
""",
            )
        )


def test_commented_out_sections_load(tmp_path):
    # A section whose keys are all commented out parses as None, not {}.
    cfg = config_mod.load(
        write(
            tmp_path,
            """
models:
  - name: base
    source: { provider: local, file: m.gguf }
    inference:
      # context_length: 8192
logging:
  # mysql:
  #   host: localhost
server:
  # port: 9000
""",
        )
    )
    assert cfg.logging.mysql is None
    assert cfg.models[0].inference.context_length == 4096
    assert cfg.server.port == 8081


def test_mysql_logging_config(tmp_path, monkeypatch):
    monkeypatch.setenv("PW", "sekrit")
    cfg = config_mod.load(
        write(
            tmp_path,
            """
models:
  - name: base
    source: { provider: local, file: m.gguf }
logging:
  mysql:
    host: 127.0.0.1
    port: 3307
    password: ${PW}
""",
        )
    )
    assert (cfg.logging.mysql.host, cfg.logging.mysql.port) == ("127.0.0.1", 3307)
    assert cfg.logging.mysql.password == "sekrit"
    assert cfg.logging.mysql.database == "simple_local"


def test_mysql_table_must_be_identifier(tmp_path):
    with pytest.raises(ValidationError, match="plain identifier"):
        config_mod.load(
            write(
                tmp_path,
                """
models:
  - name: base
    source: { provider: local, file: m.gguf }
logging:
  mysql:
    table: "log; DROP TABLE x"
""",
            )
        )


def test_unset_mysql_host_disables_logging(tmp_path, monkeypatch):
    """One config for every environment: no MYSQL_HOST means no request logging,
    rather than a connection that retries forever."""
    monkeypatch.delenv("MYSQL_HOST", raising=False)
    body = """
models:
  - name: base
    source: { provider: local, file: m.gguf }
logging:
  mysql:
    host: ${MYSQL_HOST}
    user: ${MYSQL_USER}
    password: ${MYSQL_PASSWORD}
    database: ${MYSQL_DATABASE}
    ssl: true
"""
    assert config_mod.load(write(tmp_path, body)).logging.mysql is None

    monkeypatch.setenv("MYSQL_HOST", "db.mysql.database.azure.com")
    monkeypatch.setenv("MYSQL_USER", "app")
    monkeypatch.setenv("MYSQL_DATABASE", "simple_local")
    mysql = config_mod.load(write(tmp_path, body)).logging.mysql
    assert (mysql.host, mysql.user, mysql.database) == (
        "db.mysql.database.azure.com",
        "app",
        "simple_local",
    )
    assert mysql.ssl is True


def test_env_expansion(tmp_path, monkeypatch):
    monkeypatch.setenv("TEST_KEY", "sekrit")
    cfg = config_mod.load(
        write(
            tmp_path,
            """
models:
  - name: base
    source: { provider: local, file: m.gguf }
server: { api_key: "${TEST_KEY}" }
""",
        )
    )
    assert cfg.server.api_key == "sekrit"
