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
