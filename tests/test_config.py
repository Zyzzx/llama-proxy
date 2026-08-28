import textwrap

import pytest

from llm_proxy.config import ConfigError, load_config


def _write(tmp_path, body: str):
    p = tmp_path / "cfg.yaml"
    p.write_text(textwrap.dedent(body))
    return p


def test_minimal_config(tmp_path):
    p = _write(tmp_path, """
        backends:
          - name: gpu0
            url: http://127.0.0.1:8080
            slots: 4
    """)
    cfg = load_config(p)
    assert len(cfg.backends) == 1
    assert cfg.server.port == 4000
    assert cfg.routing.strategy == "least_busy"
    assert cfg.enabled_backends == cfg.backends


def test_env_expansion(tmp_path, monkeypatch):
    monkeypatch.setenv("BACKEND_HOST", "10.1.2.3")
    p = _write(tmp_path, """
        backends:
          - name: gpu0
            url: http://${BACKEND_HOST}:8080
            slots: 2
    """)
    cfg = load_config(p)
    assert cfg.backends[0].url == "http://10.1.2.3:8080"


def test_env_expansion_default(tmp_path, monkeypatch):
    monkeypatch.delenv("MISSING_VAR", raising=False)
    p = _write(tmp_path, """
        backends:
          - name: gpu0
            url: http://${MISSING_VAR:-127.0.0.1}:8080
    """)
    cfg = load_config(p)
    assert cfg.backends[0].url == "http://127.0.0.1:8080"


def test_missing_env_var_no_default(tmp_path, monkeypatch):
    monkeypatch.delenv("NOPE", raising=False)
    p = _write(tmp_path, """
        backends:
          - name: gpu0
            url: http://x
            api_key: ${NOPE}
    """)
    with pytest.raises(ConfigError, match="NOPE"):
        load_config(p)


def test_no_backends(tmp_path):
    p = _write(tmp_path, "backends: []\n")
    with pytest.raises(ConfigError, match="at least one backend"):
        load_config(p)


def test_duplicate_names(tmp_path):
    p = _write(tmp_path, """
        backends:
          - name: gpu0
            url: http://127.0.0.1:8080
          - name: gpu0
            url: http://127.0.0.1:8081
    """)
    with pytest.raises(ConfigError, match="duplicate"):
        load_config(p)


def test_all_disabled(tmp_path):
    p = _write(tmp_path, """
        backends:
          - name: gpu0
            url: http://127.0.0.1:8080
            enabled: false
    """)
    with pytest.raises(ConfigError, match="disabled"):
        load_config(p)


def test_bad_url(tmp_path):
    p = _write(tmp_path, """
        backends:
          - name: gpu0
            url: ftp://nope
    """)
    with pytest.raises(ConfigError):
        load_config(p)


def test_url_must_not_end_in_v1(tmp_path):
    p = _write(tmp_path, """
        backends:
          - name: gpu0
            url: http://127.0.0.1:8080/v1
    """)
    with pytest.raises(ConfigError, match="must not end in /v1"):
        load_config(p)


def test_bad_strategy(tmp_path):
    p = _write(tmp_path, """
        routing:
          strategy: magic
        backends:
          - name: gpu0
            url: http://127.0.0.1:8080
    """)
    with pytest.raises(ConfigError, match="strategy"):
        load_config(p)


def test_missing_file(tmp_path):
    with pytest.raises(ConfigError, match="not found"):
        load_config(tmp_path / "nope.yaml")
