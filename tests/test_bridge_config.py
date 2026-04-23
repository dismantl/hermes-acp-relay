from __future__ import annotations

import logging
from pathlib import Path

import pytest

from hermes_acp_bridge.config import ConfigError, load_config


def test_load_config_rejects_blank_credentials(tmp_path):
    config_path = tmp_path / "config.toml"
    config_path.write_text(
        'url = "wss://example.com/acp"\n'
        'username = ""\n'
        'password = ""\n'
    )

    with pytest.raises(ConfigError, match="non-empty strings"):
        load_config(str(config_path))


def test_load_config_rejects_ws_url_with_credentials(tmp_path):
    config_path = tmp_path / "config.toml"
    config_path.write_text(
        'url = "ws://example.com/acp"\n'
        'username = "u"\n'
        'password = "p"\n'
    )

    with pytest.raises(ConfigError, match="wss"):
        load_config(str(config_path))


def test_load_config_accepts_ws_url_without_credentials(tmp_path):
    config_path = tmp_path / "config.toml"
    config_path.write_text('url = "ws://localhost:8765/acp"\n')

    cfg = load_config(str(config_path))
    assert cfg.url == "ws://localhost:8765/acp"
    assert cfg.username is None and cfg.password is None


def test_load_config_rejects_url_with_embedded_userinfo(tmp_path):
    config_path = tmp_path / "config.toml"
    config_path.write_text('url = "wss://user:pass@example.com/acp"\n')

    with pytest.raises(ConfigError, match="userinfo"):
        load_config(str(config_path))


def test_load_config_warns_on_world_readable(tmp_path, caplog):
    config_path = tmp_path / "config.toml"
    config_path.write_text('url = "wss://example.com/acp"\n')
    config_path.chmod(0o644)

    with caplog.at_level(logging.WARNING, logger="hermes_acp_bridge.config"):
        load_config(str(config_path))

    assert any("chmod 600" in r.message for r in caplog.records)


def test_load_config_no_warning_when_mode_is_600(tmp_path, caplog):
    config_path = tmp_path / "config.toml"
    config_path.write_text('url = "wss://example.com/acp"\n')
    config_path.chmod(0o600)

    with caplog.at_level(logging.WARNING, logger="hermes_acp_bridge.config"):
        load_config(str(config_path))

    assert not [r for r in caplog.records if "chmod" in r.message]


def test_load_config_missing_file_gives_helpful_error(tmp_path):
    missing = tmp_path / "nope.toml"
    with pytest.raises(ConfigError, match="Config file not found"):
        load_config(str(missing))


def test_load_config_rejects_missing_url(tmp_path):
    config_path = tmp_path / "config.toml"
    config_path.write_text('username = "u"\npassword = "p"\n')
    with pytest.raises(ConfigError, match="'url' is required"):
        load_config(str(config_path))


def test_load_config_rejects_non_string_url(tmp_path):
    config_path = tmp_path / "config.toml"
    config_path.write_text("url = 42\n")
    with pytest.raises(ConfigError, match="'url' is required"):
        load_config(str(config_path))


def test_load_config_rejects_username_without_password(tmp_path):
    config_path = tmp_path / "config.toml"
    config_path.write_text('url = "wss://example.com/acp"\nusername = "u"\n')
    with pytest.raises(ConfigError, match="both be set"):
        load_config(str(config_path))


def test_load_config_rejects_malformed_toml(tmp_path):
    config_path = tmp_path / "config.toml"
    config_path.write_text('url = "wss://example.com/acp\n')  # unterminated
    with pytest.raises(ConfigError, match="Failed to parse"):
        load_config(str(config_path))


def test_resolve_path_prefers_explicit_over_env(monkeypatch, tmp_path):
    env_cfg = tmp_path / "env.toml"
    env_cfg.write_text('url = "wss://env.example.com/acp"\n')
    explicit_cfg = tmp_path / "explicit.toml"
    explicit_cfg.write_text('url = "wss://explicit.example.com/acp"\n')

    monkeypatch.setenv("HERMES_BRIDGE_CONFIG", str(env_cfg))
    cfg = load_config(str(explicit_cfg))
    assert cfg.url == "wss://explicit.example.com/acp"


def test_resolve_path_uses_env_when_no_explicit(monkeypatch, tmp_path):
    env_cfg = tmp_path / "env.toml"
    env_cfg.write_text('url = "wss://env.example.com/acp"\n')
    monkeypatch.setenv("HERMES_BRIDGE_CONFIG", str(env_cfg))
    cfg = load_config()
    assert cfg.url == "wss://env.example.com/acp"


def test_load_config_wraps_os_errors(monkeypatch, tmp_path):
    config_path = tmp_path / "config.toml"
    config_path.write_text('url = "wss://example.com/acp"\n')

    original_open = Path.open

    def failing_open(self, *args, **kwargs):
        if self == config_path:
            raise PermissionError("permission denied")
        return original_open(self, *args, **kwargs)

    monkeypatch.setattr(Path, "open", failing_open)

    with pytest.raises(ConfigError, match="Failed to read"):
        load_config(str(config_path))
