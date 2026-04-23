from __future__ import annotations

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
