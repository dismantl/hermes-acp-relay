import json
import tomllib
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_pinned_hermes_agent_owns_the_acp_version() -> None:
    relay_config = tomllib.loads(
        (ROOT / "relay" / "pyproject.toml").read_text(encoding="utf-8")
    )
    dependencies = relay_config["project"]["dependencies"]

    assert "agent-client-protocol" in dependencies
    assert "hermes-agent[acp]" in dependencies


def test_renovate_defers_hermes_compatibility_updates() -> None:
    config = json.loads((ROOT / "renovate.json").read_text(encoding="utf-8"))
    disabled_pep621_packages = {
        package
        for rule in config["packageRules"]
        if rule.get("enabled") is False and "pep621" in rule.get("matchManagers", [])
        for package in rule.get("matchPackageNames", [])
    }

    assert {"agent-client-protocol", "hermes-agent"} <= disabled_pep621_packages
