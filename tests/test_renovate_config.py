import json
import re
import tomllib
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
WORKFLOWS = (
    ROOT / ".forgejo" / "workflows" / "ci.yml",
    ROOT / ".github" / "workflows" / "ci.yml",
)
HERMES_IMAGE = re.compile(r"nousresearch/hermes-agent@sha256:[0-9a-f]{64}")


def test_hermes_image_owns_hermes_agent_and_the_acp_version() -> None:
    """The relay installs --no-deps into the hermes-agent image, so it must not
    declare hermes-agent or pin an ACP SDK version the image could disagree
    with."""
    relay_config = tomllib.loads(
        (ROOT / "relay" / "pyproject.toml").read_text(encoding="utf-8")
    )
    dependencies = relay_config["project"]["dependencies"]

    assert "agent-client-protocol" in dependencies
    assert not any(dep.startswith("hermes-agent") for dep in dependencies)


def test_ci_workflows_pin_one_hermes_image_digest() -> None:
    """Both CI workflows run in the same digest-pinned hermes-agent image; the
    promotion PR rewrites every occurrence, so they must never drift apart."""
    digests = set()
    for workflow in WORKFLOWS:
        found = HERMES_IMAGE.findall(workflow.read_text(encoding="utf-8"))
        assert found, f"{workflow} does not run in a digest-pinned hermes-agent image"
        digests.update(found)
    assert len(digests) == 1, digests


def test_renovate_defers_hermes_compatibility_updates() -> None:
    config = json.loads((ROOT / "renovate.json").read_text(encoding="utf-8"))
    disabled = [rule for rule in config["packageRules"] if rule.get("enabled") is False]

    assert any(
        "pep621" in rule.get("matchManagers", [])
        and "agent-client-protocol" in rule.get("matchPackageNames", [])
        for rule in disabled
    )
    assert any(
        "docker" in rule.get("matchDatasources", [])
        and "nousresearch/hermes-agent" in rule.get("matchPackageNames", [])
        for rule in disabled
    )
