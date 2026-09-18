# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Helm render checks for NetworkPolicy and rayAuth defaults."""

import shutil
import subprocess
from pathlib import Path
from types import ModuleType

import pytest


def _repo_root() -> Path:
    for parent in Path(__file__).resolve().parents:
        if (parent / "charts" / "cosmos-curator" / "Chart.yaml").exists():
            return parent
    msg = "could not find repository root"
    raise RuntimeError(msg)


REPO_ROOT = _repo_root()
CHART_DIR = REPO_ROOT / "charts" / "cosmos-curator"

pytestmark = pytest.mark.helm


def _yaml() -> ModuleType:
    return pytest.importorskip("yaml")


def _render(values_file: Path, show_only: list[str] | None = None) -> list[dict[str, object]]:
    if shutil.which("helm") is None:
        pytest.skip("helm is not installed")
    args = ["helm", "template", "diag", str(CHART_DIR), "-f", str(values_file)]
    for t in show_only or []:
        args.extend(["--show-only", t])
    result = subprocess.run(args, check=False, text=True, capture_output=True, timeout=30)  # noqa: S603
    if result.returncode != 0:
        raise AssertionError(result.stderr)
    return [d for d in _yaml().safe_load_all(result.stdout) if d]


def _np(docs: list[dict[str, object]]) -> dict[str, object]:
    for doc in docs:
        if doc.get("kind") == "NetworkPolicy":
            return doc
    msg = "no NetworkPolicy in rendered output"
    raise AssertionError(msg)


def _all_covered_ports(rule: dict[str, object]) -> set[int]:
    """Expand port/endPort ranges and plain ports into a set of covered port numbers."""
    covered: set[int] = set()
    for p in rule.get("ports", []):
        lo = p["port"]
        hi = p.get("endPort", lo)
        covered |= set(range(lo, hi + 1))
    return covered


def test_networkpolicy_rendered_by_default(tmp_path: Path) -> None:
    """NetworkPolicy is on by default for standalone clusters."""
    vf = tmp_path / "v.yaml"
    vf.write_text("{}\n")
    kinds = [d.get("kind") for d in _render(vf)]
    assert "NetworkPolicy" in kinds


def test_networkpolicy_absent_when_disabled(tmp_path: Path) -> None:
    """Disabling networkPolicy emits no NetworkPolicy resource — required for NVCF deploys."""
    vf = tmp_path / "v.yaml"
    vf.write_text("networkPolicy:\n  enabled: false\n")
    kinds = [d.get("kind") for d in _render(vf)]
    assert "NetworkPolicy" not in kinds


def test_networkpolicy_egress_unrestricted(tmp_path: Path) -> None:
    """Egress must never appear in policyTypes — wide-open egress is a hard requirement."""
    vf = tmp_path / "v.yaml"
    vf.write_text("networkPolicy:\n  enabled: true\n")
    policy = _np(_render(vf, ["templates/networkpolicy.yaml"]))
    assert policy["spec"]["policyTypes"] == ["Ingress"]
    assert "egress" not in policy["spec"]


def test_networkpolicy_invocation_port_absent_without_allow_invocation_from(tmp_path: Path) -> None:
    """curator.port is unreachable from any peer when allowInvocationFrom is empty."""
    # Use a non-default port to prove the template uses the value, not a hardcoded 8000.
    vf = tmp_path / "v.yaml"
    vf.write_text("networkPolicy:\n  enabled: true\ncurator:\n  port: 18000\nmetrics:\n  enabled: false\n")
    policy = _np(_render(vf, ["templates/networkpolicy.yaml"]))
    all_covered: set[int] = set()
    for rule in policy["spec"]["ingress"]:
        all_covered |= _all_covered_ports(rule)
    assert 18000 not in all_covered, "invocation port must be unreachable when allowInvocationFrom is empty"


def test_networkpolicy_inter_pod_rule_covers_boundary_ports(tmp_path: Path) -> None:
    """Rule 1 covers ports 1, 17999, 18001, 65535 — but NOT 18000.

    A regression that drops endPort renders valid YAML but silently wrong semantics;
    this test catches it by asserting both boundaries of each range.
    """
    vf = tmp_path / "v.yaml"
    vf.write_text("networkPolicy:\n  enabled: true\ncurator:\n  port: 18000\nmetrics:\n  enabled: false\n")
    policy = _np(_render(vf, ["templates/networkpolicy.yaml"]))
    inter_pod_rules = [
        r
        for r in policy["spec"]["ingress"]
        if any(peer.get("podSelector", {}).get("matchLabels", {}) for peer in r.get("from", []))
        and _all_covered_ports(r) != {18000}
    ]
    assert inter_pod_rules, "expected at least one inter-pod range rule"
    inter_pod_covered: set[int] = set()
    for rule in inter_pod_rules:
        inter_pod_covered |= _all_covered_ports(rule)
    assert 1 in inter_pod_covered, "port 1 must be covered"
    assert 17999 in inter_pod_covered, "port immediately below curator.port must be covered"
    assert 18001 in inter_pod_covered, "port immediately above curator.port must be covered"
    assert 65535 in inter_pod_covered, "port 65535 must be covered"
    assert 18000 not in inter_pod_covered, "curator.port must NOT be in inter-pod rule"


def test_networkpolicy_allow_invocation_from_opens_only_invocation_port(tmp_path: Path) -> None:
    """AllowInvocationFrom adds a rule covering only curator.port for the listed peers."""
    vf = tmp_path / "v.yaml"
    vf.write_text(
        """
networkPolicy:
  enabled: true
  allowInvocationFrom:
    - namespaceSelector:
        matchLabels:
          kubernetes.io/metadata.name: api-gateway
curator:
  port: 18000
metrics:
  enabled: false
""".lstrip()
    )
    policy = _np(_render(vf, ["templates/networkpolicy.yaml"]))
    invocation_rules = [r for r in policy["spec"]["ingress"] if _all_covered_ports(r) == {18000}]
    assert len(invocation_rules) == 1
    peers = invocation_rules[0].get("from", [])
    assert {"namespaceSelector": {"matchLabels": {"kubernetes.io/metadata.name": "api-gateway"}}} in peers


def _otel_peer_present(tmp_path: Path, *, metrics_enabled: bool) -> bool:
    vf = tmp_path / f"v-{metrics_enabled}.yaml"
    vf.write_text(f"networkPolicy:\n  enabled: true\nmetrics:\n  enabled: {str(metrics_enabled).lower()}\n")
    policy = _np(_render(vf, ["templates/networkpolicy.yaml"]))
    all_labels = [
        peer.get("podSelector", {}).get("matchLabels", {})
        for rule in policy["spec"]["ingress"]
        for peer in rule.get("from", [])
    ]
    # Assert both name AND component so a mode change (deployment→daemonset/sidecar)
    # produces different labels and fails this test rather than silently losing the rule.
    return any(
        lbl.get("app.kubernetes.io/name") == "opentelemetry-collector"
        and lbl.get("component") == "standalone-collector"
        for lbl in all_labels
    )


def test_networkpolicy_otel_collector_rule_conditional_on_metrics_enabled(tmp_path: Path) -> None:
    """OTel collector scrape peer appears only when metrics.enabled=true."""
    assert _otel_peer_present(tmp_path, metrics_enabled=True)
    assert not _otel_peer_present(tmp_path, metrics_enabled=False)


def test_networkpolicy_extra_ingress_appended(tmp_path: Path) -> None:
    """ExtraIngress entries appear in ingress rules with correct peers."""
    vf = tmp_path / "v.yaml"
    vf.write_text(
        """
networkPolicy:
  enabled: true
  extraIngress:
    - ports:
        - port: 9999
          protocol: TCP
      from:
        - namespaceSelector:
            matchLabels:
              kubernetes.io/metadata.name: monitoring
metrics:
  enabled: false
""".lstrip()
    )
    policy = _np(_render(vf, ["templates/networkpolicy.yaml"]))
    # Port 9999 is inside the inter-pod range (8001-65535), so asserting on port is vacuous.
    # Assert on the peer that only extraIngress would add.
    monitoring_peer = {"namespaceSelector": {"matchLabels": {"kubernetes.io/metadata.name": "monitoring"}}}
    all_peers = [peer for rule in policy["spec"]["ingress"] for peer in rule.get("from", [])]
    assert monitoring_peer in all_peers, "monitoring namespace peer must appear via extraIngress"


def test_rayauth_enabled_by_default_renders_secret_and_statefulset(tmp_path: Path) -> None:
    """rayAuth.enabled=true (new default) renders a Secret and wires it into the StatefulSet."""
    vf = tmp_path / "v.yaml"
    vf.write_text("{}\n")
    docs = _render(vf)
    kinds = {d.get("kind") for d in docs}
    assert "Secret" in kinds, "ray-auth-token Secret must render with default values"
    assert "StatefulSet" in kinds, "StatefulSet must still render with rayAuth enabled"
    secrets = [d for d in docs if d.get("kind") == "Secret" and d.get("metadata", {}).get("name") == "ray-auth-token"]
    assert secrets, "Secret must be named ray-auth-token"
