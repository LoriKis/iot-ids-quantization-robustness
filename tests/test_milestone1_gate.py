import json
import pytest
from research.milestone1 import read_target, validate_request, verify_bundle


def test_unresolved_science_cannot_train_even_with_smoke_flag():
    with pytest.raises(ValueError, match="Exact paper reproduction is blocked"):
        validate_request("paper", smoke=True, dry_run=False)
    validate_request("paper", smoke=False, dry_run=True)


def test_full_artifact_execution_is_disabled():
    with pytest.raises(ValueError, match="full-budget training is disabled"):
        validate_request("artifact", smoke=False, dry_run=False)
    validate_request("artifact", smoke=True, dry_run=False)


def test_configuration_edit_cannot_remove_scientific_gate(tmp_path):
    target = read_target()
    target["status"] = "verified"
    target["unresolved"] = {}
    path = tmp_path / "target.json"
    path.write_text(json.dumps(target), encoding="utf-8")
    with pytest.raises(ValueError, match="review the scientific evidence"):
        read_target(path)


def test_empty_manifest_cannot_claim_complete_smoke(tmp_path):
    (tmp_path / "manifest.json").write_text("{}", encoding="utf-8")
    with pytest.raises(ValueError, match="Incomplete smoke manifest"):
        verify_bundle(tmp_path)
