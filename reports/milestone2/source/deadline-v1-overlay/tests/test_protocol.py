from __future__ import annotations

import copy
import json
import re

import numpy as np
import pytest

from conftest import fake_efficiency_cfg, make_split_and_csv

from m2_deadline import markers, protocol, sampling
from m2_deadline.common import (
    DataIntegrityError,
    GuardError,
    ProtocolChangedError,
    atomic_write_json,
    canonical_json,
    out_path,
    sha256_bytes,
    sha256_file,
)

pytest.importorskip("pandas", reason="read_test_labels needs pandas (reported skip, not a pass)")


def rewrite_csv(csv_path, split, transform):
    text = transform(csv_path.read_text(encoding="utf-8"))
    csv_path.write_text(text, encoding="utf-8", newline="\n")
    split = copy.deepcopy(split)
    split["dataset"]["sha256"] = sha256_file(csv_path)
    return split


# ---------------------------------------------------------------- read_test_labels
def test_read_test_labels_returns_positional_labels(tmp_path):
    split, csv_path = make_split_and_csv(tmp_path)
    ids, y = protocol.read_test_labels(split, None, csv_path)
    names = [line.split(",")[2] for line in csv_path.read_text(encoding="utf-8").splitlines()[1:]]
    assert np.array_equal(ids, np.array(split["row_ids"]["test"]))
    assert [split["dataset"]["class_names"][c] for c in y] == [names[i] for i in ids]
    assert np.bincount(y, minlength=27).tolist() == split["per_class_counts"]["test"]
    # rows really are shuffled: ids are not 0..n-1 and labels are not sorted
    assert not np.array_equal(ids, np.arange(len(ids)))
    assert list(y) != sorted(y)


def test_read_test_labels_accepts_a_path_to_the_split_json(tmp_path):
    split, csv_path = make_split_and_csv(tmp_path)
    split_path = tmp_path / "split_m2.json"
    atomic_write_json(split_path, split)
    ids, y = protocol.read_test_labels(split_path, {}, csv_path)
    assert len(ids) == len(y) == 27 * 15


def test_unknown_label_raises(tmp_path):
    split, csv_path = make_split_and_csv(tmp_path)
    split = rewrite_csv(csv_path, split, lambda t: t.replace("class_03", "class_unknown", 1))
    with pytest.raises(DataIntegrityError, match="unknown labels"):
        protocol.read_test_labels(split, None, csv_path)


def test_wrong_dataset_sha_raises(tmp_path):
    split, csv_path = make_split_and_csv(tmp_path)
    split["dataset"]["sha256"] = "0" * 64
    with pytest.raises(DataIntegrityError, match="sha256 mismatch"):
        protocol.read_test_labels(split, None, csv_path)


def test_wrong_row_count_raises(tmp_path):
    split, csv_path = make_split_and_csv(tmp_path)
    split["dataset"]["rows"] += 1
    with pytest.raises(DataIntegrityError, match="rows"):
        protocol.read_test_labels(split, None, csv_path)


def test_per_class_count_mismatch_raises(tmp_path):
    split, csv_path = make_split_and_csv(tmp_path)
    split["per_class_counts"]["test"][0] += 1
    with pytest.raises(DataIntegrityError, match="per-class counts"):
        protocol.read_test_labels(split, None, csv_path)


def test_id_hash_mismatch_raises(tmp_path):
    split, csv_path = make_split_and_csv(tmp_path)
    split["id_sha256"]["test"] = "f" * 64
    with pytest.raises(DataIntegrityError, match="id hash"):
        protocol.read_test_labels(split, None, csv_path)


def test_out_of_range_or_duplicate_ids_raise(tmp_path):
    split, csv_path = make_split_and_csv(tmp_path)
    bad = copy.deepcopy(split)
    bad["row_ids"]["test"][0] = split["dataset"]["rows"]
    with pytest.raises(DataIntegrityError, match="outside"):
        protocol.read_test_labels(bad, None, csv_path)
    dup = copy.deepcopy(split)
    dup["row_ids"]["test"][1] = dup["row_ids"]["test"][0]
    with pytest.raises(DataIntegrityError, match="duplicate"):
        protocol.read_test_labels(dup, None, csv_path)


def test_missing_split_fields_and_csv_raise(tmp_path):
    split, csv_path = make_split_and_csv(tmp_path)
    with pytest.raises(DataIntegrityError):
        protocol.read_test_labels({"dataset": split["dataset"]}, None, csv_path)
    with pytest.raises(DataIntegrityError, match="not found"):
        protocol.read_test_labels(split, None, tmp_path / "nope.csv")


# ---------------------------------------------------------------- G1 pgd5 cfg
def attacks_like():
    return {
        "attacks": {
            "fgsm": {"kind": "fgsm", "steps": 1, "restarts": 1, "random_start": False, "status": "frozen"},
            "pgd20": {"kind": "pgd", "steps": 20, "alpha_fraction": 0.25, "restarts": 3, "random_start": True, "status": "provisional"},
        },
        "expected_selector_calls_per_batch": {
            "pgd20": {"selector_calls": 65, "candidates": 64},
            "fgsm": {"selector_calls": 4, "candidates": 3},
        },
    }


def test_build_pgd5_cfg_adds_pgd5_with_derived_selector_table_and_keeps_input():
    src = attacks_like()
    before = copy.deepcopy(src)
    cfg = protocol.build_pgd5_cfg(src)
    assert src == before  # deep copy: the frozen V4 config object is untouched
    assert cfg["attacks"]["pgd5"] == {
        "kind": "pgd", "steps": 5, "alpha_fraction": 0.25, "restarts": 1, "random_start": True, "status": "deadline_v1",
    }
    # independent re-derivation: candidates = 1 + restarts * (1 + steps); selector calls = 1 + candidates
    candidates = 1 + 1 * (1 + 5)
    assert cfg["expected_selector_calls_per_batch"]["pgd5"] == {"selector_calls": 1 + candidates, "candidates": candidates}
    assert cfg["expected_selector_calls_per_batch"]["pgd5"] == {"selector_calls": 8, "candidates": 7}
    # the same formula reproduces the frozen entries of V4's file
    assert protocol.selector_calls("pgd", 20, 3) == {"selector_calls": 65, "candidates": 64}
    assert protocol.selector_calls("fgsm", 1, 1) == {"selector_calls": 4, "candidates": 3}


def test_pgd5_cfg_on_the_real_v4_attacks_file(v4):
    real = json.loads((v4 / "configs/m2/attacks.json").read_text(encoding="utf-8"))
    cfg = protocol.build_pgd5_cfg(real)
    assert cfg["expected_selector_calls_per_batch"]["pgd5"] == {"selector_calls": 8, "candidates": 7}
    assert set(cfg["attacks"]) == set(real["attacks"]) | {"pgd5"}
    for name, table in real["expected_selector_calls_per_batch"].items():
        if name in real["attacks"]:
            a = real["attacks"][name]
            assert protocol.selector_calls(a["kind"], a["steps"], a["restarts"]) == table


# ---------------------------------------------------------------- build_protocol / freeze
@pytest.fixture
def freeze_inputs(tmp_path, config):
    split, csv_path = make_split_and_csv(tmp_path)
    test_ids, y_test = protocol.read_test_labels(split, None, csv_path)
    lock = {"epochs": 10, "per_class_target": 10}
    return dict(
        config=config,
        config_sha256="c" * 64,
        lock=lock,
        lock_sha256="l" * 64,
        split=split,
        split_m2_sha256="s" * 64,
        scaler_sha256="m" * 64,
        efficiency_cfg=fake_efficiency_cfg(),
        test_ids=test_ids,
        y_test=y_test,
    )


def test_protocol_has_no_timestamp_and_records_required_fields(freeze_inputs):
    fr = protocol.build_freeze(**freeze_inputs)
    p = fr["protocol"]
    text = fr["protocol_bytes"].decode("utf-8")
    assert "frozen_utc" not in text and "created_utc" not in text
    assert not re.search(r"20\d\d-\d\d-\d\dT", text)
    assert p["test_partition_id_sha256"] == freeze_inputs["split"]["id_sha256"]["test"]
    assert p["epochs"] == 10 and p["seed"] == 1
    assert p["eps0_consistency"]["behaviour"] == "record_only" and p["eps0_consistency"]["never_fails_run"] is True
    assert "VmHWM" in p["efficiency"]["memory_metric"] and "..." not in text
    assert p["robustness"]["methods"]["int8"]["label"] == "APPROXIMATE INT8 GRADIENT (BPDA)"
    assert "dequantized weights" in p["robustness"]["methods"]["int8"]["description"]
    assert p["robustness"]["attacks"]["pgd5"]["steps"] == 5 and p["robustness"]["attacks"]["pgd5"]["restarts"] == 1
    assert p["robustness"]["attacks"]["pgd5"]["random_start"] is True
    assert p["robustness"]["not_run"] and "pgd20" in p["robustness"]["not_run"]
    assert p["robustness"]["per_class_target"] == 10 and p["robustness"]["n_actual"] == 270
    assert p["split"] == {
        "split_m2_sha256": "s" * 64, "scaler_sha256": "m" * 64,
        "n_train": 27 * 6, "n_val": 27 * 3, "n_test": 27 * 15,
    }
    assert p["efficiency"]["threads"] == 1 and p["efficiency"]["repetitions"] == 3 and p["efficiency"]["rows"] == 256
    assert p["frozen_not_chosen_by_performance"] is True


def test_epsilon_strings_are_the_single_repr_form_everywhere(freeze_inputs):
    fr = protocol.build_freeze(**freeze_inputs)
    p = fr["protocol"]
    assert list(p["robustness"]["attacks"]["pgd5"]["alpha_at_eps"]) == ["0.05", "0.1"]
    assert p["robustness"]["attacks"]["pgd5"]["alpha_at_eps"] == {"0.05": 0.0125, "0.1": 0.025}
    assert p["robustness"]["epsilons"] == [0.0, 0.05, 0.1]
    text = fr["protocol_bytes"].decode("utf-8")
    assert "0.10" not in text
    from m2_deadline import metrics
    from m2_deadline.common import DISPLAY_NAMES, MODEL_IDS

    units = {m: {"clean_eps0.0": dict(n=1, accuracy=1.0, macro_f1=1.0, robust_accuracy=1.0)} for m in MODEL_IDS}
    for m in MODEL_IDS:
        for a in ("fgsm", "pgd5"):
            for e in ("0.05", "0.1"):
                units[m]["%s_eps%s" % (a, e)] = dict(n=1, accuracy=0.5, macro_f1=0.5, robust_accuracy=0.5)
    rows = metrics.build_curves(units, MODEL_IDS, DISPLAY_NAMES)
    assert {r["epsilon"] for r in rows} == {"0.0", "0.05", "0.1"}


def test_protocol_hash_is_stable_and_every_field_matters(freeze_inputs):
    a = protocol.build_freeze(**freeze_inputs)
    b = protocol.build_freeze(**freeze_inputs)
    assert a["protocol_bytes"] == b["protocol_bytes"] and a["protocol_sha256"] == b["protocol_sha256"]
    assert a["protocol_sha256"] == sha256_bytes(canonical_json(a["protocol"]).encode())
    for changed in (
        dict(freeze_inputs, config_sha256="d" * 64),
        dict(freeze_inputs, lock_sha256="k" * 64),
        dict(freeze_inputs, lock={"epochs": 5, "per_class_target": 10}),
        dict(freeze_inputs, lock={"epochs": 10, "per_class_target": 5}),
        dict(freeze_inputs, scaler_sha256="n" * 64),
    ):
        assert protocol.build_freeze(**changed)["protocol_sha256"] != a["protocol_sha256"]


def test_subset_files_content_matches_protocol(freeze_inputs):
    fr = protocol.build_freeze(**freeze_inputs)
    rob = fr["protocol"]["robustness"]
    assert rob["subset_ids_file_sha256"] == sha256_bytes(fr["ids_csv_bytes"])
    assert rob["counts_sha256"] == sha256_bytes(fr["counts_bytes"])
    ids, labels = protocol.parse_ids_csv(fr["ids_csv_bytes"])
    assert sampling.ids_sha256(ids) == rob["subset_ids_sha256"]
    assert sampling.label_sha256(labels) == rob["subset_label_sha256"]
    assert set(ids.tolist()) <= set(freeze_inputs["split"]["row_ids"]["test"])
    assert json.loads(fr["counts_bytes"])["n_actual"] == rob["n_actual"]
    assert b"\r" not in fr["ids_csv_bytes"] + fr["counts_bytes"] + fr["protocol_bytes"]


def test_freeze_with_k5_and_a_small_class(tmp_path, config):
    counts = [15] * 27
    counts[2] = 3
    split, csv_path = make_split_and_csv(tmp_path, test_counts=counts)
    ids, y = protocol.read_test_labels(split, None, csv_path)
    fr = protocol.build_freeze(
        config=config, config_sha256="c" * 64, lock={"epochs": 5, "per_class_target": 5}, lock_sha256="l" * 64,
        split=split, split_m2_sha256="s" * 64, scaler_sha256="m" * 64, efficiency_cfg=fake_efficiency_cfg(),
        test_ids=ids, y_test=y,
    )
    assert fr["protocol"]["robustness"]["n_actual"] == 26 * 5 + 3
    assert fr["counts_doc"]["counts"]["2"] == {"available": 3, "selected": 3}


# ---------------------------------------------------------------- write_protocol / assert_frozen
def test_write_protocol_writes_files_and_sidecar(freeze_inputs, out):
    fr = protocol.build_freeze(**freeze_inputs)
    result = protocol.write_protocol(out, fr, "2026-01-01T00:00:00Z")
    assert result["written"] is True
    for key in ("protocol", "protocol_sha256", "protocol_frozen", "subset_ids", "subset_counts"):
        assert out_path(out, key).is_file(), key
    assert out_path(out, "protocol_sha256").read_text().split()[0] == sha256_file(out_path(out, "protocol"))
    assert out_path(out, "protocol_sha256").read_text().split()[1] == "protocol.json"
    sidecar = json.loads(out_path(out, "protocol_frozen").read_text())
    assert sidecar == {"frozen_utc": "2026-01-01T00:00:00Z", "protocol_sha256": fr["protocol_sha256"]}
    assert protocol.assert_frozen(out) == fr["protocol_sha256"]


def test_write_protocol_is_idempotent_and_keeps_first_timestamp(freeze_inputs, out):
    fr = protocol.build_freeze(**freeze_inputs)
    protocol.write_protocol(out, fr, "2026-01-01T00:00:00Z")
    before = {k: out_path(out, k).read_bytes() for k in ("protocol", "subset_ids", "subset_counts", "protocol_frozen")}
    again = protocol.write_protocol(out, protocol.build_freeze(**freeze_inputs), "2027-02-02T00:00:00Z")
    assert again["written"] is False
    assert before == {k: out_path(out, k).read_bytes() for k in before}


def test_write_protocol_refuses_changed_content_and_leaves_files_alone(freeze_inputs, out):
    fr = protocol.build_freeze(**freeze_inputs)
    protocol.write_protocol(out, fr, "2026-01-01T00:00:00Z")
    before = out_path(out, "protocol").read_bytes()
    changed = protocol.build_freeze(**dict(freeze_inputs, config_sha256="d" * 64))
    with pytest.raises(GuardError, match="differs"):
        protocol.write_protocol(out, changed, "2026-01-02T00:00:00Z")
    assert out_path(out, "protocol").read_bytes() == before


def frozen_out(freeze_inputs, out):
    fr = protocol.build_freeze(**freeze_inputs)
    protocol.write_protocol(out, fr, "2026-01-01T00:00:00Z")
    return fr


@pytest.mark.parametrize("key", ["protocol", "subset_ids", "subset_counts"])
def test_assert_frozen_detects_one_byte_edit(freeze_inputs, out, key):
    frozen_out(freeze_inputs, out)
    path = out_path(out, key)
    data = bytearray(path.read_bytes())
    data[len(data) // 2] ^= 0x01
    path.write_bytes(bytes(data))
    with pytest.raises(ProtocolChangedError):
        protocol.assert_frozen(out)


def test_assert_frozen_detects_edit_that_keeps_ids_but_changes_file_bytes(freeze_inputs, out):
    frozen_out(freeze_inputs, out)
    path = out_path(out, "subset_ids")
    path.write_bytes(path.read_bytes() + b"\n")
    with pytest.raises(ProtocolChangedError, match="changed after freeze"):
        protocol.assert_frozen(out)


def test_assert_frozen_checks_sha_file_identity_and_markers(freeze_inputs, out):
    fr = frozen_out(freeze_inputs, out)
    sha_file = out_path(out, "protocol_sha256")
    sha_file.write_text("0" * 64 + "  protocol.json\n")
    with pytest.raises(ProtocolChangedError, match="protocol.json.sha256"):
        protocol.assert_frozen(out)
    sha_file.write_text("%s  protocol.json\n" % fr["protocol_sha256"])
    assert protocol.assert_frozen(out) == fr["protocol_sha256"]

    identity = markers.build_run_identity(
        config_sha256="c" * 64, lock_sha256="l" * 64, epochs=10, protocol_sha256="e" * 64,
        v4_manifest_sha256="v" * 64, source_sha256={},
    )
    atomic_write_json(out_path(out, "run_identity"), identity)
    with pytest.raises(ProtocolChangedError, match="run_identity"):
        protocol.assert_frozen(out)
    identity["protocol_sha256"] = fr["protocol_sha256"]
    atomic_write_json(out_path(out, "run_identity"), identity)
    assert protocol.assert_frozen(out)

    marker_dir = out / "stage_markers"
    marker_dir.mkdir()
    (marker_dir / "freeze.ok").write_text(json.dumps({"stage": "freeze", "protocol_sha256": "9" * 64}))
    with pytest.raises(ProtocolChangedError, match="marker freeze.ok"):
        protocol.assert_frozen(out)


def test_assert_frozen_when_files_are_missing(out):
    with pytest.raises(ProtocolChangedError, match="missing"):
        protocol.assert_frozen(out)


def test_stage_freeze_style_flow_never_needs_features(freeze_inputs, out):
    """The freeze path uses labels from the CSV label column only: no arrays, scaler or model arguments."""
    import inspect

    assert list(inspect.signature(protocol.read_test_labels).parameters) == ["split", "dataset_cfg", "dataset_csv"]
    fr = protocol.build_freeze(**freeze_inputs)
    assert "x" not in fr and "features" not in json.dumps(fr["protocol"]).lower()
