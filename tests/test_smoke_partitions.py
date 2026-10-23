import json

import numpy as np
import pytest

from research import paper_described_adam as runner
from research import provenance as pv

SPLIT = json.loads((runner.ROOT / "reports/milestone1-paper-described/split_ids.json").read_text(encoding="utf-8"))
CFG = runner.read_adam_config(runner.DEFAULT_CONFIG)


@pytest.fixture(scope="module")
def labels():
    _, _, labels, _ = pv.load_dataset(runner.ROOT / CFG["dataset"]["path"], CFG)
    return labels


def test_new_smoke_selection_is_partition_safe(labels):
    split = {k: np.array(SPLIT[k]) for k in ("train", "validation", "test")}
    sel = runner.select_smoke_ids(split, labels)
    test = set(SPLIT["test"])
    assert len(sel["train_ids"]) == 270 and len(sel["validation_ids"]) == 135
    assert set(sel["train_ids"]) <= set(SPLIT["train"]) and set(sel["validation_ids"]) <= set(SPLIT["validation"])
    assert not set(sel["train_ids"]) & test and not set(sel["validation_ids"]) & test
    assert not set(sel["train_ids"]) & set(sel["validation_ids"])
    assert sel["assertions"]["no_intersection_with_saved_test"] is True


def test_selection_is_deterministic_and_balanced(labels):
    split = {k: np.array(SPLIT[k]) for k in ("train", "validation", "test")}
    a, b = runner.select_smoke_ids(split, labels), runner.select_smoke_ids(split, labels)
    assert a == b
    assert np.bincount(labels[a["train_ids"]], minlength=27).tolist() == [10] * 27
    assert np.bincount(labels[a["validation_ids"]], minlength=27).tolist() == [5] * 27


def test_contaminated_split_is_refused(labels):
    # a "train" partition that contains a test ID must still never leak test rows into the selection
    split = {k: np.array(SPLIT[k]) for k in ("train", "validation", "test")}
    leaked = dict(split)
    leaked["test"] = np.concatenate([split["test"], split["train"][:1000]])
    with pytest.raises(runner.PartitionError, match="intersects the saved test"):
        runner.select_smoke_ids(leaked, labels)


def test_too_few_rows_per_class_is_refused(labels):
    split = {k: np.array(SPLIT[k]) for k in ("train", "validation", "test")}
    split["validation"] = split["validation"][labels[split["validation"]] != 3]
    with pytest.raises(runner.PartitionError):
        runner.select_smoke_ids(split, labels)


def test_smoke_never_loads_weights_from_the_full_run_or_smoke_dir():
    src = (runner.ROOT / "research/paper_described_adam.py").read_text(encoding="utf-8")
    smoke_body = src.split("def run_smoke", 1)[1].split("# ----", 1)[0]
    assert "torch.load" in smoke_body  # only the smoke's own state-hash reload check
    assert "seed-" not in smoke_body and "epoch-" not in smoke_body
    assert runner.SMOKE_RUN != runner.DEFAULT_RUN


def test_old_smoke_overlap_reconstruction_is_exact_and_counted(labels):
    import importlib.util
    path = runner.ROOT / "scripts/reconstruct_old_smoke_overlap.py"
    spec = importlib.util.spec_from_file_location("reconstruct_old_smoke_overlap", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    res = mod.reconstruct(labels, SPLIT)
    c = res["overlap_counts"]
    assert len(res["smoke_train_ids"]) == 270 and len(res["smoke_validation_ids"]) == 135
    # every smoke row lies in exactly one saved partition
    assert c["smoke_train_in_saved_train"] + c["smoke_train_in_saved_validation"] + c["smoke_train_in_saved_test"] == 270
    assert c["smoke_validation_in_saved_train"] + c["smoke_validation_in_saved_validation"] + c["smoke_validation_in_saved_test"] == 135
    assert res["distinct_smoke_rows_in_saved_test"] == c["smoke_train_in_saved_test"] + c["smoke_validation_in_saved_test"]
