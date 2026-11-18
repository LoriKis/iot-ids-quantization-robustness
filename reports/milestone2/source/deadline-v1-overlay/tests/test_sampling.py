from __future__ import annotations

import hashlib
import re

import numpy as np
import pytest

from m2_deadline import sampling
from m2_deadline.common import DataIntegrityError


def pool(counts, seed=3):
    """Shuffled ids/labels with ``counts[c]`` rows of class c (ids are non-contiguous)."""
    labels = np.concatenate([np.full(n, c) for c, n in enumerate(counts)]).astype(np.int64)
    ids = np.arange(len(labels), dtype=np.int64) * 3 + 11
    order = np.random.default_rng(seed).permutation(len(labels))
    return ids[order], labels[order]


FULL = [40] * 27


def test_ids_sha256_hard_coded_vector_and_order_insensitive():
    assert sampling.ids_sha256([3, 1, 2]) == hashlib.sha256(b"[1,2,3]").hexdigest()
    assert sampling.ids_sha256([1, 2, 3]) == sampling.ids_sha256([3, 2, 1])
    assert sampling.ids_sha256(np.array([3, 1, 2], dtype=np.int64)) == hashlib.sha256(b"[1,2,3]").hexdigest()
    assert sampling.ids_sha256(np.array([3, 1, 2], dtype=np.int32)) == hashlib.sha256(b"[1,2,3]").hexdigest()
    assert sampling.ids_sha256([]) == hashlib.sha256(b"[]").hexdigest()


def test_label_sha256_is_little_endian_int64_bytes():
    y = [0, 1, 26]
    assert sampling.label_sha256(y) == hashlib.sha256(np.asarray(y, dtype="<i8").tobytes()).hexdigest()
    assert sampling.label_sha256(np.array(y, dtype=np.int32)) == sampling.label_sha256(y)


def test_ids_hash_equals_v4_canonical_digest_source(v4):
    """Regex check of V4 ``canonical`` / ``digest`` / ``id_hash`` source text (research.* is never imported)."""
    common_src = (v4 / "research" / "common.py").read_text(encoding="utf-8")
    assert re.search(
        r"def canonical\(value\):\s+return json\.dumps\(value, sort_keys=True, separators=\(\",\", \":\"\), allow_nan=False\)",
        common_src,
    )
    assert re.search(r"def digest\(value\):\s+return hashlib\.sha256\(canonical\(value\)\.encode\(\)\)\.hexdigest\(\)", common_src)
    split_src = (v4 / "research" / "m2_split.py").read_text(encoding="utf-8")
    assert re.search(r"def id_hash\(ids\):\s+return digest\(\[int\(i\) for i in sorted\(ids\)\]\)", split_src)


def test_deterministic_across_calls_and_row_order():
    ids, labels = pool(FULL, seed=1)
    a = sampling.stratified_subset(ids, labels, 10, 20270)
    b = sampling.stratified_subset(ids, labels, 10, 20270)
    perm = np.random.default_rng(99).permutation(len(ids))
    c = sampling.stratified_subset(ids[perm], labels[perm], 10, 20270)
    assert np.array_equal(a["ids"], b["ids"]) and a["ids_sha256"] == b["ids_sha256"]
    assert np.array_equal(a["ids"], c["ids"]) and a["ids_sha256"] == c["ids_sha256"]
    assert np.array_equal(a["labels"], c["labels"])


def test_counts_sorted_unique_and_aligned():
    ids, labels = pool(FULL)
    r = sampling.stratified_subset(ids, labels, 10, 20270)
    assert r["n_actual"] == 270 == len(r["ids"])
    assert np.all(np.diff(r["ids"]) > 0)  # sorted and unique
    assert sum(v["selected"] for v in r["counts"].values()) == r["n_actual"]
    assert all(v["selected"] == 10 and v["available"] == 40 for v in r["counts"].values())
    lookup = dict(zip(ids.tolist(), labels.tolist()))
    assert [lookup[i] for i in r["ids"].tolist()] == r["labels"].tolist()
    assert r["ids_sha256"] == sampling.ids_sha256(r["ids"])
    assert np.bincount(r["labels"], minlength=27).tolist() == [10] * 27


def test_min_k_available_and_empty_class_listed_with_zero():
    counts = list(FULL)
    counts[4] = 3
    counts[9] = 0
    ids, labels = pool(counts)
    r = sampling.stratified_subset(ids, labels, 10, 20270)
    assert r["counts"][4] == {"available": 3, "selected": 3}
    assert r["counts"][9] == {"available": 0, "selected": 0}
    assert r["classes_missing"] == [9]
    assert 9 not in r["classes_present"] and 4 in r["classes_present"]
    assert r["n_actual"] == 26 * 10 - 7
    doc = sampling.counts_document(r, 10, 20270)
    assert doc["counts"]["9"] == {"available": 0, "selected": 0}
    assert doc["classes_missing"] == [9] and doc["n_actual"] == r["n_actual"]
    assert doc["subset_ids_sha256"] == r["ids_sha256"]


def test_k5_is_not_nested_in_k10_in_general_but_both_deterministic():
    ids, labels = pool([60] * 27)
    k10 = sampling.stratified_subset(ids, labels, 10, 20270)
    k5 = sampling.stratified_subset(ids, labels, 5, 20270)
    assert k5["n_actual"] == 135
    assert not set(k5["ids"].tolist()) <= set(k10["ids"].tolist())
    again = sampling.stratified_subset(ids, labels, 5, 20270)
    assert np.array_equal(k5["ids"], again["ids"])


def test_same_id_file_reused_by_four_models_gives_identical_hash():
    ids, labels = pool(FULL)
    r = sampling.stratified_subset(ids, labels, 10, 20270)
    hashes = {sampling.ids_sha256(r["ids"].copy()) for _model in ("V1", "V2", "V3", "V4")}
    assert hashes == {r["ids_sha256"]}


def test_different_split_seed_or_seed_name_changes_selection():
    ids, labels = pool(FULL)
    a = sampling.stratified_subset(ids, labels, 10, 20270)
    b = sampling.stratified_subset(ids, labels, 10, 20271)
    c = sampling.stratified_subset(ids, labels, 10, 20270, seed_name="other")
    assert a["ids_sha256"] != b["ids_sha256"]
    assert a["ids_sha256"] != c["ids_sha256"]


def test_class_seed_matches_documented_rule():
    expected = int.from_bytes(hashlib.sha256(b"m2-deadline-robust-subset|20270|5").digest()[:8], "big")
    assert sampling.class_seed("m2-deadline-robust-subset", 20270, 5) == expected


def test_input_validation():
    ids, labels = pool(FULL)
    with pytest.raises(DataIntegrityError):
        sampling.stratified_subset(ids, labels[:-1], 10, 1)
    with pytest.raises(DataIntegrityError):
        sampling.stratified_subset(np.r_[ids, ids[0]], np.r_[labels, labels[0]], 10, 1)
    with pytest.raises(DataIntegrityError):
        sampling.stratified_subset(ids, labels + 1, 10, 1)
    with pytest.raises(DataIntegrityError):
        sampling.stratified_subset(ids, labels, 0, 1)
