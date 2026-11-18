"""Old-versus-new split comparison and the split freeze record (features and labels only, no model).

    py -B -m research.m2b_splitreport --old reports\\milestone2a\\split --new reports\\milestone2b\\split_v2 ^
        --out-dir reports\\milestone2b\\split_v2

Writes ``split_comparison_v1_v2.json/.md`` and ``split_freeze.json`` (refuses to overwrite).
"""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
import sys

import numpy as np

from . import m2_split as m2
from .common import ROOT, digest, sha256

log = logging.getLogger("m2b_splitreport")

PARTS = m2.PARTITIONS
WEAK_VECTORS = m2.V2_WEAK_GROUPS
COMPARISON = ("split_comparison_v1_v2.json", "split_comparison_v1_v2.md")
FREEZE = "split_freeze.json"
DEFAULT_BEFORE = "artifacts/m2b-integrity/before.json"


def _read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _ids(split):
    return {p: np.array(split["row_ids"][p], dtype=np.int64) for p in PARTS}


def _class_stats(ids, y, groups, n_classes):
    """Per partition and class: rows, distinct groups, largest-group share, effective vector count."""
    out = {}
    for p in PARTS:
        rows = ids[p]
        stats = []
        for c in range(n_classes):
            members = rows[y[rows] == c]
            _, per_group = np.unique(groups[members], return_counts=True)
            share = per_group / per_group.sum()
            stats.append({"rows": int(len(members)), "groups": int(len(per_group)),
                          "max_group_share": float(share.max()),
                          "effective_vectors": float(1.0 / (share ** 2).sum())})
        out[p] = stats
    return out


def _shared_groups(ids, groups):
    sets = {p: set(groups[ids[p]].tolist()) for p in PARTS}
    return {f"{a}/{b}": len(sets[a] & sets[b]) for i, a in enumerate(PARTS) for b in PARTS[i + 1:]}


def _labels_hash(y):
    return digest([int(v) for v in y])


def _labels_from_partitions(arrays, n_rows):
    y = np.full(n_rows, -1, dtype=np.int64)
    for p in PARTS:
        y[arrays[p]["ids"]] = arrays[p]["y"]
    return y


def build_comparison(old_dir, new_dir, before_path=None):
    old, new = _read(Path(old_dir) / "split_m2.json"), _read(Path(new_dir) / "split_m2.json")
    m2.verify_split_document(old)
    m2.verify_split_document(new)
    if old["dataset"] != new["dataset"]:
        raise m2.DataIntegrityError("old and new splits describe different datasets")
    dataset = new["dataset"]
    raw, y = m2.read_table(dataset)
    groups = m2.group_hashes(raw)
    names, k = dataset["class_names"], dataset["classes"]
    old_ids, new_ids = _ids(old), _ids(new)
    new_arrays = m2.load_arrays(new, _read(Path(new_dir) / "scaler_minmax.json"), PARTS,
                                purpose="range_audit", table=(raw, y))
    labels_after = _labels_from_partitions(new_arrays, len(y))
    cover = np.sort(np.concatenate([new_ids[p] for p in PARTS]))
    old_stats = _class_stats(old_ids, y, groups, k)
    new_stats = _class_stats(new_ids, y, groups, k)
    gd = new.get("group_diversity", {})
    floors = gd.get("chosen_floors", {})
    classes = []
    for c, name in enumerate(names):
        entry = {"class_id": c, "name": name, "old": {}, "new": {}}
        for p in PARTS:
            entry["old"][p] = old_stats[p][c]
            entry["new"][p] = {**new_stats[p][c], "floor": floors.get(p, {}).get(name)}
        entry["new_weak_evidence"] = any(new_stats[p][c]["groups"] < WEAK_VECTORS
                                         for p in ("validation", "test"))
        classes.append(entry)
    totals = {label: {p: {"rows": len(ids[p]), "groups": len(set(groups[ids[p]].tolist()))}
                      for p in PARTS} for label, ids in (("old", old_ids), ("new", new_ids))}
    old_files = {name: sha256(Path(old_dir) / name) for name in m2.ARTIFACTS}
    before_match = None
    if before_path is not None and Path(before_path).is_file():
        rec = _read(before_path)["files"]
        rel = Path(old_dir).resolve().relative_to(ROOT).as_posix()
        recorded = {n: rec.get(f"{rel}/{n}") for n in old_files}
        before_match = {"matches": recorded == old_files, "recorded": recorded}
    return {
        "schema_version": 1, "old_dir": str(Path(old_dir).as_posix()), "new_dir": str(Path(new_dir).as_posix()),
        "uses": "feature vectors and labels only; no model was constructed or run",
        "old_manifest_sha256": old["manifest_sha256"], "new_manifest_sha256": new["manifest_sha256"],
        "chosen_rung": gd.get("chosen_rung"), "new_split_seed": new["split_seed"],
        "weak_evidence_threshold_distinct_vectors": WEAK_VECTORS,
        "totals": totals,
        "classes_with_one_distinct_train_vector": {
            "old": [e["name"] for e in classes if e["old"]["train"]["groups"] == 1],
            "new": [e["name"] for e in classes if e["new"]["train"]["groups"] == 1]},
        "classes_with_weak_evidence_new": [e["name"] for e in classes if e["new_weak_evidence"]],
        "classes": classes,
        "shared_groups": {"old": _shared_groups(old_ids, groups), "new": _shared_groups(new_ids, groups)},
        "rows_preserved": bool(np.array_equal(cover, np.arange(len(y)))),
        "labels_sha256": {"before": _labels_hash(y), "after": _labels_hash(labels_after)},
        "labels_unchanged": bool(np.array_equal(y, labels_after)),
        "all_classes_in_every_partition": all(
            new_stats[p][c]["rows"] > 0 for p in PARTS for c in range(k)),
        "old_artifact_sha256": old_files,
        "old_artifacts_equal_before_snapshot": before_match,
    }


def comparison_markdown(cmp):
    t = cmp["totals"]
    lines = ["# Split comparison: revision 1 (milestone2a) versus revision 2", "",
             "Features and labels only; no model was run. Counts are rows and distinct feature-vector "
             "groups per class (a conflicting-label group counts toward each of its classes).", "",
             f"- Old manifest: `{cmp['old_manifest_sha256']}`", f"- New manifest: `{cmp['new_manifest_sha256']}` "
             f"(rung {cmp['chosen_rung']}, seed {cmp['new_split_seed']})",
             f"- Rows preserved: {cmp['rows_preserved']}; labels unchanged: {cmp['labels_unchanged']} "
             f"(SHA-256 before `{cmp['labels_sha256']['before']}`, after `{cmp['labels_sha256']['after']}`)",
             f"- Shared groups old: {cmp['shared_groups']['old']}; new: {cmp['shared_groups']['new']}",
             f"- Classes with exactly one distinct train vector: old {len(cmp['classes_with_one_distinct_train_vector']['old'])}, "
             f"new {len(cmp['classes_with_one_distinct_train_vector']['new'])}",
             f"- Old artifact hashes equal the pre-work snapshot: "
             f"{(cmp['old_artifacts_equal_before_snapshot'] or {}).get('matches')}", "",
             "| Partition | Old rows | Old groups | New rows | New groups |", "|---|---|---|---|---|"]
    for p in PARTS:
        lines.append(f"| {p} | {t['old'][p]['rows']} | {t['old'][p]['groups']} | "
                     f"{t['new'][p]['rows']} | {t['new'][p]['groups']} |")
    lines += ["", "Per class, rows / distinct groups, train | validation | test. `Floor` is the new rung floor "
              "(train/val/test); `Share` is the largest group's share of the class's train rows; `EffV` is the "
              "effective vector count 1 / sum(p_g^2) of the class's train rows (reported, not constrained). "
              f"Weak = fewer than {cmp['weak_evidence_threshold_distinct_vectors']} distinct vectors in "
              "validation or test.", "",
              "| Class | Name | Old tr | Old va | Old te | New tr | New va | New te | Floor | Share | EffV | Weak |",
              "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for e in cmp["classes"]:
        o, n = e["old"], e["new"]
        cell = lambda s: f"{s['rows']} / {s['groups']}"
        floor = "/".join(str(n[p]["floor"]) for p in PARTS)
        lines.append(f"| {e['class_id']} | {e['name']} | {cell(o['train'])} | {cell(o['validation'])} | "
                     f"{cell(o['test'])} | {cell(n['train'])} | {cell(n['validation'])} | {cell(n['test'])} | "
                     f"{floor} | {n['train']['max_group_share']:.3f} | {n['train']['effective_vectors']:.2f} | "
                     f"{'yes' if e['new_weak_evidence'] else ''} |")
    lines.append("")
    return "\n".join(lines)


def build_freeze(new_dir, config_path):
    new_dir = Path(new_dir)
    split = _read(new_dir / "split_m2.json")
    m2.verify_split_document(split)
    config_sha = sha256(config_path)
    if split["config_sha256"] != config_sha:
        raise m2.DataIntegrityError("split_m2.json was built from a different config file")
    artifacts = sorted(p.name for p in new_dir.iterdir() if p.is_file() and p.name != FREEZE)
    return {
        "schema_version": 1, "algorithm_version": split["algorithm"]["version"],
        "dataset_sha256": split["dataset"]["sha256"], "config_sha256": config_sha,
        "manifest_sha256": split["manifest_sha256"], "id_sha256": split["id_sha256"],
        "rung": split.get("group_diversity", {}).get("chosen_rung"), "split_seed": split["split_seed"],
        "artifact_sha256": {name: sha256(new_dir / name) for name in artifacts}}


def _write_new(path, text):
    if Path(path).exists():
        raise FileExistsError(f"Refusing to overwrite {path}")
    Path(path).write_text(text, encoding="utf-8", newline="\n")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--old", required=True, type=Path)
    parser.add_argument("--new", required=True, type=Path)
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/m2b/data_v2.json")
    parser.add_argument("--before", type=Path, default=ROOT / DEFAULT_BEFORE)
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s", stream=sys.stderr)
    out = args.out_dir
    for name in (*COMPARISON, FREEZE):
        if (out / name).exists():
            raise FileExistsError(f"Refusing to overwrite {out / name}")
    cmp = build_comparison(args.old, args.new, args.before)
    if not (cmp["rows_preserved"] and cmp["labels_unchanged"] and cmp["all_classes_in_every_partition"]
            and not any(cmp["shared_groups"]["new"].values())):
        raise m2.DataIntegrityError("new split fails the row, label, class or group-overlap check")
    if (cmp["old_artifacts_equal_before_snapshot"] or {}).get("matches") is False:
        raise m2.DataIntegrityError("old split artifacts differ from the pre-work snapshot")
    _write_new(out / COMPARISON[0], json.dumps(cmp, indent=2) + "\n")
    _write_new(out / COMPARISON[1], comparison_markdown(cmp))
    freeze = build_freeze(out, args.config)
    _write_new(out / FREEZE, json.dumps(freeze, indent=2) + "\n")
    print(json.dumps({"rung": freeze["rung"], "manifest_sha256": freeze["manifest_sha256"],
                      "one_train_vector_classes_new": cmp["classes_with_one_distinct_train_vector"]["new"],
                      "weak_evidence_classes": cmp["classes_with_weak_evidence_new"]}, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
