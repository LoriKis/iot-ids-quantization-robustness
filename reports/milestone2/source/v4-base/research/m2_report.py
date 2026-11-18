"""Milestone 2A report rendering: pilot_report.md and attack_results.csv from a finished pilot directory.

Pure presentation of files already written by research.m2_pilot; it computes nothing new except table layout.
Every report carries the framing sentence (protocol A) and the pilot non-scientific banner (protocol G).
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

BANNER = ("Pilot results are non-scientific development checks (1 seed, 2 epochs, small subsets, validation data only) "
          "and carry no accuracy or robustness conclusion.")
FRAMING = ("This is a new group-disjoint evaluation partition within a previously examined dataset, not a completely "
           "untouched external benchmark. The dataset was seen by Milestone 1; the earlier test partition was evaluated; "
           "identical feature vectors with conflicting labels remain in the data.")
ATTACK_NOTE = ("Feature-space attacks; no guarantee that perturbed vectors correspond to valid network traffic. "
               "Perturbations are tied to the original feature row (one 39-d delta broadcast to all 50 synthetic positions).")
CSV_COLUMNS = ("model_id", "attack_source", "target", "attack", "epsilon", "n", "clean_accuracy", "accuracy_under_attack",
               "robust_accuracy", "attack_success_rate", "max_linf", "attack_seconds", "valid", "note")


def _json(path: Path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _fmt(value, digits: int = 4) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def table(headers, rows) -> str:
    lines = ["| " + " | ".join(headers) + " |", "|" + "|".join("---" for _ in headers) + "|"]
    lines += ["| " + " | ".join(_fmt(c) for c in row) + " |" for row in rows]
    return "\n".join(lines)


def attack_results_csv(rows: list, path: Path | str) -> None:
    path = Path(path)
    if path.exists():
        raise FileExistsError(f"Refusing to overwrite {path}")
    with open(path, "w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=CSV_COLUMNS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _mb(n):
    return None if n is None else n / 1e6


def render_pilot_report(run: Path | str) -> str:
    run = Path(run)
    runtime, subsets = _json(run / "pilot_runtime.json"), _json(run / "subsets.json")
    clean, attacks = _json(run / "clean_eval.json"), _json(run / "attack_results.json")
    diag, budget = _json(run / "gradient_diagnostics.json"), _json(run / "attack_budget_check.json")
    eff, qat = _json(run / "efficiency/efficiency.json"), _json(run / "qat/qat_feasibility.json")
    logs = {v: _json(run / f"models/{v}/train_log.json") for v in ("V1", "V3")}
    conv = {v: _json(run / f"models/{v}/conversion.json") for v in ("V2", "V4")}
    out = ["# Milestone 2A pilot report", "", f"> **{BANNER}**", "", FRAMING, "", ATTACK_NOTE, ""]

    out += ["## Setup", "",
            f"- seed {runtime['seed']}, {runtime['epochs']} epochs, batch size {runtime['batch_size']}, "
            f"training threads {runtime['threads']['training']}, attack and evaluation threads {runtime['threads']['attack_and_evaluation']}",
            f"- split seed {subsets['split_seed']}, split manifest SHA-256 `{subsets['split_manifest_sha256']}`",
            f"- source digest `{runtime['provenance']['source_digest']}`, git revision `{runtime['provenance']['revision']}` (no commit made)",
            "- test partition: not loaded, not evaluated (guard phase `m2a` allows train and validation only)", "",
            table(("subset", "rows", "ID SHA-256"),
                  [(k, subsets[k]["n"], subsets[k]["id_sha256"][:16] + "...") for k in
                   ("train", "validation", "attack", "calibration", "e4_plumbing")]), ""]

    out += ["## Training (V1 ordinary, V3 adversarial)", "",
            table(("variant", "updates", "wall clock s", "mean step s (first discarded)", "final loss", "init state hash", "final state hash"),
                  [(v, l["optimizer_updates"], l["wall_clock_seconds"], sum(l["step_seconds"][1:]) / (len(l["step_seconds"]) - 1),
                    l["epoch_losses"][-1], l["init_state_hash"][:12], l["final_state_hash"][:12]) for v, l in logs.items()]), "",
            f"V1 and V3 share the same initial weights: **{logs['V1']['init_state_hash'] == logs['V3']['init_state_hash']}**; "
            f"equal optimizer updates: **{logs['V1']['optimizer_updates'] == logs['V3']['optimizer_updates']}**.", ""]
    cost = runtime["at_cost"]
    out += [f"AT extra compute: measured wall-clock ratio {_fmt(cost['measured_wall_clock_ratio'], 2)}x "
            f"(first step of each run discarded, {cost['steps_compared']} steps, indicative only); nominal pass-count ratio "
            f"{_fmt(cost['nominal_pass_count_ratio'], 2)}x. Counters per AT step: "
            f"{logs['V3']['pass_counters_per_step']}; first-step forward-hook counts on the real model: "
            f"{logs['V3']['first_step_hook_counts']}.", ""]

    out += ["## Clean evaluation on the validation subset", "",
            table(("model", "accuracy", "rows"), [(m, v["accuracy"], v["n"]) for m, v in clean["models"].items()]), "",
            "INT8 vs FP32 parent argmax agreement: " + ", ".join(f"{k} {_fmt(v)}" for k, v in clean["int8_agreement"].items()), ""]

    out += ["## INT8 post-training quantization evidence", ""]
    for v, c in conv.items():
        counts = c["parameter_counts"]
        out += [f"### {v} (from {'V1' if v == 'V2' else 'V3'})", "",
                f"- backend `{c['backend']}`, mode `{c['mode']}`, scope `{c['scope']}`, calibration {c['calibration']['n_rows']} train rows "
                f"(ID SHA-256 `{c['calibration']['id_sha256'][:16]}...`)",
                f"- islands converted: {len(c['requested'])} ({c['classification_path_islands']} classification path, "
                f"{c['anomaly_scorer_islands']} anomaly scorer); packed weights {c['structure_checks']['weight_dtype_inventory']} "
                f"{c['structure_checks']['weight_qscheme_inventory']}",
                f"- profiler quantized::linear calls on one forward: {c['structure_checks']['profiler']['quantized_linear_calls']}",
                f"- identity wrapper max abs error before observers: {c['architecture_wrapper_fp32_max_abs_error']}; "
                f"reload check: bitwise logits {c['reload_check']['logits_bitwise_equal']}, argmax identical {c['reload_check']['argmax_identical']} "
                f"over {c['reload_check']['rows']} rows",
                f"- INT8 weight elements {counts['int8_weight_elements']} of {counts['total_parameters']} parameters "
                f"({_fmt(100 * counts['int8_weight_share'], 2)} %); quant/dequant boundaries {c['quant_dequant_boundaries']['total']}", ""]
    out += ["Operations that remain FP32 (hybrid model, not integer-only):", ""]
    out += [f"- {op}" for op in conv["V2"]["fp32_remaining_operations"]] + [""]

    out += ["## Adversarial evaluation (validation attack subset)", "",
            f"Attack subset: {subsets['attack']['n']} rows. Invalid units: {len(attacks['invalid_units'])}. "
            "Sources are never merged; `bpda_int8` is the headline INT8 row (forward through the real INT8 model, approximate backward pass).", ""]
    for attack in ("fgsm", "pgd20"):
        for model_ids in (("V1", "V2"), ("V3", "V4")):
            rows = [r for r in attacks["rows"] if r["attack"] == attack and r["model_id"] in model_ids]
            grid = sorted({r["epsilon"] for r in rows})
            sources = []
            for r in rows:
                if (r["model_id"], r["attack_source"]) not in sources:
                    sources.append((r["model_id"], r["attack_source"]))
            body = []
            for model_id, source in sources:
                cells = {r["epsilon"]: r["robust_accuracy"] for r in rows if r["model_id"] == model_id and r["attack_source"] == source}
                body.append((f"{model_id} {source}", *[cells.get(e) for e in grid]))
            out += [f"### {attack}, models {model_ids[0]}/{model_ids[1]}: robust accuracy by epsilon", "",
                    table(("source", *[f"eps {e:g}" for e in grid]), body), ""]
    if attacks["union_rows"]:
        out += ["Per-example union rows (an example counts as robust only if no INT8 source broke it; separately labelled):", "",
                table(("target", "attack", "epsilon", "robust accuracy"),
                      [(u["target"], u["attack"], u["epsilon"], u["robust_accuracy"]) for u in attacks["union_rows"]]), ""]
    if attacks["invalid_units"]:
        out += ["Invalid units (failed an E5 check):", ""] + [f"- {u}" for u in attacks["invalid_units"]] + [""]

    out += ["### Gradient-masking diagnostics and surrogate fidelity", ""]
    for v in ("V2", "V4"):
        g, f = diag[v]["direct_int8_grad"], diag[v]["surrogate_fidelity"]
        out += [f"- {v} `direct_int8_grad`: {g['behaviour']}; undefined/zero fraction {_fmt(g['undefined_or_zero_fraction'])} "
                f"over {g['n']} rows; counted as robust: {g['counted_as_robust']}",
                f"- {v} surrogate: {f['note']}; max logit difference to the INT8 model {_fmt(f['max_logit_diff_to_int8'], 6)}; "
                f"argmax agreement {_fmt(f['argmax_agreement_with_int8'])}"]
    out += [""]

    out += ["## E4 budget-adequacy plumbing (not an adequacy decision)", "",
            f"{budget['note']}. Plumbing subset {budget['n_subset']} rows; threshold would be {budget['threshold']}. "
            f"`pgd20` stays provisional; adequacy is decided in campaign step 0 on fully trained checkpoints. "
            f"Real campaign E4 subset size for this split: {runtime['campaign_e4_subset']['n_subset']} rows "
            f"({runtime['campaign_e4_subset']['rule']}).", "",
            table(("model", "epsilon", "n_extra", "pgd20 s", "pgd50x5 s"),
                  [(c["model"], c["epsilon"], c["n_extra"], c["seconds_pgd20"], c["seconds_pgd50x5"]) for c in budget["cell_costs"]]), ""]

    out += ["## Efficiency (1 thread, fresh process per measurement)", "",
            f"Executed order: {', '.join(e['variant'] + ('*' if e['retry'] else '') for e in eff['run_order'])} "
            f"(interleaved: {eff['interleaved']}, retries flagged *). Reduced pilot repetitions: {eff['repetitions']}; "
            f"warm-up/timed {eff['warmup_and_timed']}.", "",
            table(("variant", "state_dict MB", "bs1 median ms", "bs1 p95 ms", "bs256 median ms", "bs256 samples/s",
                   "baseline RSS MB", "peak wset - baseline MB"),
                  [(v, _mb(s["state_dict_bytes"]["median"]), s["batch1_median_ms"]["median"], s["batch1_p95_ms"]["median"],
                    s["batch256_median_ms"]["median"], s["batch256_throughput_samples_per_s"]["median"],
                    _mb(s["baseline_rss_bytes"]["median"]),
                    _mb(s["peak_wset_minus_baseline_bytes"]["median"] if s["peak_wset_minus_baseline_bytes"] else None))
                   for v, s in eff["summary"].items()]), "",
            eff["no_claim"], "Process-wide RSS is coarse and includes allocator slack and imported packages.", ""]

    out += ["## QAT feasibility (optional V5, bounded check)", "",
            f"- status **{qat['status']}**, stage `{qat['stage']}`, {qat['epochs']} epoch on {qat['train_rows']} train rows, "
            f"{_fmt(qat['seconds'], 1)} s",
            f"- error: {qat['error']}; module: {qat['module']}",
            f"- packed dtypes: {qat['packed_dtypes']['weight_dtype'] if qat['packed_dtypes'] else None}; "
            f"INT8 inference: {qat.get('int8_inference')}",
            f"- {qat['claim']}", ""]

    out += ["## Phase timings (seconds)", "", table(("phase", "seconds"), list(runtime["phases"].items())), "",
            f"Peak sampled RSS of the pilot process: {_fmt(_mb(runtime['peak_rss_bytes']), 1)} MB; "
            f"minimum available RAM seen: {_fmt(_mb(runtime['min_available_ram_bytes']), 1)} MB.", "",
            f"> **{BANNER}**", ""]
    return "\n".join(out)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--run", required=True, type=Path, help="finished pilot directory")
    parser.add_argument("--out", required=True, type=Path, help="markdown file to write (never overwritten)")
    args = parser.parse_args(argv)
    if args.out.exists():
        raise FileExistsError(f"Refusing to overwrite {args.out}")
    args.out.write_text(render_pilot_report(args.run), encoding="utf-8")
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
