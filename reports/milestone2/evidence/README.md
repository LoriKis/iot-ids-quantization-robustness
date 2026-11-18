# Milestone 2 evidence index

Everything under `deadline-v1/` and `v4-base/` is a **byte-identical copy** of a file from the two read-only VM
folders. Nothing was edited. This README, the folder layout and the file renames listed below are post-run
documentation; the copied files are the executed evidence.

- `deadline-v1/` = outputs of the completed campaign `m2-deadline-v1` (VM folder `IOT-network-traffic-DL-M2-VM-deadline-v1`,
  path `artifacts/m2-deadline-v1/...`). Same relative paths as on the VM, minus the `artifacts/m2-deadline-v1/` prefix.
- `v4-base/` = frozen split, scaler, QAT pilot and package provenance from the V4 base package
  (VM folder `IOT-network-traffic-DL-M2-VM-v4`).

The table at the end lists every copied file with its VM path, size and SHA-256. The same byte-for-byte comparison was run once during the post-run work against the VM originals by a PowerShell script that needs both VM folders and the full VM file manifests of the setup step. That script is not published
because a teammate cannot run it without those inputs; its saved output is [`copy_check.txt`](../../../verification/results/copy_check.txt) (path map: [`copy_map.tsv`](../../../verification/results/copy_map.tsv)).
Without the VM folders, the SHA-256 column below is the reference.

## Where to find each piece of evidence

| What you want to check | File |
|---|---|
| Per-example predictions for all 3,764 held-out test ids (`example_id, y_true, in_robust_subset, V1_pred..V4_pred`) | [`deadline-v1/test/predictions_full_test.csv`](deadline-v1/test/predictions_full_test.csv) |
| Stored clean accuracy / macro-F1 | [`deadline-v1/test/clean_results.json`](deadline-v1/test/clean_results.json) |
| Per-example predictions on the 270-id robustness subset (model, attack, epsilon, id, label, prediction) | [`deadline-v1/robustness/predictions_subset.csv`](deadline-v1/robustness/predictions_subset.csv) |
| Stored robustness scores | [`deadline-v1/robustness/robustness_results.json`](deadline-v1/robustness/robustness_results.json), [`deadline-v1/results/robustness_curves.csv`](deadline-v1/results/robustness_curves.csv) |
| Per-unit attack arrays and attack checks (20 units) | `deadline-v1/robustness/V*/<unit>/` |
| Four-model table / percentage-point differences | [`deadline-v1/results/model_comparison.csv`](deadline-v1/results/model_comparison.csv), [`.md`](deadline-v1/results/model_comparison.md), [`pp_differences.json`](deadline-v1/results/pp_differences.json) |
| Raw efficiency measurements and settings (12 child processes) | `deadline-v1/efficiency/results/01..12.json`, `deadline-v1/efficiency/specs/01..12.json`, [`efficiency.json`](deadline-v1/efficiency/efficiency.json), [`fixed_rows.json`](deadline-v1/efficiency/fixed_rows.json) |
| Frozen protocol and robustness-subset identity | [`deadline-v1/protocol/`](deadline-v1/protocol/protocol.json) |
| Frozen configuration | [`../source/deadline-v1-overlay/configs/m2_deadline_v1.json`](../source/deadline-v1-overlay/configs/m2_deadline_v1.json), and V4 configs under [`../source/v4-base/configs/`](../source/v4-base/configs/) |
| Split and scaler identity | [`v4-base/reports/milestone2b/split_v2/split_m2.json`](v4-base/reports/milestone2b/split_v2/split_m2.json), [`scaler_minmax.json`](v4-base/reports/milestone2b/split_v2/scaler_minmax.json), [`split_report.md`](v4-base/reports/milestone2b/split_v2/split_report.md) |
| Calibration ids (540 training rows) | [`deadline-v1/calibration/calibration_ids.json`](deadline-v1/calibration/calibration_ids.json) |
| Training logs, parent link and PTQ conversion records | `deadline-v1/models/V*/train_log.json`, `deadline_meta.json`, `conversion.json`, `ckpt/latest.json` |
| Completion records | `deadline-v1/stage_markers/*.ok` (10), [`run_identity.json`](deadline-v1/run_identity.json), `validation/val_check.json`, `timing/*.json`, `logs/*.log.txt` |
| QAT pilot (separate, earlier pilot) | [`v4-base/m2a-pilot/qat/qat_feasibility.json`](v4-base/m2a-pilot/qat/qat_feasibility.json) and the other files under [`v4-base/m2a-pilot/`](v4-base/m2a-pilot/pilot_report.md) |
| Package provenance | [`v4-base/provenance/`](v4-base/provenance/PACKAGE_IDENTITY.json) |

## Changes forced by the repository layout (names only, bytes unchanged)

- `logs/*.log` on the VM are stored as `logs/*.log.txt` (and `output/m2b-v1/logs/*.log` as `.log.txt`) because the repository
  `.gitignore` ignores `*.log`. The content is identical; the table below shows the VM name.
- The VM folder `artifacts/m2a-pilot/` is stored as `v4-base/m2a-pilot/` and `artifacts/m2b-integrity/` as
  `v4-base/provenance/m2b-integrity/` because the repository `.gitignore` ignores every directory named `artifacts`.
- `../.gitattributes` (`* -text`, new file) keeps copied CRLF files byte-for-byte despite the repository's `eol=lf` rule.

## Deliberately NOT copied (kept outside Git) and how to get or locate them

| Item | Why excluded | Where it is / how to obtain it |
|---|---|---|
| Model checkpoints: `models/V1/checkpoint.pt`, `models/V3/checkpoint.pt` (1,450,583 B each), `models/V2/state.pt`, `models/V4/state.pt` (499,961 B each), `models/V*/ckpt/*.pt` (per-epoch, ~4.3 MB each) | binary model files; `.gitignore` ignores `*.pt` | VM folder `IOT-network-traffic-DL-M2-VM-deadline-v1/artifacts/m2-deadline-v1/models/`. SHA-256 recorded in the stage markers and below. Never loaded for this review. POST-RUN: the VM files were hashed directly and every recorded hash matches ([`vm_hash_check_postrun.tsv`](../../../verification/results/vm_hash_check_postrun.tsv); needs the VM folder to repeat). |
| `calibration/calibration_x.pt` | tensor of the 540 calibration rows (4,213,148 B) | same VM folder, `calibration/`. SHA-256 `ceb7c616306b1645969ec08fdb1015a12fb2b5a292421cc9a2c55e35b3d2c024` (matches `ptq.ok`). The row ids are in `calibration_ids.json`. |
| Dataset `balanced_CICIoT.csv` (29,997 rows, SHA-256 `0d869e334cc9a2795c96996fbde8fb04a8acac61f5e67c5d46aee2b0717a53ed`) | data stays outside Git | PAVE artifact data; see [`docs/SETUP.md`](../../../docs/SETUP.md) of the repository for how to obtain and hash it. A copy sits in the V4 VM folder under `upstream/PAVE_artifact/data/`; it was not copied here. |
| `v4-base/.../split_v2/group_manifest.json` (2,340,043 B, SHA-256 `12fbb4f607b24686adbc60c6b41a4d831596edfc6963583cbbf87b7c568605f9`) | 2.3 MB, redundant with `split_m2.json` | V4 VM folder, `reports/milestone2b/split_v2/`. Its hash is recorded in `split_freeze.json`. |
| `.mplconfig/`, `timing/probe_models/`, `ckpt/old/`, `m2a-pilot/attacks/`, `m2a-pilot/efficiency/`, `output/m2b-v1-smoke` and virtual environments, `upstream/` PAVE data | caches, timing probes, pilot sweeps, environments | VM folders. The smoke run's Stage 04 did not pass and is not used by the campaign (see [`../source/deadline-v1-overlay/HANDOFF.md`](../source/deadline-v1-overlay/HANDOFF.md) section 4). |
| Other V4 `reports/milestone1-*` and docs that differ from the repository only by line endings or are M1 material | M1 stays exactly as submitted | repository M1 paths |

## Hash table of the copied files

| Clone path (relative to `reports/milestone2/`) | VM path | Bytes | SHA-256 (copy = VM file) |
|---|---|---:|---|
| `evidence/deadline-v1/test/predictions_full_test.csv` | `artifacts/m2-deadline-v1/test/predictions_full_test.csv` | 92989 | `b5f2a97a6ebef9605e27d923be56219145a49e58d6b5bbf799ee28af404f44a4` |
| `evidence/deadline-v1/test/clean_results.json` | `artifacts/m2-deadline-v1/test/clean_results.json` | 812 | `dc586912c8bc203a73540d1f8a08c000963c219945f7dec8b23ebc70fddd0a00` |
| `evidence/deadline-v1/test/robustness_subset_rows.npz` | `artifacts/m2-deadline-v1/test/robustness_subset_rows.npz` | 47168 | `e41fdb5809c85c5674b55ac86ea53a54139bd8e2b91be39e21999395477fccd6` |
| `evidence/deadline-v1/robustness/predictions_subset.csv` | `artifacts/m2-deadline-v1/robustness/predictions_subset.csv` | 126602 | `20b4eb334c499fd85006326676a3bc4e2d59aeb539a9dfa2d0b35ca84bc5c273` |
| `evidence/deadline-v1/robustness/robustness_results.json` | `artifacts/m2-deadline-v1/robustness/robustness_results.json` | 4662 | `f7282357f90884a92df99f3e0388b37b62f2b8b44401c90064520a69a3522955` |
| `evidence/deadline-v1/results/model_comparison.csv` | `artifacts/m2-deadline-v1/results/model_comparison.csv` | 1059 | `4aa374adfc2ae0008cb72d89c12b91e4596e67a60884fa0545831380c3c9baf2` |
| `evidence/deadline-v1/results/model_comparison.md` | `artifacts/m2-deadline-v1/results/model_comparison.md` | 2006 | `e05821a0f0f5ff2869e89a81fd3580cf5e2a4fbe3acd7605ce199d0080f926c1` |
| `evidence/deadline-v1/results/pp_differences.json` | `artifacts/m2-deadline-v1/results/pp_differences.json` | 11762 | `151931a1262eb595fedae9f1a9fb82dc68bedeb4c9424ad1fcdfc5ca10c057f9` |
| `evidence/deadline-v1/results/robustness_curves.csv` | `artifacts/m2-deadline-v1/results/robustness_curves.csv` | 2710 | `eb1a583f25c51d82af0e6cb5ae03fbb28c8930556f006c7917280baca2d93405` |
| `evidence/deadline-v1/efficiency/efficiency.json` | `artifacts/m2-deadline-v1/efficiency/efficiency.json` | 95836 | `6a614000675dd2ffeb45a5914a7f80864a043f293b77e1be10b2688d93fc400a` |
| `evidence/deadline-v1/efficiency/fixed_rows.json` | `artifacts/m2-deadline-v1/efficiency/fixed_rows.json` | 3037 | `c04e03e798c2f99f3394fe10e711ce24a5ec345cd69e8aecba13a61b9010c02a` |
| `evidence/deadline-v1/efficiency/fixed_rows.npy` | `artifacts/m2-deadline-v1/efficiency/fixed_rows.npy` | 40064 | `0adb2960b5c29a18c3811107d439bf73c60910cd151de0c85ec2b97f7cc405df` |
| `evidence/deadline-v1/protocol/protocol.json` | `artifacts/m2-deadline-v1/protocol/protocol.json` | 4245 | `62adf121b39054f8109fd7c89a2c3577a4303f9f1610d60742521fd7dcf47b36` |
| `evidence/deadline-v1/protocol/protocol.frozen.json` | `artifacts/m2-deadline-v1/protocol/protocol.frozen.json` | 132 | `ab4be7d39508372414c11ae4dd9b16635751afd2f64fe95daf6fee55da5efd8e` |
| `evidence/deadline-v1/protocol/protocol.json.sha256` | `artifacts/m2-deadline-v1/protocol/protocol.json.sha256` | 80 | `8a85338b3e4322fb445cb9e16c0edebd8b0f10560aaa823ddd20ec114b145138` |
| `evidence/deadline-v1/protocol/robustness_subset_counts.json` | `artifacts/m2-deadline-v1/protocol/robustness_subset_counts.json` | 2187 | `ed6a07f4e3449ea4a94182a55bc96e7032a890e10d7f629803843996670235fb` |
| `evidence/deadline-v1/protocol/robustness_subset_ids.csv` | `artifacts/m2-deadline-v1/protocol/robustness_subset_ids.csv` | 2233 | `dbafdd5afa486e08fa48b8f211eedcc193bb17238ec50dbea4ead4f17cf6f919` |
| `evidence/deadline-v1/calibration/calibration_ids.json` | `artifacts/m2-deadline-v1/calibration/calibration_ids.json` | 6234 | `de11d3a7090edd8e25b4697e31d761279dc224e58c38e53fed736ee5a14400b8` |
| `evidence/deadline-v1/run_identity.json` | `artifacts/m2-deadline-v1/run_identity.json` | 1118 | `c2edd73431203beb822ecb75186bf44a96e432e2ed8800671279d23a3b961126` |
| `evidence/deadline-v1/validation/val_check.json` | `artifacts/m2-deadline-v1/validation/val_check.json` | 1050 | `2f4ba7642f121eecff64e7383c013f81b6c7313b69d7b8e52becca00e93b6601` |
| `evidence/deadline-v1/timing/timing_estimate.json` | `artifacts/m2-deadline-v1/timing/timing_estimate.json` | 6212 | `52d0b29d97e26397d29d29e5a2b4eb63b57c2a45bd9feb8c277b60f2d694c549` |
| `evidence/deadline-v1/timing/budget_lock.json` | `artifacts/m2-deadline-v1/timing/budget_lock.json` | 510 | `1b1ced84ab3a68569c2ace4654b468c6679926dc7546bf7752da4fcfd72f6ccb` |
| `evidence/deadline-v1/models/V1/train_log.json` | `artifacts/m2-deadline-v1/models/V1/train_log.json` | 28411 | `b44957afb258ea2b2366864a9dbfd06ef764eb79e195420206fa8d201e71ac37` |
| `evidence/deadline-v1/models/V3/train_log.json` | `artifacts/m2-deadline-v1/models/V3/train_log.json` | 53468 | `7f04f1d2b71d557fc7546c34f13ade3e5ab9bb937cee8b1a584df80d6a26d2b9` |
| `evidence/deadline-v1/models/V2/conversion.json` | `artifacts/m2-deadline-v1/models/V2/conversion.json` | 26162 | `f4bcc23844cc6c3813c70722fd22753085748996f9e7a72da486a4cfff21d4cd` |
| `evidence/deadline-v1/models/V4/conversion.json` | `artifacts/m2-deadline-v1/models/V4/conversion.json` | 26166 | `97a457e8c48de067e66c90e90e6a73b9f67a35da4f1918609a430f4c11b35884` |
| `evidence/deadline-v1/models/V1/ckpt/latest.json` | `artifacts/m2-deadline-v1/models/V1/ckpt/latest.json` | 124 | `96845ede53e1044381c2b18977cbac324a37866d4032ea08a33d451bd47ed74a` |
| `evidence/deadline-v1/models/V3/ckpt/latest.json` | `artifacts/m2-deadline-v1/models/V3/ckpt/latest.json` | 124 | `aa9da07c11b6bba3c6dae2dcaccf7eb2fc898a95a885bcd535a7e3fb3864b02d` |
| `evidence/deadline-v1/models/V1/deadline_meta.json` | `artifacts/m2-deadline-v1/models/V1/deadline_meta.json` | 608 | `b7aa46c59bb909156ac80a0fd05c68a8c3ec198d1ac07e1dd54b4c895708a06c` |
| `evidence/deadline-v1/models/V2/deadline_meta.json` | `artifacts/m2-deadline-v1/models/V2/deadline_meta.json` | 606 | `3d2433c153d77c7dac34e928b44ae97c1bbc43f43da908352b23a4e829a318e2` |
| `evidence/deadline-v1/models/V3/deadline_meta.json` | `artifacts/m2-deadline-v1/models/V3/deadline_meta.json` | 607 | `d1f6f6fb2c02a347469748f9b8ad1a750b9df732d52d1ee6f9c26b9b1dfbad30` |
| `evidence/deadline-v1/models/V4/deadline_meta.json` | `artifacts/m2-deadline-v1/models/V4/deadline_meta.json` | 605 | `16176dbad8875c19e5693cc4a1ac0c393a53c6716a7e4d889f512764276df859` |
| `evidence/deadline-v1/robustness/V1/clean_eps0.0/arrays.npz` | `artifacts/m2-deadline-v1/robustness/V1/clean_eps0.0/arrays.npz` | 11484 | `b45e606535a79aeefa6f9f0e50dcf14ccff1feda7b2cdac70bf2506a25ef4256` |
| `evidence/deadline-v1/robustness/V1/clean_eps0.0/attack_checks.json` | `artifacts/m2-deadline-v1/robustness/V1/clean_eps0.0/attack_checks.json` | 1569 | `511f5d3c56788b6fbea39fcbe849900952d9e629d46e02adb14944284bd7cf0c` |
| `evidence/deadline-v1/robustness/V1/clean_eps0.0/attack_result.json` | `artifacts/m2-deadline-v1/robustness/V1/clean_eps0.0/attack_result.json` | 826 | `a03588599507bfea188004e74cfea0a43af03196921121227c5de72ef58d0dd8` |
| `evidence/deadline-v1/robustness/V1/clean_eps0.0/unit_done.json` | `artifacts/m2-deadline-v1/robustness/V1/clean_eps0.0/unit_done.json` | 854 | `8399016b736a8c0d5c43dc6887aa10e00d37668ff93bb88ea1054ab415a52253` |
| `evidence/deadline-v1/robustness/V2/clean_eps0.0/arrays.npz` | `artifacts/m2-deadline-v1/robustness/V2/clean_eps0.0/arrays.npz` | 11484 | `313b86d3d5acd2b165184a5f774bd7d221316136437d450b22d7c62db940969a` |
| `evidence/deadline-v1/robustness/V2/clean_eps0.0/attack_checks.json` | `artifacts/m2-deadline-v1/robustness/V2/clean_eps0.0/attack_checks.json` | 1618 | `a16e5918c9ebb95e2dfe684af1ab9cf009eb8de7c6d8c795b2812197b91f1220` |
| `evidence/deadline-v1/robustness/V2/clean_eps0.0/attack_result.json` | `artifacts/m2-deadline-v1/robustness/V2/clean_eps0.0/attack_result.json` | 939 | `4ffc335aa4eecbb53e6cf1c7effe75832b7c21e9392988a5669fd9d7092750d6` |
| `evidence/deadline-v1/robustness/V2/clean_eps0.0/unit_done.json` | `artifacts/m2-deadline-v1/robustness/V2/clean_eps0.0/unit_done.json` | 852 | `af12143daf9a2af6ea42db793c31f618412ecaac872e3453b1f9e815ecbf67da` |
| `evidence/deadline-v1/robustness/V3/clean_eps0.0/arrays.npz` | `artifacts/m2-deadline-v1/robustness/V3/clean_eps0.0/arrays.npz` | 11484 | `2d39a9decdce9f3ab1bdee0cac421244b68261b462fca08ff798bc8aa46c3e52` |
| `evidence/deadline-v1/robustness/V3/clean_eps0.0/attack_checks.json` | `artifacts/m2-deadline-v1/robustness/V3/clean_eps0.0/attack_checks.json` | 1569 | `4dd33c35273674009c2d92447cd365d94c3ea76e146f1cbad6581745ab204ce9` |
| `evidence/deadline-v1/robustness/V3/clean_eps0.0/attack_result.json` | `artifacts/m2-deadline-v1/robustness/V3/clean_eps0.0/attack_result.json` | 826 | `d49e2caa33344f8d3fd4abeabb3e939476bdc46d963db9b3bea22d2ad85cce0b` |
| `evidence/deadline-v1/robustness/V3/clean_eps0.0/unit_done.json` | `artifacts/m2-deadline-v1/robustness/V3/clean_eps0.0/unit_done.json` | 854 | `9964f98233d1f59b71600e50f282a1384be0c8256b1e7159e6b3f2a1c1b76295` |
| `evidence/deadline-v1/robustness/V4/clean_eps0.0/arrays.npz` | `artifacts/m2-deadline-v1/robustness/V4/clean_eps0.0/arrays.npz` | 11484 | `108ad61b1f700a7cd4eb3517e655bbdc308d13155d9501a27b1fc3964e2df671` |
| `evidence/deadline-v1/robustness/V4/clean_eps0.0/attack_checks.json` | `artifacts/m2-deadline-v1/robustness/V4/clean_eps0.0/attack_checks.json` | 1618 | `55023a8147ebcd28444e7bf736f32febeba81439f3fdcb1abd8a801ed441df28` |
| `evidence/deadline-v1/robustness/V4/clean_eps0.0/attack_result.json` | `artifacts/m2-deadline-v1/robustness/V4/clean_eps0.0/attack_result.json` | 939 | `ace39edb0282ec7138afa41d92fa563f9fdffff01b8bb838fb8ba9d4abecd1ac` |
| `evidence/deadline-v1/robustness/V4/clean_eps0.0/unit_done.json` | `artifacts/m2-deadline-v1/robustness/V4/clean_eps0.0/unit_done.json` | 837 | `54717ee00bb52da7c838f300c2853a3a83217353633787091bf8466528ce9420` |
| `evidence/deadline-v1/robustness/V1/fgsm_eps0.05/arrays.npz` | `artifacts/m2-deadline-v1/robustness/V1/fgsm_eps0.05/arrays.npz` | 11484 | `67f3a71762df17575fcfa4dd7b57044f0fdecb236d1404ce1ef1b9d6c6504a4a` |
| `evidence/deadline-v1/robustness/V1/fgsm_eps0.05/attack_checks.json` | `artifacts/m2-deadline-v1/robustness/V1/fgsm_eps0.05/attack_checks.json` | 1570 | `2ace49100d9353e3b7c315bbb119750c404495012d5549fd0c32d12045c0da70` |
| `evidence/deadline-v1/robustness/V1/fgsm_eps0.05/attack_result.json` | `artifacts/m2-deadline-v1/robustness/V1/fgsm_eps0.05/attack_result.json` | 858 | `99fda52a5c9cdad2aa3a132a24c85fae86a5f4ed896710aed16c03856fcf713c` |
| `evidence/deadline-v1/robustness/V1/fgsm_eps0.05/unit_done.json` | `artifacts/m2-deadline-v1/robustness/V1/fgsm_eps0.05/unit_done.json` | 671 | `5c3b39555d651dce84a5e2402922e2b5e1e9f6dff09e20a2a44e08dcc149d8c6` |
| `evidence/deadline-v1/robustness/V2/fgsm_eps0.05/arrays.npz` | `artifacts/m2-deadline-v1/robustness/V2/fgsm_eps0.05/arrays.npz` | 11484 | `2ac59884a119da770c949fc57f8d81995e94dc2e3bdfe5db870f0ef07fc5a94b` |
| `evidence/deadline-v1/robustness/V2/fgsm_eps0.05/attack_checks.json` | `artifacts/m2-deadline-v1/robustness/V2/fgsm_eps0.05/attack_checks.json` | 1623 | `1406f4cfe497a67ae247ea8143501db1fe178b64e42860a596e4a9de044340ed` |
| `evidence/deadline-v1/robustness/V2/fgsm_eps0.05/attack_result.json` | `artifacts/m2-deadline-v1/robustness/V2/fgsm_eps0.05/attack_result.json` | 973 | `1a2dc6089cc868517cc71b795fc0dccb2356775cc2a3232a10bdc076f339cedc` |
| `evidence/deadline-v1/robustness/V2/fgsm_eps0.05/unit_done.json` | `artifacts/m2-deadline-v1/robustness/V2/fgsm_eps0.05/unit_done.json` | 670 | `892fd57c27fe140372cdf8b05e85c4c470c7f4344e442bb9ca6c069f35839b58` |
| `evidence/deadline-v1/robustness/V3/fgsm_eps0.05/arrays.npz` | `artifacts/m2-deadline-v1/robustness/V3/fgsm_eps0.05/arrays.npz` | 11484 | `fb57d6a0ecd29a700db298b837bbd3e29d841a7d488a844f068be9234ba087bf` |
| `evidence/deadline-v1/robustness/V3/fgsm_eps0.05/attack_checks.json` | `artifacts/m2-deadline-v1/robustness/V3/fgsm_eps0.05/attack_checks.json` | 1570 | `dc0fde3b0c80426341f6213899ded990413fbf3d642d435ce075ee410ff9b48e` |
| `evidence/deadline-v1/robustness/V3/fgsm_eps0.05/attack_result.json` | `artifacts/m2-deadline-v1/robustness/V3/fgsm_eps0.05/attack_result.json` | 860 | `e1b72516b074d255425986f2cbfe7cbca0ef85d3bd2d68f10d3eb992f3bb250d` |
| `evidence/deadline-v1/robustness/V3/fgsm_eps0.05/unit_done.json` | `artifacts/m2-deadline-v1/robustness/V3/fgsm_eps0.05/unit_done.json` | 673 | `c6204b102930a1a785439a52df4349c1c823454d87f1d5be422282b14d4ed3c5` |
| `evidence/deadline-v1/robustness/V4/fgsm_eps0.05/arrays.npz` | `artifacts/m2-deadline-v1/robustness/V4/fgsm_eps0.05/arrays.npz` | 11484 | `4d00bac12073e72d3ca38a94ca24218772b604524e9a6c32c81ae5f820d80898` |
| `evidence/deadline-v1/robustness/V4/fgsm_eps0.05/attack_checks.json` | `artifacts/m2-deadline-v1/robustness/V4/fgsm_eps0.05/attack_checks.json` | 1623 | `1e94c790f6dbb33ebc9ad579a95a567289eb20f5a32ecff84990168e4c237f63` |
| `evidence/deadline-v1/robustness/V4/fgsm_eps0.05/attack_result.json` | `artifacts/m2-deadline-v1/robustness/V4/fgsm_eps0.05/attack_result.json` | 975 | `f0fd8c1b36bde068444c3896e1761d4643b6ab21f13c80628f83bc4a62d5622e` |
| `evidence/deadline-v1/robustness/V4/fgsm_eps0.05/unit_done.json` | `artifacts/m2-deadline-v1/robustness/V4/fgsm_eps0.05/unit_done.json` | 670 | `c03b8fd2c130ba0ac628850a2b819a14a1e4394416f954cc623bb9fcecb18d85` |
| `evidence/deadline-v1/robustness/V1/fgsm_eps0.1/arrays.npz` | `artifacts/m2-deadline-v1/robustness/V1/fgsm_eps0.1/arrays.npz` | 11484 | `98d6bffb640870558d2ba8609117bf175ce41924cd0e9b6bdba213a9d2307270` |
| `evidence/deadline-v1/robustness/V1/fgsm_eps0.1/attack_checks.json` | `artifacts/m2-deadline-v1/robustness/V1/fgsm_eps0.1/attack_checks.json` | 1569 | `3d68e6425de73f4acb855d224d9143cb30139d814d7337ccf8f6bba8d3899daa` |
| `evidence/deadline-v1/robustness/V1/fgsm_eps0.1/attack_result.json` | `artifacts/m2-deadline-v1/robustness/V1/fgsm_eps0.1/attack_result.json` | 859 | `3285365a7994a40c3753586d0450cc8efaf114d8196c8ab57a554843110e1400` |
| `evidence/deadline-v1/robustness/V1/fgsm_eps0.1/unit_done.json` | `artifacts/m2-deadline-v1/robustness/V1/fgsm_eps0.1/unit_done.json` | 671 | `0b6e8aa604a5a3d471b1eab692ee596e0f9bdb13bbe44406401075e158ea7749` |
| `evidence/deadline-v1/robustness/V2/fgsm_eps0.1/arrays.npz` | `artifacts/m2-deadline-v1/robustness/V2/fgsm_eps0.1/arrays.npz` | 11484 | `44e9792262a6b2cb281ce5708f2128e646a50e08e5b32dc9bd60735621e61347` |
| `evidence/deadline-v1/robustness/V2/fgsm_eps0.1/attack_checks.json` | `artifacts/m2-deadline-v1/robustness/V2/fgsm_eps0.1/attack_checks.json` | 1622 | `385a6bc41cfa8914fbd0ca9efcb163120f378adb2978bd105ab2425242f39c84` |
| `evidence/deadline-v1/robustness/V2/fgsm_eps0.1/attack_result.json` | `artifacts/m2-deadline-v1/robustness/V2/fgsm_eps0.1/attack_result.json` | 971 | `f0639f496657ab1bf18294c676a517e785fb9c4e4e56cf724465205b4926152c` |
| `evidence/deadline-v1/robustness/V2/fgsm_eps0.1/unit_done.json` | `artifacts/m2-deadline-v1/robustness/V2/fgsm_eps0.1/unit_done.json` | 669 | `eab1fd15e347d50d1cc8da74b52001d0887f92937ee1e73b2c6f2cf2d1ccd257` |
| `evidence/deadline-v1/robustness/V3/fgsm_eps0.1/arrays.npz` | `artifacts/m2-deadline-v1/robustness/V3/fgsm_eps0.1/arrays.npz` | 11484 | `5945c5877cd9a328bd7ceaccd89e70c153c9f3acf8fe7f98dfdfc76c30864f36` |
| `evidence/deadline-v1/robustness/V3/fgsm_eps0.1/attack_checks.json` | `artifacts/m2-deadline-v1/robustness/V3/fgsm_eps0.1/attack_checks.json` | 1569 | `fa06a874504545bd406a8bb27dc7aaf5692c8c33e36117332f349d790029d4ff` |
| `evidence/deadline-v1/robustness/V3/fgsm_eps0.1/attack_result.json` | `artifacts/m2-deadline-v1/robustness/V3/fgsm_eps0.1/attack_result.json` | 857 | `aa359895babc550d2a2ff6e7ea2610137dc456a54e973b8287caf70f0f5e056b` |
| `evidence/deadline-v1/robustness/V3/fgsm_eps0.1/unit_done.json` | `artifacts/m2-deadline-v1/robustness/V3/fgsm_eps0.1/unit_done.json` | 669 | `65b260f77a07a552454b67cda71edb2b18e9047e76ca99255f19d4825029a5d8` |
| `evidence/deadline-v1/robustness/V4/fgsm_eps0.1/arrays.npz` | `artifacts/m2-deadline-v1/robustness/V4/fgsm_eps0.1/arrays.npz` | 11484 | `54f2ae70c073e0de10ed2b1f4ccf8a1595dddf824b3635ad6bc015239da0e11d` |
| `evidence/deadline-v1/robustness/V4/fgsm_eps0.1/attack_checks.json` | `artifacts/m2-deadline-v1/robustness/V4/fgsm_eps0.1/attack_checks.json` | 1622 | `4a8549f71ccd532113c615457c1fe665004755d729471315cdb4c64b0f74c629` |
| `evidence/deadline-v1/robustness/V4/fgsm_eps0.1/attack_result.json` | `artifacts/m2-deadline-v1/robustness/V4/fgsm_eps0.1/attack_result.json` | 972 | `729e68e50572084d1746021cc4366d8ce7b3ff85d2df44fcec0d775ac5202775` |
| `evidence/deadline-v1/robustness/V4/fgsm_eps0.1/unit_done.json` | `artifacts/m2-deadline-v1/robustness/V4/fgsm_eps0.1/unit_done.json` | 668 | `24be2561ed3261dfed068a93ec8b29b90447fdacfe32d495aa6d06ec820e19ec` |
| `evidence/deadline-v1/robustness/V1/pgd5_eps0.05/arrays.npz` | `artifacts/m2-deadline-v1/robustness/V1/pgd5_eps0.05/arrays.npz` | 11484 | `2536882be45975e40cb48d53a069106b0cb2d5e716ada52c0ea6c3f2e2610854` |
| `evidence/deadline-v1/robustness/V1/pgd5_eps0.05/attack_checks.json` | `artifacts/m2-deadline-v1/robustness/V1/pgd5_eps0.05/attack_checks.json` | 1570 | `61191de2e13cbac2d29d5734c93ad1109a9e0be43e095fe43a8a4f761bb14248` |
| `evidence/deadline-v1/robustness/V1/pgd5_eps0.05/attack_result.json` | `artifacts/m2-deadline-v1/robustness/V1/pgd5_eps0.05/attack_result.json` | 859 | `90ba80428b225e7ba82f097f0855ffe247c15860dec3b3ef4695abf2ce3def4a` |
| `evidence/deadline-v1/robustness/V1/pgd5_eps0.05/unit_done.json` | `artifacts/m2-deadline-v1/robustness/V1/pgd5_eps0.05/unit_done.json` | 672 | `39d4c24623182388ea293a90d5897b17ecf7afbecb3f57fdfa6ab25ffd8b24c7` |
| `evidence/deadline-v1/robustness/V2/pgd5_eps0.05/arrays.npz` | `artifacts/m2-deadline-v1/robustness/V2/pgd5_eps0.05/arrays.npz` | 11484 | `d2e4857318d7f6115f8e746cf89379341e77df438f95e7bfc571a655df74d1ec` |
| `evidence/deadline-v1/robustness/V2/pgd5_eps0.05/attack_checks.json` | `artifacts/m2-deadline-v1/robustness/V2/pgd5_eps0.05/attack_checks.json` | 1623 | `36e5367a306f1670e91a75de94f76b35725caeba418f4b8f487523fd7eac456e` |
| `evidence/deadline-v1/robustness/V2/pgd5_eps0.05/attack_result.json` | `artifacts/m2-deadline-v1/robustness/V2/pgd5_eps0.05/attack_result.json` | 973 | `15ab8de00c2c89c49f6ee430d2bcff0c020139bac7351f23bd160ad845d505e8` |
| `evidence/deadline-v1/robustness/V2/pgd5_eps0.05/unit_done.json` | `artifacts/m2-deadline-v1/robustness/V2/pgd5_eps0.05/unit_done.json` | 671 | `a68a31c0496c8ecff39ca37da86ec8d5ed13ce0ea5985d4b14a30fddaee5e4a7` |
| `evidence/deadline-v1/robustness/V3/pgd5_eps0.05/arrays.npz` | `artifacts/m2-deadline-v1/robustness/V3/pgd5_eps0.05/arrays.npz` | 11484 | `ad4ea39b35d4a06199b247454cbeaa830868a695059c334d57245b11446ddf1e` |
| `evidence/deadline-v1/robustness/V3/pgd5_eps0.05/attack_checks.json` | `artifacts/m2-deadline-v1/robustness/V3/pgd5_eps0.05/attack_checks.json` | 1570 | `cedad868a186cecbd21d542de604fda16e2e74cf6c3613fb668d10855f5467af` |
| `evidence/deadline-v1/robustness/V3/pgd5_eps0.05/attack_result.json` | `artifacts/m2-deadline-v1/robustness/V3/pgd5_eps0.05/attack_result.json` | 859 | `06bb48ad2e58c8f852abd47daab0998e04a78a57ee9f2b86515870c6680b24e0` |
| `evidence/deadline-v1/robustness/V3/pgd5_eps0.05/unit_done.json` | `artifacts/m2-deadline-v1/robustness/V3/pgd5_eps0.05/unit_done.json` | 670 | `3ee011aa5ab04199b91d737a162fbf3c0c1e0a6355c2535fbf6497062d6e3090` |
| `evidence/deadline-v1/robustness/V4/pgd5_eps0.05/arrays.npz` | `artifacts/m2-deadline-v1/robustness/V4/pgd5_eps0.05/arrays.npz` | 11484 | `19fb4cb4e37f5bdb0dff86d21046ab79cf9aa6201e5687605cea0d719e1768da` |
| `evidence/deadline-v1/robustness/V4/pgd5_eps0.05/attack_checks.json` | `artifacts/m2-deadline-v1/robustness/V4/pgd5_eps0.05/attack_checks.json` | 1623 | `c4cc0d1317c251ce4d02d4d2007be33cc109f1531f9c6698a1231f4eee37d214` |
| `evidence/deadline-v1/robustness/V4/pgd5_eps0.05/attack_result.json` | `artifacts/m2-deadline-v1/robustness/V4/pgd5_eps0.05/attack_result.json` | 971 | `f330161e7b82fa8d47d353337b90103947a5e1be5e9eb428f7861a510ffcc52d` |
| `evidence/deadline-v1/robustness/V4/pgd5_eps0.05/unit_done.json` | `artifacts/m2-deadline-v1/robustness/V4/pgd5_eps0.05/unit_done.json` | 668 | `5c35bb35e5b0d535fa33aaa1ca9e1229827897b0bebcc97f40b0353bf0e54d21` |
| `evidence/deadline-v1/robustness/V1/pgd5_eps0.1/arrays.npz` | `artifacts/m2-deadline-v1/robustness/V1/pgd5_eps0.1/arrays.npz` | 11484 | `3dd71af221c1afdb178d25ba8d70b487b5f2a2777a527a1a87fdf974591856bf` |
| `evidence/deadline-v1/robustness/V1/pgd5_eps0.1/attack_checks.json` | `artifacts/m2-deadline-v1/robustness/V1/pgd5_eps0.1/attack_checks.json` | 1569 | `5e5cba9f7a9b5459a8630d5404f877add9748231c60194fc7a40ca2ffb0fdae1` |
| `evidence/deadline-v1/robustness/V1/pgd5_eps0.1/attack_result.json` | `artifacts/m2-deadline-v1/robustness/V1/pgd5_eps0.1/attack_result.json` | 859 | `c436f9de08a6e21d0861b7b33b9a70f87cc3c61dbc915180ab47b2222b495772` |
| `evidence/deadline-v1/robustness/V1/pgd5_eps0.1/unit_done.json` | `artifacts/m2-deadline-v1/robustness/V1/pgd5_eps0.1/unit_done.json` | 673 | `d2bfed919c059cb51cb71aa274b8dbb2a0e503bf18db4c54c6f347fda22b331a` |
| `evidence/deadline-v1/robustness/V2/pgd5_eps0.1/arrays.npz` | `artifacts/m2-deadline-v1/robustness/V2/pgd5_eps0.1/arrays.npz` | 11484 | `7303908b81e42772bdbbf68b5d5cd58284d9f21bd05cc69ff917837447953d1a` |
| `evidence/deadline-v1/robustness/V2/pgd5_eps0.1/attack_checks.json` | `artifacts/m2-deadline-v1/robustness/V2/pgd5_eps0.1/attack_checks.json` | 1622 | `6db19e74c4e5c86cff8120909b633ea8f8da1a868a22f9c98006df2305a84b89` |
| `evidence/deadline-v1/robustness/V2/pgd5_eps0.1/attack_result.json` | `artifacts/m2-deadline-v1/robustness/V2/pgd5_eps0.1/attack_result.json` | 971 | `ce834ace39f297a6857cf6ed5a4ad507b82055c14fc45b37b5b6b3155498d408` |
| `evidence/deadline-v1/robustness/V2/pgd5_eps0.1/unit_done.json` | `artifacts/m2-deadline-v1/robustness/V2/pgd5_eps0.1/unit_done.json` | 669 | `2b91e3835c4ec59e72fbbb4e449f38f5d9a9db7a2aa523c980d836f6f7178e46` |
| `evidence/deadline-v1/robustness/V3/pgd5_eps0.1/arrays.npz` | `artifacts/m2-deadline-v1/robustness/V3/pgd5_eps0.1/arrays.npz` | 11484 | `09ff86b7b141b997a6e0a0f45f8dd4c4f5e10b57ec0ac4c0535aea368d7285d9` |
| `evidence/deadline-v1/robustness/V3/pgd5_eps0.1/attack_checks.json` | `artifacts/m2-deadline-v1/robustness/V3/pgd5_eps0.1/attack_checks.json` | 1569 | `4b0c4a0a92eb69d547d43e08fbaebbba2bd2a8eb8103541e86f2dc94acf98863` |
| `evidence/deadline-v1/robustness/V3/pgd5_eps0.1/attack_result.json` | `artifacts/m2-deadline-v1/robustness/V3/pgd5_eps0.1/attack_result.json` | 857 | `d92fc47679a9f0c6e4efd9840163f8e0b2c5db69e625c6ea5216b521c702545d` |
| `evidence/deadline-v1/robustness/V3/pgd5_eps0.1/unit_done.json` | `artifacts/m2-deadline-v1/robustness/V3/pgd5_eps0.1/unit_done.json` | 669 | `7c18c4b83b3cf41987846be60e1b089adb3947bdbb698df74ea33b717825feba` |
| `evidence/deadline-v1/robustness/V4/pgd5_eps0.1/arrays.npz` | `artifacts/m2-deadline-v1/robustness/V4/pgd5_eps0.1/arrays.npz` | 11484 | `ec770908088514d21a982aea3e131babf6eb0f776b09b929935d053aaf0a0089` |
| `evidence/deadline-v1/robustness/V4/pgd5_eps0.1/attack_checks.json` | `artifacts/m2-deadline-v1/robustness/V4/pgd5_eps0.1/attack_checks.json` | 1622 | `9e1d661c255c00b77bc8ba8f235333ce967e81170b665c9f914506c9b450a6c0` |
| `evidence/deadline-v1/robustness/V4/pgd5_eps0.1/attack_result.json` | `artifacts/m2-deadline-v1/robustness/V4/pgd5_eps0.1/attack_result.json` | 969 | `742d2449649e72f01f30911c97020a8095ba5041868300bffb62d258b177de5e` |
| `evidence/deadline-v1/robustness/V4/pgd5_eps0.1/unit_done.json` | `artifacts/m2-deadline-v1/robustness/V4/pgd5_eps0.1/unit_done.json` | 666 | `2cf31ede43d0a3ce8a49e64b44a6520701dab9bfb496020c8f012a9ddf56d783` |
| `evidence/deadline-v1/efficiency/results/01.json` | `artifacts/m2-deadline-v1/efficiency/results/01.json` | 4955 | `77ad02f3913e073954234472cbb935a484eeda8262c118fed0cfc089f927a4aa` |
| `evidence/deadline-v1/efficiency/specs/01.json` | `artifacts/m2-deadline-v1/efficiency/specs/01.json` | 697 | `532f39e4d7be761b593d33c907ac04e5b9dfbc5809385baf155886d30c29c5df` |
| `evidence/deadline-v1/efficiency/results/02.json` | `artifacts/m2-deadline-v1/efficiency/results/02.json` | 4951 | `10d4fdb9e48d21c3d3d555f6e038bb72650fd77160922b48614d4ac54f7b15c0` |
| `evidence/deadline-v1/efficiency/specs/02.json` | `artifacts/m2-deadline-v1/efficiency/specs/02.json` | 996 | `842300729477c870aadc8696f27705723ac783716c5770d8d995fc43f685f017` |
| `evidence/deadline-v1/efficiency/results/03.json` | `artifacts/m2-deadline-v1/efficiency/results/03.json` | 4939 | `c01cb84b3564d728357706a06e1bb19721e0079901a84b1eb62cd04b5964b6b9` |
| `evidence/deadline-v1/efficiency/specs/03.json` | `artifacts/m2-deadline-v1/efficiency/specs/03.json` | 697 | `e1acb2063245aabfe50f878d5cf81fec208f9ae363def2d5b3ac27aaeba7b8c6` |
| `evidence/deadline-v1/efficiency/results/04.json` | `artifacts/m2-deadline-v1/efficiency/results/04.json` | 4955 | `c8da6171bcf928eadf68a32f680d0de075c38a16b1c25299888270887d63030e` |
| `evidence/deadline-v1/efficiency/specs/04.json` | `artifacts/m2-deadline-v1/efficiency/specs/04.json` | 996 | `25722079741d33bfea82870be5275e7c7d624b1d0810f1888ea6a24f6995daf9` |
| `evidence/deadline-v1/efficiency/results/05.json` | `artifacts/m2-deadline-v1/efficiency/results/05.json` | 4954 | `9c5de444b296ffcf6c2e1cac5cd3ec1719cbaf43ef6ab1d924d7a283e621a7df` |
| `evidence/deadline-v1/efficiency/specs/05.json` | `artifacts/m2-deadline-v1/efficiency/specs/05.json` | 697 | `532f39e4d7be761b593d33c907ac04e5b9dfbc5809385baf155886d30c29c5df` |
| `evidence/deadline-v1/efficiency/results/06.json` | `artifacts/m2-deadline-v1/efficiency/results/06.json` | 4961 | `fdf5a9fe8731f8681d6341a15ecd32a87b46cd6f0d02f4b466ea8542f69031e9` |
| `evidence/deadline-v1/efficiency/specs/06.json` | `artifacts/m2-deadline-v1/efficiency/specs/06.json` | 996 | `842300729477c870aadc8696f27705723ac783716c5770d8d995fc43f685f017` |
| `evidence/deadline-v1/efficiency/results/07.json` | `artifacts/m2-deadline-v1/efficiency/results/07.json` | 4956 | `f3b4b64ac9e29100cd890663760e1a7617c7f386ee3dc1563e07118e7f1599c7` |
| `evidence/deadline-v1/efficiency/specs/07.json` | `artifacts/m2-deadline-v1/efficiency/specs/07.json` | 697 | `e1acb2063245aabfe50f878d5cf81fec208f9ae363def2d5b3ac27aaeba7b8c6` |
| `evidence/deadline-v1/efficiency/results/08.json` | `artifacts/m2-deadline-v1/efficiency/results/08.json` | 4969 | `649ffc22e226c965747fa85e0fedd89428185bc044c10df3279ded7cdf9e37c9` |
| `evidence/deadline-v1/efficiency/specs/08.json` | `artifacts/m2-deadline-v1/efficiency/specs/08.json` | 996 | `25722079741d33bfea82870be5275e7c7d624b1d0810f1888ea6a24f6995daf9` |
| `evidence/deadline-v1/efficiency/results/09.json` | `artifacts/m2-deadline-v1/efficiency/results/09.json` | 4949 | `304591cd1dfa4d28c5702eb9582be7d22255dc931ebbf2d559d8dafa8aa86d35` |
| `evidence/deadline-v1/efficiency/specs/09.json` | `artifacts/m2-deadline-v1/efficiency/specs/09.json` | 697 | `532f39e4d7be761b593d33c907ac04e5b9dfbc5809385baf155886d30c29c5df` |
| `evidence/deadline-v1/efficiency/results/10.json` | `artifacts/m2-deadline-v1/efficiency/results/10.json` | 4950 | `ebfed46a4fd728244e166c08d3d97303d706c5be200556cffb81b6a271555c59` |
| `evidence/deadline-v1/efficiency/specs/10.json` | `artifacts/m2-deadline-v1/efficiency/specs/10.json` | 996 | `842300729477c870aadc8696f27705723ac783716c5770d8d995fc43f685f017` |
| `evidence/deadline-v1/efficiency/results/11.json` | `artifacts/m2-deadline-v1/efficiency/results/11.json` | 4943 | `10c7357ade9d65bea453f54ddc249861014026e40a927b89eb7f24ffbcd38c26` |
| `evidence/deadline-v1/efficiency/specs/11.json` | `artifacts/m2-deadline-v1/efficiency/specs/11.json` | 697 | `e1acb2063245aabfe50f878d5cf81fec208f9ae363def2d5b3ac27aaeba7b8c6` |
| `evidence/deadline-v1/efficiency/results/12.json` | `artifacts/m2-deadline-v1/efficiency/results/12.json` | 4943 | `cd5247885ded016c935420c9aa2cd59b3a01ffb70b4587dc15d4e53062cc909d` |
| `evidence/deadline-v1/efficiency/specs/12.json` | `artifacts/m2-deadline-v1/efficiency/specs/12.json` | 996 | `25722079741d33bfea82870be5275e7c7d624b1d0810f1888ea6a24f6995daf9` |
| `evidence/deadline-v1/stage_markers/efficiency.ok` | `artifacts/m2-deadline-v1/stage_markers/efficiency.ok` | 673 | `28ee313990f366cc75a5f4d8233f8e8e76bf70a1d3db4e84a4226521aff89c81` |
| `evidence/deadline-v1/stage_markers/freeze.ok` | `artifacts/m2-deadline-v1/stage_markers/freeze.ok` | 1091 | `9997d262d0fe5a018a1333d58d0b27fe3324514de4e7add31319332691cb9224` |
| `evidence/deadline-v1/stage_markers/preflight.ok` | `artifacts/m2-deadline-v1/stage_markers/preflight.ok` | 608 | `6c605610d30492ebd608ed59b2b41461b0a9d95c8ce7c3d1b731ac01849b51ae` |
| `evidence/deadline-v1/stage_markers/ptq.ok` | `artifacts/m2-deadline-v1/stage_markers/ptq.ok` | 1273 | `5325adb70c026ccc1101f201d02bc95c21408647f7bc3d1427fe0d6c083b94df` |
| `evidence/deadline-v1/stage_markers/report.ok` | `artifacts/m2-deadline-v1/stage_markers/report.ok` | 1197 | `72c5c906e304db94b64674c46e132b675d60a28504eeae04f454030708289a05` |
| `evidence/deadline-v1/stage_markers/robustness.ok` | `artifacts/m2-deadline-v1/stage_markers/robustness.ok` | 3021 | `ef8b46483b95ad50d13a3c596b468f7dc3b1bbe904fadb1efa5cf6a1542556e8` |
| `evidence/deadline-v1/stage_markers/test_clean.ok` | `artifacts/m2-deadline-v1/stage_markers/test_clean.ok` | 782 | `c8a039a894744d858034311db686128cf91e3baa4c968f570123784a3e17ec14` |
| `evidence/deadline-v1/stage_markers/train_adversarial.ok` | `artifacts/m2-deadline-v1/stage_markers/train_adversarial.ok` | 779 | `dcb07094ec5f0def7f4954c391a37e340d4a465f7646c547c1375d0a25d054a8` |
| `evidence/deadline-v1/stage_markers/train_ordinary.ok` | `artifacts/m2-deadline-v1/stage_markers/train_ordinary.ok` | 776 | `297f825da4f9a55d12e2f1ca763700d9658349131f113e45b32d8bdb94292979` |
| `evidence/deadline-v1/stage_markers/val_check.ok` | `artifacts/m2-deadline-v1/stage_markers/val_check.ok` | 570 | `49c2017e3d8b5ef8d84e9bb760d5aaf0a548ca303573b7d91a419e361353361a` |
| `evidence/deadline-v1/logs/campaign-20261117T123149Z.log.txt` | `artifacts/m2-deadline-v1/logs/campaign-20261117T123149Z.log` | 716 | `547d919d303ad31bb75efd7f69044ca91601b6257bb41b58ea29cc9c31ef2637` |
| `evidence/deadline-v1/logs/campaign-20261117T123200Z.log.txt` | `artifacts/m2-deadline-v1/logs/campaign-20261117T123200Z.log` | 447 | `1c990d180c7aaf5b297f10d8b79fc9d51d0e4bd4fee771f8a2ac6ca7bd1cc771` |
| `evidence/deadline-v1/logs/campaign-20261117T123716Z.log.txt` | `artifacts/m2-deadline-v1/logs/campaign-20261117T123716Z.log` | 3122 | `551d7958b2e1374b9179b347d9b2cd5f37bffa11c5021cfae41101ca23667c30` |
| `evidence/deadline-v1/logs/launch-wrapper-20261117T123149Z.log.txt` | `artifacts/m2-deadline-v1/logs/launch-wrapper-20261117T123149Z.log` | 716 | `547d919d303ad31bb75efd7f69044ca91601b6257bb41b58ea29cc9c31ef2637` |
| `evidence/deadline-v1/logs/resume-wrapper-20261117T123200Z.log.txt` | `artifacts/m2-deadline-v1/logs/resume-wrapper-20261117T123200Z.log` | 447 | `1c990d180c7aaf5b297f10d8b79fc9d51d0e4bd4fee771f8a2ac6ca7bd1cc771` |
| `evidence/deadline-v1/logs/resume-wrapper-20261117T123716Z.log.txt` | `artifacts/m2-deadline-v1/logs/resume-wrapper-20261117T123716Z.log` | 4224 | `7a921ae12b1fd74d83f3fa10c4b8e857d9c78693acb58ab54f6aa077d8cbc271` |
| `evidence/deadline-v1/logs/timing-20261117T122524Z.log.txt` | `artifacts/m2-deadline-v1/logs/timing-20261117T122524Z.log` | 344 | `db403890a8e9d245bbc7824099933d8e2b398d2d4cd71b9bafd9bfaad4f75463` |
| `evidence/deadline-v1/logs/timing-wrapper-20261117T122524Z.log.txt` | `artifacts/m2-deadline-v1/logs/timing-wrapper-20261117T122524Z.log` | 2726 | `7c7731c66dabbc77f4f182d037ee3aec7d53277c1021ca32f23e55c84f4ec79a` |
| `evidence/deadline-v1/logs/verify-20261117T122101Z.log.txt` | `artifacts/m2-deadline-v1/logs/verify-20261117T122101Z.log` | 106 | `f66af0d510c8665dd9aef1053dfee45633425e7d202101e54fa2cd03f3e540a6` |
| `evidence/v4-base/reports/milestone2b/split_v2/split_m2.json` | `reports/milestone2b/split_v2/split_m2.json` | 418316 | `7b8da874b36cc5b7c967490388e87b050173427f0447322b3c2b2fe064b7e06e` |
| `evidence/v4-base/reports/milestone2b/split_v2/scaler_minmax.json` | `reports/milestone2b/split_v2/scaler_minmax.json` | 3545 | `de41d50dba0e7e4cfdb839d2e18f59683248a29b530ad04414b3f6cc1ea584a7` |
| `evidence/v4-base/reports/milestone2b/split_v2/split_freeze.json` | `reports/milestone2b/split_v2/split_freeze.json` | 1506 | `17a7f6218c2876ad9829dc9fbaf5f7dbbc228b386d18fc95e65e3a29b3302210` |
| `evidence/v4-base/reports/milestone2b/split_v2/split_report.md` | `reports/milestone2b/split_v2/split_report.md` | 8058 | `c9f988c3c3b69db7497b923d9c75897b94e7f605e1519b708f650c76d0d944de` |
| `evidence/v4-base/reports/milestone2b/split_v2/split_attempts.json` | `reports/milestone2b/split_v2/split_attempts.json` | 13332 | `d3657aaf69798d49c8dcbcfd9e75e659f76f84cc01f61ab4f46b39e90d7d8e5a` |
| `evidence/v4-base/reports/milestone2b/split_v2/range_audit.json` | `reports/milestone2b/split_v2/range_audit.json` | 3506 | `3921bbfba779437eff5a848420fe19a6dbe6c623c1ea5b15cba000c5a811e3dc` |
| `evidence/v4-base/reports/milestone2b/split_v2/conflicting_labels.json` | `reports/milestone2b/split_v2/conflicting_labels.json` | 59420 | `8858e72a1f2ab23ab14f0ed8c1b0caee50de9d0fa47b5fbf889d0191ee2599b3` |
| `evidence/v4-base/reports/milestone2b/split_v2/split_comparison_v1_v2.json` | `reports/milestone2b/split_v2/split_comparison_v1_v2.json` | 37385 | `9375d6e616bdc4c8b6186868df99c3a7e11ffbc938ece2e7c72c8350d44dd625` |
| `evidence/v4-base/reports/milestone2b/split_v2/split_comparison_v1_v2.md` | `reports/milestone2b/split_v2/split_comparison_v1_v2.md` | 4928 | `8aa8fddbd51cf1c4fa00ef76f4ae08315932b1679e4f268deb1aa9fdd0954190` |
| `evidence/v4-base/reports/milestone2b/fingerprint_sets.json` | `reports/milestone2b/fingerprint_sets.json` | 6573 | `76956a94e1a20739aecf10061d7a781fc4411085d4015306ec6634df8484419a` |
| `evidence/v4-base/output/m2b-v1/preflight.json` | `output/m2b-v1/preflight.json` | 5655 | `a1aa8338019f42d44d78ffdad49ddf6018c3f8b92ae4c36bcb194d0e34aced9b` |
| `evidence/v4-base/output/m2b-v1/stage_markers/03.ok` | `output/m2b-v1/stage_markers/03.ok` | 28 | `a6531a44e790cf3cf9e0d1b7ee4aa86f59b8492c1a9169c5699c3e7108cb2d06` |
| `evidence/v4-base/output/.stage_markers/01.ok` | `output/.stage_markers/01.ok` | 28 | `d9d5e3fe7bd9294bb41dcdfd831c513acac9062c20aff5a6626873a2726d1e19` |
| `evidence/v4-base/output/.stage_markers/02.ok` | `output/.stage_markers/02.ok` | 28 | `fbb1a0b90f070797b8f5829971a97377c977c86e098b240dd02d38129f56b874` |
| `evidence/v4-base/output/m2b-v1/logs/01-20261116T230411Z.log.txt` | `output/m2b-v1/logs/01-20261116T230411Z.log` | 498 | `d846661b17fd7282d0b3bd157202f2b19a7ffe14d90c6f7757b2bdc1dc33f96c` |
| `evidence/v4-base/output/m2b-v1/logs/02-20261116T230533Z.log.txt` | `output/m2b-v1/logs/02-20261116T230533Z.log` | 8537 | `55cd4dc880a5532f2c10e9e8c43cca1d54554dc18aad6aedf61bdc673f63d4f2` |
| `evidence/v4-base/output/m2b-v1/logs/03-20261116T230717Z.log.txt` | `output/m2b-v1/logs/03-20261116T230717Z.log` | 3887 | `3c2c0d5e8bde220f2fb937173b4882957b664947cde19ce806906347b0daed65` |
| `evidence/v4-base/output/m2b-v1/logs/04-20261116T230921Z.log.txt` | `output/m2b-v1/logs/04-20261116T230921Z.log` | 11545 | `77dd69486d31dc37f8ad5b5afa5d2a3d414751f99e9a670aa785959e5e751660` |
| `evidence/v4-base/m2a-pilot/PILOT_NOTICE.txt` | `artifacts/m2a-pilot/PILOT_NOTICE.txt` | 444 | `9dfc2723f7529d20d184bf860062ec2346889ca7d3593afb34111dce33820ac8` |
| `evidence/v4-base/m2a-pilot/pilot_report.md` | `artifacts/m2a-pilot/pilot_report.md` | 11163 | `02b6e198ece11ce4106e729960b53bf6917b090e67e27aba3cd39764220e558f` |
| `evidence/v4-base/m2a-pilot/status.json` | `artifacts/m2a-pilot/status.json` | 256 | `b1a9a0056724b1db8c325e05b479df23fe232819c80f3e15593f809bd89df62b` |
| `evidence/v4-base/m2a-pilot/pilot_runtime.json` | `artifacts/m2a-pilot/pilot_runtime.json` | 5058 | `ce7f90f2cb63ce7f82e7c3cad4a1aaf2458ac2b7daa0801aef92a6984ef7a700` |
| `evidence/v4-base/m2a-pilot/clean_eval.json` | `artifacts/m2a-pilot/clean_eval.json` | 971 | `19ab3b7550697ae4f624ee7a41e60f5d4d0e08f6c75dedc03608236a3a26e42d` |
| `evidence/v4-base/m2a-pilot/subsets.json` | `artifacts/m2a-pilot/subsets.json` | 83589 | `7d1c3c8a9a4b9d831d132c126722402808729b672905e8bd9301c73b09559602` |
| `evidence/v4-base/m2a-pilot/calibration_ids.json` | `artifacts/m2a-pilot/calibration_ids.json` | 6823 | `8f66ad2f1eb22695341dffdad90433153e74b1c4fdfa94be8620a62aba3c7645` |
| `evidence/v4-base/m2a-pilot/qat/qat_feasibility.json` | `artifacts/m2a-pilot/qat/qat_feasibility.json` | 6161 | `735b5da47bd5f5c801e82f6e51b95b470480f4b8e3e1c8430f1fd61f7499f4a4` |
| `evidence/v4-base/m2a-pilot/qat/qat_int8/conversion.json` | `artifacts/m2a-pilot/qat/qat_int8/conversion.json` | 26944 | `b7a4be745b98db4f9a6971397a8cc31cee0387a3f613414a19834e1f78c982ac` |
| `evidence/v4-base/provenance/PACKAGE_IDENTITY.json` | `PACKAGE_IDENTITY.json` | 23215 | `b8e2d0d0467bb840f24300fcb624c187adcb77cf504300d4e8c7ed4522162b40` |
| `evidence/v4-base/provenance/MANIFEST.sha256` | `MANIFEST.sha256` | 85966 | `5f49c93af30704a181ea77d3c09c23e99062accee40efc6c93ec9365e8a88d99` |
| `evidence/v4-base/provenance/m2b-integrity/before.json` | `artifacts/m2b-integrity/before.json` | 73335 | `6f435fdc7fd3b209a0a27467d6c0cdc7f7d3cc45c42854bdf644b1f5c37e038f` |
| `evidence/v4-base/provenance/m2b-integrity/after.json` | `artifacts/m2b-integrity/after.json` | 73493 | `1a14d825ec93fa8f6b0974a6820ca7454db936f93f750894c5504e637365b815` |
| `evidence/v4-base/provenance/m2b-integrity/after_feat005.json` | `artifacts/m2b-integrity/after_feat005.json` | 73493 | `1a14d825ec93fa8f6b0974a6820ca7454db936f93f750894c5504e637365b815` |
