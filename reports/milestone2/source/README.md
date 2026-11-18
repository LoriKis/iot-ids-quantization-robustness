# Group-authored Milestone 2 source (provenance)

The two folders below are **byte-identical copies** of group-authored files from the two read-only VM folders. They are
kept in separate trees because the two packages are different things and share two file names
(`pytest.ini`, `tests/conftest.py`) with different content.

| Folder | What it is | VM folder |
|---|---|---|
| [`v4-base/`](v4-base/) | The **V4 base package**: split, training, quantization, attack, efficiency and campaign code (`research/m2_*.py`, `research/m2b_*.py`), configs, tests, runbooks. V4 itself was verified before the campaign (its `MANIFEST.sha256` has sha256 `5f49c93af30704a181ea77d3c09c23e99062accee40efc6c93ec9365e8a88d99`, copied at [`../evidence/v4-base/provenance/MANIFEST.sha256`](../evidence/v4-base/provenance/MANIFEST.sha256)). | `IOT-network-traffic-DL-M2-VM-v4` |
| [`deadline-v1-overlay/`](deadline-v1-overlay/) | The **deadline overlay** `m2-deadline-v1`: the add-on that actually ran the completed campaign (timing, launch, resume, report commands, `m2_deadline/*.py`, config `configs/m2_deadline_v1.json`). It does not modify V4. | `IOT-network-traffic-DL-M2-VM-deadline-v1` |

Both are the code and files that existed on the VM. No file here was edited, reformatted or re-encoded, so some files have
CRLF line endings (the nested [`../.gitattributes`](../.gitattributes) keeps them unchanged).
**No post-run code adaptation is stored here.** The post-run adaptation note is in
[`../post-run-adaptation/README.md`](../post-run-adaptation/README.md).

No end-to-end reproduction from this repository layout was performed.

## Files deliberately not duplicated

- V4 `research/` modules that are identical to the already tracked M1 repository files (compared after normalising
  line endings; the raw bytes were also equal): `adversarial_domain.py`, `data.py`, `evaluation.py`, `quantization.py`,
  `provenance.py`, `validate.py`, `training.py`, `runner.py`, `__init__.py`. They are imported by the M2 code and are
  in the repository's [`research/`](../../../research/) folder.
- V4 `research/attacks.py` and `research/common.py` **differ** from the repository's M1 versions (they were extended for M2). They are copied
  to `v4-base/research/` and the repository's M1 files are left untouched. See the conflict list in the handoff.
- Upstream PAVE code and data (`upstream/`) are third-party and not copied.
- Other V4 files that differ from tracked M1 paths (for example `.gitattributes`, `configs/milestone1_artifact_full.json`, M1 docs and reports,
  `research/paper_described_adam_report.py`, `scripts/check_milestone1_adam_review.py`) are M1 material and are not copied.
  The V4 `pytest.ini` is copied only into `v4-base/`, never to the repository root.

## Link between the executed code and the saved results

`run_identity.json` in the deadline evidence records the SHA-256 of the executed `m2_attacks.py`, `m2_efficiency.py`, `m2_quant.py`,
`m2_split.py`, `m2_training.py`, `m2b_exec.py` and `quantization.py`. The verification script
([`../../../verification/verify_m2.py`](../../../verification/verify_m2.py), check group `i`) confirms that the copies here have exactly those hashes
(`quantization.py` is the tracked repository file). It also confirms that all copied V4 files match
V4's `MANIFEST.sha256` and all copied overlay files match `OVERLAY_MANIFEST.sha256`. This shows the copied bytes are the files
the campaign recorded. It does not prove that a rerun from this layout would give the same numbers.

## Hash tables

### `v4-base/` (from the V4 VM folder)

| Clone path (relative to `reports/milestone2/`) | VM path | Bytes | SHA-256 (copy = VM file) |
|---|---|---:|---|
| `source/v4-base/VM_RUNBOOK.md` | `VM_RUNBOOK.md` | 19905 | `e8068d7ccb96dac98c7e4457e43cac8da76d80297ae326b93439c97e9bf509d8` |
| `source/v4-base/requirements-linux-cpu.txt` | `requirements-linux-cpu.txt` | 1065 | `3a2c5d8f4ec29ccd3630e314ba8153ea5d2640cf271575e1021c874757310982` |
| `source/v4-base/pytest.ini` | `pytest.ini` | 385 | `c0cc174182de86b4b93751c7be8dc648fc83806301e2fb4cf2e16c951a1b4d20` |
| `source/v4-base/research/attacks.py` | `research/attacks.py` | 4104 | `94e24e2a725a5ea96d11395a4a367ffb96c125b785b21ebbd0a5190124e2fb22` |
| `source/v4-base/research/common.py` | `research/common.py` | 11406 | `8234978f1e9e4195023431c531ae713f83aa28482c199817e4ecc02281b2300a` |
| `source/v4-base/research/m2_attacks.py` | `research/m2_attacks.py` | 40124 | `d5f04c77a9c0ea0bee060891d0e1fe9cd4dfa562ecb6ca0367cb36c5c2818956` |
| `source/v4-base/research/m2_campaign.py` | `research/m2_campaign.py` | 8191 | `bb63517abb581659d878dfc0a5d4b6875e1b7de4b36c5fc2c5f576e41851617f` |
| `source/v4-base/research/m2_efficiency.py` | `research/m2_efficiency.py` | 23428 | `219b83989b032a3ee57c51f83c5f1632dee1be4056f02c425c83f9830647ffbd` |
| `source/v4-base/research/m2_estimate.py` | `research/m2_estimate.py` | 13261 | `8e0e59151dbd7eea918b2ed16c674e16d8d70e8b809b03cf6fad207ff6c5b5f3` |
| `source/v4-base/research/m2_pilot.py` | `research/m2_pilot.py` | 24796 | `ed8fc51132099f9ac4049144971fad11fa8cb93fbee1641371bb75c3cb51d68a` |
| `source/v4-base/research/m2_quant.py` | `research/m2_quant.py` | 33646 | `b62ad18dc1726d3b671c43c5b583486b302443f27244109b3dae268dc7636958` |
| `source/v4-base/research/m2_report.py` | `research/m2_report.py` | 12870 | `507e91a9c24e90ce040b071153c29c313b6a14fb0e6579423c0a8e5a7da7b931` |
| `source/v4-base/research/m2_split.py` | `research/m2_split.py` | 61482 | `bcbc99eec380c160c67c461af4621c63aa7c264fe5869ed853319282cbb6c3ff` |
| `source/v4-base/research/m2_training.py` | `research/m2_training.py` | 18987 | `5fe4fa1d03a82906826ab48328f2ff64f1fe3b0f290bb856a87d9360135a43a6` |
| `source/v4-base/research/m2b_campaign.py` | `research/m2b_campaign.py` | 54882 | `5aecd9d06e56d23aaceab355616bacebda5cf6552b64e37bd2a6b0543c9fdcff` |
| `source/v4-base/research/m2b_exec.py` | `research/m2b_exec.py` | 30401 | `e7524f06356dc4ab750f8efdee0748221e6a31267a0d4abb96a0f31156e09e24` |
| `source/v4-base/research/m2b_io.py` | `research/m2b_io.py` | 5680 | `894fc5d26cf99be2fb5fa49abedb592f093e6191dfe47e5e2697af3940979018` |
| `source/v4-base/research/m2b_plan.py` | `research/m2b_plan.py` | 10044 | `3892081435b2aa1774341a202264d1750825b2bacf246a58f4064b36e4a506ff` |
| `source/v4-base/research/m2b_preflight.py` | `research/m2b_preflight.py` | 29823 | `46aca17653a5b94e8560ea3de6d25cec82f0d8c28bc8c84ec1a5ad7d69c16ab7` |
| `source/v4-base/research/m2b_report.py` | `research/m2b_report.py` | 34306 | `e7c8df8f03c19cf8ca98f95eb46eabb20e4cce985d3f575cebc090b8da00fedd` |
| `source/v4-base/research/m2b_splitreport.py` | `research/m2b_splitreport.py` | 11322 | `741172233b430002e40267ce195e1f46eae691ef3954dc63903d5d16feebd5d4` |
| `source/v4-base/research/m2b_state.py` | `research/m2b_state.py` | 26435 | `d378b1df532e2d396f55675031db3bd03f7e318fc8d31e0c289dadcd3ee8775c` |
| `source/v4-base/scripts/check_m2_integrity.py` | `scripts/check_m2_integrity.py` | 4565 | `2a68da4c288073d83f6877af65af52a300e54b3d9105845f605b814b9848667f` |
| `source/v4-base/scripts/check_m2b_integrity.py` | `scripts/check_m2b_integrity.py` | 4863 | `149ea8599c88e73e5da718cacdfb02dcd64348ea30fe98802d874bdd6c0c6918` |
| `source/v4-base/scripts/package_m2_vm.py` | `scripts/package_m2_vm.py` | 23233 | `6d1db8ded41f3bad453e20addfb4e17c634432a6f0696375842db7e2c2f6aa3f` |
| `source/v4-base/scripts/vm/_common.sh` | `scripts/vm/_common.sh` | 2380 | `b10e0eec1bb4de52326d71b236947036191d0e2873088cdf0bb2125c193a6ec4` |
| `source/v4-base/scripts/vm/01_verify_transfer.sh` | `scripts/vm/01_verify_transfer.sh` | 535 | `782b66597c980ead85c9aeb11273a251ab931c5c67ddee5f6587a6ca9e8e4c7e` |
| `source/v4-base/scripts/vm/02_setup_env.sh` | `scripts/vm/02_setup_env.sh` | 1063 | `33629d30d84f8a321189c46b208f8f522c51cca506f10a9136c574f66d7c328d` |
| `source/v4-base/scripts/vm/03_preflight.sh` | `scripts/vm/03_preflight.sh` | 410 | `b42c349078ebe9201a56851b61bdb389bcbb451cba5c2df3c5182f9d16874ea1` |
| `source/v4-base/scripts/vm/04_tests_and_smoke.sh` | `scripts/vm/04_tests_and_smoke.sh` | 619 | `5482254eec6b7e123c3db8192eef57ba910e173aa52ba188041dd61528230d2a` |
| `source/v4-base/scripts/vm/05_seed1_validation.sh` | `scripts/vm/05_seed1_validation.sh` | 1595 | `a7c46b316175d12633eb9f585bbdafdd9869d9e7baf52c31026c7f9223edb442` |
| `source/v4-base/scripts/vm/06_resume_remaining.sh` | `scripts/vm/06_resume_remaining.sh` | 477 | `0594f81ce83912e6c8709908154b8d641d89b46f85cd810e8e6f3365ad1a800f` |
| `source/v4-base/scripts/vm/07_qat_optional.sh` | `scripts/vm/07_qat_optional.sh` | 711 | `ea3dc77b58f12407c6a03e78176df847d84afb4d34a02f6c454dcbbb346df7a9` |
| `source/v4-base/scripts/vm/08a_freeze_eval.sh` | `scripts/vm/08a_freeze_eval.sh` | 744 | `3ed1add29c4a16bb94ebadd976d3604e6deca6a33f92b6cadf808ab96c1f69eb` |
| `source/v4-base/scripts/vm/08b_run_final_test.sh` | `scripts/vm/08b_run_final_test.sh` | 683 | `e261cbdf325569bc2ff2b487899c23d87b1c7df83bf85e8f53e4a5c0ae575269` |
| `source/v4-base/scripts/vm/09a_efficiency.sh` | `scripts/vm/09a_efficiency.sh` | 632 | `99a5e8a77f6682f4057f9fc6b49a1d2cb1336452cd1467b15645da76563a9bd8` |
| `source/v4-base/scripts/vm/09b_report.sh` | `scripts/vm/09b_report.sh` | 738 | `43036615eb7054e8bdd928378f83f10e8cc8282366a59c14bdab66931317c348` |
| `source/v4-base/scripts/vm/check_runbook_commands.py` | `scripts/vm/check_runbook_commands.py` | 4823 | `fdc302fb1fe5babfc7c690d8bef30089426f85c2bf94eb44f55a17d58e012a0f` |
| `source/v4-base/scripts/vm/check_shell.py` | `scripts/vm/check_shell.py` | 3239 | `b5f584aa31f419c0398649cfa90c8f7d23e73996ccb9f3221feda8f01206e5a9` |
| `source/v4-base/scripts/vm/scan_test_dependencies.py` | `scripts/vm/scan_test_dependencies.py` | 7762 | `4930149bf8f30d8ea32ee9e9e72565aa18eb90889ee4e4f0d6fa8c99b6e99839` |
| `source/v4-base/scripts/vm/status.sh` | `scripts/vm/status.sh` | 356 | `9b7ca4bd8fc624ff3fb5b38026636ba153e7876831aaa0ca9bd3d84c698df648` |
| `source/v4-base/scripts/vm/verify_package.py` | `scripts/vm/verify_package.py` | 9710 | `ad531477829d9055ed2fefcb6fc03ece692d37073f3a1cb3df57d0bbebc3c7b0` |
| `source/v4-base/tests/conftest.py` | `tests/conftest.py` | 1090 | `3138d25cdec22a565a2df75adde02bffe33331ab35949d24a83d60d1839bddbf` |
| `source/v4-base/tests/m2_guard_support.py` | `tests/m2_guard_support.py` | 3691 | `37293c308216c9da0daa7fad720efefeb699e3439f1fd5043392c7846d210501` |
| `source/v4-base/tests/m2_helpers.py` | `tests/m2_helpers.py` | 3030 | `3c8261076a0f66334657ad781b75477854530305a418b3cf4d81d572437c1e9b` |
| `source/v4-base/tests/m2b_stub_support.py` | `tests/m2b_stub_support.py` | 3150 | `b1ea81395341396c23a6243ab45617022edaf1ac7cc7483962d4f32128a1292b` |
| `source/v4-base/tests/test_m2_attacks.py` | `tests/test_m2_attacks.py` | 34698 | `5bb5004e4ba272e1d0c0e59a5fc698e60b7ce23243b47cdbd3b4757c9dcaecf4` |
| `source/v4-base/tests/test_m2_efficiency.py` | `tests/test_m2_efficiency.py` | 11505 | `9b0d7cb1eada6f1a47ce0e28ee03b5ba00f1ecd409485433859335ac45ec306c` |
| `source/v4-base/tests/test_m2_guard.py` | `tests/test_m2_guard.py` | 12284 | `529eff69f3d9ecf907313900d1bc1038b80aec6420708ac2c24e7619e1b56274` |
| `source/v4-base/tests/test_m2_pilot.py` | `tests/test_m2_pilot.py` | 9909 | `1b2b61c0927ebc80aa28b15feeb742e82615782f7e1f91904e7172ec5c813c80` |
| `source/v4-base/tests/test_m2_quant.py` | `tests/test_m2_quant.py` | 20413 | `bf47f03135b8d31e2a93475464b2cd0efb39b6c4294a4ee69affe588e702baea` |
| `source/v4-base/tests/test_m2_split.py` | `tests/test_m2_split.py` | 15927 | `f49097fcda331ee527e5964164a68b221899e318597472ab182f2356b4da509f` |
| `source/v4-base/tests/test_m2_training.py` | `tests/test_m2_training.py` | 10579 | `243639eb96dcb3ffdc61c59fc3fffc7d795b5e62daebc22f5df8d62dcba87c9b` |
| `source/v4-base/tests/test_m2b_checkpoint.py` | `tests/test_m2b_checkpoint.py` | 6557 | `76f386791fc6f05dc5e3271b1b38b6d55a441fcedc3aaf23bfd996dca9bc135e` |
| `source/v4-base/tests/test_m2b_exec_stub.py` | `tests/test_m2b_exec_stub.py` | 14018 | `f6c5a70030c6bc70e83800582f31a762f4dbea3094373b2428ce9f8b7c0e84de` |
| `source/v4-base/tests/test_m2b_gating.py` | `tests/test_m2b_gating.py` | 12375 | `6dde551cce2df379780808476af7ddcdf78fb10e23993a8119887d38be84270c` |
| `source/v4-base/tests/test_m2b_guard_local.py` | `tests/test_m2b_guard_local.py` | 1410 | `bd4c50554269c0c2643c93fcca5684880dff832f6cf07ec28b063b1650aecd81` |
| `source/v4-base/tests/test_m2b_io.py` | `tests/test_m2b_io.py` | 3837 | `3f02bfc4728bd9c6a02b46edd94c1015580633b1218f48a37fb3f356f548e9f5` |
| `source/v4-base/tests/test_m2b_package.py` | `tests/test_m2b_package.py` | 15268 | `1bb6cab1ce4849c395eb3830a9b382362dc74154f758006a7bc24f559b77eed6` |
| `source/v4-base/tests/test_m2b_plan.py` | `tests/test_m2b_plan.py` | 6774 | `3f9b48f124c554feb3d6593c765fcd0664eb0f304dc476c71ff3e78e3ac8f91e` |
| `source/v4-base/tests/test_m2b_preflight.py` | `tests/test_m2b_preflight.py` | 4818 | `b4eb21af5163824affce98b59691fcfdcff2afc2e807f6eb8eab08c5641ccc15` |
| `source/v4-base/tests/test_m2b_report.py` | `tests/test_m2b_report.py` | 12543 | `83f7c0ca1a2475d41a700f433c2783369d230af204d87336e9a7f9aea4cddf12` |
| `source/v4-base/tests/test_m2b_split_v2.py` | `tests/test_m2b_split_v2.py` | 11102 | `d7d0df354815a70c1cfab51631cb72cb0d6477bd4fb3748dfb4468fd98955b3c` |
| `source/v4-base/tests/test_m2b_state.py` | `tests/test_m2b_state.py` | 14970 | `169ff4ad756f6f99772d1081057833794b2e678125b0801bad2c37233fbe0b15` |
| `source/v4-base/tests/test_m2b_vm_only.py` | `tests/test_m2b_vm_only.py` | 4261 | `0c49a6e843ce57c986f1a3db1d10ce1c485bf2afdffbb5b2bfdff8acbf7d0d1c` |
| `source/v4-base/configs/m2/attacks.json` | `configs/m2/attacks.json` | 3827 | `7ff7b518cc76b393a71a09574d1ec4a3f024a998edb4d81842b319a37d5fc8a4` |
| `source/v4-base/configs/m2/campaign.json` | `configs/m2/campaign.json` | 4662 | `18d853b5135c856fdb8197f11b14bdc47addc0f0cbfb53c80524289dd0e650cd` |
| `source/v4-base/configs/m2/data.json` | `configs/m2/data.json` | 3382 | `15bb457b2f4313cc64fa3fe3fd4e289a7194abf03ba51369e71b7a9915104ad0` |
| `source/v4-base/configs/m2/efficiency.json` | `configs/m2/efficiency.json` | 2707 | `c3357b941f154c813054ac0cddc76ae343b98503f919b4ca218493fd8d69f4db` |
| `source/v4-base/configs/m2/pilot.json` | `configs/m2/pilot.json` | 2376 | `36fe9028e6b535ff82d2941e57714734fccd92529dc7a73e87111a0464ad9ce3` |
| `source/v4-base/configs/m2/quantization.json` | `configs/m2/quantization.json` | 4428 | `6bb20f37e56a1a0e627f3741b8a616e37983a1c93d92b42b414d240129c655fe` |
| `source/v4-base/configs/m2/training.json` | `configs/m2/training.json` | 3349 | `7be76132a32999f980e0a0e2474a4852439f0622f2f222a6e15d74b81167aaed` |
| `source/v4-base/configs/m2b/campaign_v2.json` | `configs/m2b/campaign_v2.json` | 4412 | `c882f7ff8374440b2c01dd75eb206e4c8a05af0c13697a3e918cb918965ebb7d` |
| `source/v4-base/configs/m2b/data_v2.json` | `configs/m2b/data_v2.json` | 4692 | `fe2c49cd8cd2f36c8f7682d63e9b6a4c62c0c370b161e80030f24256071baa3f` |
| `source/v4-base/configs/m2b/efficiency_v2.json` | `configs/m2b/efficiency_v2.json` | 2840 | `11fe82a10878c4ad49f183e550ec524bc4c6629d33b4552fe5e38ad2e1efe8f7` |
| `source/v4-base/docs/M2B_VM_HANDOFF.md` | `docs/M2B_VM_HANDOFF.md` | 5449 | `ce1c6ee4ad7deb2ae715640480d3ee50fff4149e02567d2b8f19623866816c50` |
| `source/v4-base/docs/MILESTONE2A_HANDOFF.md` | `docs/MILESTONE2A_HANDOFF.md` | 20035 | `f6c44a31a660f2725b08c9ea4bed13a2cba38dc2fb085e9c842d35e1bf99bf33` |
| `source/v4-base/docs/MILESTONE2_HANDOFF.md` | `docs/MILESTONE2_HANDOFF.md` | 20035 | `f6c44a31a660f2725b08c9ea4bed13a2cba38dc2fb085e9c842d35e1bf99bf33` |
| `source/v4-base/docs/MILESTONE2_IMPLEMENTATION_AUDIT.md` | `docs/MILESTONE2_IMPLEMENTATION_AUDIT.md` | 35188 | `0efd7878e5d2a0886fe0040a38957cb0c2b1f72d3ad3796cd4327c382e445d6e` |
| `source/v4-base/docs/MILESTONE2_PROTOCOL.md` | `docs/MILESTONE2_PROTOCOL.md` | 80941 | `70bcce18decf7b111602077af64c541673a33ecba927a9a7b21cd9709853b034` |
| `source/v4-base/docs/VM_RUNBOOK.md` | `docs/VM_RUNBOOK.md` | 17906 | `412bb2756125212fffd2004169ec248fb95f6a4da159ba991f46a31a02500abc` |

### `deadline-v1-overlay/` (from the deadline VM folder)

| Clone path (relative to `reports/milestone2/`) | VM path | Bytes | SHA-256 (copy = VM file) |
|---|---|---:|---|
| `source/deadline-v1-overlay/HANDOFF.md` | `HANDOFF.md` | 13253 | `83462929ce41e6eadbca9e38a61bf09c1af5f42cd6eee023f8ff8293c808de54` |
| `source/deadline-v1-overlay/OVERLAY_IDENTITY.json` | `OVERLAY_IDENTITY.json` | 286 | `07fa1ad5cd1715b835a986b85ba7bf4b1f4e69143066c7a21d4e552ff963d69e` |
| `source/deadline-v1-overlay/OVERLAY_MANIFEST.sha256` | `OVERLAY_MANIFEST.sha256` | 3596 | `f255e691676894b09e9b9c185be7de36f1db1a54e6c5bd91155abd6c0475b0ef` |
| `source/deadline-v1-overlay/pytest.ini` | `pytest.ini` | 57 | `8fa307d1ef787f3822b949dc90f1b3c405769c4eb97bb2e813b271109d9d8a90` |
| `source/deadline-v1-overlay/configs/m2_deadline_v1.json` | `configs/m2_deadline_v1.json` | 2391 | `4f12845e86270944be7c9abd885c9af6c9cfb98282ae3a5f4581c0c0fd1c888a` |
| `source/deadline-v1-overlay/m2_deadline/__init__.py` | `m2_deadline/__init__.py` | 139 | `ede2176c34518b94f7e5a4cc967d9e6e16cd2048e135187df09ab6a8724be1bb` |
| `source/deadline-v1-overlay/m2_deadline/budget.py` | `m2_deadline/budget.py` | 17022 | `2f0765d57633a399e53125f4cbab1671c7df1c01412b545983b0033832647041` |
| `source/deadline-v1-overlay/m2_deadline/campaign.py` | `m2_deadline/campaign.py` | 14072 | `a038bd5c26d6aa3f50d1be23eca2fe7ee389ea48ef770a0b88eca0ab90bcc395` |
| `source/deadline-v1-overlay/m2_deadline/common.py` | `m2_deadline/common.py` | 22979 | `3abf164fd2a4a492409914bb9f137ca41685f71ab7533fb92a354128832d04a1` |
| `source/deadline-v1-overlay/m2_deadline/make_manifest.py` | `m2_deadline/make_manifest.py` | 3632 | `dd8ad4a58437c570a63c15796ef4684a03712162e48b128dd41bd627224bac31` |
| `source/deadline-v1-overlay/m2_deadline/manifest.py` | `m2_deadline/manifest.py` | 8071 | `9120a65ad70bc5c00a44b293d9b057dbca33aae33bf13117f67b26303a966204` |
| `source/deadline-v1-overlay/m2_deadline/markers.py` | `m2_deadline/markers.py` | 10747 | `6d4a06bfe5c96becf3b9bfe6f87174430ed0256fa96152b0fd0cef321e6c5cd2` |
| `source/deadline-v1-overlay/m2_deadline/metrics.py` | `m2_deadline/metrics.py` | 6687 | `bc4e80236f6113bd60cd4906e3cde4e2b3310818a9307a7910c634d09fa60881` |
| `source/deadline-v1-overlay/m2_deadline/protocol.py` | `m2_deadline/protocol.py` | 17382 | `f5e14a117ef8a4c0b2d53bcf3637ce927f197ae7863a6ad5247f5970c36cadec` |
| `source/deadline-v1-overlay/m2_deadline/report.py` | `m2_deadline/report.py` | 38311 | `3d30faa2358a91f286346c97f414d839a7243f3d18ce82f3e763112c30aec2e7` |
| `source/deadline-v1-overlay/m2_deadline/sampling.py` | `m2_deadline/sampling.py` | 4188 | `2a8b694cbc7249dc04a7eae98255644d94595614150f0270d2d579efb8cd3bea` |
| `source/deadline-v1-overlay/m2_deadline/stages.py` | `m2_deadline/stages.py` | 49924 | `04b8e724fc51727fd3cbd7bc9f57722f19d11b3c2b655ebbcb667eef860b9617` |
| `source/deadline-v1-overlay/m2_deadline/timing.py` | `m2_deadline/timing.py` | 26925 | `5377181569e713daeefab0a8bf7d8543cb97b10ce023bc8cfe44625fd96e4a16` |
| `source/deadline-v1-overlay/m2_deadline/verify_overlay.py` | `m2_deadline/verify_overlay.py` | 5239 | `914f15e4bd392bb55aa74871cc1aca903c27ca45d89aac6f1528a8cbada73c86` |
| `source/deadline-v1-overlay/scripts/vm_deadline/_env.sh` | `scripts/vm_deadline/_env.sh` | 1503 | `ce07b49c4884d7c4f33b86b8cb6b59bbd37332a47b38dbe81308982558d24729` |
| `source/deadline-v1-overlay/scripts/vm_deadline/launch.sh` | `scripts/vm_deadline/launch.sh` | 471 | `b01868a5c38cde01d3d678ec69aa6cabaa60c126b6ebc8133579beab8eb7cbfe` |
| `source/deadline-v1-overlay/scripts/vm_deadline/report.sh` | `scripts/vm_deadline/report.sh` | 271 | `1fcccda38a1b786f95ce6840f9cb2809f6a48d5171033bd3bda0e565e4df5d83` |
| `source/deadline-v1-overlay/scripts/vm_deadline/resume.sh` | `scripts/vm_deadline/resume.sh` | 480 | `638b00b000e3c32a2273c471c5ed2e395753b75efb1379d1533e270efbae49c3` |
| `source/deadline-v1-overlay/scripts/vm_deadline/status.sh` | `scripts/vm_deadline/status.sh` | 225 | `f1036262a171091b0281c74d350c4d9ab64f5b47e5bee594413674a65e449a94` |
| `source/deadline-v1-overlay/scripts/vm_deadline/timing.sh` | `scripts/vm_deadline/timing.sh` | 456 | `6d7e7f544e04df1841dbf1cf59934c1f61cf7294530a293c51450f2d98aa2cc0` |
| `source/deadline-v1-overlay/scripts/vm_deadline/verify.sh` | `scripts/vm_deadline/verify.sh` | 283 | `801b4df2a2b82371ba14965a001e419bdac65090124d19d7d2f9d8fcc997889a` |
| `source/deadline-v1-overlay/tests/conftest.py` | `tests/conftest.py` | 5445 | `377ce3d98de5c7d244104bf4ffb7f362d4fb08d2dd186692875a54f97efd7019` |
| `source/deadline-v1-overlay/tests/stub_support.py` | `tests/stub_support.py` | 15374 | `2ddbd044fda50e67eae2e9bf00b463658e519917ad501607f6aba5701772ca1c` |
| `source/deadline-v1-overlay/tests/test_budget.py` | `tests/test_budget.py` | 15480 | `8da7696ad2300340bcf030f762505ce06a21cb727ce7bf8ccc353449123f911c` |
| `source/deadline-v1-overlay/tests/test_campaign_guards.py` | `tests/test_campaign_guards.py` | 11458 | `f706574c9aa9c5e7e471d404709a1d3d4ada6bab6fecde002b44d04df5dbbba6` |
| `source/deadline-v1-overlay/tests/test_cli_help.py` | `tests/test_cli_help.py` | 2042 | `624816cfa69ad92deea2cf9d0c193fc49f606888fe05371a01adf21baddfeab8` |
| `source/deadline-v1-overlay/tests/test_config.py` | `tests/test_config.py` | 6387 | `e5d6a88bbe643628ada8a6a72e4ae459075eb67db8d12b4fd995373b76141028` |
| `source/deadline-v1-overlay/tests/test_manifest.py` | `tests/test_manifest.py` | 8312 | `10f1c6094ad2929d52e9bc1d0376f997f9d671fcd764c546b3399fa0b875772c` |
| `source/deadline-v1-overlay/tests/test_markers.py` | `tests/test_markers.py` | 12831 | `16ae3480f610b9260ee803d7fbb05e1bd07a62b46711ba96fab809859b1b901b` |
| `source/deadline-v1-overlay/tests/test_metrics.py` | `tests/test_metrics.py` | 6615 | `b8e100dd1742d38ffd97638ca1426236ffb57a4b6520442b9ac71c982bbc18a6` |
| `source/deadline-v1-overlay/tests/test_protocol.py` | `tests/test_protocol.py` | 16388 | `25abb34e1ad614db3a0b2f9c115e1f4f0381c023f1810b0c963ebd864dc470f8` |
| `source/deadline-v1-overlay/tests/test_report.py` | `tests/test_report.py` | 21066 | `28d0287897820fa670a66773128f74c712d71ec7a580075eebb56b00b344a5a6` |
| `source/deadline-v1-overlay/tests/test_sampling.py` | `tests/test_sampling.py` | 5911 | `9ffcf64d39afb054e0d859d2eb54756ef6318e0096ff36b09a9c4e43020b442c` |
| `source/deadline-v1-overlay/tests/test_stages_stub.py` | `tests/test_stages_stub.py` | 19762 | `ba314c62ec9e411d3c0fac9d4a2a222054a34f342311e7eb73d1fe1355a3bec3` |
| `source/deadline-v1-overlay/tests/test_syntax.py` | `tests/test_syntax.py` | 2146 | `83dd3ecb85fa6e3da89b37ae140156981fa58bddae021ebb803010445acc810a` |
| `source/deadline-v1-overlay/tests/test_timing_cli.py` | `tests/test_timing_cli.py` | 15669 | `43850d677a2184fe68f742a082aa7c8bf14c1031de6781320e75fcf0bff28030` |
