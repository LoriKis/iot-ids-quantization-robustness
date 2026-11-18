# Four-model comparison

| Model | Parent | Precision | Epochs | Accuracy, full test (n=3764) | Macro-F1, full test | State size (MB) | Latency b1 median (ms) | Latency b1 p95 (ms) | Latency b256 per example (ms) | Peak RSS (MB) | Peak RSS delta (MB) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| FP32-ordinary (V1) | - | FP32 | 25 | 61.96% | 56.80% | 1.449 | 2.8759 | 3.1439 | 1.24162 | 446.44 | 122.03 |
| INT8-PTQ-ordinary (V2) | V1 | INT8 | 25 | 61.77% | 56.89% | 0.497 | 5.3579 | 5.8583 | 1.05289 | 475.71 | 135.82 |
| FP32-adversarial (V3) | - | FP32 | 25 | 53.85% | 49.90% | 1.449 | 2.9001 | 3.2030 | 1.24674 | 446.56 | 122.04 |
| INT8-PTQ-adversarial (V4) | V3 | INT8 | 25 | 53.93% | 49.94% | 0.497 | 5.3973 | 6.0385 | 1.08363 | 471.94 | 131.63 |

Notes:
- Values are medians over 3 repetitions; each repetition runs V1..V4 interleaved, every measurement in a fresh child process, on 1 thread(s) (batch 1: 20 warm-up and 200 timed calls; batch 256: 5 warm-up and 30 timed calls; 256 fixed validation rows).
- Size: bytes of the serialized state (torch.save of the state_dict for FP32; the packed state.pt for INT8). MB = 1,000,000 bytes.
- Latency: eval mode, inference mode, one thread; timed region is row -> repeat to 50 positions -> forward -> argmax. Batch 1 and batch 256 are in the CSV; the per-example figure is the batch-256 median divided by the batch size.
- Memory: peak_rss_bytes = Linux VmHWM of a fresh child process after the timed runs; peak_rss_delta_bytes = VmHWM minus RSS right after model load (the baseline includes the loaded weights and the torch footprint, so the delta is inference-time working memory); peak_rss_sampled_delta_bytes = 10 ms sampler cross-check; weight memory = state_dict_bytes. MB = 1,000,000 bytes.
- Accuracy and macro-F1 are on the full held-out test set (macro-F1 over all 27 labels, zero_division=0). The robustness-subset clean values are separate and appear in the robustness table.
- Epochs: 25 (recipe: 25). Seed 1, one run per parent.
