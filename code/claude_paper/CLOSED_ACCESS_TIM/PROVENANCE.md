# Provenance record — what is verified, and what was corrected

Every number in the manuscript was traced back to the run notebook
`crash_FREE_T.ipynb` (source code **and** stdout log) or to the `REPORT_*.md`
tables it produced. This file records the audit outcome, including the claims
that turned out to be wrong and were removed.

---

## 1. Which results folder is authoritative

Both `cv20.html` and `crash_FREE_T.html` contain the same line:

```
BASE_DIR    = d:\python\study_1_militery\code
RESULTS_DIR = BASE_DIR / "DT_Q1_FINAL_results"
```

So **`D:\python\study_1_militery\code\DT_Q1_FINAL_results` is the authoritative
folder.** Neither notebook writes to `aftercrash\DT_Q1_FINAL_results`.

`aftercrash\DT_Q1_FINAL_results` is a **pre-crash partial copy**:

| | aftercrash (stale) | what the final run wrote |
|---|---|---|
| `tables/` | 0 files | 18 tables |
| `paper_pack/` | fingerprint only | + 28-page master PDF |
| per-model figures | C0–C11 (old roster) | C0–C7, C15–C17, P0 |
| newest file | 7 Aug | reports dated 9–10 Aug |

Its `model_log_DS1_primary.txt` records `C3 | MobileNetV3-Small | 98.77%` — the
stale duplicate row the paper excludes. **`98.77` appears nowhere in the
manuscript or tables.** Audit confirmed no paper value derives from this folder.

---

## 2. Corrections made after the provenance audit

### 2.1 Removed: cross-corpus transfer (was a headline finding)

The log line `Cross-domain: n=2562 gap=23.1pp` was read as a transfer accuracy
loss. The source shows it is not:

```python
(mc if int(yb[i]) == idx_m else nc2).append(float(probs[i].max()))
"conf_gap": float(np.mean(mc) - np.mean(nc2))
```

It is the difference in **mean maximum-softmax confidence** between two label
groups on the other corpus. No accuracy is computed. For DS1 the grouping
compares one of ten classes against the remaining nine, which is not a
military/civilian split at all.

**Action:** the Cross-Corpus Transfer subsection, its table, and the figure
panel were **deleted**. The deployment recommendation now rests only on the
corruption results, which are verified.

### 2.2 Corrected: the sealed-test fingerprint

Claimed: SHA-256 of the sorted test **file list**, "recomputed and matched at
evaluation time".

Actual: `hashlib.sha256(str(sorted(int(y) for y in labels)))` — it hashes the
sorted **label multiset**, certifying class composition only, not image content.
The pipeline also never re-compares the digest; there is no verification step.

**Action:** the protocol section now states exactly what the digest does and
does not certify. Algorithm 2 no longer claims a verification step.

### 2.3 Corrected: latency is the mean, not the median

`measure_inference` returns `(mean, std, p50, p95)`, but the reporting path uses
`lats.append(lm)` and `"latency_ms": float(np.mean(lats))`. The p50 is
discarded.

**Action:** the metrics section now says arithmetic mean over timed runs,
averaged again across seeds.

### 2.4 Corrected: multiple-comparison correction was never applied

The pipeline flags significance as `"sig": "Yes" if p < 0.05` on the **raw**
McNemar p-value. The notebook source contains zero occurrences of holm,
bonferroni, multipletests, fdr or statsmodels.

**Action:** Holm-Bonferroni is now applied as a documented post-hoc step in
`data.py` (`holm_reject`). Raw p-values are reported; significance is decided on
the corrected outcome. This **changed the results**:

| | raw | after Holm |
|---|---|---|
| DS1 significant | 7 of 10 | **6 of 10** |
| DS2 significant | 3 of 11 | **1 of 11** |

- DS1: MNASNet 1.0 (p = 0.016) loses significance.
- DS2: EfficientNet-B0 and MobileNetV4-Conv-S lose significance, so **no
  baseline is significantly better than the proposed model on DS2.**

Values the pipeline truncated to `<1e-4` are treated as 1e-4, which is
conservative. Table X marks corrected significance with a check and
raw-only significance with a circle.

### 2.5 Qualified: the "five seeds, identical recipe" claim

`REPORT_DS1` reports 8 of 12 models reused from cache; `REPORT_DS2` reports
12 of 12. For those rows the recipe identity rests on cached checkpoints, not on
the final run's log.

**Action:** the protocol section now discloses this.

---

## 3. Verified without change

Input resolution, backbone and pretrained weights, AdamW, learning rate and
cosine schedule, weight decay, batch size, 41 epochs, patience 4, label
smoothing 0.1, gradient clipping 1.0, EMA 0.999, horizontal-flip TTA, the
augmentation pipeline including camouflage jitter at p = 0.5, the five seeds,
2,000 bootstrap resamples, the 20 + 100 latency protocol, **ECE over 15 bins**
(`np.linspace(0,1,16)`, `range(15)` — the `n_bins=10` elsewhere belongs to the
reliability diagram, which the paper does not report), DirectML FP32, the five
corruption types at severities 1/3/5, and the DeiT branch being disabled because
of DirectML kernel coverage.

Architecture constants: C = 576, r = max(4, C/16) = 36, kernels {3,5,7},
k_c = 5, Sobel scaling 1/4, epsilon 1e-6, GroupNorm with one group, dropout 0.2.
Parameter overhead 83 + 3 + 2 + 5 + 20,809 + 3 = 20,905 ≈ 20.9 k = 2.2 % of
0.95 M — all recomputed and correct.

All derived quantities were recomputed independently: efficiency figures,
parameter ratios, throughput, corruption means, ECE ranks, accuracy ranks,
per-class statistics, the Pearson correlation of −0.61, and every significance
count now stated in the prose.

---

## 4. Known residual issues

1. `latex/tables/t12_transfer.tex` still exists on disk but is **not** included
   by the manuscript. Delete it; it is the removed transfer table.
2. AdamW β = (0.9, 0.999) is the PyTorch default, not an explicit setting in the
   run. Reported as a value, which is true, but it was not a deliberate choice.
3. The 1.46 ms attention overhead subtracts baseline C4 (MobileNetV3-Small)
   latency, not ablation variant A0. No A0 latency was recorded.
4. The headline accuracy is 98.45 % (five-seed mean) while the ablation table
   shows 98.44 % for A6–A8 (single seed). Both are traceable; the paper does not
   reconcile them explicitly.
