"""
Verified experimental data extracted from crash_FREE_T.ipynb (run v24.0).
Every number below is transcribed from the [INLINE PREVIEW] tables printed by
STEP 8/9 of the pipeline. Do not edit by hand -- re-extract from the notebook.

NOTE on DS1 model list: the resumed run reused progress/*.pkl written under an
older model list, which inserted ShuffleNetV2 x0.5 at index C1 and shifted the
remainder. The .pkl 'name' fields (authoritative, written at training time)
confirm DS1 C1=ShuffleNetV2 x1.0, C2=MobileNetV2, C3=MobileNetV3-Small.
Slot C4 then re-trained MobileNetV3-Small from scratch, producing a duplicate.
The stale C3 row is therefore EXCLUDED from the paper; DS1 reports 10 distinct
baselines + the proposed model. DS1 has no ShuffleNetV2 x0.5 result.
"""

# ---------------------------------------------------------------- DS1 (10-way)
# ID, Model, Year, Type, Params(M), Acc, Acc_std, F1, F1_std, MCC, AUC,
# Spec, G-Mean, Kappa, Brier, Lat(ms), ECE
DS1 = [
    ("C0",  "SqueezeNet 1.1",      2016, "CNN",      0.73, 95.94, 0.55, 95.93, 0.56, 0.9549, 0.9986, 99.43, 97.10, 0.9429, 0.0109, 0.88, 0.1549),
    ("C1",  "ShuffleNetV2 x1.0",   2018, "CNN",      1.26, 97.09, 0.25, 97.09, 0.25, 0.9677, 0.9990, 99.67, 98.35, 0.9672, 0.0068, 3.33, 0.1331),
    ("C2",  "MobileNetV2",         2018, "CNN",      2.24, 98.11, 0.23, 98.11, 0.23, 0.9791, 0.9995, 99.78, 98.90, 0.9782, 0.0046, 2.73, 0.1163),
    ("C4",  "MobileNetV3-Small",   2019, "CNN",      1.53, 98.98, 0.15, 98.98, 0.15, 0.9887, 0.9998, 99.90, 99.51, 0.9903, 0.0031, 2.29, 0.0985),
    ("C5",  "MobileNetV3-Large",   2019, "CNN",      4.21, 99.11, 0.12, 99.11, 0.12, 0.9902, 0.9997, 99.92, 99.60, 0.9921, 0.0023, 2.92, 0.0980),
    ("C6",  "MNASNet 1.0",         2019, "CNN",      3.12, 98.73, 0.15, 98.73, 0.15, 0.9859, 0.9998, 99.88, 99.41, 0.9882, 0.0031, 2.29, 0.0877),
    ("C7",  "EfficientNet-B0",     2019, "CNN",      4.02, 99.17, 0.09, 99.17, 0.09, 0.9908, 0.9994, 99.92, 99.59, 0.9919, 0.0022, 3.95, 0.0894),
    ("C15", "RepViT-M0.9",         2024, "Hybrid",   4.73, 98.75, 0.12, 98.74, 0.13, 0.9861, 0.9998, 99.85, 99.26, 0.9853, 0.0041, 6.04, 0.1228),
    ("C16", "MobileNetV4-Conv-S",  2024, "CNN",      2.51, 98.29, 0.24, 98.29, 0.24, 0.9810, 0.9996, 99.80, 99.01, 0.9803, 0.0044, 2.35, 0.1073),
    ("C17", "EdgeNeXt-XX-Small",   2022, "CNN",      1.16, 96.92, 0.37, 96.91, 0.38, 0.9657, 0.9988, 99.59, 97.91, 0.9586, 0.0065, 4.48, 0.0817),
    ("P0",  "MS-EGCA (proposed)",  2026, "Proposed", 0.95, 98.45, 0.07, 98.45, 0.07, 0.9828, 0.9996, 99.83, 99.13, 0.9827, 0.0035, 3.75, 0.0916),
]

# ---------------------------------------------------------------- DS2 (binary)
DS2 = [
    ("C0",  "SqueezeNet 1.1",      2016, "CNN",      0.72, 98.95, 0.19, 98.95, 0.19, 0.9790, 0.9992, 99.10, 99.10, 0.9820, 0.94, 0.0613),
    ("C1",  "ShuffleNetV2 x0.5",   2018, "CNN",      0.34, 98.03, 0.42, 98.03, 0.42, 0.9607, 0.9981, 97.27, 97.26, 0.9454, 3.60, 0.0562),
    ("C2",  "ShuffleNetV2 x1.0",   2018, "CNN",      1.26, 98.99, 0.22, 98.99, 0.22, 0.9799, 0.9986, 98.67, 98.67, 0.9735, 2.97, 0.0535),
    ("C3",  "MobileNetV2",         2018, "CNN",      2.23, 99.38, 0.02, 99.38, 0.02, 0.9877, 0.9999, 99.38, 99.38, 0.9875, 2.49, 0.0541),
    ("C4",  "MobileNetV3-Small",   2019, "CNN",      1.52, 99.07, 0.21, 99.07, 0.21, 0.9815, 0.9989, 99.22, 99.22, 0.9844, 2.54, 0.0493),
    ("C5",  "MobileNetV3-Large",   2019, "CNN",      4.20, 99.45, 0.14, 99.45, 0.14, 0.9889, 0.9996, 99.18, 99.18, 0.9836, 3.01, 0.0519),
    ("C6",  "MNASNet 1.0",         2019, "CNN",      3.10, 97.86, 2.53, 97.85, 2.55, 0.9591, 0.9999, 98.79, 98.78, 0.9758, 2.41, 0.0508),
    ("C7",  "EfficientNet-B0",     2019, "CNN",      4.01, 99.59, 0.13, 99.59, 0.13, 0.9917, 0.9998, 99.57, 99.57, 0.9914, 3.88, 0.0504),
    ("C15", "RepViT-M0.9",         2024, "Hybrid",   4.72, 99.57, 0.08, 99.57, 0.08, 0.9914, 0.9999, 99.45, 99.45, 0.9891, 5.93, 0.0538),
    ("C16", "MobileNetV4-Conv-S",  2024, "CNN",      2.50, 98.35, 0.58, 98.35, 0.58, 0.9669, 0.9978, 98.44, 98.44, 0.9688, 2.31, 0.0527),
    ("C17", "EdgeNeXt-XX-Small",   2022, "CNN",      1.16, 98.20, 1.05, 98.20, 1.05, 0.9646, 0.9989, 99.18, 99.18, 0.9836, 4.38, 0.0489),
    ("P0",  "MS-EGCA (proposed)",  2026, "Proposed", 0.95, 98.91, 0.13, 98.91, 0.13, 0.9782, 0.9983, 99.10, 99.10, 0.9820, 3.57, 0.0498),
]

# -------------------------------------- DS1 extended metrics (REPORT_DS1 §3)
# model -> (Specificity%, NPV%, FPR%, FNR%, G-Mean%, LR+, LR-, DOR, Kappa, Brier)
EXTENDED_DS1 = {
    "SqueezeNet 1.1":      (99.43, 99.43, 0.57, 5.14, 97.10,  263.162, 0.0517, 5800.69, 0.9429, 0.0109),
    "ShuffleNetV2 x1.0":   (99.67, 99.67, 0.33, 2.95, 98.35,  389.021, 0.0296, 8435.57, 0.9672, 0.0068),
    "MobileNetV2":         (99.78, 99.78, 0.22, 1.96, 98.90,  625.409, 0.0196, 9188.01, 0.9782, 0.0046),
    "MobileNetV3-Small":   (99.90, 99.90, 0.10, 0.87, 99.51, 2285.636, 0.0087, 9999.00, 0.9903, 0.0031),
    "MobileNetV3-Large":   (99.92, 99.92, 0.08, 0.71, 99.60, 4021.155, 0.0071, 9999.00, 0.9921, 0.0023),
    "MNASNet 1.0":         (99.88, 99.88, 0.12, 1.06, 99.41, 2535.855, 0.0106, 9999.00, 0.9882, 0.0031),
    "EfficientNet-B0":     (99.92, 99.92, 0.08, 0.73, 99.59, 4832.901, 0.0073, 9999.00, 0.9919, 0.0022),
    "RepViT-M0.9":         (99.85, 99.85, 0.15, 1.32, 99.26,  879.722, 0.0132, 9794.42, 0.9853, 0.0041),
    "MobileNetV4-Conv-S":  (99.80, 99.80, 0.20, 1.77, 99.01, 1023.482, 0.0177, 9505.35, 0.9803, 0.0044),
    "EdgeNeXt-XX-Small":   (99.59, 99.59, 0.41, 3.73, 97.91,  272.147, 0.0374, 6930.75, 0.9586, 0.0065),
    "MS-EGCA (proposed)":  (99.83, 99.83, 0.17, 1.56, 99.13,  895.589, 0.0156, 9705.61, 0.9827, 0.0035),
}

# -------------------------------------- DS2 extended metrics (REPORT_DS2 §3)
# model -> (Specificity%, NPV%, FPR%, FNR%, G-Mean%, LR+, LR-, DOR, Kappa)
# Brier is NaN throughout the DS2 run and is therefore not reported for DS2.
EXTENDED_DS2 = {
    "SqueezeNet 1.1":      (99.10, 99.10, 0.90, 0.90, 99.10, 121.558, 0.0091, 9999.00, 0.9820),
    "ShuffleNetV2 x0.5":   (97.27, 97.29, 2.73, 2.73, 97.26,  43.385, 0.0279, 1552.29, 0.9454),
    "ShuffleNetV2 x1.0":   (98.67, 98.68, 1.33, 1.33, 98.67,  81.299, 0.0134, 6051.78, 0.9735),
    "MobileNetV2":         (99.38, 99.38, 0.62, 0.62, 99.38, 169.667, 0.0063, 9999.00, 0.9875),
    "MobileNetV3-Small":   (99.22, 99.22, 0.78, 0.78, 99.22, 151.119, 0.0079, 9999.00, 0.9844),
    "MobileNetV3-Large":   (99.18, 99.18, 0.82, 0.82, 99.18, 166.375, 0.0082, 9999.00, 0.9836),
    "MNASNet 1.0":         (98.79, 98.81, 1.21, 1.21, 98.78, 335.052, 0.0121, 9999.00, 0.9758),
    "EfficientNet-B0":     (99.57, 99.57, 0.43, 0.43, 99.57, 250.464, 0.0043, 9999.00, 0.9914),
    "RepViT-M0.9":         (99.45, 99.45, 0.55, 0.55, 99.45, 182.000, 0.0055, 9999.00, 0.9891),
    "MobileNetV4-Conv-S":  (98.44, 98.44, 1.56, 1.56, 98.44,  71.712, 0.0158, 4530.12, 0.9688),
    "EdgeNeXt-XX-Small":   (99.18, 99.19, 0.82, 0.82, 99.18, 246.000, 0.0082, 9999.00, 0.9836),
    "MS-EGCA (proposed)":  (99.10, 99.11, 0.90, 0.90, 99.10, 242.117, 0.0090, 9999.00, 0.9820),
}

# ---------------------------------------------------- DS2 per-class (proposed)
# Class, TP, FP, FN, Precision, Recall, Specificity, FPR, FNR, G-Mean
PERCLASS_DS2 = [
    ("Military", 1278, 20,  3, 0.9846, 0.9977, 0.9844, 0.0156, 0.0023, 0.9910),
    ("Other",    1261,  3, 20, 0.9976, 0.9844, 0.9977, 0.0023, 0.0156, 0.9910),
]

# Both corpora are exactly class-balanced. The pipeline therefore recorded
# weighted-F1 identical to macro-F1, and balanced accuracy identical to plain
# accuracy, for every model on both corpora (verified against REPORT_*.md).
# We report the macro forms only and state the equality in the text.
BALANCED_METRIC_IDENTITY = True

# ------------------------------------------------------------------- ablation
# ID, Config, Sobel, Attn, DeiT, XAttn, DS1 Acc, DS1 F1, DS2 Acc, DS2 F1
ABL = [
    ("A0", "Backbone only",                    "--", "None",    "--", "--", 98.33, 98.32, 99.18, 99.18),
    ("A1", "Backbone + SE",                    "--", "SE",      "--", "--", 98.11, 98.11, 98.56, 98.56),
    ("A2", "Backbone + CBAM",                  "--", "CBAM",    "--", "--", 98.51, 98.51, 98.71, 98.71),
    ("A3", "Backbone + ECA",                   "--", "ECA",     "--", "--", 98.30, 98.29, 98.95, 98.95),
    ("A4", "Backbone + CoordAtt",              "--", "CA",      "--", "--", 98.18, 98.18, 98.63, 98.63),
    ("A5", "Backbone + Sobel",                 "Y",  "None",    "--", "--", 98.37, 98.37, 98.71, 98.71),
    ("A6", "Backbone + Sobel + \\textbf{MS-EGCA}", "Y", "MS-EGCA", "--", "--", 98.44, 98.44, 99.10, 99.10),
    ("A7", "A6 + DeiT$^{\\dagger}$",              "Y", "MS-EGCA", "Y",  "--", 98.44, 98.44, 99.10, 99.10),
    ("A8", "Proposed (full)$^{\\dagger}$",     "Y",  "MS-EGCA", "Y",  "Y",  98.44, 98.44, 99.10, 99.10),
]

# ---------------------------------------------------- DS1 per-class (proposed)
# Class, TP, FP, FN, Precision, Recall, Specificity, FPR, FNR, G-Mean
PERCLASS = [
    ("Anti-aircraft",                    424,  6,  0, 0.9860, 1.0000, 0.9984, 0.0016, 0.0000, 0.9992),
    ("Armd. combat support veh.",        411, 17, 13, 0.9603, 0.9693, 0.9955, 0.0045, 0.0307, 0.9824),
    ("Armd. personnel carriers",         420,  6,  4, 0.9859, 0.9906, 0.9984, 0.0016, 0.0094, 0.9945),
    ("Infantry fighting veh.",           409,  8, 15, 0.9808, 0.9646, 0.9979, 0.0021, 0.0354, 0.9811),
    ("Light armored veh.",               408,  4, 16, 0.9903, 0.9623, 0.9990, 0.0010, 0.0377, 0.9804),
    ("Mine-protected veh.",              420,  3,  4, 0.9929, 0.9906, 0.9992, 0.0008, 0.0094, 0.9949),
    ("Prime movers & trucks",            422,  5,  2, 0.9883, 0.9953, 0.9987, 0.0013, 0.0047, 0.9970),
    ("Self-propelled artillery",         419,  2,  5, 0.9952, 0.9882, 0.9995, 0.0005, 0.0118, 0.9938),
    ("Light utility veh.",               419, 13,  5, 0.9699, 0.9882, 0.9966, 0.0034, 0.0118, 0.9924),
    ("Tanks",                            422,  2,  2, 0.9953, 0.9953, 0.9995, 0.0005, 0.0047, 0.9974),
]

# ------------------------------------------- paired stats: P0 vs each baseline
# baseline, b, c, p-value(str), dF1%, CI_lo, CI_hi, Cohen_d, sig
STATS_DS1 = [
    ("SqueezeNet 1.1",     21, 173, "$<10^{-4}$",  3.607,  2.991,  4.241,  0.170, True),
    ("ShuffleNetV2 x1.0",  38,  97, "$<10^{-4}$",  1.398,  0.869,  1.932,  0.078, True),
    ("MobileNetV2",        33,  50, "0.079",       0.408, -0.023,  0.847,  0.029, False),
    ("MobileNetV3-Small",  41,  12, "$1.2\\!\\times\\!10^{-4}$", -0.687, -1.025, -0.345, -0.061, True),
    ("MobileNetV3-Large",  46,  10, "$<10^{-4}$", -0.849, -1.196, -0.495, -0.074, True),
    ("MNASNet 1.0",        45,  24, "0.016",      -0.496, -0.873, -0.108, -0.039, True),
    ("EfficientNet-B0",    46,  11, "$<10^{-4}$", -0.829, -1.206, -0.492, -0.071, True),
    ("RepViT-M0.9",        36,  26, "0.253",      -0.236, -0.594,  0.126, -0.020, False),
    ("MobileNetV4-Conv-S", 36,  45, "0.374",       0.221, -0.194,  0.634,  0.015, False),
    ("EdgeNeXt-XX-Small",  33, 125, "$<10^{-4}$",  2.183,  1.598,  2.780,  0.113, True),
]

STATS_DS2 = [
    ("SqueezeNet 1.1",     16, 16, "0.860",  0.002, -0.430, 0.430,  0.000, False),
    ("ShuffleNetV2 x0.5",  13, 60, "$<10^{-4}$", 1.840, 1.210, 2.498, 0.109, True),
    ("ShuffleNetV2 x1.0",  15, 26, "0.118",  0.424, -0.078, 0.898,  0.034, False),
    ("MobileNetV2",        18, 11, "0.265", -0.275, -0.664, 0.117, -0.026, False),
    ("MobileNetV3-Small",  17, 14, "0.719", -0.121, -0.547, 0.312, -0.011, False),
    ("MobileNetV3-Large",  18, 16, "0.864", -0.085, -0.547, 0.351, -0.007, False),
    ("MNASNet 1.0",        20, 28, "0.312",  0.307, -0.234, 0.859,  0.023, False),
    ("EfficientNet-B0",    19,  7, "0.031", -0.476, -0.898, -0.078, -0.047, True),
    ("RepViT-M0.9",        19, 10, "0.137", -0.353, -0.781, 0.039, -0.033, False),
    ("MobileNetV4-Conv-S", 16, 33, "0.022",  0.657,  0.118, 1.210,  0.048, True),
    ("EdgeNeXt-XX-Small",  19, 17, "0.868", -0.084, -0.547, 0.390, -0.007, False),
]

# ------------------------------------------- multiple-comparison correction --
# PROVENANCE NOTE. The experimental pipeline (`mcnemar_test` + `make_table_stats`
# in the run notebook) records the RAW McNemar p-value with continuity
# correction and flags significance as `p < 0.05`. It applies NO
# multiple-comparison correction: the source contains no occurrence of holm,
# bonferroni, multipletests, fdr or statsmodels. Because ten comparisons are
# made on DS1 and eleven on DS2, we apply Holm-Bonferroni here as a documented
# post-hoc step, and report the raw p-value alongside the adjusted decision.
#
# The pipeline truncates very small p-values to the string "<1e-4". We therefore
# substitute the upper bound 1e-4, which is conservative: the true p is smaller,
# so a rejection at 1e-4 remains a rejection at the true value.

def _p_numeric(s):
    """Recover a numeric p-value from the string recorded by the pipeline."""
    import re
    s = str(s)
    if "<" in s:
        return 1e-4
    m = re.search(r"([\d.]+)\s*\\!?\\times\\!?10\^\{?(-?\d+)\}?", s)
    if m:
        return float(m.group(1)) * 10 ** int(m.group(2))
    return float(re.sub(r"[^0-9.eE+-]", "", s))


def holm_reject(pvals, alpha=0.05):
    """Holm-Bonferroni step-down. Returns a list of booleans."""
    m = len(pvals)
    order = sorted(range(m), key=lambda i: pvals[i])
    rej = [False] * m
    for k, i in enumerate(order):
        if pvals[i] <= alpha / (m - k):
            rej[i] = True
        else:
            break                      # step-down: stop at the first failure
    return rej


def with_holm(stats_rows, alpha=0.05):
    """Return [(name, b, c, p_str, dF1, lo, hi, d, raw_sig, holm_sig), ...]."""
    ps = [_p_numeric(r[3]) for r in stats_rows]
    rej = holm_reject(ps, alpha)
    return [tuple(r) + (h,) for r, h in zip(stats_rows, rej)]


STATS_DS1_HOLM = with_holm(STATS_DS1)
STATS_DS2_HOLM = with_holm(STATS_DS2)

# ---------------------------------------------------------------- corruptions
# corruption -> {severity: (DS1 acc, DS1 f1, DS2 acc, DS2 f1)}
CORRUPT = {
    "Haze":      {1: (97.49, 97.48, 99.34, 99.34), 3: (85.28, 85.10, 97.88, 97.88), 5: (51.93, 50.50, 85.52, 85.46)},
    "Blur":      {1: (97.03, 97.03, 99.44, 99.44), 3: (61.98, 61.75, 95.94, 95.94), 5: (37.21, 35.63, 90.95, 90.95)},
    "Noise":     {1: (91.53, 91.52, 98.94, 98.94), 3: (52.83, 53.07, 96.91, 96.91), 5: (28.90, 27.33, 84.61, 84.33)},
    "Occlusion": {1: (97.60, 97.60, 99.38, 99.38), 3: (94.64, 94.63, 98.85, 98.85), 5: (69.04, 69.21, 95.66, 95.66)},
    "Mud":       {1: (97.90, 97.90, 99.38, 99.38), 3: (97.74, 97.74, 99.06, 99.06), 5: (96.90, 96.90, 98.97, 98.97)},
}

# ------------------------------------------------------------------- datasets
SPLITS_DS1 = [("Train", 11890, 1189), ("Validation", 5090, 509), ("Test (sealed)", 4240, 424)]
SPLITS_DS2 = [("Train", 12090, 6045), ("Validation", 3204, 1602), ("Test (sealed)", 2562, 1281)]

FINGERPRINT_DS1 = "446ff4fb7338e740"

# PROVENANCE WARNING -- DO NOT REPORT THESE AS ACCURACY.
# The run logs "Cross-domain: n=... gap=...pp", and we originally read that as a
# transfer accuracy loss. Inspection of the source shows it is NOT:
#
#     (mc if int(yb[i]) == idx_m else nc2).append(float(probs[i].max()))
#     "conf_gap": float(np.mean(mc) - np.mean(nc2))
#
# The quantity is the difference in MEAN MAXIMUM SOFTMAX CONFIDENCE between
# images whose label equals idx_m and all others, on the other corpus. No
# accuracy is computed in that block. For DS1 the grouping compares one of ten
# classes against the remaining nine, which is not a military/civilian split at
# all. The numbers are therefore not interpretable as a transfer measurement and
# are EXCLUDED from the paper. Retained here only to document the decision.
_UNUSED_CONF_GAP_DS1 = 23.1   # confidence gap, not accuracy; see note above
_UNUSED_CONF_GAP_DS2 = 0.4    # confidence gap, not accuracy; see note above
