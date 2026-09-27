#!/usr/bin/env python3
"""
Emit every LaTeX table for the MS-EGCA manuscript directly from data.py.
Each file contains a complete float environment and is \input at the exact
point in paper.tex where the table should appear.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from data import (DS1, DS2, ABL, PERCLASS, STATS_DS1, STATS_DS2, CORRUPT,
                  SPLITS_DS1, SPLITS_DS2, FINGERPRINT_DS1, EXTENDED_DS1)

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "tables")
os.makedirs(OUT, exist_ok=True)

PROP = "MS-EGCA (proposed)"


def esc(s):
    return s.replace("&", "\\&").replace("_", "\\_")


def w(name, body):
    with open(os.path.join(OUT, name), "w", encoding="utf-8") as f:
        f.write(body)
    print("  wrote", name)


def bold(s):
    return "\\textbf{" + str(s) + "}"


# ---------------------------------------------------------------- Table I
def t_datasets():
    r = []
    r.append("\\begin{table}[H]")
    r.append("\\centering")
    r.append("\\caption{Composition of the two evaluation corpora. Both are "
             "exactly class-balanced; the test partition of each is sealed "
             "before any training and touched once.}")
    r.append("\\label{tab:datasets}")
    r.append("\\setlength{\\tabcolsep}{4pt}")
    r.append("\\begin{tabular}{llrrr}")
    r.append("\\toprule")
    r.append("Corpus & Partition & Images & Classes & Per class \\\\")
    r.append("\\midrule")
    r.append("\\multirow{3}{*}{\\shortstack[l]{DS1\\\\(ten-way)}}")
    for i, (sp, tot, per) in enumerate(SPLITS_DS1):
        r.append(f" & {sp} & {tot:,} & 10 & {per:,} \\\\")
    r.append("\\midrule")
    r.append("\\multirow{3}{*}{\\shortstack[l]{DS2\\\\(binary)}}")
    for sp, tot, per in SPLITS_DS2:
        r.append(f" & {sp} & {tot:,} & 2 & {per:,} \\\\")
    r.append("\\bottomrule")
    r.append("\\end{tabular}")
    r.append("\\\\[2pt]")
    r.append("\\footnotesize DS1 sealed-test SHA-256 fingerprint: "
             "\\texttt{" + FINGERPRINT_DS1 + "} ($n=4{,}240$).")
    r.append("\\end{table}")
    w("tab1_datasets.tex", "\n".join(r))


# --------------------------------------------------------------- Table II
def t_config():
    rows = [
        ("Input resolution", "$224\\times224$ RGB"),
        ("Backbone initialisation", "ImageNet-1k pretrained"),
        ("Optimiser", "AdamW, $\\beta=(0.9,0.999)$"),
        ("Learning rate / schedule", "$1\\times10^{-4}$, cosine annealing"),
        ("Weight decay", "$1\\times10^{-2}$"),
        ("Batch size", "16"),
        ("Maximum epochs", "41 (early stop, patience 4)"),
        ("Loss", "cross-entropy, label smoothing $0.1$"),
        ("Gradient clipping", "$\\ell_2$ norm $1.0$"),
        ("Weight averaging", "EMA, decay $0.999$"),
        ("Test-time augmentation", "horizontal flip"),
        ("Augmentation", "resize 256, centre crop 224, camouflage jitter ($p{=}0.5$)"),
        ("Random seeds", "$\\{42,123,2026,7,999\\}$"),
        ("Bootstrap resamples", "2{,}000"),
        ("Latency protocol", "20 warm-up + 100 timed runs, batch 1"),
        ("Hardware", "AMD GPU via DirectML, FP32"),
    ]
    r = ["\\begin{table}[H]", "\\centering",
         "\\caption{Training and evaluation configuration, held identical "
         "across every architecture and both corpora.}",
         "\\label{tab:config}", "\\setlength{\\tabcolsep}{3pt}",
         "\\footnotesize",
         "\\begin{tabular}{@{}l p{0.56\\columnwidth}@{}}", "\\toprule",
         "Setting & Value \\\\", "\\midrule"]
    for k, v in rows:
        r.append(f"{k} & {v} \\\\")
    r += ["\\bottomrule", "\\end{tabular}", "\\end{table}"]
    w("tab2_config.tex", "\n".join(r))


# -------------------------------------------------------------- Table III
def t_main():
    """Wide double-column benchmark: both datasets side by side."""
    d2 = {r[1]: r for r in DS2}
    r = ["\\begin{table*}[!t]", "\\centering",
         "\\caption{Sealed-test benchmark of eleven efficient architectures and "
         "the proposed model on both corpora. Accuracy and macro-F1 are the mean "
         "$\\pm$ standard deviation over five seeds; all other columns are computed "
         "from the pooled predictions. Best value per column in \\textbf{bold}; "
         "the proposed row is shaded. ECE is expected calibration error (lower is "
         "better) and latency is single-image inference time.}",
         "\\label{tab:main}", "\\setlength{\\tabcolsep}{4pt}",
         "\\footnotesize",
         "\\begin{tabular}{llcc|cccc|cccc}", "\\toprule",
         "& & & & \\multicolumn{4}{c|}{\\textbf{DS1 -- ten-way vehicle type}} "
         "& \\multicolumn{4}{c}{\\textbf{DS2 -- military vs.\\ civilian}} \\\\",
         "\\cmidrule(lr){5-8}\\cmidrule(lr){9-12}",
         "ID & Architecture & Type & Par.(M) "
         "& Acc.(\\%) & F1(\\%) & ECE & Lat.(ms) "
         "& Acc.(\\%) & F1(\\%) & ECE & Lat.(ms) \\\\",
         "\\midrule"]

    best1 = {"acc": max(x[5] for x in DS1), "f1": max(x[7] for x in DS1),
             "mcc": max(x[9] for x in DS1), "ece": min(x[16] for x in DS1),
             "lat": min(x[15] for x in DS1), "par": min(x[4] for x in DS1)}
    best2 = {"acc": max(x[5] for x in DS2), "f1": max(x[7] for x in DS2),
             "mcc": max(x[9] for x in DS2), "ece": min(x[15] for x in DS2),
             "lat": min(x[14] for x in DS2), "par": min(x[4] for x in DS2)}

    def f(v, ref, fmt, lower=False):
        s = fmt.format(v)
        return bold(s) if abs(v - ref) < 1e-9 else s

    for row in DS1:
        idx, name, yr, typ, par, acc, astd, f1, fstd, mcc = row[:10]
        ece, lat = row[16], row[15]
        prop = typ == "Proposed"
        pre = "\\rowcolor{propshade} " if prop else ""
        nm = bold(esc(name)) if prop else esc(name)

        def accf(a, s, ref):
            core = bold(f"{a:.2f}") if abs(a - ref) < 1e-9 else f"{a:.2f}"
            return core + f"\\,{{\\tiny$\\pm${s:.2f}}}"

        c1 = [accf(acc, astd, best1["acc"]), f(f1, best1["f1"], "{:.2f}"),
              f(ece, best1["ece"], "{:.4f}"), f(lat, best1["lat"], "{:.2f}")]

        if name in d2:
            q = d2[name]
            c2 = [accf(q[5], q[6], best2["acc"]), f(q[7], best2["f1"], "{:.2f}"),
                  f(q[15], best2["ece"], "{:.4f}"), f(q[14], best2["lat"], "{:.2f}")]
        else:
            c2 = ["--"] * 4
        r.append(pre + " & ".join(
            [idx, nm, typ, f"{par:.2f}"] + c1 + c2) + " \\\\")

    # DS2-only model
    q = d2["ShuffleNetV2 x0.5"]
    r.append("\\midrule")
    r.append(" & ".join(
        ["C1$^{\\ast}$", "ShuffleNetV2 x0.5", "CNN", bold(f"{q[4]:.2f}")]
        + ["--"] * 4
        + [f"{q[5]:.2f}\\,{{\\tiny$\\pm${q[6]:.2f}}}", f"{q[7]:.2f}",
           f"{q[15]:.4f}", f"{q[14]:.2f}"]) + " \\\\")
    r += ["\\bottomrule", "\\end{tabular}", "\\\\[2pt]",
          "\\footnotesize Parameter counts are for the ten-class head; the "
          "binary head differs by $<0.01$\\,M. $^{\\ast}$Evaluated on DS2 only "
          "(Sec.~\\ref{sec:protocol}).",
          "\\end{table*}"]
    w("tab3_main.tex", "\n".join(r))


# --------------------------------------------------------------- Table IV
def t_extended():
    r = ["\\begin{table}[H]", "\\centering",
         "\\caption{Extended discriminative metrics on the DS1 sealed test set "
         "(macro-averaged one-vs-rest). DOR is the diagnostic odds ratio, "
         "capped at $10^4$.}",
         "\\label{tab:extended}", "\\setlength{\\tabcolsep}{3pt}",
         "\\footnotesize", "\\begin{tabular}{lccccc}", "\\toprule",
         "Architecture & Spec.(\\%) & G-Mean(\\%) & $\\kappa$ & LR$^{+}$ & Brier \\\\",
         "\\midrule"]
    ext = {k: (v[0], v[4], v[8], v[5], v[9])
           for k, v in EXTENDED_DS1.items()}
    for row in DS1:
        name = row[1]
        sp, gm, kp, lr, br = ext[name]
        prop = row[3] == "Proposed"
        pre = "\\rowcolor{propshade} " if prop else ""
        nm = bold(esc(name)) if prop else esc(name)
        r.append(pre + f"{nm} & {sp:.2f} & {gm:.2f} & {kp:.4f} & "
                 f"{lr:.1f} & {br:.4f} \\\\")
    r += ["\\bottomrule", "\\end{tabular}", "\\end{table}"]
    w("tab4_extended.tex", "\n".join(r))


# ---------------------------------------------------------------- Table V
def t_stats():
    r = ["\\begin{table*}[!t]", "\\centering",
         "\\caption{Paired comparison of the proposed model against every "
         "baseline on the pooled sealed-test predictions. $b$ and $c$ are the "
         "McNemar discordant counts (baseline-only correct / proposed-only "
         "correct). $\\Delta$F1 is proposed $-$ baseline in percentage points, "
         "with a 95\\,\\% bootstrap confidence interval over 2{,}000 resamples. "
         "$p$-values are Holm-corrected within each corpus (ten comparisons on DS1, "
         "eleven on DS2).}",
         "\\label{tab:stats}", "\\setlength{\\tabcolsep}{4pt}", "\\footnotesize",
         "\\begin{tabular}{l|ccrl|ccrl}", "\\toprule",
         "& \\multicolumn{4}{c|}{\\textbf{DS1 -- ten-way}} "
         "& \\multicolumn{4}{c}{\\textbf{DS2 -- binary}} \\\\",
         "\\cmidrule(lr){2-5}\\cmidrule(lr){6-9}",
         "Baseline & $b$/$c$ & $\\Delta$F1 & 95\\,\\% CI & $p$ "
         "& $b$/$c$ & $\\Delta$F1 & 95\\,\\% CI & $p$ \\\\",
         "\\midrule"]
    m2 = {s[0]: s for s in STATS_DS2}

    def cell(t):
        name, b, c, p, d, lo, hi, cd, sig = t
        dd = bold(f"{d:+.2f}") if sig else f"{d:+.2f}"
        pp = p + ("\\,\\ding{51}" if sig else "")
        return [f"{b}/{c}", dd, f"[{lo:+.2f},\\,{hi:+.2f}]", pp]

    for s in STATS_DS1:
        name = s[0]
        c2 = cell(m2[name]) if name in m2 else ["--"] * 4
        r.append(" & ".join([esc(name)] + cell(s) + c2) + " \\\\")
    r.append("\\midrule")
    r.append(" & ".join(["ShuffleNetV2 x0.5"] + ["--"] * 4 +
                        cell(m2["ShuffleNetV2 x0.5"])) + " \\\\")
    r += ["\\bottomrule", "\\end{tabular}", "\\\\[2pt]",
          "\\footnotesize \\ding{51} marks significance at $\\alpha=0.05$ after "
          "Holm correction; bold $\\Delta$F1 likewise. Positive $\\Delta$F1 "
          "favours the proposed model.",
          "\\end{table*}"]
    w("tab5_stats.tex", "\n".join(r))


# --------------------------------------------------------------- Table VI
def t_ablation():
    r = ["\\begin{table}[H]", "\\centering",
         "\\caption{Component ablation on both corpora. All variants share the "
         "MobileNetV3-Small backbone and the training recipe of "
         "Table~\\ref{tab:config}. $\\Delta$F1 is relative to A0.}",
         "\\label{tab:ablation}", "\\setlength{\\tabcolsep}{2.6pt}", "\\footnotesize",
         "\\begin{tabular}{@{}ll|cc|cc@{}}", "\\toprule",
         "& & \\multicolumn{2}{c|}{DS1} & \\multicolumn{2}{c}{DS2} \\\\",
         "\\cmidrule(lr){3-4}\\cmidrule(lr){5-6}",
         "ID & Configuration & F1(\\%) & $\\Delta$ & F1(\\%) & $\\Delta$ \\\\",
         "\\midrule"]
    b1, b2 = ABL[0][7], ABL[0][9]
    for a in ABL:
        aid, cfg, sob, attn, deit, xat, a1, f1, a2, f2 = a
        prop = aid in ("A6", "A7", "A8")
        pre = "\\rowcolor{propshade} " if aid == "A8" else ""
        nm = bold(cfg) if aid == "A8" else cfg
        r.append(pre + f"{aid} & {nm} & "
                 f"{f1:.2f} & {f1-b1:+.2f} & {f2:.2f} & {f2-b2:+.2f} \\\\")
    r += ["\\bottomrule", "\\end{tabular}", "\\\\[2pt]",
          "\\footnotesize $^{\\dagger}$The DeiT-Tiny cross-attention branch was "
          "disabled on the DirectML backend (Sec.~\\ref{sec:deit}); A7 and A8 "
          "therefore reduce to A6 and are reported for completeness.",
          "\\end{table}"]
    w("tab6_ablation.tex", "\n".join(r))


# -------------------------------------------------------------- Table VII
def t_perclass():
    r = ["\\begin{table}[H]", "\\centering",
         "\\caption{Per-class sealed-test performance of the proposed model on "
         "DS1. Each class contributes 424 test images.}",
         "\\label{tab:perclass}", "\\setlength{\\tabcolsep}{3pt}", "\\footnotesize",
         "\\begin{tabular}{lrrrccc}", "\\toprule",
         "Class & TP & FP & FN & Prec. & Rec. & G-Mean \\\\", "\\midrule"]
    worst = min(PERCLASS, key=lambda p: p[9])[0]
    for c in PERCLASS:
        nm, tp, fp, fn, pr, rc, sp, fpr, fnr, gm = c
        lab = esc(nm)
        r.append(f"{lab} & {tp} & {fp} & {fn} & {pr:.4f} & {rc:.4f} & "
                 f"{gm:.4f} \\\\")
    n = len(PERCLASS)
    mp = sum(c[4] for c in PERCLASS) / n
    mr = sum(c[5] for c in PERCLASS) / n
    mg = sum(c[9] for c in PERCLASS) / n
    stp = sum(c[1] for c in PERCLASS)
    sfp = sum(c[2] for c in PERCLASS)
    sfn = sum(c[3] for c in PERCLASS)
    r += ["\\midrule",
          f"\\textit{{Macro average}} & {stp} & {sfp} & {sfn} & "
          f"{mp:.4f} & {mr:.4f} & {mg:.4f} \\\\",
          "\\bottomrule", "\\end{tabular}", "\\end{table}"]
    w("tab7_perclass.tex", "\n".join(r))


# ------------------------------------------------------------- Table VIII
def t_corruption():
    r = ["\\begin{table}[H]", "\\centering",
         "\\caption{Accuracy (\\%) of the proposed model under five synthetic "
         "field corruptions at three severities, on both sealed test sets.}",
         "\\label{tab:corruption}", "\\setlength{\\tabcolsep}{4pt}",
         "\\footnotesize", "\\begin{tabular}{l|ccc|ccc}", "\\toprule",
         "& \\multicolumn{3}{c|}{DS1 -- ten-way} & \\multicolumn{3}{c}{DS2 -- binary} \\\\",
         "\\cmidrule(lr){2-4}\\cmidrule(lr){5-7}",
         "Corruption & $s{=}1$ & $s{=}3$ & $s{=}5$ & $s{=}1$ & $s{=}3$ & $s{=}5$ \\\\",
         "\\midrule"]
    for name, sev in CORRUPT.items():
        v = [sev[s] for s in (1, 3, 5)]
        r.append(f"{name} & {v[0][0]:.2f} & {v[1][0]:.2f} & {v[2][0]:.2f} & "
                 f"{v[0][2]:.2f} & {v[1][2]:.2f} & {v[2][2]:.2f} \\\\")
    # mean row
    m1 = [sum(CORRUPT[k][s][0] for k in CORRUPT) / len(CORRUPT) for s in (1, 3, 5)]
    m2 = [sum(CORRUPT[k][s][2] for k in CORRUPT) / len(CORRUPT) for s in (1, 3, 5)]
    r.append("\\midrule")
    r.append("\\textit{Mean} & " + " & ".join(f"{x:.2f}" for x in m1) + " & " +
             " & ".join(f"{x:.2f}" for x in m2) + " \\\\")
    r += ["\\bottomrule", "\\end{tabular}", "\\end{table}"]
    w("tab8_corruption.tex", "\n".join(r))


if __name__ == "__main__":
    print("Generating tables ->", OUT)
    t_datasets(); t_config(); t_main(); t_extended()
    t_stats(); t_ablation(); t_perclass(); t_corruption()
    print("done")
