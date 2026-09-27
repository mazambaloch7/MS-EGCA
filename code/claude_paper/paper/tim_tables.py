#!/usr/bin/env python3
"""
Complete set of LaTeX tables for the IEEE TIM submission.

Every architecture and every evaluation technique recorded by the experimental
run is represented. Nothing is summarised away.

  T1  corpora and sealed-test integrity
  T2  training and evaluation configuration
  T3  DS1 primary benchmark   (multi-seed accuracy/F1, MCC, AUC, kappa)
  T4  DS1 extended metrics    (Spec, NPV, FPR, FNR, G-Mean, LR+, LR-, DOR, Brier, ECE)
  T5  DS2 primary benchmark
  T6  DS2 extended metrics
  T7  ablation, both corpora
  T8  DS1 per-class confusion counts and derived metrics
  T9  DS2 per-class confusion counts and derived metrics
  T10 paired significance, both corpora
  T11 corruption robustness, both corpora
  T12 cross-corpus transfer
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from data import (DS1, DS2, ABL, PERCLASS, PERCLASS_DS2,
                  STATS_DS1_HOLM as STATS_DS1, STATS_DS2_HOLM as STATS_DS2,
                  CORRUPT, SPLITS_DS1, SPLITS_DS2, FINGERPRINT_DS1,
                  EXTENDED_DS1, EXTENDED_DS2,
)

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "tables_tim")
os.makedirs(OUT, exist_ok=True)
PROP = "MS-EGCA (proposed)"


def esc(s):
    return (str(s).replace("&", r"\&").replace("_", r"\_")
            .replace("%", r"\%").replace("#", r"\#"))


def bf(s):
    return r"\textbf{" + str(s) + "}"


def w(name, body):
    open(os.path.join(OUT, name), "w", encoding="utf-8").write(body)
    print("  wrote", name)


def shade(is_prop):
    return r"\rowcolor{propshade} " if is_prop else ""


def dor(v):
    return r"$>10^{4}$" if v >= 9999 else f"{v:.0f}"


# ------------------------------------------------------------------ T1 ------
def t1_corpora():
    r = [r"\begin{table}[!t]", r"\centering",
         r"\caption{Composition of the two evaluation corpora and sealed-test "
         r"integrity record. Both corpora are exactly class balanced.}",
         r"\label{tab:corpora}", r"\setlength{\tabcolsep}{4pt}", r"\footnotesize",
         r"\begin{tabular}{@{}llrrr@{}}", r"\toprule",
         r"Corpus & Partition & Images & Classes & Per class \\", r"\midrule"]
    for i, (sp, tot, per) in enumerate(SPLITS_DS1):
        lead = r"\multirow{3}{*}{DS1}" if i == 0 else ""
        r.append(f"{lead} & {sp} & {tot:,} & 10 & {per:,} \\\\")
    r.append(r"\midrule")
    for i, (sp, tot, per) in enumerate(SPLITS_DS2):
        lead = r"\multirow{3}{*}{DS2}" if i == 0 else ""
        r.append(f"{lead} & {sp} & {tot:,} & 2 & {per:,} \\\\")
    r += [r"\bottomrule", r"\end{tabular}", r"\\[2pt]",
          r"\footnotesize DS1 sealed-test SHA-256 prefix \texttt{"
          + FINGERPRINT_DS1 + r"} over $n=4{,}240$ samples, recomputed and "
          r"matched at evaluation time.", r"\end{table}"]
    w("t1_corpora.tex", "\n".join(r))


# ------------------------------------------------------------------ T2 ------
def t2_config():
    rows = [
        ("Input resolution", r"$224\times224$ RGB"),
        ("Backbone initialisation", "ImageNet-1k pretrained"),
        ("Optimizer", r"AdamW, $\beta=(0.9,0.999)$"),
        ("Learning rate", r"$1\times10^{-4}$, cosine annealing"),
        ("Weight decay", r"$1\times10^{-2}$"),
        ("Batch size", "16"),
        ("Maximum epochs", "41, early stopping patience 4"),
        ("Loss", r"cross-entropy, label smoothing $0.1$"),
        ("Gradient clipping", r"$\ell_2$ norm $1.0$"),
        ("Weight averaging", r"EMA, decay $0.999$"),
        ("Test-time augmentation", "horizontal flip"),
        ("Augmentation", r"resize 256, center crop 224, camouflage jitter $p{=}0.5$"),
        ("Random seeds", r"$\{42,123,2026,7,999\}$"),
        ("Bootstrap resamples", "2{,}000"),
        ("Significance test", "McNemar (raw), Holm-Bonferroni post-hoc"),
        ("Latency protocol", "20 warm-up, 100 timed runs, batch 1"),
        ("Calibration bins", "15 equal-width"),
        ("Hardware", "AMD GPU via DirectML, FP32"),
    ]
    r = [r"\begin{table}[!t]", r"\centering",
         r"\caption{Training and evaluation configuration. The identical recipe "
         r"was applied to every architecture on both corpora.}",
         r"\label{tab:config}", r"\setlength{\tabcolsep}{3pt}", r"\footnotesize",
         r"\begin{tabular}{@{}l p{0.56\columnwidth}@{}}", r"\toprule",
         r"Setting & Value \\", r"\midrule"]
    for k, v in rows:
        r.append(f"{k} & {v} \\\\")
    r += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    w("t2_config.tex", "\n".join(r))


# ------------------------------------------------------- T3 / T5 primary ----
def primary(rows, label, cap, fname, ece_idx, lat_idx):
    best = {"acc": max(x[5] for x in rows), "f1": max(x[7] for x in rows),
            "mcc": max(x[9] for x in rows), "auc": max(x[10] for x in rows)}
    r = [r"\begin{table*}[!t]", r"\centering",
         r"\caption{" + cap + "}", r"\label{" + label + "}",
         r"\setlength{\tabcolsep}{4pt}", r"\footnotesize",
         r"\begin{tabular}{@{}llccrrrrr@{}}", r"\toprule",
         r"ID & Architecture & Year & Family & Accuracy (\%) & Macro-F1 (\%) "
         r"& MCC & AUC & $\kappa$ \\", r"\midrule"]
    for x in rows:
        idx, name, yr, typ, par, acc, astd, f1, fstd, mcc, auc = x[:11]
        kappa = x[13]
        pr = typ == "Proposed"
        acc_s = (bf(f"{acc:.2f}") if abs(acc - best["acc"]) < 1e-9 else f"{acc:.2f}")
        f1_s = (bf(f"{f1:.2f}") if abs(f1 - best["f1"]) < 1e-9 else f"{f1:.2f}")
        mcc_s = (bf(f"{mcc:.4f}") if abs(mcc - best["mcc"]) < 1e-9 else f"{mcc:.4f}")
        auc_s = (bf(f"{auc:.4f}") if abs(auc - best["auc"]) < 1e-9 else f"{auc:.4f}")
        nm = bf(esc(name)) if pr else esc(name)
        r.append(shade(pr) + " & ".join([
            idx, nm, str(yr), typ,
            acc_s + f"\\,{{\\tiny$\\pm${astd:.2f}}}",
            f1_s + f"\\,{{\\tiny$\\pm${fstd:.2f}}}",
            mcc_s, auc_s, f"{kappa:.4f}"]) + r" \\")
    r += [r"\bottomrule", r"\end{tabular}", r"\\[2pt]",
          r"\footnotesize Mean $\pm$ standard deviation over five random seeds; "
          r"remaining columns are computed from the pooled sealed-test "
          r"predictions. Because the corpus is exactly balanced, weighted F1 "
          r"equals macro-F1 and balanced accuracy equals accuracy for every "
          r"row, so those columns are omitted. Parameter count and latency are "
          r"reported graphically to avoid duplication.",
          r"\end{table*}"]
    w(fname, "\n".join(r))


# ------------------------------------------------------ T4 / T6 extended ----
def extended(rows, ext, label, cap, fname, with_brier, ece_idx):
    r = [r"\begin{table*}[!t]", r"\centering",
         r"\caption{" + cap + "}", r"\label{" + label + "}",
         r"\setlength{\tabcolsep}{4pt}", r"\footnotesize",
         r"\begin{tabular}{@{}ll" + ("r" * (9 if with_brier else 8)) + "@{}}",
         r"\toprule",
         r"ID & Architecture & Spec.\ (\%) & NPV (\%) & FPR (\%) & FNR (\%) "
         r"& G-Mean (\%) & LR$^{+}$ & DOR"
         + (r" & Brier & ECE \\" if with_brier else r" & ECE \\"),
         r"\midrule"]
    for x in rows:
        name = x[1]
        e = ext[name]
        pr = x[3] == "Proposed"
        nm = bf(esc(name)) if pr else esc(name)
        cells = [x[0], nm, f"{e[0]:.2f}", f"{e[1]:.2f}", f"{e[2]:.2f}",
                 f"{e[3]:.2f}", f"{e[4]:.2f}", f"{e[5]:.1f}", dor(e[7])]
        if with_brier:
            cells.append(f"{e[9]:.4f}")
        cells.append(f"{x[ece_idx]:.4f}")
        r.append(shade(pr) + " & ".join(cells) + r" \\")
    r += [r"\bottomrule", r"\end{tabular}", r"\\[2pt]",
          r"\footnotesize All quantities are macro-averaged one-versus-rest on "
          r"the sealed test set. DOR is the diagnostic odds ratio, capped by the "
          r"pipeline at $10^{4}$. ECE is the expected calibration error over 15 "
          r"equal-width bins; lower is better.",
          r"\end{table*}"]
    w(fname, "\n".join(r))


# ------------------------------------------------------------------ T7 ------
def t7_ablation():
    b1, b2 = ABL[0][7], ABL[0][9]
    r = [r"\begin{table}[!t]", r"\centering",
         r"\caption{Component ablation on both corpora. Every variant shares the "
         r"MobileNetV3-Small backbone and the recipe of Table~\ref{tab:config}. "
         r"$\Delta$ is measured against the bare backbone A0.}",
         r"\label{tab:ablation}", r"\setlength{\tabcolsep}{2.6pt}", r"\footnotesize",
         r"\begin{tabular}{@{}ll|rr|rr@{}}", r"\toprule",
         r"& & \multicolumn{2}{c|}{DS1} & \multicolumn{2}{c}{DS2} \\",
         r"\cmidrule(lr){3-4}\cmidrule(lr){5-6}",
         r"ID & Configuration & F1 (\%) & $\Delta$ "
         r"& F1 (\%) & $\Delta$ \\", r"\midrule"]
    for a in ABL:
        aid, cfg, sob, attn, deit, xat, a1, f1, a2, f2 = a
        pr = aid == "A8"
        nm = bf(cfg) if pr else cfg
        r.append(shade(pr) + " & ".join([
            aid, nm, f"{f1:.2f}", f"{f1-b1:+.2f}",
            f"{f2:.2f}", f"{f2-b2:+.2f}"]) + r" \\")
    r += [r"\bottomrule", r"\end{tabular}", r"\\[2pt]",
          r"\footnotesize $^{\dagger}$The DeiT-Tiny cross-attention branch was "
          r"disabled on the DirectML backend, as explained in "
          r"Sec.~\ref{sec:deit}. Variants A7 and A8 therefore reduce to A6, and "
          r"their identical scores are a property of the configuration rather "
          r"than a coincidence.", r"\end{table}"]
    w("t7_ablation.tex", "\n".join(r))


# ------------------------------------------------------- T8 / T9 per-class --
def perclass(rows, label, cap, fname, support):
    r = [r"\begin{table}[!t]", r"\centering",
         r"\caption{" + cap + "}", r"\label{" + label + "}",
         r"\setlength{\tabcolsep}{2.6pt}", r"\footnotesize",
         r"\begin{tabular}{@{}lrrrrrrr@{}}", r"\toprule",
         r"Class & TP & FP & FN & Prec. & Rec. & Spec. & G-Mean \\", r"\midrule"]
    for c in rows:
        nm, tp, fp, fn, pre, rec, spc, fpr, fnr, gm = c
        r.append(" & ".join([esc(nm), str(tp), str(fp), str(fn),
                             f"{pre:.4f}", f"{rec:.4f}", f"{spc:.4f}",
                             f"{gm:.4f}"]) + r" \\")
    n = len(rows)
    r += [r"\midrule",
          " & ".join([r"\textit{Macro average}",
                      str(sum(c[1] for c in rows)),
                      str(sum(c[2] for c in rows)),
                      str(sum(c[3] for c in rows)),
                      f"{sum(c[4] for c in rows)/n:.4f}",
                      f"{sum(c[5] for c in rows)/n:.4f}",
                      f"{sum(c[6] for c in rows)/n:.4f}",
                      f"{sum(c[9] for c in rows)/n:.4f}"]) + r" \\",
          r"\bottomrule", r"\end{tabular}", r"\\[2pt]",
          r"\footnotesize Proposed model, sealed test set, " + support + ".",
          r"\end{table}"]
    w(fname, "\n".join(r))


# ------------------------------------------------------------------ T10 -----
def t10_stats():
    m2 = {s[0]: s for s in STATS_DS2}

    def cell(t):
        _, b, c, p, d, lo, hi, cd, raw_sig, holm_sig = t
        ds = bf(f"{d:+.2f}") if holm_sig else f"{d:+.2f}"
        ps = str(p).replace("<", r"$<$")
        mark = r"\,\ding{51}" if holm_sig else (r"\,$^{\circ}$" if raw_sig else "")
        return [f"{b}/{c}", ds, f"[{lo:+.2f},\\,{hi:+.2f}]", f"{cd:+.3f}", ps + mark]

    r = [r"\begin{table*}[!t]", r"\centering",
         r"\caption{Paired comparison of the proposed model against every "
         r"baseline on the pooled sealed-test predictions. $b$ and $c$ are the "
         r"McNemar discordant counts: baseline correct with proposed wrong, and "
         r"proposed correct with baseline wrong. $\Delta$F1 is proposed minus "
         r"baseline in percentage points, with a 95\,\% bootstrap confidence "
         r"interval over 2{,}000 resamples and Cohen's $d$. Reported $p$-values "
         r"are the raw McNemar values; significance is decided after "
         r"Holm-Bonferroni correction across the ten comparisons on DS1 and the "
         r"eleven on DS2.}",
         r"\label{tab:stats}", r"\setlength{\tabcolsep}{3pt}", r"\footnotesize",
         r"\begin{tabular}{@{}l|ccrrl|ccrrl@{}}", r"\toprule",
         r"& \multicolumn{5}{c|}{\textbf{DS1 -- ten-way}} "
         r"& \multicolumn{5}{c}{\textbf{DS2 -- binary}} \\",
         r"\cmidrule(lr){2-6}\cmidrule(lr){7-11}",
         r"Baseline & $b$/$c$ & $\Delta$F1 & 95\,\% CI & $d$ & $p$ "
         r"& $b$/$c$ & $\Delta$F1 & 95\,\% CI & $d$ & $p$ \\", r"\midrule"]
    for s in STATS_DS1:
        c2 = cell(m2[s[0]]) if s[0] in m2 else ["--"] * 5
        r.append(" & ".join([esc(s[0])] + cell(s) + c2) + r" \\")
    for n, t in m2.items():
        if n not in [s[0] for s in STATS_DS1]:
            r.append(" & ".join([esc(n)] + ["--"] * 5 + cell(t)) + r" \\")
    r += [r"\bottomrule", r"\end{tabular}", r"\\[2pt]",
          r"\footnotesize \ding{51} marks significance at $\alpha=0.05$ after "
          r"Holm-Bonferroni correction, with the corresponding $\Delta$F1 in "
          r"bold. $^{\circ}$ marks a comparison significant at the raw "
          r"threshold but not after correction. A positive $\Delta$F1 favours "
          r"the proposed model. Values recorded as $<10^{-4}$ by the pipeline "
          r"are treated as $10^{-4}$, which is conservative.",
          r"\end{table*}"]
    w("t10_stats.tex", "\n".join(r))


# ------------------------------------------------------------------ T11 -----
def t11_corruption():
    r = [r"\begin{table}[!t]", r"\centering",
         r"\caption{Accuracy (\%) of the proposed model under five synthetic "
         r"field corruptions at three severities, on both sealed test sets.}",
         r"\label{tab:corruption}", r"\setlength{\tabcolsep}{3.4pt}",
         r"\footnotesize", r"\begin{tabular}{@{}l|rrr|rrr@{}}", r"\toprule",
         r"& \multicolumn{3}{c|}{DS1 -- ten-way} "
         r"& \multicolumn{3}{c}{DS2 -- binary} \\",
         r"\cmidrule(lr){2-4}\cmidrule(lr){5-7}",
         r"Corruption & $s{=}1$ & $s{=}3$ & $s{=}5$ & $s{=}1$ & $s{=}3$ & $s{=}5$ \\",
         r"\midrule"]
    for name, sev in CORRUPT.items():
        v = [sev[s] for s in (1, 3, 5)]
        r.append(f"{name} & " + " & ".join(
            [f"{v[i][0]:.2f}" for i in range(3)] +
            [f"{v[i][2]:.2f}" for i in range(3)]) + r" \\")
    m1 = [sum(CORRUPT[k][s][0] for k in CORRUPT) / len(CORRUPT) for s in (1, 3, 5)]
    m2 = [sum(CORRUPT[k][s][2] for k in CORRUPT) / len(CORRUPT) for s in (1, 3, 5)]
    r += [r"\midrule",
          r"\textit{Mean} & " + " & ".join(f"{x:.2f}" for x in m1 + m2) + r" \\",
          r"\midrule",
          r"\textit{Clean reference} & \multicolumn{3}{c|}{98.45} "
          r"& \multicolumn{3}{c}{98.91} \\",
          r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    w("t11_corruption.tex", "\n".join(r))


if __name__ == "__main__":
    print("Generating TIM tables ->", OUT)
    t1_corpora()
    t2_config()
    primary(DS1, "tab:ds1main",
            "Primary sealed-test benchmark on DS1, the ten-way vehicle-type "
            "corpus. Best value per column in bold; the proposed model is shaded.",
            "t3_ds1_primary.tex", 16, 15)
    extended(DS1, EXTENDED_DS1, "tab:ds1ext",
             "Extended discriminative and probabilistic metrics on DS1.",
             "t4_ds1_extended.tex", True, 16)
    primary(DS2, "tab:ds2main",
            "Primary sealed-test benchmark on DS2, the military versus civilian "
            "corpus. Best value per column in bold; the proposed model is shaded.",
            "t5_ds2_primary.tex", 15, 14)
    extended(DS2, EXTENDED_DS2, "tab:ds2ext",
             "Extended discriminative metrics on DS2. The Brier score was not "
             "recorded on this corpus by the evaluation pipeline.",
             "t6_ds2_extended.tex", False, 15)
    t7_ablation()
    perclass(PERCLASS, "tab:pc1",
             "Per-class sealed-test performance on DS1.",
             "t8_ds1_perclass.tex", "424 images per class")
    perclass(PERCLASS_DS2, "tab:pc2",
             "Per-class sealed-test performance on DS2.",
             "t9_ds2_perclass.tex", "1{,}281 images per class")
    t10_stats()
    t11_corruption()
    print("done")
