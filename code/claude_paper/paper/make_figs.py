#!/usr/bin/env python3
"""
Publication-quality figures for the MS-EGCA manuscript (IEEE TIM, two-column).

All figures are vector PDF at IEEE column widths. In-figure text is plain
matplotlib mathtext -- no LaTeX escape sequences (they would render literally).
"""
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from data import DS1, DS2, ABL, PERCLASS, STATS_DS1, STATS_DS2, CORRUPT

FIG = os.path.join(os.path.dirname(os.path.abspath(__file__)), "figs")
os.makedirs(FIG, exist_ok=True)

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Nimbus Roman No9 L", "Times New Roman", "DejaVu Serif"],
    "mathtext.fontset": "cm",
    "font.size": 8,
    "axes.labelsize": 8,
    "axes.titlesize": 8.5,
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,
    "legend.fontsize": 6.8,
    "axes.linewidth": 0.6,
    "xtick.major.width": 0.5,
    "ytick.major.width": 0.5,
    "xtick.major.size": 2.4,
    "ytick.major.size": 2.4,
    "grid.linewidth": 0.35,
    "lines.linewidth": 1.15,
    "savefig.dpi": 600,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.015,
})

COL1, COL2 = 3.45, 7.16                  # IEEE single / double column (inches)
CP = "#b52b2b"                           # proposed (crimson)
CB = "#20303f"                           # ink
CG = "#95a5a6"                           # muted grey
BLUE, ORANGE, TEAL = "#2b6ca3", "#d98032", "#1a8f79"
PALETTE = ["#2b6ca3", "#d98032", "#1a8f79", "#8e5fa8", "#c0392b"]


def style(ax, grid="both"):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color("#5d6d7e")
    if grid in ("both", "y"):
        ax.grid(axis="y", alpha=0.30, ls=":", color=CG)
    if grid in ("both", "x"):
        ax.grid(axis="x", alpha=0.30, ls=":", color=CG)
    ax.set_axisbelow(True)


def save(fig, name):
    fig.savefig(os.path.join(FIG, name + ".pdf"))
    plt.close(fig)
    print("  wrote", name + ".pdf")


# ==================================================== Fig 1: overall pipeline
def fig_architecture():
    fig, ax = plt.subplots(figsize=(COL2, 1.80))
    ax.set_xlim(0, 100); ax.set_ylim(0, 31); ax.axis("off")

    def box(x, y, w, h, label, fc, sub=None, fs=7.3, ec=CB, lw=0.75):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.35",
                                    fc=fc, ec=ec, lw=lw))
        ax.text(x + w / 2, y + h / 2 + (1.5 if sub else 0), label, ha="center",
                va="center", fontsize=fs, weight="bold", color=CB)
        if sub:
            ax.text(x + w / 2, y + h / 2 - 2.5, sub, ha="center", va="center",
                    fontsize=6.0, style="italic", color="#41576b")

    def arrow(x1, y1, x2, y2, ls="-", c=CB, lw=0.9):
        ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>",
                                     mutation_scale=8, lw=lw, ls=ls, color=c,
                                     shrinkA=0, shrinkB=0))

    y0, h0 = 12, 9.5
    box(0.5, y0, 11.5, h0, "Input", "#eef2f5", "224 x 224 x 3")
    arrow(12.0, y0 + h0 / 2, 15.6, y0 + h0 / 2)
    box(15.6, y0, 14.5, h0, "Sobel Gate", "#fdeedd", "edge prior  S")
    arrow(30.1, y0 + h0 / 2, 33.7, y0 + h0 / 2)
    box(33.7, y0, 17.5, h0, "MobileNetV3-S", "#e2eef8", "backbone F, 576-d")
    arrow(51.2, y0 + h0 / 2, 54.8, y0 + h0 / 2)
    box(54.8, y0, 15.5, h0, "MS-EGCA", "#fadedd", "attention (ours)",
        ec=CP, lw=1.2)
    arrow(70.3, y0 + h0 / 2, 73.9, y0 + h0 / 2)
    box(73.9, y0, 10.0, h0, "GAP", "#eef2f5")
    arrow(83.9, y0 + h0 / 2, 87.0, y0 + h0 / 2)
    box(87.0, y0, 12.5, h0, "Head", "#eef2f5", "dropout + FC")

    # optional, disabled branch
    box(45.0, 2.4, 27.0, 6.6, "DeiT-Tiny + cross-attention", "#f4f6f7",
        fs=6.7, ec=CG, lw=0.7)
    ax.text(58.5, 0.5, "optional branch, disabled in this study (Sec. III-E)",
            ha="center", va="center", fontsize=6.0, style="italic", color=CG)
    arrow(21.0, y0, 45.0, 6.6, ls=(0, (2.2, 2.2)), c=CG, lw=0.7)
    arrow(72.0, 6.6, 79.0, y0, ls=(0, (2.2, 2.2)), c=CG, lw=0.7)

    ax.text(50, 28.4, "Proposed lightweight recognition pipeline  "
            "(0.95 M parameters, 3.75 ms / image)",
            ha="center", fontsize=7.7, weight="bold", color=CB)
    save(fig, "fig1_architecture")


# ======================================================= Fig 2: MS-EGCA block
def fig_msegca():
    fig, ax = plt.subplots(figsize=(COL1, 2.25))
    ax.set_xlim(0, 100); ax.set_ylim(0, 74); ax.axis("off")

    def box(x, y, w, h, label, fc, fs=6.3, ec=CB):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.3",
                                    fc=fc, ec=ec, lw=0.65))
        ax.text(x + w / 2, y + h / 2, label, ha="center", va="center",
                fontsize=fs, color=CB)

    def arrow(x1, y1, x2, y2, c=CB):
        ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>",
                                     mutation_scale=6, lw=0.7, color=c,
                                     shrinkA=0, shrinkB=0))

    box(0.5, 31, 13.5, 12, r"$\mathbf{X}$" + "\n" + r"$C\times H\times W$",
        "#eef2f5", 6.2)

    # edge branch (top)
    ax.text(46, 71.5, "multi-scale edge branch", ha="center", fontsize=6.2,
            style="italic", color="#a9691f")
    arrow(14, 40, 22, 60)
    box(22, 54, 19, 12, "channel mean\n" + r"$g=\bar{\mathbf{X}}_c$", "#fdeedd")
    arrow(41, 60, 47.5, 60)
    box(47.5, 51.5, 26, 17,
        r"$3{\times}3,\,5{\times}5,\,7{\times}7$" + "\n" +
        r"$1{\times}1$ fuse" + "\n" + r"GN $+\ \sigma\ \rightarrow\ \mathbf{E}$",
        "#fdeedd", 6.1)

    # channel branch (middle)
    ax.text(46, 46.8, "ECA-style channel branch", ha="center", fontsize=6.2,
            style="italic", color="#1f618d")
    arrow(14, 37, 29, 37)
    box(29, 30, 34, 13.5,
        r"GAP $\rightarrow$ 1-D conv ($k$ adaptive)" + "\n" +
        r"$\rightarrow\ \sigma\ \rightarrow\ \mathbf{A}$", "#e2eef8", 6.1)

    # gate branch (bottom)
    ax.text(46, 1.8, "adaptive residual gate", ha="center", fontsize=6.2,
            style="italic", color="#a93226")
    arrow(14, 34, 27, 17)
    box(27, 8.5, 32, 13.5, r"GAP $\rightarrow$ MLP $\rightarrow\ \sigma$" + "\n" +
        r"$\rightarrow\ \alpha\in(0,1)$", "#fadedd", 6.1)

    # fusion
    box(77, 28, 22, 19, r"$\mathbf{X}\odot[(1-\alpha)$" + "\n" +
        r"$+\ \alpha\,\mathbf{E}\odot\mathbf{A}]$", "#e4f5ef", 6.3, ec=CP)
    arrow(73.5, 58, 88, 47)
    arrow(63, 37, 77, 37.5)
    arrow(59, 16, 88, 28)
    save(fig, "fig2_msegca")


# ============================================ Fig 3: accuracy-parameter Pareto
def fig_pareto():
    """2x2 efficiency panel: Pareto on both corpora + parameter efficiency +
    latency-accuracy plane. Replaces the former separate Figs. 3 and 8."""
    fig = plt.figure(figsize=(COL2, 3.70))
    gs = fig.add_gridspec(2, 2, hspace=0.42, wspace=0.22)
    axes = [fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1])]
    ax_eff = fig.add_subplot(gs[1, 0])
    ax_lat = fig.add_subplot(gs[1, 1])

    OFF1 = {  # DS1 offsets (dx, dy) in points
        "SqueezeNet 1.1": (7, -2), "ShuffleNetV2 x1.0": (7, 4),
        "MobileNetV2": (3, -10), "MobileNetV3-Small": (7, 4),
        "MobileNetV3-Large": (8, -3), "MNASNet 1.0": (-3, -10),
        "EfficientNet-B0": (-5, 8), "RepViT-M0.9": (0, -10),
        "MobileNetV4-Conv-S": (7, 1), "EdgeNeXt-XX-Small": (7, -7),
        "MS-EGCA (proposed)": (1, 12),
    }
    OFF2 = {
        "SqueezeNet 1.1": (-4, 9), "ShuffleNetV2 x0.5": (7, -2),
        "ShuffleNetV2 x1.0": (6, 5), "MobileNetV2": (-6, 7),
        "MobileNetV3-Small": (6, -6), "MobileNetV3-Large": (-6, -9),
        "MNASNet 1.0": (7, 0), "EfficientNet-B0": (2, 10),
        "RepViT-M0.9": (5, -3), "MobileNetV4-Conv-S": (7, 0),
        "EdgeNeXt-XX-Small": (-7, -5), "MS-EGCA (proposed)": (-7, -12),
    }

    for ax, rows, title, lo, off in zip(
            axes, [DS1, DS2],
            ["(a) DS1 – ten-way vehicle type",
             "(b) DS2 – military vs. civilian"],
            [95.4, 97.5], [OFF1, OFF2]):

        # Pareto staircase
        pts = sorted((r[4], r[5]) for r in rows)
        fx, fy, best = [], [], -1e9
        for px, py in pts:
            if py > best:
                best = py; fx.append(px); fy.append(py)
        fx.append(5.4); fy.append(fy[-1])
        ax.step(fx, fy, where="post", color=CG, lw=0.8, ls="--", zorder=1)
        ax.fill_between(fx, lo, fy, step="post", color=CG, alpha=0.07, zorder=0)

        for r in rows:
            x, y, name, prop = r[4], r[5], r[1], r[3] == "Proposed"
            ax.scatter(x, y, s=95 if prop else 26,
                       marker="*" if prop else "o",
                       c=CP if prop else "white",
                       edgecolors=CP if prop else BLUE,
                       linewidths=1.0 if prop else 0.85,
                       zorder=5 if prop else 3)
            dx, dy = off[name]
            ax.annotate(name.replace(" (proposed)", "\n(proposed)"), (x, y),
                        textcoords="offset points", xytext=(dx, dy),
                        fontsize=5.7, ha="left" if dx > 0 else "right",
                        va="center", zorder=6,
                        color=CP if prop else "#33475b",
                        weight="bold" if prop else "normal")
        ax.set_xlabel("Parameters (M)")
        ax.set_xlim(0, 5.4); ax.set_ylim(lo, 100.05)
        ax.set_title(title, fontsize=7.9, pad=3)
        style(ax)
    axes[0].set_ylabel("Sealed-test accuracy (%)")

    # ---- (c) parameter efficiency ----------------------------------------
    rows = sorted(DS1, key=lambda r: r[7] / r[4])
    labels = [r[1] for r in rows]
    eff = [r[7] / r[4] for r in rows]
    cols = [CP if r[3] == "Proposed" else "#9fb2c0" for r in rows]
    yy = np.arange(len(labels))
    ax_eff.barh(yy, eff, 0.62, color=cols, ec=CB, lw=0.35)
    ax_eff.set_yticks(yy); ax_eff.set_yticklabels(labels, fontsize=6.0)
    ax_eff.set_xlabel("Macro-F1 per million parameters (DS1)")
    ax_eff.set_title("(c) Parameter efficiency", fontsize=7.9, pad=3)
    for i, v in enumerate(eff):
        ax_eff.text(v + 1.8, yy[i], f"{v:.1f}", va="center", fontsize=5.8,
                    color=CP if cols[i] == CP else "#41576b")
    ax_eff.set_xlim(0, 152)
    style(ax_eff, grid="x")

    # ---- (d) latency-accuracy plane --------------------------------------
    for r in DS1:
        prop = r[3] == "Proposed"
        ax_lat.scatter(r[15], r[5], s=95 if prop else 26,
                       marker="*" if prop else "o",
                       c=CP if prop else "white",
                       edgecolors=CP if prop else BLUE,
                       linewidths=1.0 if prop else 0.85, zorder=5 if prop else 3)
    offl = {"SqueezeNet 1.1": (7, 0), "ShuffleNetV2 x1.0": (7, 0),
            "MobileNetV2": (-6, -6), "MobileNetV3-Small": (-6, 5),
            "MobileNetV3-Large": (-6, 6), "MNASNet 1.0": (-6, 0),
            "EfficientNet-B0": (5, 2), "RepViT-M0.9": (-5, 6),
            "MobileNetV4-Conv-S": (-6, -5), "EdgeNeXt-XX-Small": (6, 0),
            "MS-EGCA (proposed)": (7, -8)}
    for r in DS1:
        dx, dy = offl[r[1]]
        prop = r[3] == "Proposed"
        ax_lat.annotate(r[1].replace(" (proposed)", "\n(proposed)"),
                        (r[15], r[5]), textcoords="offset points",
                        xytext=(dx, dy), fontsize=5.7,
                        ha="left" if dx > 0 else "right", va="center",
                        color=CP if prop else "#33475b",
                        weight="bold" if prop else "normal", zorder=6)
    ax_lat.axvspan(0, 4.0, color=TEAL, alpha=0.06, zorder=0)
    ax_lat.text(2.0, 99.62, "< 4 ms  (> 250 FPS)", fontsize=5.9, ha="center",
                color=TEAL, style="italic")
    ax_lat.set_xlabel("Inference latency per image (ms)")
    ax_lat.set_ylabel("Sealed-test accuracy (%)")
    ax_lat.set_title("(d) Latency-accuracy plane, DS1", fontsize=7.9, pad=3)
    ax_lat.set_xlim(0, 6.9); ax_lat.set_ylim(95.5, 99.75)
    style(ax_lat)

    save(fig, "fig3_pareto")


# =================================================== Fig 4: significance forest
def fig_forest():
    fig, axes = plt.subplots(1, 2, figsize=(COL2, 2.00))
    for ax, st, title in zip(axes, [STATS_DS1, STATS_DS2],
                             ["(a) DS1 – ten-way", "(b) DS2 – binary"]):
        names = [s[0] for s in st][::-1]
        d = [s[4] for s in st][::-1]
        lo = [s[5] for s in st][::-1]
        hi = [s[6] for s in st][::-1]
        sig = [s[8] for s in st][::-1]
        y = np.arange(len(names))
        ax.axvspan(-2.4, 0, color=BLUE, alpha=0.045, zorder=0)
        for i in range(len(names)):
            c = CP if sig[i] else CG
            ax.plot([lo[i], hi[i]], [y[i], y[i]], color=c, lw=1.15, zorder=2,
                    solid_capstyle="round")
            for e in (lo[i], hi[i]):
                ax.plot([e, e], [y[i] - .17, y[i] + .17], color=c, lw=0.8)
            ax.scatter(d[i], y[i], s=19 if sig[i] else 15, zorder=3,
                       c=c if sig[i] else "white", edgecolors=c, linewidths=0.9,
                       marker="s" if sig[i] else "o")
        ax.axvline(0, color=CB, lw=0.75, ls="--", zorder=1)
        ax.set_yticks(y); ax.set_yticklabels(names, fontsize=6.0)
        ax.set_xlabel("$\\Delta$F1  (proposed $-$ baseline, pp)")
        ax.set_title(title, fontsize=7.9, pad=3)
        ax.set_xlim(-2.4, 4.7)
        style(ax, grid="x")
        ax.tick_params(labelleft=True)
    fig.text(0.5, -0.045, "filled squares: significant at $p<0.05$ "
             "(McNemar, Holm-corrected);  open circles: not significant;  "
             "bars are 95% bootstrap CIs",
             ha="center", fontsize=6.1, style="italic", color="#41576b")
    fig.tight_layout(pad=0.35, w_pad=1.6)
    save(fig, "fig4_forest")


# ========================================================= Fig 5: ablation bars
def fig_ablation():
    fig, ax = plt.subplots(figsize=(COL1, 1.95))
    ids = [a[0] for a in ABL]
    d1 = [a[7] for a in ABL]
    d2 = [a[9] for a in ABL]
    x = np.arange(len(ids)); w = 0.39
    c1 = [CP if a[0] in ("A6", "A7", "A8") else BLUE for a in ABL]
    c2 = [ORANGE if a[0] not in ("A6", "A7", "A8") else "#e08c8c" for a in ABL]
    ax.bar(x - w / 2, d1, w, color=c1, ec=CB, lw=0.4, label="DS1 (ten-way)")
    ax.bar(x + w / 2, d2, w, color=c2, ec=CB, lw=0.4, label="DS2 (binary)")
    ax.axhline(ABL[0][7], color=BLUE, lw=0.7, ls=":", zorder=1)
    ax.axhline(ABL[0][9], color=ORANGE, lw=0.7, ls=":", zorder=1)
    ax.annotate("A0 baseline", (8.55, ABL[0][9] + 0.03), fontsize=5.6,
                color=ORANGE, ha="right")
    ax.set_xticks(x); ax.set_xticklabels(ids, fontsize=6.6)
    ax.set_ylim(97.8, 99.5)
    ax.set_ylabel("Macro-F1 (%)")
    ax.set_xlabel("Ablation variant")
    ax.legend(frameon=False, ncol=2, loc="upper left", fontsize=6.2,
              handlelength=1.1, columnspacing=1.0)
    style(ax, grid="y")
    fig.tight_layout(pad=0.3)
    save(fig, "fig5_ablation")


# ================================================== Fig 6: per-class + calibration
def fig_perclass_ece():
    fig, axes = plt.subplots(1, 2, figsize=(COL2, 2.00))

    ax = axes[0]
    names = [p[0] for p in PERCLASS][::-1]
    prec = np.array([p[4] for p in PERCLASS][::-1]) * 100
    rec = np.array([p[5] for p in PERCLASS][::-1]) * 100
    y = np.arange(len(names)); h = 0.38
    ax.barh(y + h / 2, prec, h, color=BLUE, ec=CB, lw=0.35, label="Precision")
    ax.barh(y - h / 2, rec, h, color=CP, ec=CB, lw=0.35, label="Recall")
    ax.set_yticks(y); ax.set_yticklabels(names, fontsize=6.0)
    ax.set_xlim(94, 100.6); ax.set_xlabel("%")
    ax.legend(frameon=False, fontsize=6.3, loc="lower left", handlelength=1.0)
    ax.set_title("(a) Per-class performance, DS1", fontsize=7.9, pad=3)
    style(ax, grid="x")

    ax = axes[1]
    rows = sorted(DS1, key=lambda r: r[16])
    labels = [r[1] for r in rows]
    ece = [r[16] for r in rows]
    cols = [CP if r[3] == "Proposed" else "#9fb2c0" for r in rows]
    yy = np.arange(len(labels))
    ax.barh(yy, ece, 0.62, color=cols, ec=CB, lw=0.35)
    ax.set_yticks(yy); ax.set_yticklabels(labels, fontsize=6.0)
    ax.set_xlabel("Expected calibration error (DS1, lower is better)")
    ax.set_title("(b) Calibration quality", fontsize=7.9, pad=3)
    for i, v in enumerate(ece):
        ax.text(v + 0.0035, yy[i], f"{v:.3f}", va="center", fontsize=5.7,
                color=CP if cols[i] == CP else "#41576b")
    ax.set_xlim(0, 0.19)
    style(ax, grid="x")
    fig.tight_layout(pad=0.35, w_pad=1.4)
    save(fig, "fig6_perclass_ece")


# ==================================================== Fig 7: corruption curves
def fig_corruption():
    fig, axes = plt.subplots(1, 2, figsize=(COL1, 1.75), sharey=True)
    marks = ["o", "s", "^", "D", "v"]
    for ax, idx, title in zip(axes, [0, 2],
                              ["(a) DS1 – ten-way",
                               "(b) DS2 – binary"]):
        for (name, sev), m, c in zip(CORRUPT.items(), marks, PALETTE):
            xs = sorted(sev)
            ys = [sev[s][idx] for s in xs]
            ax.plot(xs, ys, marker=m, ms=3.4, color=c, label=name, mfc="white",
                    mew=0.9)
        ax.set_xticks([1, 3, 5])
        ax.set_xlabel("Corruption severity")
        ax.set_title(title, fontsize=7.9, pad=3)
        ax.set_ylim(20, 103)
        style(ax)
    axes[0].set_ylabel("Accuracy (%)")
    axes[0].legend(frameon=False, fontsize=6.2, loc="lower left", ncol=2,
                   handlelength=1.3, columnspacing=0.9)
    fig.tight_layout(pad=0.35, w_pad=1.0)
    save(fig, "fig7_corruption")


# ================================================== Fig 8: efficiency trade-off
def fig_efficiency():
    """(a) parameter efficiency (F1 per M params); (b) latency-accuracy plane."""
    fig, axes = plt.subplots(1, 2, figsize=(COL2, 2.00))

    # ---- (a) F1 per million parameters, DS1
    ax = axes[0]
    rows = sorted(DS1, key=lambda r: r[7] / r[4])
    labels = [r[1] for r in rows]
    eff = [r[7] / r[4] for r in rows]
    cols = [CP if r[3] == "Proposed" else "#9fb2c0" for r in rows]
    yy = np.arange(len(labels))
    ax.barh(yy, eff, 0.62, color=cols, ec=CB, lw=0.35)
    ax.set_yticks(yy); ax.set_yticklabels(labels, fontsize=6.0)
    ax.set_xlabel("Macro-F1 per million parameters (DS1)")
    ax.set_title("(a) Parameter efficiency", fontsize=7.9, pad=3)
    for i, v in enumerate(eff):
        ax.text(v + 1.8, yy[i], f"{v:.1f}", va="center", fontsize=5.8,
                color=CP if cols[i] == CP else "#41576b")
    ax.set_xlim(0, 152)
    style(ax, grid="x")

    # ---- (b) latency vs accuracy
    ax = axes[1]
    for r in DS1:
        prop = r[3] == "Proposed"
        ax.scatter(r[15], r[5], s=95 if prop else 26,
                   marker="*" if prop else "o",
                   c=CP if prop else "white",
                   edgecolors=CP if prop else BLUE,
                   linewidths=1.0 if prop else 0.85, zorder=5 if prop else 3)
    off = {"SqueezeNet 1.1": (7, 0), "ShuffleNetV2 x1.0": (7, 0),
           "MobileNetV2": (-6, -6), "MobileNetV3-Small": (-6, 5),
           "MobileNetV3-Large": (-6, 6), "MNASNet 1.0": (-6, 0),
           "EfficientNet-B0": (5, 2), "RepViT-M0.9": (-5, 6),
           "MobileNetV4-Conv-S": (-6, -5), "EdgeNeXt-XX-Small": (6, 0),
           "MS-EGCA (proposed)": (7, -8)}
    for r in DS1:
        dx, dy = off[r[1]]
        prop = r[3] == "Proposed"
        ax.annotate(r[1].replace(" (proposed)", "\n(proposed)"), (r[15], r[5]),
                    textcoords="offset points", xytext=(dx, dy), fontsize=5.7,
                    ha="left" if dx > 0 else "right", va="center",
                    color=CP if prop else "#33475b",
                    weight="bold" if prop else "normal", zorder=6)
    ax.axvspan(0, 4.0, color=TEAL, alpha=0.06, zorder=0)
    ax.text(2.0, 99.62, "< 4 ms  (> 250 FPS)", fontsize=5.9, ha="center",
            color=TEAL, style="italic")
    ax.set_xlabel("Inference latency per image (ms)")
    ax.set_ylabel("Sealed-test accuracy (%)")
    ax.set_title("(b) Latency-accuracy plane, DS1", fontsize=7.9, pad=3)
    ax.set_xlim(0, 6.9); ax.set_ylim(95.5, 99.75)
    style(ax)
    fig.tight_layout(pad=0.35, w_pad=1.4)
    save(fig, "fig8_efficiency")


if __name__ == "__main__":
    print("Generating figures ->", FIG)
    fig_architecture()
    fig_msegca()
    fig_pareto()
    fig_forest()
    fig_ablation()
    fig_perclass_ece()
    fig_corruption()
    print("done")
