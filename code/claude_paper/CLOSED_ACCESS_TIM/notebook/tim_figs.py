#!/usr/bin/env python3
"""
Figures for the IEEE TIM submission.

Design rule (author's instruction): a quantity shown in a FIGURE must not be
repeated in a TABLE, and vice versa.

  FIGURES carry : cost axes (parameters, latency, efficiency), per-class
                  behaviour, calibration ranking, corruption response,
                  cross-corpus trends.
  TABLES  carry : accuracy / macro-F1 with dispersion, MCC, AUC, kappa,
                  specificity, G-mean, likelihood ratio, Brier, and all
                  paired-significance statistics.

Five figures:
  F1  method       : pipeline + MS-EGCA block
  F2  DS1 cost     : Pareto (params), efficiency ranking, latency plane
  F3  DS1 behaviour: per-class precision/recall, calibration ranking, corruption
  F4  DS2          : cost + behaviour on the binary corpus
  F5  cross-corpus : parity, robustness gap, transfer asymmetry, significance
"""
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from data import (DS1, DS2, PERCLASS, CORRUPT,
                  STATS_DS1_HOLM as STATS_DS1, STATS_DS2_HOLM as STATS_DS2)

FIG = os.path.join(os.path.dirname(os.path.abspath(__file__)), "figs_tim")
os.makedirs(FIG, exist_ok=True)

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Nimbus Roman No9 L", "Times New Roman", "DejaVu Serif"],
    "mathtext.fontset": "cm",
    "font.size": 8, "axes.labelsize": 7.8, "axes.titlesize": 8.2,
    "xtick.labelsize": 6.8, "ytick.labelsize": 6.8, "legend.fontsize": 6.5,
    "axes.linewidth": 0.6, "xtick.major.width": 0.5, "ytick.major.width": 0.5,
    "xtick.major.size": 2.2, "ytick.major.size": 2.2,
    "grid.linewidth": 0.35, "lines.linewidth": 1.15,
    "savefig.dpi": 600, "savefig.bbox": "tight", "savefig.pad_inches": 0.015,
})
COL1, COL2 = 3.45, 7.16
INK, CP, GREY = "#1b2a38", "#c0392b", "#9aa8b2"
BLUE, ORANGE, TEAL, PURPLE = "#2c6fa6", "#d9822b", "#178f7a", "#8e5fa8"
PAL = [BLUE, ORANGE, TEAL, PURPLE, CP]

D1 = {r[1]: r for r in DS1}
D2 = {r[1]: r for r in DS2}
P1, P2 = D1["MS-EGCA (proposed)"], D2["MS-EGCA (proposed)"]


def short(n):
    return n.replace(" (proposed)", "")


def style(ax, grid="both"):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color("#7c8b96")
    if grid in ("both", "y"):
        ax.grid(axis="y", alpha=.30, ls=":", color=GREY)
    if grid in ("both", "x"):
        ax.grid(axis="x", alpha=.30, ls=":", color=GREY)
    ax.set_axisbelow(True)


def save(fig, name):
    fig.savefig(os.path.join(FIG, name + ".pdf"))
    plt.close(fig)
    print("  wrote", name + ".pdf")


# --------------------------------------------------------------- declutter --
def declutter(fig, ax, anns, pts, iters=240, pad=1.5):
    try:
        fig.canvas.draw()
        r = fig.canvas.get_renderer()
    except Exception:
        return
    offs = [list(a.get_position()) for a in anns]
    pxy = [ax.transData.transform(p) for p in pts]
    ab = ax.get_window_extent(r)
    for _ in range(iters):
        bs = [a.get_window_extent(r) for a in anns]
        sh = [[0., 0.] for _ in anns]
        moved = False
        for i in range(len(anns)):
            for j in range(i + 1, len(anns)):
                A, B = bs[i], bs[j]
                ox = min(A.x1, B.x1) - max(A.x0, B.x0) + pad
                oy = min(A.y1, B.y1) - max(A.y0, B.y0) + pad
                if ox > 0 and oy > 0:
                    moved = True
                    dx = (A.x0 + A.x1) / 2 - (B.x0 + B.x1) / 2
                    dy = (A.y0 + A.y1) / 2 - (B.y0 + B.y1) / 2
                    if abs(dy) * 1.4 >= abs(dx):
                        s = oy / 2 * (1 if dy >= 0 else -1)
                        sh[i][1] += s; sh[j][1] -= s
                    else:
                        s = ox / 2 * (1 if dx >= 0 else -1)
                        sh[i][0] += s; sh[j][0] -= s
        for i, bb in enumerate(bs):
            for (mx, my) in pxy:
                if bb.x0 - 2 < mx < bb.x1 + 2 and bb.y0 - 2 < my < bb.y1 + 2:
                    moved = True
                    sh[i][1] += 3.0 if (bb.y0 + bb.y1) / 2 >= my else -3.0
        if not moved:
            break
        for i, a in enumerate(anns):
            offs[i][0] += sh[i][0] * .55
            offs[i][1] += sh[i][1] * .55
            bb = bs[i]
            if bb.x0 + sh[i][0] < ab.x0: offs[i][0] += (ab.x0 - bb.x0) + 2
            if bb.x1 + sh[i][0] > ab.x1: offs[i][0] -= (bb.x1 - ab.x1) + 2
            if bb.y0 + sh[i][1] < ab.y0: offs[i][1] += (ab.y0 - bb.y0) + 2
            if bb.y1 + sh[i][1] > ab.y1: offs[i][1] -= (bb.y1 - ab.y1) + 2
            a.set_position(tuple(offs[i]))
        fig.canvas.draw()


def label_pts(fig, ax, xs, ys, labs, props):
    anns = []
    for x, y, l, pr in zip(xs, ys, labs, props):
        anns.append(ax.annotate(
            l, (x, y), textcoords="offset points", xytext=(8, 5),
            fontsize=5.6, ha="left", va="center", zorder=6,
            color=CP if pr else "#33475b", weight="bold" if pr else "normal",
            arrowprops=dict(arrowstyle="-", lw=.35, color="#b6c2cb",
                            shrinkA=0, shrinkB=3)))
    declutter(fig, ax, anns, list(zip(xs, ys)))


# ================================================================== FIG 1 ====
def fig1_method():
    fig = plt.figure(figsize=(COL2, 3.5))
    gs = fig.add_gridspec(2, 1, height_ratios=[1.0, 1.35], hspace=.14)

    # ---- (a) pipeline
    ax = fig.add_subplot(gs[0]); ax.set_xlim(0, 100); ax.set_ylim(0, 30); ax.axis("off")

    def box(x, y, w, h, lab, fc, sub=None, fs=7.2, ec=INK, lw=.75):
        ax.add_patch(FancyBboxPatch((x + .5, y - .5), w, h, boxstyle="round,pad=0.32",
                                    fc="#00000010", ec="none", zorder=1))
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.32",
                                    fc=fc, ec=ec, lw=lw, zorder=2))
        ax.text(x + w / 2, y + h / 2 + (1.5 if sub else 0), lab, ha="center",
                va="center", fontsize=fs, weight="bold", color=INK, zorder=3)
        if sub:
            ax.text(x + w / 2, y + h / 2 - 2.4, sub, ha="center", va="center",
                    fontsize=5.8, style="italic", color="#41576b", zorder=3)

    def arr(x1, y1, x2, y2, ls="-", c=INK, lw=.9):
        ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>",
                                     mutation_scale=7.5, lw=lw, ls=ls, color=c,
                                     shrinkA=0, shrinkB=0, zorder=4))

    y0, h0 = 11, 9.5
    box(0.5, y0, 11.5, h0, "Input", "#eef3f7", "224$\\times$224$\\times$3")
    arr(12.0, y0 + h0 / 2, 15.4, y0 + h0 / 2)
    box(15.4, y0, 14.6, h0, "Sobel Gate", "#fdefdd", "edge prior $\\mathcal{S}$")
    arr(30.0, y0 + h0 / 2, 33.4, y0 + h0 / 2)
    box(33.4, y0, 17.6, h0, "MobileNetV3-S", "#e3eff9", "576$\\times$7$\\times$7")
    arr(51.0, y0 + h0 / 2, 54.4, y0 + h0 / 2)
    box(54.4, y0, 15.6, h0, "MS-EGCA", "#fadedd", "attention (ours)", ec=CP, lw=1.3)
    arr(70.0, y0 + h0 / 2, 73.4, y0 + h0 / 2)
    box(73.4, y0, 10.2, h0, "GAP", "#eef3f7", "576-d")
    arr(83.6, y0 + h0 / 2, 86.8, y0 + h0 / 2)
    box(86.8, y0, 12.7, h0, "Head", "#eef3f7", "FC $\\rightarrow K$")
    box(45.0, 1.6, 27.0, 6.4, "DeiT-Tiny + cross-attention", "#f3f6f7", fs=6.5,
        ec=GREY, lw=.7)
    ax.text(58.5, 0.0, "optional branch, disabled in this study",
            ha="center", va="center", fontsize=5.7, style="italic", color=GREY)
    arr(21.0, y0, 45.0, 5.6, ls=(0, (2.2, 2.2)), c=GREY, lw=.7)
    arr(72.0, 5.6, 79.0, y0, ls=(0, (2.2, 2.2)), c=GREY, lw=.7)
    ax.text(1, 27.2, "(a)  Measurement pipeline", fontsize=7.6, weight="bold",
            color=INK, ha="left")

    # ---- (b) MS-EGCA block
    ax = fig.add_subplot(gs[1]); ax.set_xlim(0, 100); ax.set_ylim(0, 72); ax.axis("off")

    def b2(x, y, w, h, lab, fc, fs=6.2, ec=INK):
        ax.add_patch(FancyBboxPatch((x + .4, y - .4), w, h, boxstyle="round,pad=0.28",
                                    fc="#0000000e", ec="none", zorder=1))
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.28",
                                    fc=fc, ec=ec, lw=.65, zorder=2))
        ax.text(x + w / 2, y + h / 2, lab, ha="center", va="center",
                fontsize=fs, color=INK, zorder=3)

    def a2(x1, y1, x2, y2, c=INK):
        ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>",
                                     mutation_scale=6, lw=.7, color=c,
                                     shrinkA=0, shrinkB=0, zorder=4))

    ax.text(1, 69, "(b)  MS-EGCA module", fontsize=7.6, weight="bold",
            color=INK, ha="left")
    b2(0.5, 27, 12.5, 12, r"$\mathbf{X}$" + "\n" + r"$C\!\times\!H\!\times\!W$", "#eef3f7")
    ax.text(41, 63.5, "multi-scale edge branch", ha="center", fontsize=6.0,
            style="italic", color="#a9691f")
    a2(13, 36, 20, 55)
    b2(20, 49, 17, 11, "channel mean\n" + r"$\mathbf{g}$", "#fdefdd")
    a2(37, 54.5, 43, 54.5)
    b2(43, 46, 25, 17, r"$3{\times}3,\ 5{\times}5,\ 7{\times}7$" + "\n"
       + r"$1{\times}1$ fuse" + "\n" + r"GN, $\sigma \rightarrow \mathbf{E}$", "#fdefdd", 6.0)
    ax.text(41, 42.5, "ECA-style channel branch", ha="center", fontsize=6.0,
            style="italic", color="#1f618d")
    a2(13, 33, 26, 33)
    b2(26, 26, 33, 13, r"GAP $\rightarrow$ Conv1D$_{k_c}$" + "\n"
       + r"$\rightarrow \sigma \rightarrow \mathbf{a}$", "#e3eff9", 6.0)
    ax.text(41, 1.0, "adaptive residual gate", ha="center", fontsize=6.0,
            style="italic", color="#a93226")
    a2(13, 30, 24, 15)
    b2(24, 6.5, 31, 13, r"GAP $\rightarrow$ MLP $\rightarrow \sigma$" + "\n"
       + r"$\rightarrow \alpha\in(0,1)$", "#fadedd", 6.0)
    b2(74, 24, 25, 19, r"$(1\!-\!\alpha)\mathbf{X}$" + "\n"
       + r"$+\ \alpha\,(\mathbf{X}\!\odot\!\mathbf{E}\!\odot\!\mathbf{a})$",
       "#e4f5ef", 6.2, ec=CP)
    a2(68, 54, 86, 43); a2(59, 33, 74, 33.5); a2(55, 13, 86, 24)
    save(fig, "fig1_method")


# ================================================================== FIG 2 ====
def _pareto(ax, fig, rows, xcol, xlabel, title):
    xs = [r[xcol] for r in rows]; ys = [r[5] for r in rows]
    pts = sorted(zip(xs, ys)); fx, fy, best = [], [], -1e9
    for px, py in pts:
        if py > best:
            best = py; fx.append(px); fy.append(py)
    xmax = max(xs) * 1.30
    fx.append(xmax); fy.append(fy[-1])
    lo = min(ys) - .55
    ax.set_xlim(0, xmax); ax.set_ylim(lo, max(ys) + 1.05)
    ax.step(fx, fy, where="post", color=GREY, lw=.85, ls="--", zorder=1)
    ax.fill_between(fx, lo, fy, step="post", color=GREY, alpha=.07, zorder=0)
    for r in rows:
        pr = r[3] == "Proposed"
        ax.scatter(r[xcol], r[5], s=115 if pr else 26, marker="*" if pr else "o",
                   c=CP if pr else "white", edgecolors=CP if pr else BLUE,
                   linewidths=1.0 if pr else .85, zorder=5 if pr else 3)
    label_pts(fig, ax, xs, ys, [short(r[1]) for r in rows],
              [r[3] == "Proposed" for r in rows])
    ax.set_xlabel(xlabel); ax.set_title(title, fontsize=8.0, pad=3)
    style(ax)


def _efficiency(ax, rows, note):
    e = sorted(rows, key=lambda r: r[7] / r[4])
    vals = [r[7] / r[4] for r in e]
    cols = [CP if r[3] == "Proposed" else "#a9bcc9" for r in e]
    y = np.arange(len(e))
    ax.barh(y, vals, .64, color=cols, ec=INK, lw=.35)
    ax.set_yticks(y); ax.set_yticklabels([short(r[1]) for r in e], fontsize=5.9)
    for i, v in enumerate(vals):
        ax.text(v + max(vals) * .015, y[i], f"{v:.0f}", va="center", fontsize=5.6,
                color=CP if cols[i] == CP else "#41576b")
    ax.set_xlim(0, max(vals) * 1.16)
    ax.set_xlabel("Macro-F1 per million parameters")
    ax.set_title(note, fontsize=8.0, pad=3)
    style(ax, "x")


def fig2_ds1_cost():
    fig = plt.figure(figsize=(COL2, 4.3))
    gs = fig.add_gridspec(2, 2, hspace=.42, wspace=.26)
    ax = fig.add_subplot(gs[0, 0])
    _pareto(ax, fig, DS1, 4, "Parameters (M)", "(a) Accuracy vs. model size")
    ax.set_ylabel("Sealed-test accuracy (%)")
    ax = fig.add_subplot(gs[0, 1])
    _pareto(ax, fig, DS1, 15, "Latency per image (ms)", "(b) Accuracy vs. latency")
    ax.axvspan(0, 4.0, color=TEAL, alpha=.06, zorder=0)
    ax = fig.add_subplot(gs[1, 0])
    _efficiency(ax, DS1, "(c) Parameter efficiency")
    ax = fig.add_subplot(gs[1, 1])
    lat = sorted(DS1, key=lambda r: -r[15])
    vals = [1000.0 / r[15] for r in lat]
    cols = [CP if r[3] == "Proposed" else "#a9bcc9" for r in lat]
    y = np.arange(len(lat))
    ax.barh(y, vals, .64, color=cols, ec=INK, lw=.35)
    ax.set_yticks(y); ax.set_yticklabels([short(r[1]) for r in lat], fontsize=5.9)
    for i, v in enumerate(vals):
        ax.text(v + max(vals) * .015, y[i], f"{v:.0f}", va="center", fontsize=5.6,
                color=CP if cols[i] == CP else "#41576b")
    ax.axvline(250, color=TEAL, lw=.8, ls="--")
    ax.text(262, len(lat) - 0.6, "250 FPS", fontsize=5.6, color=TEAL,
            style="italic", va="top")
    ax.set_xlim(0, max(vals) * 1.18)
    ax.set_xlabel("Throughput (frames per second)")
    ax.set_title("(d) Measured throughput", fontsize=8.0, pad=3)
    style(ax, "x")
    save(fig, "fig2_ds1_cost")


# ================================================================== FIG 3 ====
def _corruption(ax, idx, title, ylo):
    marks = ["o", "s", "^", "D", "v"]
    for i, (name, sev) in enumerate(CORRUPT.items()):
        xs = sorted(sev); ys = [sev[s][idx] for s in xs]
        ax.plot(xs, ys, marker=marks[i], ms=3.2, color=PAL[i], label=name,
                mfc="white", mew=.9)
    ax.set_xticks([1, 3, 5]); ax.set_xlabel("Corruption severity")
    ax.set_ylabel("Accuracy (%)")
    ax.set_ylim(ylo, 103); ax.set_title(title, fontsize=8.0, pad=3)
    ax.legend(frameon=False, fontsize=5.9, loc="lower left", ncol=2,
              handlelength=1.2, columnspacing=.8)
    style(ax)


def _calibration(ax, rows, title):
    e = sorted(rows, key=lambda r: r[16] if len(r) > 16 else r[15])
    key = 16 if len(rows[0]) > 16 else 15
    vals = [r[key] for r in e]
    cols = [CP if r[3] == "Proposed" else "#a9bcc9" for r in e]
    y = np.arange(len(e))
    ax.barh(y, vals, .64, color=cols, ec=INK, lw=.35)
    ax.set_yticks(y); ax.set_yticklabels([short(r[1]) for r in e], fontsize=5.9)
    for i, v in enumerate(vals):
        ax.text(v + max(vals) * .02, y[i], f"{v:.3f}", va="center", fontsize=5.5,
                color=CP if cols[i] == CP else "#41576b")
    ax.set_xlim(0, max(vals) * 1.24)
    ax.set_xlabel("Expected calibration error")
    ax.set_title(title, fontsize=8.0, pad=3)
    style(ax, "x")


def fig3_ds1_behaviour():
    fig = plt.figure(figsize=(COL2, 4.1))
    gs = fig.add_gridspec(2, 2, hspace=.46, wspace=.30)

    ax = fig.add_subplot(gs[0, 0])
    d = PERCLASS[::-1]
    lab = [c[0] for c in d]
    pre = np.array([c[4] for c in d]) * 100
    rec = np.array([c[5] for c in d]) * 100
    y = np.arange(len(d)); h = .38
    ax.barh(y + h / 2, pre, h, color=BLUE, ec=INK, lw=.32, label="Precision")
    ax.barh(y - h / 2, rec, h, color=CP, ec=INK, lw=.32, label="Recall")
    ax.set_yticks(y); ax.set_yticklabels(lab, fontsize=5.8)
    ax.set_xlim(94, 100.6); ax.set_xlabel("%")
    ax.legend(frameon=False, fontsize=6.0, loc="lower left", handlelength=1.0)
    ax.set_title("(a) Per-class precision and recall", fontsize=8.0, pad=3)
    style(ax, "x")

    ax = fig.add_subplot(gs[0, 1])
    fn = [c[3] for c in PERCLASS]; fp = [c[2] for c in PERCLASS]
    idx = np.argsort(np.array(fn) + np.array(fp))
    names = [PERCLASS[i][0] for i in idx]
    y = np.arange(len(idx))
    ax.barh(y - .19, [PERCLASS[i][3] for i in idx], .38, color=ORANGE, ec=INK,
            lw=.32, label="False negatives")
    ax.barh(y + .19, [PERCLASS[i][2] for i in idx], .38, color=PURPLE, ec=INK,
            lw=.32, label="False positives")
    ax.set_yticks(y); ax.set_yticklabels(names, fontsize=5.8)
    ax.set_xlabel("Count out of 424 test images per class")
    ax.legend(frameon=False, fontsize=6.0, loc="lower right", handlelength=1.0)
    ax.set_title("(b) Error composition", fontsize=8.0, pad=3)
    style(ax, "x")

    ax = fig.add_subplot(gs[1, 0])
    _calibration(ax, DS1, "(c) Calibration ranking")
    ax = fig.add_subplot(gs[1, 1])
    _corruption(ax, 0, "(d) Corruption response", 20)
    save(fig, "fig3_ds1_behaviour")


# ================================================================== FIG 4 ====
def fig4_ds2():
    fig = plt.figure(figsize=(COL2, 4.1))
    gs = fig.add_gridspec(2, 2, hspace=.44, wspace=.28)
    ax = fig.add_subplot(gs[0, 0])
    _pareto(ax, fig, DS2, 4, "Parameters (M)", "(a) Accuracy vs. model size")
    ax.set_ylabel("Sealed-test accuracy (%)")
    ax = fig.add_subplot(gs[0, 1])
    _pareto(ax, fig, DS2, 14, "Latency per image (ms)", "(b) Accuracy vs. latency")
    ax.axvspan(0, 4.0, color=TEAL, alpha=.06, zorder=0)
    ax = fig.add_subplot(gs[1, 0])
    _efficiency(ax, DS2, "(c) Parameter efficiency")
    ax = fig.add_subplot(gs[1, 1])
    _corruption(ax, 2, "(d) Corruption response", 80)
    save(fig, "fig4_ds2")


# ================================================================== FIG 5 ====
def fig5_cross():
    fig = plt.figure(figsize=(COL2, 4.3))
    gs = fig.add_gridspec(2, 2, hspace=.46, wspace=.30)

    # (a) accuracy parity across the two corpora
    ax = fig.add_subplot(gs[0, 0])
    common = [n for n in D1 if n in D2]
    xs = [D1[n][5] for n in common]; ys = [D2[n][5] for n in common]
    for n, x, y in zip(common, xs, ys):
        pr = D1[n][3] == "Proposed"
        ax.scatter(x, y, s=115 if pr else 26, marker="*" if pr else "o",
                   c=CP if pr else "white", edgecolors=CP if pr else BLUE,
                   linewidths=1.0 if pr else .85, zorder=5 if pr else 3)
    lim = [min(xs + ys) - .4, 100.05]
    ax.plot(lim, lim, ls="--", lw=.8, color=GREY, zorder=1)
    ax.text(lim[1] - .15, lim[1] - .30, "equal accuracy", fontsize=5.6,
            color=GREY, style="italic", rotation=45, ha="right", va="top",
            rotation_mode="anchor")
    # Every architecture is labelled. Full names would overlap badly in a
    # square parity panel, so we use the model identifiers of Tables II-III;
    # the identifier-to-name mapping is given there.
    label_pts(fig, ax, xs, ys, [D1[n][0] for n in common],
              [D1[n][3] == "Proposed" for n in common])
    ax.set_xlim(*lim); ax.set_ylim(*lim)
    ax.set_xlabel("DS1 accuracy (%)"); ax.set_ylabel("DS2 accuracy (%)")
    ax.set_title("(a) Accuracy parity across corpora", fontsize=8.0, pad=3)
    style(ax)

    # (b) robustness gap between the two tasks
    ax = fig.add_subplot(gs[0, 1])
    names = list(CORRUPT)
    x = np.arange(len(names)); w = .38
    for k, (s, c, lab) in enumerate([(3, BLUE, "severity 3"), (5, ORANGE, "severity 5")]):
        gap = [CORRUPT[n][s][2] - CORRUPT[n][s][0] for n in names]
        ax.bar(x + (k - .5) * w, gap, w, color=c, ec=INK, lw=.35, label=lab)
    ax.set_xticks(x); ax.set_xticklabels(names, fontsize=6.2, rotation=20, ha="right")
    ax.set_ylabel("DS2 $-$ DS1 accuracy (pp)")
    ax.legend(frameon=False, fontsize=6.0, loc="upper left", handlelength=1.1)
    ax.set_title("(b) Robustness gap between tasks", fontsize=8.0, pad=3)
    style(ax, "y")

    # (c) parameter efficiency on both corpora
    ax = fig.add_subplot(gs[1, 0])
    e1 = {r[1]: r[7] / r[4] for r in DS1}
    e2 = {r[1]: r[7] / r[4] for r in DS2}
    names = sorted([n for n in e1 if n in e2], key=lambda n: e1[n])
    y = np.arange(len(names))
    ax.barh(y - .19, [e1[n] for n in names], .38, color=BLUE, ec=INK, lw=.32,
            label="DS1")
    ax.barh(y + .19, [e2[n] for n in names], .38, color=ORANGE, ec=INK, lw=.32,
            label="DS2")
    ax.set_yticks(y); ax.set_yticklabels([short(n) for n in names], fontsize=5.8)
    ax.set_xlabel("Macro-F1 per million parameters")
    ax.legend(frameon=False, fontsize=6.0, loc="lower right", handlelength=1.1)
    ax.set_title("(c) Parameter efficiency, both corpora", fontsize=8.0, pad=3)
    style(ax, "x")

    # (d) paired significance, both corpora
    ax = fig.add_subplot(gs[1, 1])
    s1 = {s[0]: s for s in STATS_DS1}
    s2 = {s[0]: s for s in STATS_DS2}
    order = [n for n in s2 if n in s1][::-1]
    y = np.arange(len(order))
    for k, (S, c, off, lab) in enumerate([(s1, BLUE, -.17, "DS1"),
                                          (s2, ORANGE, .17, "DS2")]):
        for i, n in enumerate(order):
            r = S[n]
            sig = r[9]          # Holm-corrected decision
            ax.plot([r[5], r[6]], [y[i] + off] * 2, color=c, lw=1.0,
                    solid_capstyle="round", zorder=2)
            ax.scatter(r[4], y[i] + off, s=16 if sig else 12,
                       c=c if sig else "white", edgecolors=c, linewidths=.8,
                       marker="s" if sig else "o", zorder=3,
                       label=lab if i == 0 else None)
    ax.axvline(0, color=INK, lw=.75, ls="--", zorder=1)
    ax.set_yticks(y); ax.set_yticklabels(order, fontsize=5.8)
    ax.set_xlabel(r"$\Delta$F1 (proposed $-$ baseline, pp)")
    ax.legend(frameon=False, fontsize=6.0, loc="lower right", handlelength=1.1)
    ax.set_title("(d) Paired differences, both corpora", fontsize=8.0, pad=3)
    style(ax, "x")
    save(fig, "fig5_cross")


if __name__ == "__main__":
    print("Generating TIM figures ->", FIG)
    fig1_method(); fig2_ds1_cost(); fig3_ds1_behaviour(); fig4_ds2(); fig5_cross()
    print("done")
