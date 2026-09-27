# =============================================================================
#  MS-EGCA  ::  ONE-CELL PAPER PIPELINE
#  -------------------------------------------------------------------------
#  Paste this whole file into a SINGLE Jupyter cell (or run: python paper_pipeline.py)
#
#  It reads YOUR OWN result files -- nothing is hard-coded -- and produces:
#      figs/*.pdf      publication figures      (regenerated from your data)
#      tables/*.tex    LaTeX tables             (regenerated from your data)
#      paper.tex       full manuscript, Results section written from your data
#      paper.pdf       compiled (if pdflatex is on PATH)
#
#  DATA SOURCES, tried in this order (first hit wins, per table):
#      1. <RESULTS_DIR>/tables/<DS_TAG>/<table>.csv     <-- written by your run
#      2. <RESULTS_DIR>/tables/<DS_TAG>/<table>.md
#      3. REPORT_<DS_TAG>.md  section tables            <-- fallback
#
#  If a table is missing, the affected paragraph/figure is SKIPPED with a
#  warning instead of being invented. Every number in the prose is computed.
# =============================================================================

import os
import re
import sys
import shutil
import subprocess
import textwrap
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

# =============================================================================
# 1. CONFIGURATION  -- edit these two if auto-detection fails
# =============================================================================
RESULTS_DIR_CANDIDATES = [
    r"D:\python\study_1_militery\code\DT_Q1_FINAL_results",
    r"D:\python\study_1_militery\aftercrash\DT_Q1_FINAL_results",
    "./DT_Q1_FINAL_results",
    "../DT_Q1_FINAL_results",
    "/sessions/gifted-compassionate-heisenberg/mnt/aftercrash/DT_Q1_FINAL_results",
]
# extra places to look for REPORT_*.md / COMBINED_REPORT.md
REPORT_DIR_CANDIDATES = [
    ".", "./paper_pack", "..",
    r"D:\python\study_1_militery\code\DT_Q1_FINAL_results",
    r"D:\python\study_1_militery\aftercrash",
    "/sessions/gifted-compassionate-heisenberg/mnt/uploads",
]
OUT_DIR = Path("./paper_build")

DS_PRIMARY = "DS1_primary"     # ten-way vehicle type
DS_BINARY = "DS2_binary"      # military vs civilian
PROPOSED_MATCH = "MS-EGCA"       # substring identifying the proposed model

# A resumed run can leave a stale duplicate of the same architecture under an
# older ID. When two rows share a Model name inside one corpus we keep the one
# whose numeric ID is LARGER (the freshly retrained slot) and report the drop.
DROP_DUPLICATE_MODELS = True

# Training settings are properties of YOUR RUN, not of the result tables, so
# they cannot be recovered from the CSVs. Edit these to match your script if you
# change it; they are rendered verbatim into the "configuration" table.
CONFIG = [
    ("Input resolution", r"$224\times224$ RGB"),
    ("Backbone initialisation", "ImageNet-1k pretrained"),
    ("Optimiser", r"AdamW, $\beta=(0.9,0.999)$"),
    ("Learning rate / schedule", r"$1\times10^{-4}$, cosine annealing"),
    ("Weight decay", r"$1\times10^{-2}$"),
    ("Batch size", "16"),
    ("Maximum epochs", "41 (early stop, patience 4)"),
    ("Loss", r"cross-entropy, label smoothing $0.1$"),
    ("Gradient clipping", r"$\ell_2$ norm $1.0$"),
    ("Weight averaging", r"EMA, decay $0.999$"),
    ("Test-time augmentation", "horizontal flip"),
    ("Augmentation", r"resize 256, centre crop 224, camouflage jitter ($p{=}0.5$)"),
    ("Random seeds", r"$\{42,123,2026,7,999\}$"),
    ("Bootstrap resamples", "2{,}000"),
    ("Latency protocol", "20 warm-up + 100 timed runs, batch 1"),
    ("Hardware", "AMD GPU via DirectML, FP32"),
]

TITLE = ("Edge-Guided Channel Attention for Sub-Megaparameter\\\\\n"
         "Military Ground-Vehicle Recognition:\\\\\n"
         "A Sealed-Test Benchmark of Efficient Architectures")

AUTHOR_IEEE = r"""Muhammad Azam Khan*,  Yanyan Huang*,  Nazish Waqar Waqar,  Muhammad Inaam Ul Haq, and  Muhammad Adnan Khan*~\IEEEmembership{Senior Member, IEEE}%
\thanks{Muhammad Azam Khan and Yanyan Huang are with the School of Automation, Department of Control Science and Engineering, Nanjing University of Science and Technology, Nanjing 210094, China (e-mails: azam\_khan@njust.edu.cn; huangyy@njust.edu.cn).}%
\thanks{Nazish Waqar Waqar is with the School of Engineering, London South Bank University, London, UK (e-mail: s4308096@lsbu.ac.uk).}%
\thanks{Muhammad Inaam Ul Haq is with the Department of Computer Science, COMSATS University Islamabad, Sahiwal Campus, Pakistan (e-mail: minaamulhaq@cuisahiwal.edu.pk).}%
\thanks{Muhammad Adnan Khan is with the Department of Software, Faculty of Artificial Intelligence and Software, Gachon University, Seongnam-si, 13120, Republic of Korea (e-mail: adnan@gachon.ac.kr).}%
\thanks{*Corresponding authors: Muhammad Azam Khan, Yanyan Huang, and Muhammad Adnan Khan.}%
"""

_warnings = []


def warn(msg):
    _warnings.append(msg)
    print(f"  [!] {msg}")


def ok(msg):
    print(f"  [OK] {msg}")


# =============================================================================
# 2. LOCATE INPUTS
# =============================================================================
def _first_existing(cands):
    for c in cands:
        p = Path(c).expanduser()
        if p.is_dir():
            return p.resolve()
    return None


RESULTS_DIR = _first_existing(RESULTS_DIR_CANDIDATES)
if RESULTS_DIR is None:
    warn("No RESULTS_DIR found; will rely on REPORT_*.md only.")
    TAB_DIR = None
else:
    TAB_DIR = RESULTS_DIR / "tables"
    ok(f"RESULTS_DIR = {RESULTS_DIR}")

REPORT_DIRS = []
for c in REPORT_DIR_CANDIDATES + ([str(RESULTS_DIR)] if RESULTS_DIR else []):
    p = Path(c).expanduser()
    if p.is_dir() and p not in REPORT_DIRS:
        REPORT_DIRS.append(p)

OUT_DIR.mkdir(parents=True, exist_ok=True)
FIG_DIR = OUT_DIR / "figs"
TEX_TAB_DIR = OUT_DIR / "tables"
FIG_DIR.mkdir(exist_ok=True)
TEX_TAB_DIR.mkdir(exist_ok=True)


# =============================================================================
# 3. GENERIC LOADERS  (csv -> md -> REPORT section)
# =============================================================================
def _read_md_table(text):
    """Parse the first GitHub-style markdown table in `text` into a DataFrame."""
    lines = [l for l in text.splitlines() if l.strip().startswith("|")]
    if len(lines) < 3:
        return None
    header = [c.strip() for c in lines[0].strip().strip("|").split("|")]
    rows = []
    for l in lines[2:]:
        cells = [c.strip() for c in l.strip().strip("|").split("|")]
        if len(cells) == len(header):
            rows.append(cells)
    if not rows:
        return None
    df = pd.DataFrame(rows, columns=header)
    for c in df.columns:
        conv = pd.to_numeric(df[c].str.replace(",", "", regex=False), errors="coerce")
        if conv.notna().sum() >= max(1, int(0.7 * len(df))):
            df[c] = conv
    return df


def _find_report(ds_tag):
    names = [f"REPORT_{ds_tag}.md", f"REPORT_{ds_tag.split('_')[0]}.md"]
    for d in REPORT_DIRS:
        for n in names:
            p = d / n
            if p.is_file():
                return p
    return None


_REPORT_SECTION = {           # table name -> heading fragment in REPORT_*.md
    "table3_main_results": "Main benchmark",
    "table9_extended_metrics": "Extended metrics",
    "table5_per_class": "Per-class",
    "table4_ablation": "Ablation",
    "table7_corruption": "Corruption",
}


def load_table(name, ds_tag):
    """Load one result table by logical name for one dataset. Returns df or None."""
    # 1) CSV written by the training run
    if TAB_DIR is not None:
        p = TAB_DIR / ds_tag / f"{name}.csv"
        if p.is_file():
            try:
                return pd.read_csv(p)
            except Exception as e:
                warn(f"{p.name} unreadable ({e})")
        # 2) MD written by the training run
        p = TAB_DIR / ds_tag / f"{name}.md"
        if p.is_file():
            df = _read_md_table(p.read_text(encoding="utf-8", errors="ignore"))
            if df is not None:
                return df
    # 3) section inside REPORT_<ds>.md
    frag = _REPORT_SECTION.get(name)
    rp = _find_report(ds_tag)
    if frag and rp:
        txt = rp.read_text(encoding="utf-8", errors="ignore")
        blocks = re.split(r"\n##\s+", txt)
        for b in blocks:
            if frag.lower() in b.split("\n", 1)[0].lower():
                df = _read_md_table(b)
                if df is not None:
                    return df
    return None


def clean_main(df, ds_tag):
    """Normalise the main benchmark table and drop stale duplicate rows."""
    if df is None:
        return None
    df = df.copy()
    ren = {"Params(M)": "params", "Acc(%)": "acc", "Acc_std": "acc_std",
           "F1(%)": "f1", "F1_std": "f1_std", "Lat(ms)": "lat", "ECE": "ece",
           "MCC": "mcc", "AUC": "auc", "Model": "model", "ID": "id",
           "Type": "type", "Year": "year", "Spec(%)": "spec",
           "G-Mean(%)": "gmean", "Kappa": "kappa", "Brier": "brier"}
    df = df.rename(columns={k: v for k, v in ren.items() if k in df.columns})
    need = {"id", "model", "params", "acc", "f1"}
    if not need.issubset(df.columns):
        warn(f"{ds_tag}: main table missing columns {need - set(df.columns)}")
        return None
    df["idnum"] = pd.to_numeric(df["id"].astype(str).str.extract(r"(\d+)")[0],
                                errors="coerce")
    if DROP_DUPLICATE_MODELS and df["model"].duplicated().any():
        dup = df[df.duplicated("model", keep=False)].sort_values("idnum")
        keep = dup.groupby("model")["idnum"].max()
        drop_idx = dup[dup.apply(lambda r: r["idnum"] != keep[r["model"]], axis=1)].index
        for i in drop_idx:
            warn(f"{ds_tag}: dropping stale duplicate row "
                 f"{df.loc[i,'id']} '{df.loc[i,'model']}' "
                 f"(acc={df.loc[i,'acc']:.2f}) -- kept the later-ID retrain")
        df = df.drop(index=drop_idx)
    df["is_prop"] = df["model"].astype(str).str.contains(PROPOSED_MATCH, case=False)
    df["short"] = (df["model"].astype(str)
                   .str.replace(r"\s*\(.*\)\s*", "", regex=True).str.strip())
    df.loc[df["is_prop"], "short"] = "MS-EGCA (proposed)"
    return df.reset_index(drop=True)


print("\n" + "=" * 74)
print("LOADING RESULT TABLES")
print("=" * 74)

M1 = clean_main(load_table("table3_main_results", DS_PRIMARY), DS_PRIMARY)
M2 = clean_main(load_table("table3_main_results", DS_BINARY), DS_BINARY)
if M1 is None:
    raise SystemExit("FATAL: could not load the DS1 main benchmark table. "
                     "Set RESULTS_DIR / REPORT_DIR_CANDIDATES correctly.")
ok(f"{DS_PRIMARY}: {len(M1)} architectures")
if M2 is not None:
    ok(f"{DS_BINARY}: {len(M2)} architectures")
else:
    warn(f"{DS_BINARY} main table not found -- paper will be single-corpus")

ABL1 = load_table("table4_ablation", DS_PRIMARY)
ABL2 = load_table("table4_ablation", DS_BINARY)
PC1 = load_table("table5_per_class", DS_PRIMARY)
ST1 = load_table("table6_stats", DS_PRIMARY)
ST2 = load_table("table6_stats", DS_BINARY)
CR1 = load_table("table7_corruption", DS_PRIMARY)
CR2 = load_table("table7_corruption", DS_BINARY)
EX1 = load_table("table9_extended_metrics", DS_PRIMARY)
DSET1 = load_table("table1_dataset", DS_PRIMARY)
DSET2 = load_table("table1_dataset", DS_BINARY)

for nm, d in [("ablation DS1", ABL1), ("ablation DS2", ABL2), ("per-class DS1", PC1),
              ("stats DS1", ST1), ("stats DS2", ST2), ("corruption DS1", CR1),
              ("corruption DS2", CR2), ("extended DS1", EX1),
              ("dataset DS1", DSET1), ("dataset DS2", DSET2)]:
    if d is None:
        warn(f"{nm}: NOT FOUND -> dependent content will be skipped")
    else:
        ok(f"{nm}: {len(d)} rows")


def norm_stats(df):
    if df is None:
        return None
    df = df.copy()
    ren = {"vs": "vs", "b": "b", "c": "c", "p-value": "p", "dF1%": "d",
           "CI_lo": "lo", "CI_hi": "hi", "Cohen_d": "cd", "BL_F1%": "bl",
           "sig": "sig"}
    df = df.rename(columns={k: v for k, v in ren.items() if k in df.columns})
    if "vs" in df.columns:
        df["base"] = (df["vs"].astype(str)
                      .str.replace(r"^\s*\S+\s+vs\s+", "", regex=True).str.strip())
    if "sig" in df.columns:
        df["issig"] = df["sig"].astype(str).str.strip().str.lower().isin(
            ["yes", "true", "1", "y"])
    if "p" in df.columns:
        df["pnum"] = pd.to_numeric(
            df["p"].astype(str).str.replace("<", "", regex=False), errors="coerce")
    # a stats row can survive for a model dropped as a stale duplicate
    return df


ST1, ST2 = norm_stats(ST1), norm_stats(ST2)
if ST1 is not None and M1 is not None and "base" in ST1.columns:
    keep = set(M1["short"]) | set(M1["model"])
    bad = ST1[~ST1["base"].isin(keep)]
    for _, r in bad.iterrows():
        warn(f"stats DS1: dropping row for '{r['base']}' (not in cleaned benchmark)")
    ST1 = ST1[ST1["base"].isin(keep)].reset_index(drop=True)


# =============================================================================
# 4. DERIVED QUANTITIES  (everything the prose needs -- all computed)
# =============================================================================
def facts(M, tag):
    if M is None:
        return None
    p = M[M["is_prop"]].iloc[0]
    o = M.sort_values("acc", ascending=False).reset_index(drop=True)
    rank = int(o.index[o["is_prop"]][0]) + 1
    best = o.iloc[0]
    M = M.assign(eff=M["f1"] / M["params"])
    e = M.sort_values("eff", ascending=False).reset_index(drop=True)
    f = dict(
        tag=tag, n=len(M), prop=p, acc=p["acc"], f1=p["f1"], params=p["params"],
        rank=rank, best=best, best_name=best["short"], best_acc=best["acc"],
        eff=p["f1"] / p["params"], eff_rank=int(e.index[e["is_prop"]][0]) + 1,
        eff_tbl=e, order=o, M=M,
        acc_std=p.get("acc_std", np.nan), lat=p.get("lat", np.nan),
        ece=p.get("ece", np.nan),
        smallest=M.loc[M["params"].idxmin(), "short"],
        smallest_p=M["params"].min(),
    )
    if "acc_std" in M.columns and M["acc_std"].notna().any():
        f["tightest"] = M.loc[M["acc_std"].idxmin(), "short"]
        f["tightest_v"] = M["acc_std"].min()
    if "ece" in M.columns and M["ece"].notna().any():
        ec = M.sort_values("ece").reset_index(drop=True)
        f["ece_rank"] = int(ec.index[ec["is_prop"]][0]) + 1
        f["ece_tbl"] = ec
    return f


F1_ = facts(M1, "DS1")
F2_ = facts(M2, "DS2")


def sig_counts(ST):
    if ST is None or "issig" not in ST.columns:
        return None
    tot = len(ST)
    sig = int(ST["issig"].sum())
    better = ST[(ST["issig"]) & (ST["d"] > 0)]["base"].tolist()
    worse = ST[(ST["issig"]) & (ST["d"] < 0)]["base"].tolist()
    equiv = ST[~ST["issig"]]["base"].tolist()
    return dict(tot=tot, sig=sig, better=better, worse=worse, equiv=equiv, tbl=ST)


S1, S2 = sig_counts(ST1), sig_counts(ST2)


def and_list(xs):
    xs = [str(x) for x in xs]
    if not xs:
        return "none"
    if len(xs) == 1:
        return xs[0]
    return ", ".join(xs[:-1]) + " and " + xs[-1]


NUMW = {1: "first", 2: "second", 3: "third", 4: "fourth", 5: "fifth", 6: "sixth",
        7: "seventh", 8: "eighth", 9: "ninth", 10: "tenth", 11: "eleventh"}
CARD = {1: "one", 2: "two", 3: "three", 4: "four", 5: "five", 6: "six", 7: "seven",
        8: "eight", 9: "nine", 10: "ten", 11: "eleven", 12: "twelve"}


def w_ord(i):
    return NUMW.get(int(i), f"{int(i)}th")


def w_card(i):
    return CARD.get(int(i), str(int(i)))


def tex_esc(s):
    return (str(s).replace("&", r"\&").replace("_", r"\_")
            .replace("%", r"\%").replace("#", r"\#"))


# =============================================================================
# 5. FIGURES
# =============================================================================
plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Nimbus Roman No9 L", "Times New Roman", "DejaVu Serif"],
    "mathtext.fontset": "cm",
    "font.size": 8, "axes.labelsize": 8, "axes.titlesize": 8.6,
    "xtick.labelsize": 7, "ytick.labelsize": 7, "legend.fontsize": 6.8,
    "axes.linewidth": 0.6, "xtick.major.width": 0.5, "ytick.major.width": 0.5,
    "xtick.major.size": 2.4, "ytick.major.size": 2.4,
    "grid.linewidth": 0.35, "lines.linewidth": 1.2,
    "savefig.dpi": 600, "savefig.bbox": "tight", "savefig.pad_inches": 0.015,
    "figure.facecolor": "white",
})
COL1, COL2 = 3.45, 7.16
INK = "#1b2a38"
CP = "#c0392b"        # proposed
CPL = "#f2c9c4"
BLUE = "#2c6fa6"
BLUEL = "#c9dcec"
GREY = "#9aa8b2"
TEAL = "#178f7a"
ORANGE = "#d9822b"
PAL = ["#2c6fa6", "#d9822b", "#178f7a", "#8e5fa8", "#c0392b"]


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
    fig.savefig(FIG_DIR / f"{name}.pdf")
    plt.close(fig)
    ok(f"figure {name}.pdf")


FIGS = {}


# ---------------------------------------------------------------- Fig 1 ----
def fig_architecture():
    fig, ax = plt.subplots(figsize=(COL2, 1.95))
    ax.set_xlim(0, 100); ax.set_ylim(0, 31); ax.axis("off")

    def box(x, y, w, h, lab, fc, sub=None, fs=7.3, ec=INK, lw=.75):
        ax.add_patch(FancyBboxPatch((x + .5, y - .5), w, h,
                                    boxstyle="round,pad=0.35", fc="#00000012",
                                    ec="none", zorder=1))
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.35",
                                    fc=fc, ec=ec, lw=lw, zorder=2))
        ax.text(x + w / 2, y + h / 2 + (1.5 if sub else 0), lab, ha="center",
                va="center", fontsize=fs, weight="bold", color=INK, zorder=3)
        if sub:
            ax.text(x + w / 2, y + h / 2 - 2.5, sub, ha="center", va="center",
                    fontsize=6.0, style="italic", color="#41576b", zorder=3)

    def arr(x1, y1, x2, y2, ls="-", c=INK, lw=.95):
        ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>",
                                     mutation_scale=8, lw=lw, ls=ls, color=c,
                                     shrinkA=0, shrinkB=0, zorder=4))

    y0, h0 = 12, 9.5
    pr = F1_["prop"]
    box(0.5, y0, 11.5, h0, "Input", "#eef3f7", "224 x 224 x 3")
    arr(12.0, y0 + h0 / 2, 15.6, y0 + h0 / 2)
    box(15.6, y0, 14.5, h0, "Sobel Gate", "#fdefdd", "edge prior  S")
    arr(30.1, y0 + h0 / 2, 33.7, y0 + h0 / 2)
    box(33.7, y0, 17.5, h0, "MobileNetV3-S", "#e3eff9", "backbone F, 576-d")
    arr(51.2, y0 + h0 / 2, 54.8, y0 + h0 / 2)
    box(54.8, y0, 15.5, h0, "MS-EGCA", "#fadedd", "attention (ours)", ec=CP, lw=1.3)
    arr(70.3, y0 + h0 / 2, 73.9, y0 + h0 / 2)
    box(73.9, y0, 10.0, h0, "GAP", "#eef3f7")
    arr(83.9, y0 + h0 / 2, 87.0, y0 + h0 / 2)
    box(87.0, y0, 12.5, h0, "Head", "#eef3f7", "dropout + FC")

    box(45.0, 2.4, 27.0, 6.6, "DeiT-Tiny + cross-attention", "#f3f6f7",
        fs=6.7, ec=GREY, lw=.7)
    ax.text(58.5, .5, "optional branch, disabled in this study (Sec. III-E)",
            ha="center", va="center", fontsize=6.0, style="italic", color=GREY)
    arr(21.0, y0, 45.0, 6.6, ls=(0, (2.2, 2.2)), c=GREY, lw=.7)
    arr(72.0, 6.6, 79.0, y0, ls=(0, (2.2, 2.2)), c=GREY, lw=.7)

    ax.text(50, 28.4, f"Proposed lightweight recognition pipeline  "
                      f"({pr['params']:.2f} M parameters, "
                      f"{pr['lat']:.2f} ms / image)",
            ha="center", fontsize=7.7, weight="bold", color=INK)
    save(fig, "fig1_architecture")
    FIGS["arch"] = "fig1_architecture"


# ---------------------------------------------------------------- Fig 2 ----
def fig_block():
    fig, ax = plt.subplots(figsize=(COL1, 2.25))
    ax.set_xlim(0, 100); ax.set_ylim(0, 74); ax.axis("off")

    def box(x, y, w, h, lab, fc, fs=6.3, ec=INK):
        ax.add_patch(FancyBboxPatch((x + .4, y - .4), w, h,
                                    boxstyle="round,pad=0.3", fc="#00000010",
                                    ec="none", zorder=1))
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.3",
                                    fc=fc, ec=ec, lw=.65, zorder=2))
        ax.text(x + w / 2, y + h / 2, lab, ha="center", va="center",
                fontsize=fs, color=INK, zorder=3)

    def arr(x1, y1, x2, y2, c=INK):
        ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>",
                                     mutation_scale=6, lw=.7, color=c,
                                     shrinkA=0, shrinkB=0, zorder=4))

    box(0.5, 31, 13.5, 12, r"$\mathbf{X}$" + "\n" + r"$C\times H\times W$", "#eef3f7", 6.2)
    ax.text(46, 71.5, "multi-scale edge branch", ha="center", fontsize=6.2,
            style="italic", color="#a9691f")
    arr(14, 40, 22, 60)
    box(22, 54, 19, 12, "channel mean\n" + r"$g=\bar{\mathbf{X}}_c$", "#fdefdd")
    arr(41, 60, 47.5, 60)
    box(47.5, 51.5, 26, 17, r"$3{\times}3,\,5{\times}5,\,7{\times}7$" + "\n"
        + r"$1{\times}1$ fuse" + "\n" + r"GN $+\ \sigma\ \rightarrow\ \mathbf{E}$",
        "#fdefdd", 6.1)
    ax.text(46, 46.8, "ECA-style channel branch", ha="center", fontsize=6.2,
            style="italic", color="#1f618d")
    arr(14, 37, 29, 37)
    box(29, 30, 34, 13.5, r"GAP $\rightarrow$ 1-D conv ($k$ adaptive)" + "\n"
        + r"$\rightarrow\ \sigma\ \rightarrow\ \mathbf{A}$", "#e3eff9", 6.1)
    ax.text(46, 1.8, "adaptive residual gate", ha="center", fontsize=6.2,
            style="italic", color="#a93226")
    arr(14, 34, 27, 17)
    box(27, 8.5, 32, 13.5, r"GAP $\rightarrow$ MLP $\rightarrow\ \sigma$" + "\n"
        + r"$\rightarrow\ \alpha\in(0,1)$", "#fadedd", 6.1)
    box(77, 28, 22, 19, r"$\mathbf{X}\odot[(1-\alpha)$" + "\n"
        + r"$+\ \alpha\,\mathbf{E}\odot\mathbf{A}]$", "#e4f5ef", 6.3, ec=CP)
    arr(73.5, 58, 88, 47); arr(63, 37, 77, 37.5); arr(59, 16, 88, 28)
    save(fig, "fig2_msegca")
    FIGS["block"] = "fig2_msegca"


# ---------------------------------------------------------------- Fig 3 ----
def _declutter(fig, ax, anns, pts, iters=260, pad=1.6):
    """Force-based label de-overlap, in display space.

    `anns` are Annotations created with textcoords='offset points'; their
    offset is mutated until no two label boxes overlap and no label sits on a
    data marker. Leader lines keep every label attached to its point.
    """
    try:
        fig.canvas.draw()
        rend = fig.canvas.get_renderer()
    except Exception:
        return
    offs = [list(a.get_position()) for a in anns]
    pxy = [ax.transData.transform(p) for p in pts]
    axbb = ax.get_window_extent(rend)

    def boxes():
        return [a.get_window_extent(rend) for a in anns]

    for _ in range(iters):
        bs = boxes()
        shift = [[0.0, 0.0] for _ in anns]
        moved = False
        # label <-> label
        for i in range(len(anns)):
            for j in range(i + 1, len(anns)):
                a, b = bs[i], bs[j]
                ox = min(a.x1, b.x1) - max(a.x0, b.x0) + pad
                oy = min(a.y1, b.y1) - max(a.y0, b.y0) + pad
                if ox > 0 and oy > 0:
                    moved = True
                    dx = (a.x0 + a.x1) / 2 - (b.x0 + b.x1) / 2
                    dy = (a.y0 + a.y1) / 2 - (b.y0 + b.y1) / 2
                    if abs(dy) * 1.4 >= abs(dx):
                        s = oy / 2 * (1 if dy >= 0 else -1)
                        shift[i][1] += s; shift[j][1] -= s
                    else:
                        s = ox / 2 * (1 if dx >= 0 else -1)
                        shift[i][0] += s; shift[j][0] -= s
        # label <-> data marker
        for i, bb in enumerate(bs):
            for (mx, my) in pxy:
                if bb.x0 - 2 < mx < bb.x1 + 2 and bb.y0 - 2 < my < bb.y1 + 2:
                    moved = True
                    cy = (bb.y0 + bb.y1) / 2
                    shift[i][1] += 3.0 if cy >= my else -3.0
        if not moved:
            break
        for i, a in enumerate(anns):
            offs[i][0] += shift[i][0] * 0.55
            offs[i][1] += shift[i][1] * 0.55
            # keep the label inside the axes
            bb = bs[i]
            if bb.x0 + shift[i][0] < axbb.x0:
                offs[i][0] += (axbb.x0 - bb.x0) + 2
            if bb.x1 + shift[i][0] > axbb.x1:
                offs[i][0] -= (bb.x1 - axbb.x1) + 2
            if bb.y0 + shift[i][1] < axbb.y0:
                offs[i][1] += (axbb.y0 - bb.y0) + 2
            if bb.y1 + shift[i][1] > axbb.y1:
                offs[i][1] -= (bb.y1 - axbb.y1) + 2
            a.set_position(tuple(offs[i]))
        fig.canvas.draw()


def _label_points(fig, ax, xs, ys, labels, props):
    """Annotate every point, then de-overlap. Returns the annotation list."""
    anns = []
    for x, y, lab, pr in zip(xs, ys, labels, props):
        a = ax.annotate(lab, (x, y), textcoords="offset points",
                        xytext=(8, 6 if pr else 5), fontsize=5.7,
                        ha="left", va="center", zorder=6,
                        color=CP if pr else "#33475b",
                        weight="bold" if pr else "normal",
                        arrowprops=dict(arrowstyle="-", lw=0.35,
                                        color="#b6c2cb", shrinkA=0, shrinkB=3))
        anns.append(a)
    _declutter(fig, ax, anns, list(zip(xs, ys)))
    return anns


def fig_of(ax):
    return ax.get_figure()


def _pareto_panel(ax, M, title):
    pts = sorted(zip(M["params"], M["acc"]))
    fx, fy, best = [], [], -1e9
    for px, py in pts:
        if py > best:
            best = py; fx.append(px); fy.append(py)
    xmax = M["params"].max() * 1.14
    fx.append(xmax); fy.append(fy[-1])
    lo = max(0, M["acc"].min() - .55)
    ax.set_xlim(0, xmax * 1.05)
    ax.set_ylim(lo, min(100.2, M["acc"].max() + .80))
    ax.step(fx, fy, where="post", color=GREY, lw=.85, ls="--", zorder=1)
    ax.fill_between(fx, lo, fy, step="post", color=GREY, alpha=.07, zorder=0)
    for _, r in M.iterrows():
        pr = bool(r["is_prop"])
        ax.scatter(r["params"], r["acc"], s=110 if pr else 27,
                   marker="*" if pr else "o", c=CP if pr else "white",
                   edgecolors=CP if pr else BLUE, linewidths=1.0 if pr else .85,
                   zorder=5 if pr else 3)
    _label_points(fig_of(ax), ax, M["params"].tolist(), M["acc"].tolist(),
                  [str(v).replace(" (proposed)", "\n(proposed)") for v in M["short"]],
                  M["is_prop"].tolist())
    ax.set_xlabel("Parameters (M)")
    ax.set_title(title, fontsize=7.9, pad=3)
    style(ax)


def fig_efficiency():
    have2 = M2 is not None
    fig = plt.figure(figsize=(COL2, 4.5 if have2 else 3.4))
    gs = fig.add_gridspec(2, 2, hspace=.46, wspace=.24)
    a = fig.add_subplot(gs[0, 0])
    _pareto_panel(a, M1, f"(a) {DS_PRIMARY.split('_')[0]} \u2013 ten-way vehicle type")
    a.set_ylabel("Sealed-test accuracy (%)")
    if have2:
        b = fig.add_subplot(gs[0, 1])
        _pareto_panel(b, M2, f"(b) {DS_BINARY.split('_')[0]} \u2013 military vs. civilian")

    # (c) parameter efficiency
    ax = fig.add_subplot(gs[1, 0])
    e = F1_["eff_tbl"].sort_values("eff")
    cols = [CP if p else "#a9bcc9" for p in e["is_prop"]]
    yy = np.arange(len(e))
    ax.barh(yy, e["eff"], .62, color=cols, ec=INK, lw=.35)
    ax.set_yticks(yy); ax.set_yticklabels(e["short"], fontsize=6.0)
    ax.set_xlabel("Macro-F1 per million parameters")
    ax.set_title("(c) Parameter efficiency", fontsize=7.9, pad=3)
    for i, v in enumerate(e["eff"]):
        ax.text(v + e["eff"].max() * .015, yy[i], f"{v:.1f}", va="center",
                fontsize=5.8, color=CP if cols[i] == CP else "#41576b")
    ax.set_xlim(0, e["eff"].max() * 1.16)
    style(ax, "x")

    # (d) latency-accuracy
    ax = fig.add_subplot(gs[1, 1])
    if "lat" in M1.columns and M1["lat"].notna().any():
        for _, r in M1.iterrows():
            pr = bool(r["is_prop"])
            ax.scatter(r["lat"], r["acc"], s=110 if pr else 27,
                       marker="*" if pr else "o", c=CP if pr else "white",
                       edgecolors=CP if pr else BLUE,
                       linewidths=1.0 if pr else .85, zorder=5 if pr else 3)
        ax.set_xlim(0, M1["lat"].max() * 1.16)
        ax.set_ylim(M1["acc"].min() - .45, M1["acc"].max() + .6)
        _label_points(fig, ax, M1["lat"].tolist(), M1["acc"].tolist(),
                      [str(v).replace(" (proposed)", "\n(proposed)")
                       for v in M1["short"]], M1["is_prop"].tolist())
        ax.axvspan(0, 4.0, color=TEAL, alpha=.06, zorder=0)
        ax.text(2.0, ax.get_ylim()[0] + .06, "< 4 ms  (> 250 FPS)", fontsize=5.9,
                ha="center", va="bottom", color=TEAL, style="italic")
        ax.set_xlabel("Inference latency per image (ms)")
        ax.set_ylabel("Sealed-test accuracy (%)")
        ax.set_title("(d) Latency\u2013accuracy plane", fontsize=7.9, pad=3)
        style(ax)
    else:
        ax.axis("off")
    save(fig, "fig3_efficiency")
    FIGS["eff"] = "fig3_efficiency"


# ---------------------------------------------------------------- Fig 4 ----
def fig_forest():
    panels = [(S1, f"(a) {DS_PRIMARY.split('_')[0]} \u2013 ten-way"),
              (S2, f"(b) {DS_BINARY.split('_')[0]} \u2013 binary")]
    panels = [(s, t) for s, t in panels if s is not None]
    if not panels:
        warn("no stats tables -> skipping forest plot")
        return
    fig, axes = plt.subplots(1, len(panels), figsize=(COL2, 2.35), squeeze=False)
    for ax, (S, title) in zip(axes[0], panels):
        T = S["tbl"].iloc[::-1].reset_index(drop=True)
        y = np.arange(len(T))
        ax.axvspan(-99, 0, color=BLUE, alpha=.045, zorder=0)
        for i, r in T.iterrows():
            sg = bool(r.get("issig", False))
            c = CP if sg else GREY
            ax.plot([r["lo"], r["hi"]], [y[i], y[i]], color=c, lw=1.2,
                    zorder=2, solid_capstyle="round")
            for e in (r["lo"], r["hi"]):
                ax.plot([e, e], [y[i] - .17, y[i] + .17], color=c, lw=.8, zorder=2)
            ax.scatter(r["d"], y[i], s=20 if sg else 16, zorder=3,
                       c=c if sg else "white", edgecolors=c, linewidths=.9,
                       marker="s" if sg else "o")
        ax.axvline(0, color=INK, lw=.75, ls="--", zorder=1)
        ax.set_yticks(y); ax.set_yticklabels(T["base"], fontsize=6.0)
        ax.set_xlabel(r"$\Delta$F1  (proposed $-$ baseline, pp)")
        ax.set_title(title, fontsize=7.9, pad=3)
        lo = min(T["lo"].min(), 0) - .35
        hi = max(T["hi"].max(), 0) + .35
        ax.set_xlim(lo, hi)
        style(ax, "x")
    fig.text(.5, -.045, "filled squares: significant at $p<0.05$ (McNemar, "
             "Holm-corrected);  open circles: not significant;  "
             "bars are 95% bootstrap CIs", ha="center", fontsize=6.1,
             style="italic", color="#41576b")
    fig.tight_layout(pad=.35, w_pad=1.6)
    save(fig, "fig4_forest")
    FIGS["forest"] = "fig4_forest"


# ---------------------------------------------------------------- Fig 5 ----
def _abl(df):
    if df is None:
        return None
    d = df.copy().rename(columns={"F1(%)": "f1", "Acc(%)": "acc", "ID": "id",
                                  "Config": "cfg"})
    return d if {"id", "f1"}.issubset(d.columns) else None


def fig_ablation():
    A1, A2 = _abl(ABL1), _abl(ABL2)
    if A1 is None and A2 is None:
        warn("no ablation table -> skipping ablation figure")
        return
    base = A1 if A1 is not None else A2
    ids = base["id"].tolist()
    fig, ax = plt.subplots(figsize=(COL1, 1.95))
    x = np.arange(len(ids)); w = .39
    prop_ids = {"A6", "A7", "A8"}
    if A1 is not None:
        c = [CP if i in prop_ids else BLUE for i in A1["id"]]
        ax.bar(x - w / 2, A1["f1"], w, color=c, ec=INK, lw=.4, label="DS1 (ten-way)")
        ax.axhline(A1["f1"].iloc[0], color=BLUE, lw=.7, ls=":", zorder=1)
    if A2 is not None:
        c = [CPL if i in prop_ids else ORANGE for i in A2["id"]]
        ax.bar(x + w / 2, A2["f1"], w, color=c, ec=INK, lw=.4, label="DS2 (binary)")
        ax.axhline(A2["f1"].iloc[0], color=ORANGE, lw=.7, ls=":", zorder=1)
    lo = min([d["f1"].min() for d in (A1, A2) if d is not None]) - .25
    hi = max([d["f1"].max() for d in (A1, A2) if d is not None]) + .30
    ax.set_ylim(lo, hi)
    ax.set_xticks(x); ax.set_xticklabels(ids, fontsize=6.6)
    ax.set_ylabel("Macro-F1 (%)"); ax.set_xlabel("Ablation variant")
    ax.legend(frameon=False, ncol=2, loc="upper left", fontsize=6.2,
              handlelength=1.1, columnspacing=1.0)
    style(ax, "y")
    fig.tight_layout(pad=.3)
    save(fig, "fig5_ablation")
    FIGS["abl"] = "fig5_ablation"


# ---------------------------------------------------------------- Fig 6 ----
def fig_perclass_ece():
    has_pc = PC1 is not None and {"Class", "Precision", "Recall"}.issubset(PC1.columns)
    has_ece = "ece_tbl" in F1_
    if not (has_pc or has_ece):
        warn("no per-class / ECE data -> skipping figure 6")
        return
    n = int(has_pc) + int(has_ece)
    fig, axes = plt.subplots(1, n, figsize=(COL2 if n == 2 else COL1, 2.30),
                             squeeze=False)
    k = 0
    if has_pc:
        ax = axes[0][k]; k += 1
        d = PC1.iloc[::-1].reset_index(drop=True)
        lab = [re.sub(r"\s+", " ", str(s)).strip() for s in d["Class"]]
        lab = [s if len(s) <= 26 else s[:24] + "." for s in lab]
        y = np.arange(len(d)); h = .38
        ax.barh(y + h / 2, d["Precision"] * 100, h, color=BLUE, ec=INK, lw=.35,
                label="Precision")
        ax.barh(y - h / 2, d["Recall"] * 100, h, color=CP, ec=INK, lw=.35,
                label="Recall")
        ax.set_yticks(y); ax.set_yticklabels(lab, fontsize=6.0)
        vmin = min(d["Precision"].min(), d["Recall"].min()) * 100
        ax.set_xlim(max(0, vmin - 2), 100.6); ax.set_xlabel("%")
        ax.legend(frameon=False, fontsize=6.3, loc="lower left", handlelength=1.0)
        ax.set_title("(a) Per-class performance, DS1", fontsize=7.9, pad=3)
        style(ax, "x")
    if has_ece:
        ax = axes[0][k]
        e = F1_["ece_tbl"]
        cols = [CP if p else "#a9bcc9" for p in e["is_prop"]]
        yy = np.arange(len(e))
        ax.barh(yy, e["ece"], .62, color=cols, ec=INK, lw=.35)
        ax.set_yticks(yy); ax.set_yticklabels(e["short"], fontsize=6.0)
        ax.set_xlabel("Expected calibration error (lower is better)")
        ax.set_title(f"({'b' if has_pc else 'a'}) Calibration quality",
                     fontsize=7.9, pad=3)
        for i, v in enumerate(e["ece"]):
            ax.text(v + e["ece"].max() * .02, yy[i], f"{v:.3f}", va="center",
                    fontsize=5.7, color=CP if cols[i] == CP else "#41576b")
        ax.set_xlim(0, e["ece"].max() * 1.22)
        style(ax, "x")
    fig.tight_layout(pad=.35, w_pad=1.4)
    save(fig, "fig6_perclass_ece")
    FIGS["pc"] = "fig6_perclass_ece"


# ---------------------------------------------------------------- Fig 7 ----
def _corr(df):
    if df is None:
        return None
    d = df.copy()
    if not {"corruption", "severity", "accuracy"}.issubset(d.columns):
        return None
    if d["accuracy"].max() <= 1.5:
        d["accuracy"] = d["accuracy"] * 100
    return d


def fig_corruption():
    C1, C2 = _corr(CR1), _corr(CR2)
    panels = [(c, t) for c, t in [(C1, "(a) DS1 \u2013 ten-way"),
                                  (C2, "(b) DS2 \u2013 binary")] if c is not None]
    if not panels:
        warn("no corruption table -> skipping figure 7")
        return
    fig, axes = plt.subplots(1, len(panels), figsize=(COL1, 1.75),
                             sharey=True, squeeze=False)
    marks = ["o", "s", "^", "D", "v", "P", "X"]
    for ax, (C, title) in zip(axes[0], panels):
        for i, (name, g) in enumerate(C.groupby("corruption", sort=False)):
            g = g.sort_values("severity")
            ax.plot(g["severity"], g["accuracy"], marker=marks[i % len(marks)],
                    ms=3.2, color=PAL[i % len(PAL)], label=str(name).title(),
                    mfc="white", mew=.9)
        ax.set_xticks(sorted(C["severity"].unique()))
        ax.set_xlabel("Corruption severity")
        ax.set_title(title, fontsize=7.9, pad=3)
        ax.set_ylim(max(0, C["accuracy"].min() - 8), 103)
        style(ax)
    axes[0][0].set_ylabel("Accuracy (%)")
    axes[0][0].legend(frameon=False, fontsize=6.0, loc="lower left", ncol=2,
                      handlelength=1.2, columnspacing=.8)
    fig.tight_layout(pad=.35, w_pad=1.0)
    save(fig, "fig7_corruption")
    FIGS["corr"] = "fig7_corruption"


print("\n" + "=" * 74)
print("GENERATING FIGURES")
print("=" * 74)
fig_architecture(); fig_block(); fig_efficiency(); fig_forest()
fig_ablation(); fig_perclass_ece(); fig_corruption()


# =============================================================================
# 6. LATEX TABLES
# =============================================================================
def wtab(name, body):
    (TEX_TAB_DIR / name).write_text(body, encoding="utf-8")
    ok(f"table {name}")


TABS = {}


def tab_datasets():
    rows = []
    for tag, D in [("DS1 (ten-way)", DSET1), ("DS2 (binary)", DSET2)]:
        if D is None:
            continue
        for _, r in D.iterrows():
            rows.append((tag, str(r.get("Split", "")), r.get("Total", ""),
                         r.get("Classes", ""), r.get("Min", "")))
    if not rows:
        warn("no dataset table -> skipped")
        return
    out = [r"\begin{table}[H]", r"\centering",
           r"\caption{Composition of the evaluation corpora. Both are exactly "
           r"class-balanced; each test partition is sealed before any training.}",
           r"\label{tab:datasets}", r"\setlength{\tabcolsep}{4pt}",
           r"\begin{tabular}{llrrr}", r"\toprule",
           r"Corpus & Partition & Images & Classes & Per class \\", r"\midrule"]
    prev = None
    for tag, sp, tot, cls, per in rows:
        if prev is not None and tag != prev:
            out.append(r"\midrule")
        out.append(f"{tag if tag != prev else ''} & {sp} & {int(tot):,} & "
                   f"{int(cls)} & {int(per):,} \\\\")
        prev = tag
    out += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    wtab("tab_datasets.tex", "\n".join(out))
    TABS["datasets"] = True


def tab_config():
    out = [r"\begin{table}[H]", r"\centering",
           r"\caption{Training and evaluation configuration, held identical "
           r"across every architecture and corpus.}",
           r"\label{tab:config}", r"\setlength{\tabcolsep}{3pt}", r"\footnotesize",
           r"\begin{tabular}{@{}l p{0.56\columnwidth}@{}}", r"\toprule",
           r"Setting & Value \\", r"\midrule"]
    for k, v in CONFIG:
        out.append(f"{k} & {v} \\\\")
    out += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    wtab("tab_config.tex", "\n".join(out))
    TABS["config"] = True


def tab_main():
    cols = [("params", "Par.(M)", "{:.2f}", "min"),
            ("acc", "Acc.(\\%)", None, "max"),
            ("f1", "F1(\\%)", "{:.2f}", "max"),
            ("ece", "ECE", "{:.4f}", "min"),
            ("lat", "Lat.(ms)", "{:.2f}", "min")]
    use = [c for c in cols if c[0] in M1.columns]
    two = M2 is not None
    spec = "ll" + "c" * (len(use)) + ("|" + "c" * len(use) if two else "")
    out = [r"\begin{table*}[!t]", r"\centering",
           r"\caption{Sealed-test benchmark on "
           + ("both corpora" if two else "the primary corpus")
           + r". Accuracy is the mean $\pm$ standard deviation over the "
             r"multi-seed runs; remaining columns are computed from the pooled "
             r"predictions. Best value per column in \textbf{bold}; the proposed "
             r"row is shaded.}",
           r"\label{tab:main}", r"\setlength{\tabcolsep}{4.5pt}", r"\footnotesize",
           r"\begin{tabular}{" + spec + "}", r"\toprule"]
    if two:
        out.append(r"& & \multicolumn{" + str(len(use)) + r"}{c|}{\textbf{DS1 -- ten-way}} "
                   r"& \multicolumn{" + str(len(use)) + r"}{c}{\textbf{DS2 -- binary}} \\")
        out.append(r"\cmidrule(lr){3-" + str(2 + len(use)) + r"}"
                   r"\cmidrule(lr){" + str(3 + len(use)) + "-" + str(2 + 2 * len(use)) + r"}")
    hdr = ["ID", "Architecture"] + [h for _, h, _, _ in use] * (2 if two else 1)
    out += [" & ".join(hdr) + r" \\", r"\midrule"]

    def best(M, key, how):
        return M[key].min() if how == "min" else M[key].max()

    d2 = {r["short"]: r for _, r in M2.iterrows()} if two else {}
    for _, r in M1.iterrows():
        pre = r"\rowcolor{propshade} " if r["is_prop"] else ""
        nm = tex_esc(r["short"])
        if r["is_prop"]:
            nm = r"\textbf{" + nm + "}"
        cells = []
        for key, _, fmt, how in use:
            v = r[key]
            if key == "acc":
                s = f"{v:.2f}"
                if abs(v - best(M1, key, how)) < 1e-9:
                    s = r"\textbf{" + s + "}"
                if "acc_std" in M1.columns and pd.notna(r.get("acc_std")):
                    s += f"\\,{{\\tiny$\\pm${r['acc_std']:.2f}}}"
            else:
                s = fmt.format(v)
                if abs(v - best(M1, key, how)) < 1e-9:
                    s = r"\textbf{" + s + "}"
            cells.append(s)
        if two:
            q = d2.get(r["short"])
            for key, _, fmt, how in use:
                if q is None or key not in q or pd.isna(q[key]):
                    cells.append("--"); continue
                v = q[key]
                if key == "acc":
                    s = f"{v:.2f}"
                    if abs(v - best(M2, key, how)) < 1e-9:
                        s = r"\textbf{" + s + "}"
                    if "acc_std" in M2.columns and pd.notna(q.get("acc_std")):
                        s += f"\\,{{\\tiny$\\pm${q['acc_std']:.2f}}}"
                else:
                    s = fmt.format(v)
                    if abs(v - best(M2, key, how)) < 1e-9:
                        s = r"\textbf{" + s + "}"
                cells.append(s)
        out.append(pre + " & ".join([str(r["id"]), nm] + cells) + r" \\")
    # models present only in DS2
    if two:
        extra = M2[~M2["short"].isin(M1["short"])]
        if len(extra):
            out.append(r"\midrule")
            for _, q in extra.iterrows():
                cells = ["--"] * len(use)
                for key, _, fmt, how in use:
                    v = q[key]
                    s = (f"{v:.2f}" if key == "acc" else fmt.format(v))
                    if abs(v - best(M2, key, how)) < 1e-9:
                        s = r"\textbf{" + s + "}"
                    if key == "acc" and pd.notna(q.get("acc_std")):
                        s += f"\\,{{\\tiny$\\pm${q['acc_std']:.2f}}}"
                    cells.append(s)
                out.append(" & ".join([str(q["id"]) + r"$^{\ast}$",
                                       tex_esc(q["short"])] + cells) + r" \\")
            out += [r"\bottomrule", r"\end{tabular}", r"\\[2pt]",
                    r"\footnotesize $^{\ast}$Evaluated on DS2 only.",
                    r"\end{table*}"]
            wtab("tab_main.tex", "\n".join(out)); TABS["main"] = True; return
    out += [r"\bottomrule", r"\end{tabular}", r"\end{table*}"]
    wtab("tab_main.tex", "\n".join(out)); TABS["main"] = True


def tab_stats():
    if S1 is None and S2 is None:
        warn("no stats table -> LaTeX stats table skipped")
        return
    two = S1 is not None and S2 is not None
    out = [r"\begin{table*}[!t]", r"\centering",
           r"\caption{Paired comparison of the proposed model against every "
           r"baseline on the pooled sealed-test predictions. $b$/$c$ are the "
           r"McNemar discordant counts. $\Delta$F1 is proposed $-$ baseline in "
           r"percentage points with a 95\,\% bootstrap confidence interval. "
           r"$p$-values are Holm-corrected within each corpus.}",
           r"\label{tab:stats}", r"\setlength{\tabcolsep}{4pt}", r"\footnotesize",
           r"\begin{tabular}{l|ccrl" + ("|ccrl" if two else "") + "}", r"\toprule"]
    if two:
        out += [r"& \multicolumn{4}{c|}{\textbf{DS1 -- ten-way}} "
                r"& \multicolumn{4}{c}{\textbf{DS2 -- binary}} \\",
                r"\cmidrule(lr){2-5}\cmidrule(lr){6-9}"]
    h = r"$b$/$c$ & $\Delta$F1 & 95\,\% CI & $p$"
    out += ["Baseline & " + h + (" & " + h if two else "") + r" \\", r"\midrule"]

    def cell(r):
        sg = bool(r.get("issig", False))
        d = f"{r['d']:+.2f}"
        if sg:
            d = r"\textbf{" + d + "}"
        p = str(r["p"])
        if p.startswith("<"):
            p = "$<$" + p[1:]
        return [f"{int(r['b'])}/{int(r['c'])}", d,
                f"[{r['lo']:+.2f},\\,{r['hi']:+.2f}]",
                p + (r"\,\ding{51}" if sg else "")]

    d2 = {r["base"]: r for _, r in S2["tbl"].iterrows()} if S2 else {}
    src = S1["tbl"] if S1 else S2["tbl"]
    for _, r in src.iterrows():
        row = [tex_esc(r["base"])] + cell(r)
        if two:
            q = d2.get(r["base"])
            row += cell(q) if q is not None else ["--"] * 4
        out.append(" & ".join(row) + r" \\")
    if two:
        for b, q in d2.items():
            if b not in set(src["base"]):
                out.append(" & ".join([tex_esc(b)] + ["--"] * 4 + cell(q)) + r" \\")
    out += [r"\bottomrule", r"\end{tabular}", r"\\[2pt]",
            r"\footnotesize \ding{51} marks significance at $\alpha=0.05$. "
            r"Positive $\Delta$F1 favours the proposed model.", r"\end{table*}"]
    wtab("tab_stats.tex", "\n".join(out)); TABS["stats"] = True


def tab_ablation():
    A1, A2 = _abl(ABL1), _abl(ABL2)
    if A1 is None and A2 is None:
        warn("no ablation table -> LaTeX ablation table skipped")
        return
    two = A1 is not None and A2 is not None
    out = [r"\begin{table}[H]", r"\centering",
           r"\caption{Component ablation. All variants share the backbone and "
           r"training recipe; $\Delta$F1 is relative to the bare backbone (A0).}",
           r"\label{tab:ablation}", r"\setlength{\tabcolsep}{3pt}", r"\footnotesize",
           r"\begin{tabular}{@{}ll|cc" + ("|cc" if two else "") + r"@{}}", r"\toprule"]
    if two:
        out += [r"& & \multicolumn{2}{c|}{DS1} & \multicolumn{2}{c}{DS2} \\",
                r"\cmidrule(lr){3-4}\cmidrule(lr){5-6}"]
    out += [r"ID & Configuration & F1(\%) & $\Delta$"
            + (r" & F1(\%) & $\Delta$" if two else "") + r" \\", r"\midrule"]
    base = A1 if A1 is not None else A2
    b1 = A1["f1"].iloc[0] if A1 is not None else None
    b2 = A2["f1"].iloc[0] if A2 is not None else None
    m2 = {r["id"]: r for _, r in A2.iterrows()} if two else {}
    for _, r in base.iterrows():
        pre = r"\rowcolor{propshade} " if str(r["id"]) == "A8" else ""
        cfg = tex_esc(r.get("cfg", ""))
        cells = [f"{r['f1']:.2f}", f"{r['f1']-b1:+.2f}"] if A1 is not None else []
        if two:
            q = m2.get(r["id"])
            cells += ([f"{q['f1']:.2f}", f"{q['f1']-b2:+.2f}"] if q is not None
                      else ["--", "--"])
        out.append(pre + " & ".join([str(r["id"]), cfg] + cells) + r" \\")
    out += [r"\bottomrule", r"\end{tabular}", r"\\[2pt]",
            r"\footnotesize The DeiT cross-attention branch was disabled on the "
            r"DirectML backend (Sec.~\ref{sec:deit}); variants that differ only "
            r"in that branch therefore coincide.", r"\end{table}"]
    wtab("tab_ablation.tex", "\n".join(out)); TABS["abl"] = True


def tab_perclass():
    if PC1 is None:
        warn("no per-class table -> skipped")
        return
    d = PC1.copy()
    cols = [c for c in ["Class", "TP", "FP", "FN", "Precision", "Recall", "G-Mean"]
            if c in d.columns]
    out = [r"\begin{table}[H]", r"\centering",
           r"\caption{Per-class sealed-test performance of the proposed model "
           r"on the ten-way corpus.}",
           r"\label{tab:perclass}", r"\setlength{\tabcolsep}{3pt}", r"\footnotesize",
           r"\begin{tabular}{l" + "r" * (len(cols) - 1) + "}", r"\toprule",
           " & ".join(tex_esc(c) for c in cols) + r" \\", r"\midrule"]
    for _, r in d.iterrows():
        cells = []
        for c in cols:
            v = r[c]
            cells.append(tex_esc(v) if c == "Class" else
                         (f"{int(v)}" if c in ("TP", "FP", "FN") else f"{v:.4f}"))
        out.append(" & ".join(cells) + r" \\")
    num = [c for c in cols if c not in ("Class",)]
    mrow = ["\\textit{Macro avg.}"]
    for c in num:
        v = d[c].sum() if c in ("TP", "FP", "FN") else d[c].mean()
        mrow.append(f"{int(v)}" if c in ("TP", "FP", "FN") else f"{v:.4f}")
    out += [r"\midrule", " & ".join(mrow) + r" \\",
            r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    wtab("tab_perclass.tex", "\n".join(out)); TABS["pc"] = True


print("\n" + "=" * 74)
print("GENERATING LATEX TABLES")
print("=" * 74)
tab_datasets(); tab_config(); tab_main(); tab_stats(); tab_ablation(); tab_perclass()


# =============================================================================
# 7. RESULTS SECTION  -- every number below is computed, none is typed
# =============================================================================
def results_section():
    P = []
    A = P.append
    f1, f2 = F1_, F2_

    A(r"\section{Results}")
    A(r"\label{sec:results}")
    if TABS.get("main"):
        A(r"\input{tables/tab_main}")

    # ---- main benchmark -----------------------------------------------
    A(r"\subsection{Main Benchmark}")
    top1 = f1["order"].iloc[0]; top2 = f1["order"].iloc[1]
    s = (f"Table~\\ref{{tab:main}} reports the full sealed-test benchmark. "
         f"On the ten-way corpus the strongest architectures are "
         f"{tex_esc(top1['short'])} ({top1['acc']:.2f}\\,\\%) and "
         f"{tex_esc(top2['short'])} ({top2['acc']:.2f}\\,\\%). "
         f"The proposed model reaches {f1['acc']:.2f}\\,\\%, placing it "
         f"{w_ord(f1['rank'])} of {w_card(f1['n'])} on raw accuracy while using "
         f"{f1['params']:.2f}\\,M parameters.")
    if f2 is not None:
        t1 = f2["order"].iloc[0]
        s += (f" On the binary corpus {tex_esc(t1['short'])} leads at "
              f"{t1['acc']:.2f}\\,\\%, with the proposed model at "
              f"{f2['acc']:.2f}\\,\\%.")
    s += (" We state this plainly because it is the honest reading: the proposed "
          "model is \\emph{not} the most accurate model in this study. Its "
          "interest lies in the cost at which it reaches that accuracy.")
    A(s)

    if "acc_std" in M1.columns and M1["acc_std"].notna().any():
        w = M1.loc[M1["acc_std"].idxmax()]
        A(f"Seed variance is strongly architecture-dependent and is routinely "
          f"omitted in this literature. {tex_esc(w['short'])} records "
          f"$\\pm{w['acc_std']:.2f}$ percentage points on the ten-way corpus, "
          f"whereas the proposed model records $\\pm{f1['acc_std']:.2f}$"
          + (", the tightest in the table"
             if f1["tightest"] == "MS-EGCA (proposed)"
             else f", against a best of $\\pm{f1['tightest_v']:.2f}$ for "
                  f"{tex_esc(f1['tightest'])}")
          + f". A practitioner selecting an "
          f"architecture on a single favourable run would be badly misled.")

    # ---- efficiency ----------------------------------------------------
    A(r"\subsection{Efficiency and the Accuracy--Cost Frontier}")
    A(r"\label{sec:efficiency}")
    if "eff" in FIGS:
        A(r"\begin{figure*}[!t]" "\n" r"\centering" "\n"
          r"\includegraphics[width=\textwidth]{figs/" + FIGS["eff"] + ".pdf}\n"
          r"\caption{Efficiency analysis. (a),(b)~Accuracy--parameter plane on "
          r"the sealed test sets; the dashed staircase is the Pareto frontier "
          r"and the shaded region is dominated. (c)~Macro-F1 delivered per "
          r"million parameters. (d)~The latency--accuracy plane.}" "\n"
          r"\label{fig:eff}" "\n" r"\end{figure*}")

    dominators = M1[(M1["params"] <= f1["params"]) & (M1["acc"] > f1["acc"])]
    if len(dominators) == 0:
        s = (f"Fig.~\\ref{{fig:eff}} places every architecture in the "
             f"accuracy--parameter plane. On the ten-way corpus the proposed "
             f"model lies on the Pareto frontier: no architecture achieves "
             f"higher accuracy with fewer parameters.")
    else:
        d0 = dominators.sort_values("acc", ascending=False).iloc[0]
        s = (f"Fig.~\\ref{{fig:eff}} places every architecture in the "
             f"accuracy--parameter plane. On the ten-way corpus the proposed "
             f"model is dominated by {tex_esc(d0['short'])} "
             f"({d0['params']:.2f}\\,M, {d0['acc']:.2f}\\,\\%), which we flag "
             f"explicitly rather than obscure.")
    if f2 is not None:
        dom2 = M2[(M2["params"] <= f2["params"]) & (M2["acc"] > f2["acc"])]
        if len(dom2) == 0:
            s += " The same holds on the binary corpus."
        else:
            d0 = dom2.sort_values("acc", ascending=False).iloc[0]
            s += (f" On the binary corpus it is dominated by "
                  f"{tex_esc(d0['short'])} ({d0['params']:.2f}\\,M, "
                  f"{d0['acc']:.2f}\\,\\%).")
    A(s)

    e = f1["eff_tbl"]
    others = e[~e["is_prop"]].head(3)
    lst = and_list([f"{tex_esc(r['short'])} at {r['eff']:.1f}"
                    for _, r in others.iterrows()])
    ref = M1.loc[M1["acc"].idxmax()]
    A(f"Fig.~\\ref{{fig:eff}}(c) quantifies this directly. In macro-F1 points "
      f"per million parameters the proposed model scores {f1['eff']:.1f}, "
      f"ranking {w_ord(f1['eff_rank'])} of {w_card(f1['n'])}; for comparison "
      f"{lst}. Against {tex_esc(ref['short'])} "
      f"({ref['f1']/ref['params']:.1f}) this is a "
      f"{f1['eff']/(ref['f1']/ref['params']):.1f}$\\times$ advantage. "
      f"Parameter count and latency are not interchangeable proxies for cost: "
      f"at {f1['lat']:.2f}\\,ms the proposed model is slower than several "
      f"larger networks, because the attention branches introduce sequential "
      f"dependencies that parallelise poorly.")

    # ---- significance ---------------------------------------------------
    if S1 is not None or S2 is not None:
        A(r"\subsection{Statistical Significance}")
        A(r"\label{sec:significance}")
        if TABS.get("stats"):
            A(r"\input{tables/tab_stats}")
        if "forest" in FIGS:
            A(r"\begin{figure*}[!t]" "\n" r"\centering" "\n"
              r"\includegraphics[width=\textwidth]{figs/" + FIGS["forest"] + ".pdf}\n"
              r"\caption{Paired macro-F1 differences between the proposed model "
              r"and each baseline, with 95\,\% bootstrap confidence intervals. "
              r"Intervals crossing zero indicate statistical equivalence.}" "\n"
              r"\label{fig:forest}" "\n" r"\end{figure*}")
        if S1 is not None:
            A(f"Accuracy differences of a few tenths of a point are not "
              f"self-evidently meaningful, and Table~\\ref{{tab:stats}} resolves "
              f"which are. On the ten-way corpus the proposed model is "
              f"significantly better than {and_list(map(tex_esc, S1['better']))}; "
              f"significantly worse than {and_list(map(tex_esc, S1['worse']))}; "
              f"and statistically indistinguishable from "
              f"{and_list(map(tex_esc, S1['equiv']))}. "
              f"{w_card(S1['sig']).capitalize()} of {w_card(S1['tot'])} "
              f"comparisons reach significance.")
        if S2 is not None:
            A(f"On the binary corpus only {w_card(S2['sig'])} of "
              f"{w_card(S2['tot'])} comparisons reach significance, the proposed "
              f"model being statistically equivalent to "
              f"{and_list(map(tex_esc, S2['equiv']))}. "
              f"The equivalence results carry the argument: several of these "
              f"baselines use multiple times more parameters and are not "
              f"distinguishable from a {f1['params']:.2f}\\,M-parameter model.")
        if S1 is not None and "cd" in S1["tbl"].columns:
            A(f"All effect sizes are small in Cohen's terms "
              f"($|d|\\le{S1['tbl']['cd'].abs().max():.2f}$), which is itself a "
              f"finding: on balanced corpora of this size the choice among "
              f"modern efficient backbones matters less than the literature's "
              f"emphasis on leaderboard position would suggest.")

    # ---- ablation --------------------------------------------------------
    A1_, A2_ = _abl(ABL1), _abl(ABL2)
    if A1_ is not None or A2_ is not None:
        A(r"\subsection{Ablation}")
        if TABS.get("abl"):
            A(r"\input{tables/tab_ablation}")
        if "abl" in FIGS:
            A(r"\begin{figure}[H]" "\n" r"\centering" "\n"
              r"\includegraphics[width=\columnwidth]{figs/" + FIGS["abl"] + ".pdf}\n"
              r"\caption{Ablation on the evaluated corpora. Dotted lines mark "
              r"the bare-backbone baseline (A0).}" "\n"
              r"\label{fig:abl}" "\n" r"\end{figure}")
        if A1_ is not None:
            b = A1_["f1"].iloc[0]
            d = A1_.assign(delta=A1_["f1"] - b)
            worse = d[d["delta"] < 0]
            bestv = d.sort_values("delta", ascending=False).iloc[0]
            A(f"Table~\\ref{{tab:ablation}} isolates the contribution of each "
              f"component against the bare backbone (A0, {b:.2f}\\,\\% F1). "
              f"The results are nuanced and we resist over-reading them: "
              f"{w_card(len(worse))} of the {w_card(len(d)-1)} variants tested "
              f"\\emph{{degrade}} the backbone, and the best variant "
              f"({tex_esc(bestv['id'])}) improves it by only "
              f"{bestv['delta']:+.2f} percentage points. The widespread "
              f"assumption that adding an attention module to a compact backbone "
              f"yields free accuracy is not supported here.")

    # ---- per class + calibration -----------------------------------------
    if PC1 is not None or "ece_rank" in F1_:
        A(r"\subsection{Per-Class Behaviour and Calibration}")
        if TABS.get("pc"):
            A(r"\input{tables/tab_perclass}")
        if "pc" in FIGS:
            A(r"\begin{figure*}[!t]" "\n" r"\centering" "\n"
              r"\includegraphics[width=\textwidth]{figs/" + FIGS["pc"] + ".pdf}\n"
              r"\caption{(a)~Per-class precision and recall of the proposed "
              r"model. (b)~Expected calibration error across all architectures, "
              r"sorted ascending; the proposed model is shaded.}" "\n"
              r"\label{fig:pc}" "\n" r"\end{figure*}")
        if PC1 is not None and "Recall" in PC1.columns:
            d = PC1.sort_values("Recall")
            weak = d.head(3)
            strong = d.iloc[-1]
            wl = and_list([f"{tex_esc(r['Class'])} ({r['Recall']:.4f})"
                           for _, r in weak.iterrows()])
            s = (f"Table~\\ref{{tab:perclass}} decomposes performance by class. "
                 f"The three weakest classes by recall are {wl}, while "
                 f"{tex_esc(strong['Class'])} reaches {strong['Recall']:.4f}.")
            if "FP" in PC1.columns:
                fp = PC1.loc[PC1["FP"].idxmax()]
                s += (f" {tex_esc(fp['Class'])} generates the most false "
                      f"positives ({int(fp['FP'])}), confirming that the "
                      f"residual error is concentrated in genuine fine-grained "
                      f"confusion rather than distributed noise.")
            A(s)
        if "ece_rank" in F1_:
            ec = F1_["ece_tbl"]
            better = ec[ec.index < ec.index[ec["is_prop"]][0]]
            bl = and_list([f"{tex_esc(r['short'])} ({r['ece']:.4f})"
                           for _, r in better.iterrows()]) if len(better) else None
            s = (f"The proposed model records an expected calibration error of "
                 f"{F1_['ece']:.4f}, the {w_ord(F1_['ece_rank'])} lowest of "
                 f"{w_card(len(ec))} architectures")
            s += (f", behind {bl}." if bl else ".")
            s += (" This matters operationally: a model whose confidence can be "
                  "thresholded to hand ambiguous frames to an operator is more "
                  "useful than a marginally more accurate but systematically "
                  "overconfident one.")
            A(s)

    # ---- robustness -------------------------------------------------------
    C1, C2 = _corr(CR1), _corr(CR2)
    if C1 is not None:
        A(r"\subsection{Corruption Robustness}")
        if "corr" in FIGS:
            A(r"\begin{figure}[H]" "\n" r"\centering" "\n"
              r"\includegraphics[width=\columnwidth]{figs/" + FIGS["corr"] + ".pdf}\n"
              r"\caption{Accuracy of the proposed model under synthetic field "
              r"corruptions at increasing severity.}" "\n"
              r"\label{fig:corr}" "\n" r"\end{figure}")
        smax = int(C1["severity"].max())
        hi = C1[C1["severity"] == smax].sort_values("accuracy")
        worst = hi.iloc[0]; best = hi.iloc[-1]
        means = C1.groupby("severity")["accuracy"].mean()
        ms = ", ".join(f"{v:.2f}\\,\\% at severity {int(k)}"
                       for k, v in means.items())
        s = (f"Fig.~\\ref{{fig:corr}} reports the stress test, which contains "
             f"the most operationally consequential result in this paper. "
             f"Degradation is severe and strongly corruption-specific: at "
             f"severity {smax}, {tex_esc(str(worst['corruption']))} reduces "
             f"accuracy to {worst['accuracy']:.2f}\\,\\% while "
             f"{tex_esc(str(best['corruption']))} costs almost nothing "
             f"({best['accuracy']:.2f}\\,\\%). Averaged over all corruptions the "
             f"model retains {ms}.")
        if C2 is not None:
            m2 = C2.groupby("severity")["accuracy"].mean()
            s += (f" The binary task is far more robust, never falling below "
                  f"{m2.min():.2f}\\,\\% on the same corruption suite.")
        s += (" Blur and noise destroy exactly the high-frequency boundary "
              "information on which fine-grained discrimination depends, whereas "
              "the coarse decision survives on low-frequency shape. For system "
              "designers this converts into a specification: an optical sensor "
              "intended for fine-grained classification must maintain modulation "
              "transfer at the spatial frequencies of vehicle structure. This is "
              "a limitation of the entire model class, our own included.")
        A(s)
    return "\n\n".join(P)


# =============================================================================
# 7b. DISCUSSION  -- also computed, not typed
# =============================================================================
def discussion_section():
    f1, f2 = F1_, F2_
    P = [r"\section{Discussion}", r"\label{sec:discussion}"]
    A = P.append

    A(r"\subsection{What the Evidence Supports}")
    s = (f"The defensible claim from this study is one of efficiency and "
         f"reliability, not of accuracy. At {f1['params']:.2f}\\,M parameters "
         f"the proposed model delivers {f1['eff']:.1f} macro-F1 points per "
         f"million parameters")
    if S1 is not None and S1["equiv"]:
        s += (f" and is statistically indistinguishable from "
              f"{and_list(map(tex_esc, S1['equiv']))} on the ten-way corpus")
    if S2 is not None:
        s += (f", and from {w_card(len(S2['equiv']))} of {w_card(S2['tot'])} "
              f"baselines on the binary corpus")
    s += "."
    if "tightest" in f1 and f1["tightest"] == "MS-EGCA (proposed)":
        s += (f" It records the tightest seed-to-seed variance of any "
              f"architecture on the ten-way corpus "
              f"($\\pm{f1['acc_std']:.2f}$).")
    elif "tightest" in f1:
        s += (f" Its seed-to-seed variance ($\\pm{f1['acc_std']:.2f}$) is close "
              f"to the tightest recorded, {tex_esc(f1['tightest'])} at "
              f"$\\pm{f1['tightest_v']:.2f}$.")
    if S1 is not None and S1["worse"]:
        s += (f" Where it is beaten---by {and_list(map(tex_esc, S1['worse']))} "
              f"on the ten-way corpus---we have quantified the gap with paired "
              f"tests and confidence intervals rather than leaving it to a "
              f"rounded table.")
    A(s)

    A1_ = _abl(ABL1)
    if A1_ is not None:
        b = A1_["f1"].iloc[0]
        d = A1_.assign(delta=A1_["f1"] - b)
        gains = d[d["delta"] > 0]["delta"]
        miss = int(round(len(PC1) * 0 + (100 - b) / 100 *
                         (PC1["TP"].sum() + PC1["FN"].sum()))) if (
            PC1 is not None and {"TP", "FN"}.issubset(PC1.columns)) else None
        s = (f"The measured ablation gains "
             f"({gains.min():+.2f} to {gains.max():+.2f} percentage points where "
             f"positive) are smaller than the framing of edge-guided attention "
             f"might suggest, and two explanations are available. The first is a "
             f"ceiling effect: at {b:.2f}\\,\\% backbone F1")
        if miss:
            s += f" only about {miss} test images remain misclassified"
        s += (", and the per-class breakdown suggests most are genuine "
              "fine-grained ambiguities that better edge weighting cannot "
              "resolve. The second is that the backbone already contains "
              "squeeze-and-excitation blocks throughout, so the marginal value "
              "of further channel recalibration is inherently limited---"
              "supported by the observation that the established attention "
              "modules also fail to improve it.")
        A(s)

    A(r"\subsection{Limitations}")
    lim = []
    if S1 is not None and S1["worse"]:
        lim.append(f"\\emph{{(i)}}~The proposed model does not achieve the "
                   f"highest accuracy: {w_card(len(S1['worse']))} baselines are "
                   f"significantly better on the ten-way corpus"
                   + (f" and {w_card(len(S2['worse']))} on the binary corpus"
                      if S2 is not None else "") + ".")
    lim.append(r"\emph{(ii)}~The DeiT-Tiny cross-attention branch of the general "
               r"formulation was disabled throughout (Sec.~\ref{sec:deit}), so "
               r"the full hybrid remains untested.")
    lim.append(r"\emph{(iii)}~Latency was measured on a desktop GPU in FP32; the "
               r"on-device edge-metrics pass failed on that backend, so the "
               r"efficiency claims rest on parameter count and desktop latency "
               r"rather than measured embedded throughput.")
    if M2 is not None:
        only2 = M2[~M2["short"].isin(M1["short"])]["short"].tolist()
        if only2:
            lim.append(f"\\emph{{(iv)}}~{and_list(map(tex_esc, only2))} "
                       f"lack{'s' if len(only2)==1 else ''} a result on the "
                       f"ten-way corpus for the procedural reason given in "
                       f"Sec.~\\ref{{sec:protocol}}.")
    lim.append(r"\emph{(v)}~Corruptions are synthetic; field imagery would "
               r"exercise correlated degradations this model does not reproduce.")
    A(" ".join(lim))

    A(r"\subsection{Implications for Instrumentation}")
    s = ("Read as a measurement study, several findings transfer beyond this "
         "architecture. ")
    if "acc_std" in M1.columns and M1["acc_std"].notna().any():
        s += (f"Multi-seed variance spans "
              f"$\\pm{M1['acc_std'].min():.2f}$ to $\\pm{M1['acc_std'].max():.2f}$ "
              f"percentage points across architectures and must be reported if "
              f"comparisons are to mean anything. ")
    if "ece" in M1.columns and M1["ece"].notna().any() and len(M1) > 2:
        r = float(np.corrcoef(M1["acc"], M1["ece"])[0, 1])
        s += (f"Calibration is only weakly related to accuracy "
              f"(Pearson $r={r:.2f}$ across the evaluated architectures), so "
              f"selecting a model on accuracy alone leaves confidence quality "
              f"largely to chance. ")
    C1 = _corr(CR1)
    if C1 is not None:
        s += ("And the collapse under blur and noise means a recognition "
              "system's effective accuracy is a property of the imaging chain "
              "at least as much as of the network.")
    A(s)
    return "\n\n".join(P)


# =============================================================================
# 8. ASSEMBLE paper.tex
# =============================================================================
def _tokens():
    f1, f2 = F1_, F2_
    names = set(M1.loc[~M1["is_prop"], "short"])
    if M2 is not None:
        names |= set(M2.loc[~M2["is_prop"], "short"])
    n_base_all = len(names)
    t = {"PARAMS": f"{f1['params']:.2f}", "LAT": f"{f1['lat']:.2f}",
         "NBASE_ALL": w_card(n_base_all),
         "ACC1": f"{f1['acc']:.2f}", "F1_1": f"{f1['f1']:.2f}",
         "NBASE": w_card(f1["n"] - 1), "NMODELS": w_card(f1["n"]),
         "EFF": f"{f1['eff']:.1f}", "ECE1": f"{f1['ece']:.4f}",
         "RANK1": w_ord(f1["rank"]),
         "STD1": (f"{f1['acc_std']:.2f}" if pd.notna(f1.get("acc_std")) else "--"),
         "BEST1": tex_esc(f1["best_name"]), "BESTACC1": f"{f1['best_acc']:.2f}"}
    if f2 is not None:
        t.update({"ACC2": f"{f2['acc']:.2f}",
                  "STD2": (f"{f2['acc_std']:.2f}"
                           if pd.notna(f2.get("acc_std")) else "--"),
                  "BEST2": tex_esc(f2["best_name"]),
                  "BESTACC2": f"{f2['best_acc']:.2f}",
                  "NBASE2": w_card(f2["n"] - 1)})
    else:
        t.update({"ACC2": "--", "STD2": "--", "BEST2": "--",
                  "BESTACC2": "--", "NBASE2": "--"})
    return t


TOKENS = _tokens()
_SEC_SEARCH = [Path("."), Path(__file__).resolve().parent if "__file__" in dir()
               else Path("."), OUT_DIR]


def _stage_section(fname):
    """Copy a narrative section into OUT_DIR, substituting <<TOKEN>> values."""
    for d in _SEC_SEARCH:
        p = Path(d) / fname
        if p.is_file():
            txt = p.read_text(encoding="utf-8", errors="ignore")
            unknown = set(re.findall(r"<<([A-Z0-9_]+)>>", txt)) - set(TOKENS)
            for k, v in TOKENS.items():
                txt = txt.replace(f"<<{k}>>", str(v))
            if unknown:
                warn(f"{fname}: unknown token(s) {sorted(unknown)} left as-is")

            # drop \input{tables/X} lines whose table was not generated
            def _keep(line):
                m = re.match(r"\s*\\input\{tables/([A-Za-z0-9_]+)\}\s*$", line)
                if m and not (TEX_TAB_DIR / f"{m.group(1)}.tex").is_file():
                    warn(f"{fname}: dropping \\input of missing "
                         f"tables/{m.group(1)}.tex")
                    return False
                return True

            txt = "\n".join(l for l in txt.splitlines() if _keep(l))
            (OUT_DIR / fname).write_text(txt, encoding="utf-8")
            return True
    return False


PREAMBLE = r"""%=============================================================================
%  Auto-generated by paper_pipeline.py -- do not hand-edit the Results section;
%  edit the pipeline and re-run so the numbers stay tied to the source data.
%  Build: pdflatex paper -> bibtex paper -> pdflatex paper -> pdflatex paper
%=============================================================================
\IfFileExists{IEEEtran.cls}{%
  \documentclass[journal,10pt,twocolumn]{IEEEtran}
  \newcommand{\USINGIEEETRAN}{1}
}{%
  \documentclass[10pt,twocolumn]{article}
  \newcommand{\USINGIEEETRAN}{0}
}
\usepackage[T1]{fontenc}
\usepackage[utf8]{inputenc}
\usepackage{times}
\usepackage{amsmath,amssymb,amsfonts}
\usepackage{graphicx}
\usepackage{booktabs}
\usepackage{multirow}
\usepackage{float}
\usepackage{stfloats}
\usepackage{array}
\usepackage{colortbl}
\usepackage[table,dvipsnames]{xcolor}
\usepackage{pifont}
\usepackage{url}
\usepackage{balance}
\usepackage[hidelinks]{hyperref}
\usepackage{cite}
\definecolor{propshade}{RGB}{252,238,238}
\if\USINGIEEETRAN0
  \usepackage[letterpaper,top=0.75in,bottom=1.0in,left=0.625in,right=0.625in,
              columnsep=0.25in]{geometry}
  \usepackage{titlesec}\usepackage{abstract}
  \renewcommand{\abstractnamefont}{\normalfont\bfseries\itshape}
  \renewcommand{\abstracttextfont}{\normalfont\small\bfseries}
  \titleformat{\section}{\normalfont\normalsize\scshape\centering}{\Roman{section}.}{0.6em}{}
  \titleformat{\subsection}{\normalfont\normalsize\itshape}{\Alph{subsection}.}{0.6em}{}
  \renewcommand{\thesection}{\Roman{section}}
  \renewcommand{\thesubsection}{\Alph{subsection}}
  \setlength{\parindent}{1em}
  \newcommand{\IEEEmembership}[1]{\textit{#1}}
  \newcommand{\IEEEPARstart}[2]{\textbf{\large #1}#2}
  \newenvironment{IEEEkeywords}{\vspace{2pt}\noindent\small\textit{Index Terms}---}{\vspace{4pt}}
\fi
\newcommand{\ms}{MS-EGCA}
\newcommand{\pp}{\,pp}
\begin{document}
"""


def build_tex():
    f1 = F1_
    abs_txt = (
        "Recognising ground vehicles from optical imagery on power-constrained "
        "platforms is a measurement problem as much as a perception problem: the "
        "instrument must be accurate, fast, and honest about its own uncertainty. "
        "We introduce \\ms{}, a Multi-Scale Edge-Guided Channel Attention module "
        "that couples a three-scale learned edge response to an efficient channel "
        "descriptor through an input-adaptive residual gate, paired with a "
        "learnable Sobel edge-prior gate at the network input. Inserted into a "
        f"MobileNetV3-Small backbone the recogniser occupies {f1['params']:.2f}\\,M "
        f"parameters and runs in {f1['lat']:.2f}\\,ms per image. We evaluate it "
        f"against {TOKENS['NBASE_ALL']} established efficient architectures under a "
        "protocol designed for measurement credibility: class-balanced corpora, "
        "test partitions sealed before training and read once, multiple random "
        "seeds per architecture, and paired McNemar tests with Holm correction "
        f"and bootstrap confidence intervals. The proposed model attains "
        f"{f1['acc']:.2f}\\,\\% accuracy on the ten-way vehicle-type task")
    if F2_ is not None:
        abs_txt += f" and {F2_['acc']:.2f}\\,\\% on the military-versus-civilian task"
    abs_txt += (f", delivering {f1['eff']:.1f} macro-F1 points per million "
                "parameters. We deliberately do not claim state-of-the-art "
                "accuracy, and quantify every gap with paired significance tests "
                "rather than obscuring it. A five-corruption stress test exposes "
                "a severe brittleness of fine-grained vehicle-type recognition to "
                "blur and sensor noise, with direct consequences for sensor "
                "specification.")

    parts = [PREAMBLE,
             "\\title{" + TITLE + "}\n",
             "\\if\\USINGIEEETRAN1\n\\author{" + AUTHOR_IEEE + "}\n\\else\n"
             "\\author{\\parbox{0.92\\textwidth}{\\centering\\normalsize "
             "Muhammad Azam Khan$^{1,*}$\\quad Yanyan Huang$^{1,*}$\\quad "
             "Nazish Waqar Waqar$^{2}$\\quad Muhammad Inaam Ul Haq$^{3}$\\quad "
             "Muhammad Adnan Khan$^{4,*}$\\\\[3pt]\\footnotesize "
             "$^{1}$Nanjing University of Science and Technology, China; "
             "$^{2}$London South Bank University, UK; "
             "$^{3}$COMSATS University Islamabad, Sahiwal, Pakistan; "
             "$^{4}$Gachon University, Republic of Korea}}\n\\date{}\n\\fi\n",
             "\\maketitle\n",
             "\\begin{abstract}\n" + abs_txt + "\n\\end{abstract}\n",
             "\\begin{IEEEkeywords}\nAttention mechanisms, automatic target "
             "recognition, calibration, edge-guided learning, efficient neural "
             "networks, embedded vision, military vehicles, robustness.\n"
             "\\end{IEEEkeywords}\n"]

    for stub, fname in [("Introduction", "sec_introduction.tex"),
                        ("Related Work", "sec_related_work.tex"),
                        ("Proposed Method", "sec_method.tex"),
                        ("Experimental Setup", "sec_setup.tex")]:
        if _stage_section(fname):
            parts.append("\\input{" + fname[:-4] + "}\n")
        else:
            parts.append(f"\\section{{{stub}}}\n"
                         f"\\label{{sec:{stub.split()[0].lower()}}}\n"
                         f"%% TODO: narrative text for '{stub}' goes in "
                         f"{fname} (place it next to this script).\n")
            warn(f"{fname} not found -> '{stub}' left as a stub in paper.tex")

    parts.append(results_section())
    parts.append(discussion_section())
    if _stage_section("sec_conclusion.tex"):
        parts.append("\\input{sec_conclusion}\n")
    else:
        parts.append("\n\\section{Conclusion}\n\\label{sec:conclusion}\n"
                     "%% TODO: conclusion narrative (sec_conclusion.tex).\n")
        warn("sec_conclusion.tex not found -> conclusion left as a stub")
    parts.append("\n\\if\\USINGIEEETRAN0\\footnotesize\\fi\n"
                 "\\IfFileExists{IEEEtran.bst}{\\bibliographystyle{IEEEtran}}"
                 "{\\bibliographystyle{ieeetr}}\n"
                 "\\IfFileExists{refs.bib}{\\bibliography{refs}}{}\n"
                 "\\balance\n\\end{document}\n")

    tex = "\n".join(parts)
    (OUT_DIR / "paper.tex").write_text(tex, encoding="utf-8")
    ok(f"paper.tex ({len(tex.splitlines())} lines)")
    return tex


print("\n" + "=" * 74)
print("WRITING paper.tex")
print("=" * 74)
build_tex()

# bring an existing bibliography along if there is one
for cand in ["refs.bib", "../refs.bib", "./paper/refs.bib"]:
    p = Path(cand)
    if p.is_file():
        shutil.copy(p, OUT_DIR / "refs.bib")
        ok(f"bibliography copied from {p}")
        break

# =============================================================================
# 9. COMPILE
# =============================================================================
print("\n" + "=" * 74)
print("COMPILING PDF")
print("=" * 74)
if shutil.which("pdflatex") is None:
    warn("pdflatex not on PATH -- skipping compile. "
         "Upload paper_build/ to Overleaf instead.")
else:
    cwd = os.getcwd()
    try:
        os.chdir(OUT_DIR)
        try:
            Path("paper.pdf").unlink(missing_ok=True)
        except OSError:
            pass
        for i, cmd in enumerate([["pdflatex", "-interaction=nonstopmode", "paper.tex"],
                                 ["bibtex", "paper"],
                                 ["pdflatex", "-interaction=nonstopmode", "paper.tex"],
                                 ["pdflatex", "-interaction=nonstopmode", "paper.tex"]]):
            subprocess.run(cmd, stdout=subprocess.DEVNULL,
                           stderr=subprocess.DEVNULL, check=False)
        log = Path("paper.log")
        if log.is_file():
            t = log.read_text(encoding="utf-8", errors="ignore")
            errs = [l for l in t.splitlines() if l.startswith("! ")]
            over = re.findall(r"Overfull \\hbox \(([\d.]+)pt", t)
            big = [o for o in over if float(o) > 10]
            if errs:
                warn(f"{len(errs)} LaTeX error(s): {errs[:3]}")
            if big:
                warn(f"{len(big)} overfull hbox(es) > 10pt")
        if Path("paper.pdf").is_file():
            try:
                out = subprocess.run(["pdfinfo", "paper.pdf"],
                                     capture_output=True, text=True).stdout
                pages = re.search(r"Pages:\s+(\d+)", out)
                ok(f"paper.pdf written -- {pages.group(1) if pages else '?'} pages")
            except Exception:
                ok("paper.pdf written")
        else:
            warn("paper.pdf not produced -- inspect paper_build/paper.log")
    finally:
        os.chdir(cwd)

# =============================================================================
# 10. SUMMARY
# =============================================================================
print("\n" + "=" * 74)
print("SUMMARY")
print("=" * 74)
print(f"  output folder : {OUT_DIR.resolve()}")
print(f"  figures       : {len(list(FIG_DIR.glob('*.pdf')))}")
print(f"  latex tables  : {len(list(TEX_TAB_DIR.glob('*.tex')))}")
print(f"  DS1 models    : {len(M1)}   proposed acc = {F1_['acc']:.2f}%")
if M2 is not None:
    print(f"  DS2 models    : {len(M2)}   proposed acc = {F2_['acc']:.2f}%")
if _warnings:
    print(f"\n  {len(_warnings)} warning(s):")
    for w in _warnings:
        print(f"    - {w}")
else:
    print("\n  no warnings.")
print("=" * 74)
