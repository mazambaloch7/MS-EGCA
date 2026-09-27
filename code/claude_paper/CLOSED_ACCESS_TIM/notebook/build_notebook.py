#!/usr/bin/env python3
"""
Builds `generate_paper_assets.ipynb`.

The notebook reads YOUR result files and regenerates every figure and every
LaTeX table. No experimental value is written into the notebook by hand.

Run:  python build_notebook.py
"""
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))

# =============================================================================
CELLS = []


def md(text):
    CELLS.append({"cell_type": "markdown", "metadata": {},
                  "source": text.strip("\n").split("\n")})


def code(text):
    CELLS.append({"cell_type": "code", "execution_count": None,
                  "metadata": {}, "outputs": [],
                  "source": text.strip("\n").split("\n")})


# =============================================================================
md(r"""
# Paper asset generator — IEEE TIM submission

This notebook rebuilds **every figure and every table** of the manuscript
directly from the result files produced by your training run.

**No experimental value is hard-coded anywhere in this notebook.**
Each table is located by searching, in order:

1. `<RESULTS_DIR>/tables/<DS_TAG>/<name>.csv`  ← written by `save_table_md_csv_tex`
2. `<RESULTS_DIR>/tables/<DS_TAG>/<name>.md`
3. the matching section of `REPORT_<DS_TAG>.md`

If a table cannot be found, the dependent figure or column is **skipped with a
warning** rather than invented.

Run the cells top to bottom. Outputs land in `../figures/` and `../tables/`.
""")

# --------------------------------------------------------------- cell: config
code(r'''
# =============================================================================
# 1. CONFIGURATION  -- edit only these paths
# =============================================================================
from pathlib import Path

RESULTS_DIR_CANDIDATES = [
    r"D:\python\study_1_militery\code\DT_Q1_FINAL_results",
    r"D:\python\study_1_militery\aftercrash\DT_Q1_FINAL_results",
    "./DT_Q1_FINAL_results", "../DT_Q1_FINAL_results",
    "../results_source",
]
REPORT_DIR_CANDIDATES = [
    ".", "..", "../results_source", "./paper_pack",
    r"D:\python\study_1_militery\code\DT_Q1_FINAL_results",
    r"D:\python\study_1_militery\aftercrash",
]

FIG_OUT = Path("../latex/figures")
TAB_OUT = Path("../latex/tables")

DS_PRIMARY = "DS1_primary"     # ten-way vehicle type
DS_BINARY  = "DS2_binary"      # military vs civilian
PROPOSED_MATCH = "MS-EGCA"     # substring identifying the proposed model

# A resumed run can leave a stale duplicate of one architecture under an older
# ID. When two rows share a model name inside one corpus we keep the one whose
# numeric ID is LARGER (the freshly retrained slot) and print what was dropped.
DROP_DUPLICATE_MODELS = True

# Training settings are properties of the run, not of the result tables, so they
# cannot be recovered from the CSVs. Edit if you change the training script.
CONFIG_ROWS = [
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
    ("Significance test", "McNemar, Holm-corrected"),
    ("Latency protocol", "20 warm-up, 100 timed runs, batch 1"),
    ("Calibration bins", "15 equal-width"),
    ("Hardware", "AMD GPU via DirectML, FP32"),
]

FIG_OUT.mkdir(parents=True, exist_ok=True)
TAB_OUT.mkdir(parents=True, exist_ok=True)
print("figures ->", FIG_OUT.resolve())
print("tables  ->", TAB_OUT.resolve())
''')

# ---------------------------------------------------------------- cell: loader
md("## 2. Locate the result files and load every table")

code(r'''
import os, re, sys, math
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

WARN = []
def warn(m):
    WARN.append(m); print("  [!]", m)
def ok(m):
    print("  [OK]", m)

def _first_dir(cands):
    for c in cands:
        p = Path(c).expanduser()
        if p.is_dir():
            return p.resolve()
    return None

RESULTS_DIR = _first_dir(RESULTS_DIR_CANDIDATES)
TAB_DIR = (RESULTS_DIR / "tables") if RESULTS_DIR else None
if RESULTS_DIR: ok(f"RESULTS_DIR = {RESULTS_DIR}")
else:           warn("no RESULTS_DIR found; will rely on REPORT_*.md")

REPORT_DIRS = []
for c in REPORT_DIR_CANDIDATES + ([str(RESULTS_DIR)] if RESULTS_DIR else []):
    p = Path(c).expanduser()
    if p.is_dir() and p not in REPORT_DIRS:
        REPORT_DIRS.append(p)

def _read_md_table(text):
    """Parse the first GitHub-style markdown table into a DataFrame."""
    lines = [l for l in text.splitlines() if l.strip().startswith("|")]
    if len(lines) < 3: return None
    hdr = [c.strip() for c in lines[0].strip().strip("|").split("|")]
    rows = []
    for l in lines[2:]:
        cells = [c.strip() for c in l.strip().strip("|").split("|")]
        if len(cells) == len(hdr): rows.append(cells)
    if not rows: return None
    df = pd.DataFrame(rows, columns=hdr)
    for c in df.columns:
        conv = pd.to_numeric(df[c].str.replace(",", "", regex=False), errors="coerce")
        if conv.notna().sum() >= max(1, int(0.7 * len(df))):
            df[c] = conv
    return df

def _find_report(ds_tag):
    for d in REPORT_DIRS:
        for n in (f"REPORT_{ds_tag}.md", f"REPORT_{ds_tag.split('_')[0]}.md"):
            if (d / n).is_file(): return d / n
    return None

_SECTION = {"table3_main_results": "Main benchmark",
            "table9_extended_metrics": "Extended metrics",
            "table5_per_class": "Per-class",
            "table4_ablation": "Ablation",
            "table7_corruption": "Corruption"}

def load_table(name, ds_tag):
    """CSV -> MD -> REPORT section. Returns a DataFrame or None."""
    if TAB_DIR is not None:
        p = TAB_DIR / ds_tag / f"{name}.csv"
        if p.is_file():
            try: return pd.read_csv(p)
            except Exception as e: warn(f"{p.name} unreadable ({e})")
        p = TAB_DIR / ds_tag / f"{name}.md"
        if p.is_file():
            df = _read_md_table(p.read_text(encoding="utf-8", errors="ignore"))
            if df is not None: return df
    frag, rp = _SECTION.get(name), _find_report(ds_tag)
    if frag and rp:
        txt = rp.read_text(encoding="utf-8", errors="ignore")
        for b in re.split(r"\n##\s+", txt):
            if frag.lower() in b.split("\n", 1)[0].lower():
                df = _read_md_table(b)
                if df is not None: return df
    return None
print("loader ready")
''')

code(r'''
# ---- normalise the main benchmark and remove stale duplicate rows -----------
REN = {"Params(M)":"params","Acc(%)":"acc","Acc_std":"acc_std","F1(%)":"f1",
       "F1_std":"f1_std","Lat(ms)":"lat","ECE":"ece","MCC":"mcc","AUC":"auc",
       "Model":"model","ID":"id","Type":"type","Year":"year","Spec(%)":"spec",
       "G-Mean(%)":"gmean","Kappa":"kappa","Brier":"brier","BAcc(%)":"bacc",
       "F1w(%)":"f1w"}

def clean_main(df, ds_tag):
    if df is None: return None
    df = df.rename(columns={k:v for k,v in REN.items() if k in df.columns}).copy()
    need = {"id","model","params","acc","f1"}
    if not need.issubset(df.columns):
        warn(f"{ds_tag}: main table missing {need-set(df.columns)}"); return None
    df["idnum"] = pd.to_numeric(df["id"].astype(str).str.extract(r"(\d+)")[0],
                                errors="coerce")
    if DROP_DUPLICATE_MODELS and df["model"].duplicated().any():
        dup  = df[df.duplicated("model", keep=False)]
        keep = dup.groupby("model")["idnum"].max()
        bad  = dup[dup.apply(lambda r: r["idnum"] != keep[r["model"]], axis=1)].index
        for i in bad:
            warn(f"{ds_tag}: dropped stale duplicate {df.loc[i,'id']} "
                 f"'{df.loc[i,'model']}' (acc={df.loc[i,'acc']:.2f})")
        df = df.drop(index=bad)
    df["is_prop"] = df["model"].astype(str).str.contains(PROPOSED_MATCH, case=False)
    df["short"] = df["model"].astype(str).str.replace(r"\s*\(.*\)\s*", "", regex=True).str.strip()
    df.loc[df["is_prop"], "short"] = "MS-EGCA (proposed)"
    return df.reset_index(drop=True)

def norm_stats(df):
    if df is None: return None
    df = df.rename(columns={"vs":"vs","b":"b","c":"c","p-value":"p","dF1%":"d",
                            "CI_lo":"lo","CI_hi":"hi","Cohen_d":"cd",
                            "BL_F1%":"bl","sig":"sig"}).copy()
    if "vs" in df.columns:
        df["base"] = df["vs"].astype(str).str.replace(r"^\s*\S+\s+vs\s+","",regex=True).str.strip()
    if "sig" in df.columns:
        df["issig"] = df["sig"].astype(str).str.strip().str.lower().isin(["yes","true","1","y"])
    return df

def norm_corr(df):
    if df is None: return None
    d = df.copy()
    if not {"corruption","severity","accuracy"}.issubset(d.columns): return None
    if d["accuracy"].max() <= 1.5:
        d["accuracy"] = d["accuracy"] * 100
        if "f1" in d.columns: d["f1"] = d["f1"] * 100
    return d

print("=" * 70); print("LOADING")
M1  = clean_main(load_table("table3_main_results", DS_PRIMARY), DS_PRIMARY)
M2  = clean_main(load_table("table3_main_results", DS_BINARY),  DS_BINARY)
if M1 is None:
    raise SystemExit("FATAL: DS1 main benchmark not found. Fix RESULTS_DIR_CANDIDATES.")
ok(f"{DS_PRIMARY}: {len(M1)} architectures")
if M2 is not None: ok(f"{DS_BINARY}: {len(M2)} architectures")

EX1  = load_table("table9_extended_metrics", DS_PRIMARY)
EX2  = load_table("table9_extended_metrics", DS_BINARY)
PC1  = load_table("table5_per_class", DS_PRIMARY)
PC2  = load_table("table5_per_class", DS_BINARY)
AB1  = load_table("table4_ablation", DS_PRIMARY)
AB2  = load_table("table4_ablation", DS_BINARY)
CR1  = norm_corr(load_table("table7_corruption", DS_PRIMARY))
CR2  = norm_corr(load_table("table7_corruption", DS_BINARY))
ST1  = norm_stats(load_table("table6_stats", DS_PRIMARY))
ST2  = norm_stats(load_table("table6_stats", DS_BINARY))
DSET1 = load_table("table1_dataset", DS_PRIMARY)
DSET2 = load_table("table1_dataset", DS_BINARY)

for nm, d in [("extended DS1",EX1),("extended DS2",EX2),("per-class DS1",PC1),
              ("per-class DS2",PC2),("ablation DS1",AB1),("ablation DS2",AB2),
              ("corruption DS1",CR1),("corruption DS2",CR2),
              ("stats DS1",ST1),("stats DS2",ST2),
              ("dataset DS1",DSET1),("dataset DS2",DSET2)]:
    (ok if d is not None else warn)(f"{nm}: {len(d) if d is not None else 'NOT FOUND'}")
''')

# ------------------------------------------------------------- cell: style
md("## 3. Figure style and the label de-overlap routine")

code(r'''
plt.rcParams.update({
    "font.family":"serif",
    "font.serif":["Nimbus Roman No9 L","Times New Roman","DejaVu Serif"],
    "mathtext.fontset":"cm",
    "font.size":8,"axes.labelsize":7.8,"axes.titlesize":8.2,
    "xtick.labelsize":6.8,"ytick.labelsize":6.8,"legend.fontsize":6.5,
    "axes.linewidth":0.6,"xtick.major.width":0.5,"ytick.major.width":0.5,
    "xtick.major.size":2.2,"ytick.major.size":2.2,
    "grid.linewidth":0.35,"lines.linewidth":1.15,
    "savefig.dpi":600,"savefig.bbox":"tight","savefig.pad_inches":0.015,
    "figure.facecolor":"white",
})
COL1, COL2 = 3.45, 7.16                      # IEEE single / double column
INK, CP, GREY = "#1b2a38", "#c0392b", "#9aa8b2"
BLUE, ORANGE, TEAL, PURPLE = "#2c6fa6", "#d9822b", "#178f7a", "#8e5fa8"
PAL = [BLUE, ORANGE, TEAL, PURPLE, CP]

def style(ax, grid="both"):
    for s in ("top","right"): ax.spines[s].set_visible(False)
    for s in ("left","bottom"): ax.spines[s].set_color("#7c8b96")
    if grid in ("both","y"): ax.grid(axis="y", alpha=.30, ls=":", color=GREY)
    if grid in ("both","x"): ax.grid(axis="x", alpha=.30, ls=":", color=GREY)
    ax.set_axisbelow(True)

def save(fig, name):
    fig.savefig(FIG_OUT / (name + ".pdf"))
    fig.savefig(FIG_OUT / (name + ".png"), dpi=300)
    print("  wrote", name + ".pdf/.png")

def declutter(fig, ax, anns, pts, iters=240, pad=1.5):
    """Force-based label separation in display space."""
    try:
        fig.canvas.draw(); r = fig.canvas.get_renderer()
    except Exception:
        return
    offs = [list(a.get_position()) for a in anns]
    pxy  = [ax.transData.transform(p) for p in pts]
    ab   = ax.get_window_extent(r)
    for _ in range(iters):
        bs = [a.get_window_extent(r) for a in anns]
        sh = [[0.,0.] for _ in anns]; moved = False
        for i in range(len(anns)):
            for j in range(i+1, len(anns)):
                A,B = bs[i],bs[j]
                ox = min(A.x1,B.x1)-max(A.x0,B.x0)+pad
                oy = min(A.y1,B.y1)-max(A.y0,B.y0)+pad
                if ox>0 and oy>0:
                    moved=True
                    dx=(A.x0+A.x1)/2-(B.x0+B.x1)/2
                    dy=(A.y0+A.y1)/2-(B.y0+B.y1)/2
                    if abs(dy)*1.4>=abs(dx):
                        s=oy/2*(1 if dy>=0 else -1); sh[i][1]+=s; sh[j][1]-=s
                    else:
                        s=ox/2*(1 if dx>=0 else -1); sh[i][0]+=s; sh[j][0]-=s
        for i,bb in enumerate(bs):
            for mx,my in pxy:
                if bb.x0-2<mx<bb.x1+2 and bb.y0-2<my<bb.y1+2:
                    moved=True
                    sh[i][1] += 3.0 if (bb.y0+bb.y1)/2>=my else -3.0
        if not moved: break
        for i,a in enumerate(anns):
            offs[i][0]+=sh[i][0]*.55; offs[i][1]+=sh[i][1]*.55
            bb=bs[i]
            if bb.x0+sh[i][0]<ab.x0: offs[i][0]+=(ab.x0-bb.x0)+2
            if bb.x1+sh[i][0]>ab.x1: offs[i][0]-=(bb.x1-ab.x1)+2
            if bb.y0+sh[i][1]<ab.y0: offs[i][1]+=(ab.y0-bb.y0)+2
            if bb.y1+sh[i][1]>ab.y1: offs[i][1]-=(bb.y1-ab.y1)+2
            a.set_position(tuple(offs[i]))
        fig.canvas.draw()

def label_pts(fig, ax, xs, ys, labs, props):
    anns=[]
    for x,y,l,pr in zip(xs,ys,labs,props):
        anns.append(ax.annotate(l,(x,y),textcoords="offset points",xytext=(8,5),
            fontsize=5.6,ha="left",va="center",zorder=6,
            color=CP if pr else "#33475b", weight="bold" if pr else "normal",
            arrowprops=dict(arrowstyle="-",lw=.35,color="#b6c2cb",
                            shrinkA=0,shrinkB=3)))
    declutter(fig, ax, anns, list(zip(xs,ys)))

def short(n): return str(n).replace(" (proposed)","")
print("style ready")
''')

# ------------------------------------------------------------ cell: figure 1
md("## 4. Figure 1 — method diagram\n\nDrawn from the architecture description; "
   "the two annotated numbers (parameter count and latency) are read from the "
   "loaded benchmark, not typed.")

code(r'''
def fig1_method():
    pr = M1[M1["is_prop"]].iloc[0]
    fig = plt.figure(figsize=(COL2, 3.5))
    gs  = fig.add_gridspec(2,1,height_ratios=[1.0,1.35],hspace=.14)

    ax = fig.add_subplot(gs[0]); ax.set_xlim(0,100); ax.set_ylim(0,30); ax.axis("off")
    def box(x,y,w,h,lab,fc,sub=None,fs=7.2,ec=INK,lw=.75):
        ax.add_patch(FancyBboxPatch((x+.5,y-.5),w,h,boxstyle="round,pad=0.32",
                                    fc="#00000010",ec="none",zorder=1))
        ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle="round,pad=0.32",
                                    fc=fc,ec=ec,lw=lw,zorder=2))
        ax.text(x+w/2,y+h/2+(1.5 if sub else 0),lab,ha="center",va="center",
                fontsize=fs,weight="bold",color=INK,zorder=3)
        if sub: ax.text(x+w/2,y+h/2-2.4,sub,ha="center",va="center",
                        fontsize=5.8,style="italic",color="#41576b",zorder=3)
    def arr(x1,y1,x2,y2,ls="-",c=INK,lw=.9):
        ax.add_patch(FancyArrowPatch((x1,y1),(x2,y2),arrowstyle="-|>",
                     mutation_scale=7.5,lw=lw,ls=ls,color=c,shrinkA=0,shrinkB=0,zorder=4))
    y0,h0 = 11,9.5
    box(0.5,y0,11.5,h0,"Input","#eef3f7",r"224$\times$224$\times$3")
    arr(12.0,y0+h0/2,15.4,y0+h0/2)
    box(15.4,y0,14.6,h0,"Sobel Gate","#fdefdd",r"edge prior $\mathcal{S}$")
    arr(30.0,y0+h0/2,33.4,y0+h0/2)
    box(33.4,y0,17.6,h0,"MobileNetV3-S","#e3eff9",r"576$\times$7$\times$7")
    arr(51.0,y0+h0/2,54.4,y0+h0/2)
    box(54.4,y0,15.6,h0,"MS-EGCA","#fadedd","attention (ours)",ec=CP,lw=1.3)
    arr(70.0,y0+h0/2,73.4,y0+h0/2)
    box(73.4,y0,10.2,h0,"GAP","#eef3f7","576-d")
    arr(83.6,y0+h0/2,86.8,y0+h0/2)
    box(86.8,y0,12.7,h0,"Head","#eef3f7",r"FC $\rightarrow K$")
    box(45.0,1.6,27.0,6.4,"DeiT-Tiny + cross-attention","#f3f6f7",fs=6.5,ec=GREY,lw=.7)
    ax.text(58.5,0.0,"optional branch, disabled in this study",ha="center",
            va="center",fontsize=5.7,style="italic",color=GREY)
    arr(21.0,y0,45.0,5.6,ls=(0,(2.2,2.2)),c=GREY,lw=.7)
    arr(72.0,5.6,79.0,y0,ls=(0,(2.2,2.2)),c=GREY,lw=.7)
    lat = f", {pr['lat']:.2f} ms/image" if "lat" in M1.columns and pd.notna(pr.get("lat")) else ""
    ax.text(1,27.2,f"(a)  Measurement pipeline  ({pr['params']:.2f} M parameters{lat})",
            fontsize=7.6,weight="bold",color=INK,ha="left")

    ax = fig.add_subplot(gs[1]); ax.set_xlim(0,100); ax.set_ylim(0,72); ax.axis("off")
    def b2(x,y,w,h,lab,fc,fs=6.2,ec=INK):
        ax.add_patch(FancyBboxPatch((x+.4,y-.4),w,h,boxstyle="round,pad=0.28",
                                    fc="#0000000e",ec="none",zorder=1))
        ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle="round,pad=0.28",
                                    fc=fc,ec=ec,lw=.65,zorder=2))
        ax.text(x+w/2,y+h/2,lab,ha="center",va="center",fontsize=fs,color=INK,zorder=3)
    def a2(x1,y1,x2,y2,c=INK):
        ax.add_patch(FancyArrowPatch((x1,y1),(x2,y2),arrowstyle="-|>",
                     mutation_scale=6,lw=.7,color=c,shrinkA=0,shrinkB=0,zorder=4))
    ax.text(1,69,"(b)  MS-EGCA module",fontsize=7.6,weight="bold",color=INK,ha="left")
    b2(0.5,27,12.5,12,r"$\mathbf{X}$"+"\n"+r"$C\!\times\!H\!\times\!W$","#eef3f7")
    ax.text(41,63.5,"multi-scale edge branch",ha="center",fontsize=6.0,
            style="italic",color="#a9691f")
    a2(13,36,20,55); b2(20,49,17,11,"channel mean\n"+r"$\mathbf{g}$","#fdefdd")
    a2(37,54.5,43,54.5)
    b2(43,46,25,17,r"$3{\times}3,\ 5{\times}5,\ 7{\times}7$"+"\n"+r"$1{\times}1$ fuse"
       +"\n"+r"GN, $\sigma \rightarrow \mathbf{E}$","#fdefdd",6.0)
    ax.text(41,42.5,"ECA-style channel branch",ha="center",fontsize=6.0,
            style="italic",color="#1f618d")
    a2(13,33,26,33)
    b2(26,26,33,13,r"GAP $\rightarrow$ Conv1D$_{k_c}$"+"\n"
       +r"$\rightarrow \sigma \rightarrow \mathbf{a}$","#e3eff9",6.0)
    ax.text(41,1.0,"adaptive residual gate",ha="center",fontsize=6.0,
            style="italic",color="#a93226")
    a2(13,30,24,15)
    b2(24,6.5,31,13,r"GAP $\rightarrow$ MLP $\rightarrow \sigma$"+"\n"
       +r"$\rightarrow \alpha\in(0,1)$","#fadedd",6.0)
    b2(74,24,25,19,r"$(1\!-\!\alpha)\mathbf{X}$"+"\n"
       +r"$+\ \alpha\,(\mathbf{X}\!\odot\!\mathbf{E}\!\odot\!\mathbf{a})$",
       "#e4f5ef",6.2,ec=CP)
    a2(68,54,86,43); a2(59,33,74,33.5); a2(55,13,86,24)
    save(fig,"fig1_method")

fig1_method()
''')

# ------------------------------------------------------------ cell: figure 2/4
md("## 5. Figures 2 and 4 — cost and efficiency, one per corpus")

code(r'''
def pareto_panel(ax, fig, M, xcol, xlabel, title):
    xs, ys = M[xcol].tolist(), M["acc"].tolist()
    pts = sorted(zip(xs, ys)); fx, fy, best = [], [], -1e9
    for px, py in pts:
        if py > best: best = py; fx.append(px); fy.append(py)
    xmax = max(xs)*1.30; fx.append(xmax); fy.append(fy[-1])
    lo = min(ys)-.55
    ax.set_xlim(0,xmax); ax.set_ylim(lo, max(ys)+1.05)
    ax.step(fx,fy,where="post",color=GREY,lw=.85,ls="--",zorder=1)
    ax.fill_between(fx,lo,fy,step="post",color=GREY,alpha=.07,zorder=0)
    for _,r in M.iterrows():
        pr = bool(r["is_prop"])
        ax.scatter(r[xcol],r["acc"],s=115 if pr else 26,marker="*" if pr else "o",
                   c=CP if pr else "white",edgecolors=CP if pr else BLUE,
                   linewidths=1.0 if pr else .85,zorder=5 if pr else 3)
    label_pts(fig,ax,xs,ys,[short(v) for v in M["short"]],M["is_prop"].tolist())
    ax.set_xlabel(xlabel); ax.set_title(title,fontsize=8.0,pad=3); style(ax)

def eff_panel(ax, M, title):
    e = M.assign(eff=M["f1"]/M["params"]).sort_values("eff")
    cols = [CP if p else "#a9bcc9" for p in e["is_prop"]]
    y = np.arange(len(e))
    ax.barh(y, e["eff"], .64, color=cols, ec=INK, lw=.35)
    ax.set_yticks(y); ax.set_yticklabels([short(v) for v in e["short"]], fontsize=5.9)
    for i,v in enumerate(e["eff"]):
        ax.text(v+e["eff"].max()*.015, y[i], f"{v:.0f}", va="center", fontsize=5.6,
                color=CP if cols[i]==CP else "#41576b")
    ax.set_xlim(0, e["eff"].max()*1.16)
    ax.set_xlabel("Macro-F1 per million parameters")
    ax.set_title(title, fontsize=8.0, pad=3); style(ax,"x")

def fps_panel(ax, M, title):
    if "lat" not in M.columns or M["lat"].isna().all():
        ax.axis("off"); warn("no latency column -> throughput panel skipped"); return
    L = M.sort_values("lat", ascending=False)
    vals = (1000.0/L["lat"]).tolist()
    cols = [CP if p else "#a9bcc9" for p in L["is_prop"]]
    y = np.arange(len(L))
    ax.barh(y, vals, .64, color=cols, ec=INK, lw=.35)
    ax.set_yticks(y); ax.set_yticklabels([short(v) for v in L["short"]], fontsize=5.9)
    for i,v in enumerate(vals):
        ax.text(v+max(vals)*.015, y[i], f"{v:.0f}", va="center", fontsize=5.6,
                color=CP if cols[i]==CP else "#41576b")
    ax.axvline(250, color=TEAL, lw=.8, ls="--")
    ax.text(max(vals)*.02+250, len(L)-0.6, "250 FPS", fontsize=5.6, color=TEAL,
            style="italic", va="top")
    ax.set_xlim(0, max(vals)*1.18)
    ax.set_xlabel("Throughput (frames per second)")
    ax.set_title(title, fontsize=8.0, pad=3); style(ax,"x")

def corr_panel(ax, C, title):
    if C is None: ax.axis("off"); return
    marks = ["o","s","^","D","v","P","X"]
    for i,(name,g) in enumerate(C.groupby("corruption", sort=False)):
        g = g.sort_values("severity")
        ax.plot(g["severity"], g["accuracy"], marker=marks[i%len(marks)], ms=3.2,
                color=PAL[i%len(PAL)], label=str(name).title(), mfc="white", mew=.9)
    ax.set_xticks(sorted(C["severity"].unique()))
    ax.set_xlabel("Corruption severity"); ax.set_ylabel("Accuracy (%)")
    ax.set_ylim(max(0, C["accuracy"].min()-8), 103)
    ax.set_title(title, fontsize=8.0, pad=3)
    ax.legend(frameon=False, fontsize=5.9, loc="lower left", ncol=2,
              handlelength=1.2, columnspacing=.8)
    style(ax)

def cost_figure(M, C, name, corpus):
    fig = plt.figure(figsize=(COL2, 4.3))
    gs  = fig.add_gridspec(2,2,hspace=.44,wspace=.26)
    ax = fig.add_subplot(gs[0,0])
    pareto_panel(ax,fig,M,"params","Parameters (M)","(a) Accuracy vs. model size")
    ax.set_ylabel("Sealed-test accuracy (%)")
    ax = fig.add_subplot(gs[0,1])
    if "lat" in M.columns and M["lat"].notna().any():
        pareto_panel(ax,fig,M,"lat","Latency per image (ms)","(b) Accuracy vs. latency")
        ax.axvspan(0,4.0,color=TEAL,alpha=.06,zorder=0)
    else: ax.axis("off")
    ax = fig.add_subplot(gs[1,0]); eff_panel(ax,M,"(c) Parameter efficiency")
    ax = fig.add_subplot(gs[1,1])
    if C is not None: corr_panel(ax,C,"(d) Corruption response")
    else:             fps_panel(ax,M,"(d) Measured throughput")
    save(fig,name)

cost_figure(M1, None, "fig2_ds1_cost", "DS1")
if M2 is not None:
    cost_figure(M2, CR2, "fig4_ds2", "DS2")
''')

# ------------------------------------------------------------ cell: figure 3
md("## 6. Figure 3 — DS1 behaviour: per-class, error composition, calibration, corruption")

code(r'''
def fig3_behaviour():
    fig = plt.figure(figsize=(COL2, 4.1))
    gs  = fig.add_gridspec(2,2,hspace=.46,wspace=.30)

    ax = fig.add_subplot(gs[0,0])
    if PC1 is not None and {"Class","Precision","Recall"}.issubset(PC1.columns):
        d = PC1.iloc[::-1].reset_index(drop=True)
        lab = [re.sub(r"\s+"," ",str(s)).strip() for s in d["Class"]]
        lab = [s if len(s)<=26 else s[:24]+"." for s in lab]
        y = np.arange(len(d)); h=.38
        ax.barh(y+h/2, d["Precision"]*100, h, color=BLUE, ec=INK, lw=.32, label="Precision")
        ax.barh(y-h/2, d["Recall"]*100,    h, color=CP,   ec=INK, lw=.32, label="Recall")
        ax.set_yticks(y); ax.set_yticklabels(lab, fontsize=5.8)
        v = min(d["Precision"].min(), d["Recall"].min())*100
        ax.set_xlim(max(0,v-2), 100.6); ax.set_xlabel("%")
        ax.legend(frameon=False, fontsize=6.0, loc="lower left", handlelength=1.0)
        ax.set_title("(a) Per-class precision and recall", fontsize=8.0, pad=3)
        style(ax,"x")
    else: ax.axis("off"); warn("per-class DS1 missing -> panel (a) skipped")

    ax = fig.add_subplot(gs[0,1])
    if PC1 is not None and {"FP","FN","Class"}.issubset(PC1.columns):
        d = PC1.assign(tot=PC1["FP"]+PC1["FN"]).sort_values("tot")
        y = np.arange(len(d))
        ax.barh(y-.19, d["FN"], .38, color=ORANGE, ec=INK, lw=.32, label="False negatives")
        ax.barh(y+.19, d["FP"], .38, color=PURPLE, ec=INK, lw=.32, label="False positives")
        ax.set_yticks(y); ax.set_yticklabels(d["Class"], fontsize=5.8)
        ax.set_xlabel("Misclassified images")
        ax.legend(frameon=False, fontsize=6.0, loc="lower right", handlelength=1.0)
        ax.set_title("(b) Error composition", fontsize=8.0, pad=3)
        style(ax,"x")
    else: ax.axis("off")

    ax = fig.add_subplot(gs[1,0])
    if "ece" in M1.columns and M1["ece"].notna().any():
        e = M1.sort_values("ece")
        cols = [CP if p else "#a9bcc9" for p in e["is_prop"]]
        y = np.arange(len(e))
        ax.barh(y, e["ece"], .64, color=cols, ec=INK, lw=.35)
        ax.set_yticks(y); ax.set_yticklabels([short(v) for v in e["short"]], fontsize=5.9)
        for i,v in enumerate(e["ece"]):
            ax.text(v+e["ece"].max()*.02, y[i], f"{v:.3f}", va="center", fontsize=5.5,
                    color=CP if cols[i]==CP else "#41576b")
        ax.set_xlim(0, e["ece"].max()*1.24)
        ax.set_xlabel("Expected calibration error")
        ax.set_title("(c) Calibration ranking", fontsize=8.0, pad=3); style(ax,"x")
    else: ax.axis("off"); warn("no ECE column -> panel (c) skipped")

    ax = fig.add_subplot(gs[1,1])
    corr_panel(ax, CR1, "(d) Corruption response")
    save(fig,"fig3_ds1_behaviour")

fig3_behaviour()
''')

# ------------------------------------------------------------ cell: figure 5
md("## 7. Figure 5 — cross-corpus synthesis")

code(r'''
def fig5_cross():
    if M2 is None:
        warn("DS2 missing -> figure 5 skipped"); return
    D1 = {r["short"]: r for _,r in M1.iterrows()}
    D2 = {r["short"]: r for _,r in M2.iterrows()}
    common = [n for n in D1 if n in D2]
    fig = plt.figure(figsize=(COL2, 4.3))
    gs  = fig.add_gridspec(2,2,hspace=.46,wspace=.30)

    ax = fig.add_subplot(gs[0,0])
    xs=[D1[n]["acc"] for n in common]; ys=[D2[n]["acc"] for n in common]
    for n,x,y in zip(common,xs,ys):
        pr = bool(D1[n]["is_prop"])
        ax.scatter(x,y,s=115 if pr else 26,marker="*" if pr else "o",
                   c=CP if pr else "white",edgecolors=CP if pr else BLUE,
                   linewidths=1.0 if pr else .85,zorder=5 if pr else 3)
    lim=[min(xs+ys)-.4, 100.05]
    ax.plot(lim,lim,ls="--",lw=.8,color=GREY,zorder=1)
    ax.text(lim[1]-.15,lim[1]-.30,"equal accuracy",fontsize=5.6,color=GREY,
            style="italic",rotation=45,ha="right",va="top",rotation_mode="anchor")
    # every model is labelled, using the identifiers of the benchmark tables
    label_pts(fig,ax,xs,ys,[str(D1[n]["id"]) for n in common],
              [bool(D1[n]["is_prop"]) for n in common])
    ax.set_xlim(*lim); ax.set_ylim(*lim)
    ax.set_xlabel("DS1 accuracy (%)"); ax.set_ylabel("DS2 accuracy (%)")
    ax.set_title("(a) Accuracy parity across corpora",fontsize=8.0,pad=3); style(ax)

    ax = fig.add_subplot(gs[0,1])
    if CR1 is not None and CR2 is not None:
        a=CR1.set_index(["corruption","severity"])["accuracy"]
        b=CR2.set_index(["corruption","severity"])["accuracy"]
        names=list(dict.fromkeys(CR1["corruption"]))
        sev=sorted(set(CR1["severity"]))[-2:]
        x=np.arange(len(names)); w=.38
        for k,(s,c) in enumerate(zip(sev,[BLUE,ORANGE])):
            gap=[b.get((n,s),np.nan)-a.get((n,s),np.nan) for n in names]
            ax.bar(x+(k-.5)*w,gap,w,color=c,ec=INK,lw=.35,label=f"severity {s}")
        ax.set_xticks(x); ax.set_xticklabels([str(n).title() for n in names],
                                             fontsize=6.2,rotation=20,ha="right")
        ax.set_ylabel("DS2 $-$ DS1 accuracy (pp)")
        ax.legend(frameon=False,fontsize=6.0,loc="upper left",handlelength=1.1)
        ax.set_title("(b) Robustness gap between tasks",fontsize=8.0,pad=3)
        style(ax,"y")
    else: ax.axis("off")

    ax = fig.add_subplot(gs[1,0])
    e1 = M1.assign(eff=M1["f1"]/M1["params"])
    e2 = M2.assign(eff=M2["f1"]/M2["params"])
    m1 = {r["short"]: r["eff"] for _,r in e1.iterrows()}
    m2 = {r["short"]: r["eff"] for _,r in e2.iterrows()}
    names=sorted(common, key=lambda n: m1[n])
    y=np.arange(len(names))
    ax.barh(y-.19,[m1[n] for n in names],.38,color=BLUE,ec=INK,lw=.32,label="DS1")
    ax.barh(y+.19,[m2[n] for n in names],.38,color=ORANGE,ec=INK,lw=.32,label="DS2")
    ax.set_yticks(y); ax.set_yticklabels([short(n) for n in names],fontsize=5.8)
    ax.set_xlabel("Macro-F1 per million parameters")
    ax.legend(frameon=False,fontsize=6.0,loc="lower right",handlelength=1.1)
    ax.set_title("(c) Parameter efficiency, both corpora",fontsize=8.0,pad=3)
    style(ax,"x")

    ax = fig.add_subplot(gs[1,1])
    if ST1 is not None or ST2 is not None:
        s1 = {r["base"]: r for _,r in ST1.iterrows()} if ST1 is not None else {}
        s2 = {r["base"]: r for _,r in ST2.iterrows()} if ST2 is not None else {}
        order=[n for n in (s2 or s1)][::-1]
        y=np.arange(len(order))
        for S,c,off,lab in [(s1,BLUE,-.17,"DS1"),(s2,ORANGE,.17,"DS2")]:
            first=True
            for i,n in enumerate(order):
                if n not in S: continue
                r=S[n]; sig=bool(r.get("issig",False))
                ax.plot([r["lo"],r["hi"]],[y[i]+off]*2,color=c,lw=1.0,
                        solid_capstyle="round",zorder=2)
                ax.scatter(r["d"],y[i]+off,s=16 if sig else 12,
                           c=c if sig else "white",edgecolors=c,linewidths=.8,
                           marker="s" if sig else "o",zorder=3,
                           label=lab if first else None)
                first=False
        ax.axvline(0,color=INK,lw=.75,ls="--",zorder=1)
        ax.set_yticks(y); ax.set_yticklabels(order,fontsize=5.8)
        ax.set_xlabel(r"$\Delta$F1 (proposed $-$ baseline, pp)")
        ax.legend(frameon=False,fontsize=6.0,loc="lower right",handlelength=1.1)
        ax.set_title("(d) Paired differences, both corpora",fontsize=8.0,pad=3)
        style(ax,"x")
    else:
        ax.axis("off"); warn("stats tables missing -> panel (d) skipped")
    save(fig,"fig5_cross")

fig5_cross()
''')

# ------------------------------------------------------------ cell: tables
md("## 8. LaTeX tables\n\nAll tables are written from the same loaded frames.")

code(r'''
def esc(s):
    return (str(s).replace("&",r"\&").replace("_",r"\_")
            .replace("%",r"\%").replace("#",r"\#"))
def bf(s): return r"\textbf{"+str(s)+"}"
def shade(p): return r"\rowcolor{propshade} " if p else ""
def wtab(name, body):
    (TAB_OUT/name).write_text(body, encoding="utf-8"); print("  wrote", name)

def t_primary(M, label, cap, fname):
    cols=[c for c in ["mcc","auc","kappa"] if c in M.columns]
    best={"acc":M["acc"].max(),"f1":M["f1"].max()}
    for c in cols: best[c]=M[c].max()
    hdr=["ID","Architecture"]
    if "year" in M.columns: hdr.append("Year")
    if "type" in M.columns: hdr.append("Family")
    hdr += [r"Accuracy (\%)", r"Macro-F1 (\%)"] + \
           [{"mcc":"MCC","auc":"AUC","kappa":r"$\kappa$"}[c] for c in cols]
    spec="@{}ll"+("c" if "year" in M.columns else "")+("c" if "type" in M.columns else "")+"rr"+"r"*len(cols)+"@{}"
    r=[r"\begin{table*}[!t]",r"\centering",r"\caption{"+cap+"}",r"\label{"+label+"}",
       r"\setlength{\tabcolsep}{4pt}",r"\footnotesize",
       r"\begin{tabular}{"+spec+"}",r"\toprule"," & ".join(hdr)+r" \\",r"\midrule"]
    for _,x in M.iterrows():
        pr=bool(x["is_prop"])
        cells=[str(x["id"]), bf(esc(x["short"])) if pr else esc(x["short"])]
        if "year" in M.columns: cells.append(str(int(x["year"])))
        if "type" in M.columns: cells.append(str(x["type"]))
        a=f"{x['acc']:.2f}"
        if abs(x["acc"]-best["acc"])<1e-9: a=bf(a)
        if "acc_std" in M.columns and pd.notna(x.get("acc_std")):
            a+=f"\\,{{\\tiny$\\pm${x['acc_std']:.2f}}}"
        f=f"{x['f1']:.2f}"
        if abs(x["f1"]-best["f1"])<1e-9: f=bf(f)
        if "f1_std" in M.columns and pd.notna(x.get("f1_std")):
            f+=f"\\,{{\\tiny$\\pm${x['f1_std']:.2f}}}"
        cells += [a,f]
        for c in cols:
            v=f"{x[c]:.4f}"
            if abs(x[c]-best[c])<1e-9: v=bf(v)
            cells.append(v)
        r.append(shade(pr)+" & ".join(cells)+r" \\")
    r+=[r"\bottomrule",r"\end{tabular}",r"\\[2pt]",
        r"\footnotesize Mean $\pm$ standard deviation over the multi-seed runs. "
        r"Parameter count and latency are reported graphically to avoid duplication.",
        r"\end{table*}"]
    wtab(fname,"\n".join(r))

def t_extended(M, EX, label, cap, fname):
    if EX is None: warn(f"{fname}: extended metrics missing -> skipped"); return
    E=EX.rename(columns={"Specificity(%)":"spec","NPV(%)":"npv","FPR(%)":"fpr",
                         "FNR(%)":"fnr","G-Mean(%)":"gmean","LR+":"lrp",
                         "DOR":"dor","Kappa":"kappa","Brier":"brier",
                         "Model":"model","ID":"id"}).copy()
    keep=[c for c in ["spec","npv","fpr","fnr","gmean","lrp","dor","brier"] if c in E.columns]
    for c in keep:                       # "nan" may arrive as a string
        E[c] = pd.to_numeric(E[c], errors="coerce")
    keep=[c for c in keep if E[c].notna().any()]
    ece = "ece" in M.columns and M["ece"].notna().any()
    lab={"spec":r"Spec.\ (\%)","npv":r"NPV (\%)","fpr":r"FPR (\%)","fnr":r"FNR (\%)",
         "gmean":r"G-Mean (\%)","lrp":r"LR$^{+}$","dor":"DOR","brier":"Brier"}
    hdr=["ID","Architecture"]+[lab[c] for c in keep]+([ "ECE"] if ece else [])
    r=[r"\begin{table*}[!t]",r"\centering",r"\caption{"+cap+"}",r"\label{"+label+"}",
       r"\setlength{\tabcolsep}{4pt}",r"\footnotesize",
       r"\begin{tabular}{@{}ll"+"r"*(len(keep)+(1 if ece else 0))+"@{}}",
       r"\toprule"," & ".join(hdr)+r" \\",r"\midrule"]
    emap={str(a).strip(): b for a,b in zip(E.get("model",E.get("Model")), E.index)}
    for _,x in M.iterrows():
        key=[k for k in emap if k.startswith(str(x["model"])[:18])]
        if not key: continue
        e=E.loc[emap[key[0]]]
        pr=bool(x["is_prop"])
        cells=[str(x["id"]), bf(esc(x["short"])) if pr else esc(x["short"])]
        for c in keep:
            v=e[c]
            cells.append(r"$>10^{4}$" if c=="dor" and v>=9999 else
                         (f"{v:.1f}" if c=="lrp" else f"{v:.4f}" if c=="brier" else f"{v:.2f}"))
        if ece: cells.append(f"{x['ece']:.4f}")
        r.append(shade(pr)+" & ".join(cells)+r" \\")
    r+=[r"\bottomrule",r"\end{tabular}",r"\\[2pt]",
        r"\footnotesize Macro-averaged one-versus-rest on the sealed test set. "
        r"DOR is capped by the pipeline at $10^{4}$.",r"\end{table*}"]
    wtab(fname,"\n".join(r))

def t_perclass(PC, label, cap, fname):
    if PC is None: warn(f"{fname}: per-class missing -> skipped"); return
    cols=[c for c in ["Class","TP","FP","FN","Precision","Recall","Specificity","G-Mean"]
          if c in PC.columns]
    r=[r"\begin{table}[!t]",r"\centering",r"\caption{"+cap+"}",r"\label{"+label+"}",
       r"\setlength{\tabcolsep}{2.6pt}",r"\footnotesize",
       r"\begin{tabular}{@{}l"+"r"*(len(cols)-1)+"@{}}",r"\toprule",
       " & ".join(esc(c) for c in cols)+r" \\",r"\midrule"]
    for _,x in PC.iterrows():
        cells=[]
        for c in cols:
            v=x[c]
            cells.append(esc(v) if c=="Class" else
                         (f"{int(v)}" if c in ("TP","FP","FN") else f"{v:.4f}"))
        r.append(" & ".join(cells)+r" \\")
    mrow=[r"\textit{Macro average}"]
    for c in cols[1:]:
        v=PC[c].sum() if c in ("TP","FP","FN") else PC[c].mean()
        mrow.append(f"{int(v)}" if c in ("TP","FP","FN") else f"{v:.4f}")
    r+=[r"\midrule"," & ".join(mrow)+r" \\",r"\bottomrule",r"\end{tabular}",r"\end{table}"]
    wtab(fname,"\n".join(r))

def t_ablation(fname):
    A1 = AB1.rename(columns={"F1(%)":"f1","ID":"id","Config":"cfg"}) if AB1 is not None else None
    A2 = AB2.rename(columns={"F1(%)":"f1","ID":"id","Config":"cfg"}) if AB2 is not None else None
    if A1 is None and A2 is None: warn("ablation missing -> skipped"); return
    base = A1 if A1 is not None else A2
    b1 = A1["f1"].iloc[0] if A1 is not None else None
    b2 = A2["f1"].iloc[0] if A2 is not None else None
    two = A1 is not None and A2 is not None
    m2 = {r["id"]: r for _,r in A2.iterrows()} if two else {}
    r=[r"\begin{table}[!t]",r"\centering",
       r"\caption{Component ablation. All variants share the backbone and the "
       r"training recipe; $\Delta$ is measured against A0.}",
       r"\label{tab:ablation}",r"\setlength{\tabcolsep}{2.6pt}",r"\footnotesize",
       r"\begin{tabular}{@{}ll|rr"+("|rr" if two else "")+r"@{}}",r"\toprule"]
    if two:
        r+=[r"& & \multicolumn{2}{c|}{DS1} & \multicolumn{2}{c}{DS2} \\",
            r"\cmidrule(lr){3-4}\cmidrule(lr){5-6}"]
    r+=[r"ID & Configuration & F1 (\%) & $\Delta$"+(r" & F1 (\%) & $\Delta$" if two else "")+r" \\",
        r"\midrule"]
    for _,x in base.iterrows():
        pr = str(x["id"])=="A8"
        cells=[str(x["id"]), bf(esc(x["cfg"])) if pr else esc(x["cfg"])]
        if A1 is not None:
            cells += [f"{x['f1']:.2f}", f"{x['f1']-b1:+.2f}"]
        if two:
            q=m2.get(x["id"])
            cells += [f"{q['f1']:.2f}", f"{q['f1']-b2:+.2f}"] if q is not None else ["--","--"]
        r.append(shade(pr)+" & ".join(cells)+r" \\")
    r+=[r"\bottomrule",r"\end{tabular}",r"\end{table}"]
    wtab(fname,"\n".join(r))

def t_stats(fname):
    if ST1 is None and ST2 is None: warn("stats missing -> skipped"); return
    two = ST1 is not None and ST2 is not None
    def cell(x):
        sig=bool(x.get("issig",False))
        d=f"{x['d']:+.2f}"
        return [f"{int(x['b'])}/{int(x['c'])}", bf(d) if sig else d,
                f"[{x['lo']:+.2f},\\,{x['hi']:+.2f}]",
                str(x["p"]).replace("<",r"$<$")+(r"\,\ding{51}" if sig else "")]
    src = ST1 if ST1 is not None else ST2
    m2  = {r["base"]: r for _,r in ST2.iterrows()} if two else {}
    r=[r"\begin{table*}[!t]",r"\centering",
       r"\caption{Paired comparison of the proposed model against every baseline "
       r"on the pooled sealed-test predictions. $b$ and $c$ are the McNemar "
       r"discordant counts. $\Delta$F1 is proposed minus baseline in percentage "
       r"points, with a 95\,\% bootstrap confidence interval.}",
       r"\label{tab:stats}",r"\setlength{\tabcolsep}{4pt}",r"\footnotesize",
       r"\begin{tabular}{@{}l|ccrl"+("|ccrl" if two else "")+r"@{}}",r"\toprule"]
    if two:
        r+=[r"& \multicolumn{4}{c|}{\textbf{DS1}} & \multicolumn{4}{c}{\textbf{DS2}} \\",
            r"\cmidrule(lr){2-5}\cmidrule(lr){6-9}"]
    h=r"$b$/$c$ & $\Delta$F1 & 95\,\% CI & $p$"
    r+=["Baseline & "+h+(" & "+h if two else "")+r" \\",r"\midrule"]
    for _,x in src.iterrows():
        row=[esc(x["base"])]+cell(x)
        if two:
            q=m2.get(x["base"]); row += cell(q) if q is not None else ["--"]*4
        r.append(" & ".join(row)+r" \\")
    if two:
        for n,q in m2.items():
            if n not in set(src["base"]):
                r.append(" & ".join([esc(n)]+["--"]*4+cell(q))+r" \\")
    r+=[r"\bottomrule",r"\end{tabular}",r"\\[2pt]",
        r"\footnotesize \ding{51} marks significance at $\alpha=0.05$ after Holm "
        r"correction. Positive $\Delta$F1 favours the proposed model.",r"\end{table*}"]
    wtab(fname,"\n".join(r))

def t_corruption(fname):
    if CR1 is None: warn("corruption missing -> skipped"); return
    sev=sorted(CR1["severity"].unique())
    two=CR2 is not None
    a=CR1.set_index(["corruption","severity"])["accuracy"]
    b=CR2.set_index(["corruption","severity"])["accuracy"] if two else None
    names=list(dict.fromkeys(CR1["corruption"]))
    r=[r"\begin{table}[!t]",r"\centering",
       r"\caption{Accuracy (\%) of the proposed model under synthetic field "
       r"corruptions.}",r"\label{tab:corruption}",
       r"\setlength{\tabcolsep}{3.4pt}",r"\footnotesize",
       r"\begin{tabular}{@{}l|"+"r"*len(sev)+("|"+"r"*len(sev) if two else "")+r"@{}}",
       r"\toprule"]
    if two:
        r+=[r"& \multicolumn{%d}{c|}{DS1} & \multicolumn{%d}{c}{DS2} \\"%(len(sev),len(sev)),
            r"\cmidrule(lr){2-%d}\cmidrule(lr){%d-%d}"%(1+len(sev),2+len(sev),1+2*len(sev))]
    r+=["Corruption & "+" & ".join(f"$s{{=}}{s}$" for s in sev)+
        ((" & "+" & ".join(f"$s{{=}}{s}$" for s in sev)) if two else "")+r" \\",r"\midrule"]
    for n in names:
        cells=[f"{a.get((n,s),float('nan')):.2f}" for s in sev]
        if two: cells+=[f"{b.get((n,s),float('nan')):.2f}" for s in sev]
        r.append(str(n).title()+" & "+" & ".join(cells)+r" \\")
    m1=[CR1[CR1["severity"]==s]["accuracy"].mean() for s in sev]
    cells=[f"{v:.2f}" for v in m1]
    if two: cells+=[f"{CR2[CR2['severity']==s]['accuracy'].mean():.2f}" for s in sev]
    r+=[r"\midrule",r"\textit{Mean} & "+" & ".join(cells)+r" \\",
        r"\bottomrule",r"\end{tabular}",r"\end{table}"]
    wtab(fname,"\n".join(r))

def t_config(fname):
    r=[r"\begin{table}[!t]",r"\centering",
       r"\caption{Training and evaluation configuration, applied identically to "
       r"every architecture on both corpora.}",r"\label{tab:config}",
       r"\setlength{\tabcolsep}{3pt}",r"\footnotesize",
       r"\begin{tabular}{@{}l p{0.56\columnwidth}@{}}",r"\toprule",
       r"Setting & Value \\",r"\midrule"]
    for k,v in CONFIG_ROWS: r.append(f"{k} & {v} \\\\")
    r+=[r"\bottomrule",r"\end{tabular}",r"\end{table}"]
    wtab(fname,"\n".join(r))

def t_corpora(fname):
    if DSET1 is None and DSET2 is None:
        warn("dataset tables missing -> corpora table skipped"); return
    r=[r"\begin{table}[!t]",r"\centering",
       r"\caption{Composition of the evaluation corpora. Both are exactly class "
       r"balanced.}",r"\label{tab:corpora}",r"\setlength{\tabcolsep}{4pt}",
       r"\footnotesize",r"\begin{tabular}{@{}llrrr@{}}",r"\toprule",
       r"Corpus & Partition & Images & Classes & Per class \\",r"\midrule"]
    for tag,D in [("DS1",DSET1),("DS2",DSET2)]:
        if D is None: continue
        for i,(_,x) in enumerate(D.iterrows()):
            lead=r"\multirow{%d}{*}{%s}"%(len(D),tag) if i==0 else ""
            r.append(f"{lead} & {x['Split']} & {int(x['Total']):,} & "
                     f"{int(x['Classes'])} & {int(x['Min']):,} \\\\")
        r.append(r"\midrule")
    r=r[:-1]
    r+=[r"\bottomrule",r"\end{tabular}",r"\end{table}"]
    wtab(fname,"\n".join(r))

print("="*70); print("WRITING TABLES")
t_corpora("t1_corpora.tex")
t_config("t2_config.tex")
t_primary(M1,"tab:ds1main","Primary sealed-test benchmark on DS1, the ten-way "
          "vehicle-type corpus. Best value per column in bold; the proposed "
          "model is shaded.","t3_ds1_primary.tex")
t_extended(M1,EX1,"tab:ds1ext","Extended discriminative and probabilistic "
           "metrics on DS1.","t4_ds1_extended.tex")
t_perclass(PC1,"tab:pc1","Per-class sealed-test performance on DS1.","t5_ds1_perclass.tex")
if M2 is not None:
    t_primary(M2,"tab:ds2main","Primary sealed-test benchmark on DS2, the "
              "military versus civilian corpus.","t6_ds2_primary.tex")
    t_extended(M2,EX2,"tab:ds2ext","Extended discriminative metrics on DS2.",
               "t7_ds2_extended.tex")
    t_perclass(PC2,"tab:pc2","Per-class sealed-test performance on DS2.","t8_ds2_perclass.tex")
t_ablation("t9_ablation.tex")
t_stats("t10_stats.tex")
t_corruption("t11_corruption.tex")
''')

# ------------------------------------------------------------ cell: summary
md("## 9. Verification summary")

code(r'''
print("="*70); print("SUMMARY"); print("="*70)
figs=sorted(p.name for p in FIG_OUT.glob("*.pdf"))
tabs=sorted(p.name for p in TAB_OUT.glob("*.tex"))
print(f"  figures ({len(figs)}): "+", ".join(figs))
print(f"  tables  ({len(tabs)}): "+", ".join(tabs))
p1=M1[M1["is_prop"]].iloc[0]
print(f"\n  DS1: {len(M1)} architectures, proposed acc = {p1['acc']:.2f}%"
      f"  params = {p1['params']:.2f} M")
if M2 is not None:
    p2=M2[M2["is_prop"]].iloc[0]
    print(f"  DS2: {len(M2)} architectures, proposed acc = {p2['acc']:.2f}%")
rank=int((M1["acc"]>p1["acc"]).sum())+1
print(f"  proposed accuracy rank on DS1: {rank} of {len(M1)}")
print(f"  macro-F1 per million parameters: {p1['f1']/p1['params']:.1f}")
if WARN:
    print(f"\n  {len(WARN)} warning(s):")
    for w in WARN: print("    -",w)
else:
    print("\n  no warnings.")
''')

# =============================================================================
nb = {"cells": CELLS,
      "metadata": {"kernelspec": {"display_name": "Python 3",
                                  "language": "python", "name": "python3"},
                   "language_info": {"name": "python", "version": "3.10"}},
      "nbformat": 4, "nbformat_minor": 5}

out = os.path.join(HERE, "generate_paper_assets.ipynb")
with open(out, "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)
print("wrote", out, f"({len(CELLS)} cells)")
