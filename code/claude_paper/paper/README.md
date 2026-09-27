# MS-EGCA manuscript — build & submission notes

**Target venue:** IEEE Transactions on Instrumentation and Measurement
(subscription / closed access, two-column, 12-page threshold).
**Current build: 12 pages**, 7 figures, 6 tables, 44 references.

## Files
| File | Purpose |
|---|---|
| `paper.tex` | The manuscript. Compiles as-is. |
| `related_work.tex` | Section II, `\input` by `paper.tex`. |
| `refs.bib` | 44 references (all cited; no unused entries). |
| `paper.bbl` | Pre-built bibliography (some publishers want this instead of `.bib`). |
| `MS-EGCA_paper.pdf` | Compiled output. |
| `data.py` | **Single source of truth.** Every number in the paper comes from here. |
| `make_tables.py` | Generates `tables/*.tex` from `data.py`. |
| `make_figs.py` | Generates `figs/*.pdf` from `data.py`. |
| `tables/`, `figs/` | Generated assets, `\input`/`\includegraphics` at fixed positions. |

## Build
```
pdflatex paper && bibtex paper && pdflatex paper && pdflatex paper
```
To regenerate assets after editing `data.py`:
```
python make_figs.py && python make_tables.py
```

## Overleaf
Upload the whole folder. Overleaf has the real `IEEEtran` class, which the
preamble auto-detects — you get true IEEE formatting with no edits. Locally,
if `IEEEtran.cls` is absent, it falls back to a geometry-matched two-column
`article` (which is how the included PDF was produced). Expect the page count
to stay at 12 or drop slightly under real IEEEtran.

## Float positioning
Single-column floats use `[H]` (from the `float` package) so they appear
exactly where they are written, MS-Word style. Double-column floats
(`table*`/`figure*`) cannot use `[H]` — LaTeX forbids it — so they use `[!t]`
with `stfloats` loaded; they land at the top of the column-pair nearest their
call site.

## Things to check before submitting
1. `figs/fig8_efficiency.pdf` and `figs/fig8_radar.pdf` are **stale, unused**
   artefacts from earlier drafts. Delete them.
2. `tables/tab4_extended.tex` and `tables/tab8_corruption.tex` are generated
   but no longer `\input` (cut for length). Keep them for the supplement or
   delete.
3. Latency was measured on an AMD GPU via DirectML in FP32; the embedded
   INT8/edge-metrics pass failed on that backend and is declared as a
   limitation. If you can rerun on CUDA or a Jetson, that closes the biggest
   reviewer objection.
4. The DeiT-Tiny cross-attention branch was disabled throughout — ablations
   A7/A8 therefore equal A6. This is stated openly in Sec. III-E and in the
   Table V footnote. Do not remove those disclosures.
5. DS1 excludes a stale duplicate MobileNetV3-Small row and has no
   ShuffleNetV2 x0.5 result; both are declared in Sec. IV-B. See the header
   comment in `data.py` for the full provenance.
