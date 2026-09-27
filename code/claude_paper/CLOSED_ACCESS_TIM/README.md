# Closed-access submission package — IEEE TIM

Everything needed to submit, plus the notebook that regenerates the assets.

## Target journal
**IEEE Transactions on Instrumentation and Measurement.**
NJUST high-quality catalogue **category A**, entry 121, covering
控制科学与工程 / 计算机科学与技术 / 仪器科学与技术.
Hybrid journal, **default publication mode is subscription (closed access)** —
which is what you asked for. Do nothing at acceptance and it stays closed.

## Folder map
```
CLOSED_ACCESS_TIM/
  MS-EGCA_TIM_submission.pdf     compiled manuscript, send this to your professor
  latex/                         upload this folder to Overleaf
    tim_paper.tex                main document
    tim_preamble.tex             class + packages, auto-detects IEEEtran
    sec_related_work.tex         Section II
    refs_final.bib               85 references, 66 cited
    tim_paper.bbl                pre-built bibliography
    figures/                     5 vector figures
    tables/                      12 tables
  notebook/
    generate_paper_assets.ipynb  regenerates every figure and table
    build_notebook.py            rebuilds the notebook itself
    tim_figs.py, tim_tables.py   the same logic as plain scripts
    data.py                      verified transcription, for cross-checking
  results_source/                REPORT_*.md fallback inputs
```

## Rebuilding the assets
Open `notebook/generate_paper_assets.ipynb` and run all cells.
It writes straight into `latex/figures/` and `latex/tables/`, so recompiling
the LaTeX afterwards picks up the new versions automatically.

**No experimental number is hard-coded in the notebook.** For each table it
searches, in order:
1. `<RESULTS_DIR>/tables/<DS_TAG>/<name>.csv`
2. `<RESULTS_DIR>/tables/<DS_TAG>/<name>.md`
3. the matching section of `REPORT_<DS_TAG>.md`

Point `RESULTS_DIR_CANDIDATES` at
`D:\python\study_1_militery\code\DT_Q1_FINAL_results` to pick up the **stats**
and **dataset** tables as well. Those two are missing from `results_source/`,
so the significance panel of Fig. 5 and two tables are currently skipped with a
warning when the notebook runs from the bundled fallback data.

## Building the PDF
```
pdflatex tim_paper && bibtex tim_paper && pdflatex tim_paper && pdflatex tim_paper
```
Overleaf has `IEEEtran.cls` and `algorithm.sty`, so you get true IEEE
formatting and proper Algorithm environments with no edits. Locally the
preamble falls back to a geometry-matched two-column article.

## Submission facts (verified from ieee-ims.org, November 2025 terms)
- IEEE double-column Transactions format.
- Figures and tables **must be inline at their correct positions** — already done.
- Minimum 5 pages. **Free limit 8 pages.**
- Overlength: **$265 per page** non-member, **$220** IMS member, mandatory.
- Corresponding authors from lower-middle-income countries receive a 25–50 %
  discount; low-income countries a 100 % waiver.
- At least one author must be a graduate student or a full IMS member.
- Two EDICS classifications required at submission.
- Single self-contained PDF, under 20 MB.
- Open access would be $2,800 (2026). Not needed for closed access.

## Current status
15 pages, 5 figures, 12 tables, 2 algorithms, 66 citations.
Zero LaTeX errors, zero undefined references, no overfull boxes above 15 pt.
Float order is ascending for both figures and tables.
