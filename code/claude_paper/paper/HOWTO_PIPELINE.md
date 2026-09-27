# One-cell pipeline — how to run

## Quick start
Put these files in one folder and run the pipeline:

```
paper_pipeline.py        <- paste this whole file into ONE Jupyter cell, or: python paper_pipeline.py
sec_introduction.tex     <- narrative sections (edit the prose freely)
sec_related_work.tex
sec_method.tex
sec_setup.tex
sec_conclusion.tex
refs.bib
```

Output lands in `paper_build/`:
`figs/*.pdf`, `tables/*.tex`, `paper.tex`, `paper.pdf`.

## Where the numbers come from
**Nothing is hard-coded.** For each table the pipeline tries, in order:

1. `<RESULTS_DIR>/tables/<DS_TAG>/<table>.csv`   ← written by your training run
2. `<RESULTS_DIR>/tables/<DS_TAG>/<table>.md`
3. the matching section of `REPORT_<DS_TAG>.md`

Set `RESULTS_DIR_CANDIDATES` at the top of the script if auto-detection misses.
On your machine the CSVs live under
`D:\python\study_1_militery\code\DT_Q1_FINAL_results\tables\` — point the script
there and it will also pick up the **stats table** and **dataset table**, which
add the significance forest plot, Table (stats) and Table (datasets). Those are
the two things missing from the test build (10 pages); with them the paper lands
at ~12 pages.

## What is generated vs. written by hand
| Part | Source |
|---|---|
| Results section prose | **computed** from your tables (ranks, gaps, counts, means, correlations) |
| Discussion section prose | **computed** (equivalence sets, limitations, variance range, calibration correlation) |
| Abstract headline numbers | **computed** |
| All figures and tables | **computed** |
| Introduction / Related Work / Method / Setup / Conclusion | your `sec_*.tex` files |

Inside the `sec_*.tex` files you can use `<<TOKEN>>` placeholders and the
pipeline substitutes the real value: `<<PARAMS>> <<LAT>> <<ACC1>> <<STD1>>
<<ACC2>> <<STD2>> <<EFF>> <<ECE1>> <<RANK1>> <<NBASE_ALL>> <<BEST1>>
<<BESTACC1>> <<BEST2>> <<BESTACC2>> <<NMODELS>>`.

## Safety behaviour
- A resumed run can leave a **stale duplicate** of one architecture. The
  pipeline detects duplicate model names inside a corpus, keeps the
  later-ID retrain, and prints exactly what it dropped. (On your data it drops
  DS1 `C3 MobileNetV3-Small, acc=98.77`.)
- If a table is missing, the dependent figure/paragraph is **skipped with a
  warning** rather than invented.
- Every run ends with a warning list — read it before submitting.

## Changing the settings table
`CONFIG` near the top holds the training hyperparameters. These are properties
of your run, not of the result CSVs, so they cannot be recovered automatically —
edit them if you change the training script.
