# Overleaf bundle — IEEE TIM submission

## Upload
Upload this whole folder to Overleaf. Set **`tim_paper.tex` as the main
document**. Overleaf provides `IEEEtran.cls`, `IEEEtran.bst`, `algorithm.sty`
and `algpseudocode.sty`, so you get true IEEE formatting and proper
`Algorithm 1 / 2` environments with no edits.

## Files
| File | Role |
|---|---|
| `tim_paper.tex` | main document |
| `tim_preamble.tex` | class + packages, auto-detects IEEEtran and algorithmicx |
| `sec_related_work.tex` | Section II |
| `refs_final.bib` | 85 references, 66 cited |
| `tim_paper.bbl` | pre-built bibliography (some publishers require it) |
| `figs_tim/` | 5 vector figures |
| `tables_tim/` | 12 generated tables |

## Build
`pdflatex -> bibtex -> pdflatex -> pdflatex`

## Journal facts (verified from ieee-ims.org, Nov 2025 terms)
- **Format**: IEEE double-column Transactions. Figures and tables must be
  **inline at their correct positions** — this manuscript already does that.
- **Free page limit for Regular Papers: 8 pages.**
- Overlength charge: **$265 per page** (non-IMS member) or **$220** (IMS member),
  for each page beyond 8. Mandatory, not waivable except by country discount.
- Corresponding authors from lower-middle-income countries get a 25-50 %
  discount; low-income countries get a 100 % waiver. The discount follows the
  **corresponding author**, so listing the COMSATS (Pakistan) co-author as a
  corresponding author would qualify for a discount.
- At least one author must be a graduate student or a full IMS member.
- Minimum 5 pages. Single self-contained PDF under 20 MB.
- Two EDICS classifications must be chosen at submission.
- TIM is hybrid; the **default is subscription (closed access)**, which is what
  you asked for. Do nothing and it stays closed. Open access would cost
  $2,800 (2026 rate).

## Current length
**15 pages** -> 7 overlength pages -> approximately **$1,855** (or $1,540 for an
IMS member). See the length options in the chat summary if you want this lower.

## Scope warning
TIM rejects machine-learning papers that are not framed as measurement. This
manuscript is framed that way throughout: the recognizer is treated as an
instrument, and calibration, dispersion and failure modes are first-class
results. Do not remove that framing when editing.
