# Ultra Master Loop Prompt v7.2 — AMD DirectML Only + IEEE 12-Page Combined Results
## Lightweight military-vehicle image classification · Strict professor coverage · No CUDA dependency

> Copy the full block into Arena Agent Mode. This is the AMD-only final version. It deliberately does **not** require CUDA, Mamba, or any server-only experiment.

```text
MISSION
=======
Act as an Arena autonomous research-coding agent, senior PyTorch lightweight-computer-vision engineer, reproducibility/leakage auditor, statistical reviewer, and Q1-journal manuscript critic.

Create a NEW guided multi-cell notebook:

    lightweight_military_vehicle_amd_directml_q1.ipynb

and a complete archive in `lightweight_results/`.

Goal: evaluate a proposed <=3M-parameter EdgeMIL-CA CNN against broad, AMD-compatible lightweight CNN, Transformer, CNN–Transformer hybrid, and graph-vision representatives on a leakage-audited ten-class public military-vehicle image benchmark.

This is a public-image academic classification benchmark only. Do not claim surveillance, operational target recognition, autonomous weapons relevance, military deployment, real-time capability, safety, or clinical/field validity.

Never promise Q1 acceptance, 97%/99% accuracy, novelty, significance, or a latency target. Report the actual evidence, including failures and negative outcomes.

AMD-ONLY NON-NEGOTIABLE CONSTRAINT
==================================
The available system is Windows + AMD RX 6750 XT (12GB) via torch-directml and Ryzen 5 9600X. There is no CUDA server.

Set and assert:

    ALLOW_CUDA = False
    RUN_CUDA_ONLY_SERVER_MODELS = False
    RUN_SSM_MAMBA = False

Never install, import, or attempt CUDA-only Mamba/SSM, Triton, xFormers, FlashAttention, CUDA PyG extensions, or `torch.cuda.amp`. Do not list Mamba as a failed local experiment; list it once in Related Work / Hardware Scope as `OUT OF SCOPE — CUDA custom kernels unavailable in the declared AMD DirectML environment`.

Use device priority DirectML > CPU. Run a real DirectML matmul smoke test. Set num_workers=0. Use FP32. If DirectML timing cannot synchronize safely, label timing APPROXIMATE.

RESEARCH-INTEGRITY RULES
========================
Never:
- modify, delete, or overwrite the legacy MilAttn-Net source; create a backup first;
- fabricate metrics, model status, citations, versions, licences, parameter counts, FLOPs, latency, CIs, p-values, or Q1 readiness;
- silently replace, skip, rename, or zero-score a configured model;
- use test labels/results to choose models, architecture, augmentation, epochs, hyperparameters, thresholds, calibration, or claims;
- return placeholder black/zero tensors for corrupt images;
- call frozen foundation/vision-language probes lightweight at inference;
- use unapproved image datasets, ImageNet-C, ImageNet-Hard, or outside data;
- treat model-family breadth as novelty.

Use cautious wording: leakage-cleaned public image-level benchmark; AMD DirectML measurement; synthetic perturbation analysis; research-use classifier; proposed method.

DATA AND CLASS CONTRACT
=======================
Use only Kaggle `amanrajbose/millitary-vechiles`, expected local root:
    D:\python\study_1\dataset_balanced_all

Safely resolve train/, validation/ or valid/, and test/. Recount real supported images; reported source counts are train=11,890, validation=5,090, test=4,240.

Use fixed class IDs:
0 Anti-aircraft
1 Armored combat support vehicles
2 Armored personnel carriers
3 Infantry fighting vehicles
4 Light armored vehicles
5 Mine-protected vehicles
6 Prime movers and trucks
7 Self-propelled artillery
8 light utility vehicles
9 tanks

Record local root, package URL, access date, available version/licence metadata, actual class/split counts, rejected files and manifest SHA-256. If licence is unknown, state HUMAN REVIEW REQUIRED.

BOOLEAN RUN SYSTEM
==================
Implement a frozen validated dataclass:

RUN_QUICK: bool = True
RUN_FULL: bool = False
QUICK_EPOCHS: int = 1
FULL_MAX_EPOCHS: int = 40
ENABLE_EARLY_STOPPING: bool = True
EARLY_STOPPING_PATIENCE: int = 7
QUICK_BOOTSTRAPS: int = 500
FULL_BOOTSTRAPS: int = 2000
QUICK_SEEDS: list[int] = [42]
FULL_PRIMARY_SEEDS: list[int] = [42, 123, 2026]
PRIMARY_MULTI_SEED_MODELS: list[str] = ["C3", "P0"]
SECONDARY_MODEL_SEEDS: list[int] = [42]
RUN_MULTI_SEED_BREADTH: bool = False
IMAGE_SIZE: int = 224
BATCH_SIZE: int = 32
FINAL_CONFIGURATION_LOCKED: bool = False

RUN_DUPLICATE_AUDIT: bool = True
RUN_CALIBRATION: bool = True
RUN_XAI: bool = True
RUN_SYNTHETIC_STRESS: bool = True
RUN_ABLATIONS: bool = RUN_FULL
RUN_CURRENT_CANDIDATES: bool = True
RUN_GNN: bool = True
RUN_FOUNDATION_PROBE: bool = False
RUN_VISION_LANGUAGE_PROBE: bool = False
SAVE_MODELS: bool = True
SHOW_ALL_INLINE: bool = True

assert RUN_QUICK != RUN_FULL, "Exactly one of RUN_QUICK/RUN_FULL must be True."
assert not ALLOW_CUDA and not RUN_CUDA_ONLY_SERVER_MODELS and not RUN_SSM_MAMBA

Derived values:
- QUICK = one epoch; all results carry “PRELIMINARY — 1 EPOCH ONLY — NOT MANUSCRIPT EVIDENCE”.
- FULL = common maximum 40 epochs and predeclared early stopping. Report actual epochs.
- C3 and P0 receive seeds [42,123,2026] in FULL; all other primary models receive seed 42 unless RUN_MULTI_SEED_BREADTH=True.
- QUICK and FULL show the same sections, dashboards, figures, artifact hierarchy and checkpoint ledger. In QUICK, unavailable final analyses show PRELIMINARY/N-A with reason.
- `assert_not_preliminary_for_manuscript()` blocks result-aware manuscript drafting, final ranking, final paired-inference claims and final conclusion when RUN_QUICK=True.

PHASE 0: PROMPT SELF-AUDIT — MANDATORY
=======================================
Before writing notebook code or training:
A0 extract every model, family, package, Boolean, output, metric, dataset rule and test-lock requirement into REQUIREMENTS_TABLE.
A1 find contradictions in model counts, run modes, lightweight definitions, output rules, test locking and names.
A2 create hardware/dependency COMPATIBILITY_MATRIX for every model.
A3 check data restriction, leakage policy, predeclared primary endpoint/comparator, calibration rule and ablation logic.
A4 check that every metric/statistic can reconstruct from retained predictions; require paired IDs, bootstrap seed, CI, correction and failure statuses.
A5 estimate QUICK/FULL compute and create execution tiers.
A6 check paper integrity: citations, claims, limitations, licence/provenance, original prose, no Q1 promise.
A7 repair all blocking/major issues and repeat A1–A6, maximum three rounds.
A8 display a PROMPT RELEASE CARD. If blockers remain, label DRAFT WITH PROMPT BLOCKERS and stop before training.

MODEL REGISTRY — AMD PRIMARY BENCHMARK
======================================
Definitions:
- ultra-light <=1.5M measured parameters;
- light >1.5M and <=3M;
- compact >3M and <=6M;
- non-light >6M: not eligible for the main lightweight ranking.

Primary registry, eligible for the fair compact ranking only when COMPLETED on identical locked IDs:

C0 SqueezeNet1_1
C1 ShuffleNetV2_x1_0
C2 MobileNetV2
C3 MobileNetV3_Small             # PREDECLARED primary P0 comparator
C4 MNASNet1_0
C5 EfficientNet_B0
C6 RegNetX_400MF
C7 GhostNetV2_1_0                 # verify first
C8 StarNet_S1                     # verify first
C9 MobileOne_S0 OR FasterNet_T0   # select one before lock, record other NOT_SELECTED
T0 DeiT_Tiny                      # pure Transformer, verify first
H0 MobileViT_v2_XXS_or_XS         # hybrid, verify first
H1 EdgeNeXt_XXS OR FastViT_T8     # hybrid, select one before lock
G0 ViG_Ti                          # graph vision, pure PyTorch DirectML only
P0 EdgeMIL_CA                     # proposed <=3M CNN

Configured AMD primary count: 15 architecture families.

Separate optional AMD-only representation probes, OFF by default and NEVER mixed into primary lightweight ranking:
F0 DINOv2_ViT_S_frozen_linear_probe
V0 MobileCLIP OR TinyCLIP_frozen_linear_probe

These may be enabled only after dependency, download, DirectML, full-backbone-complexity and protocol checks. They use the same cleaned military data. They are labelled EXPLORATORY FROZEN PROBES, not lightweight architecture comparisons.

CAL_P0 and A0–A4 are configurations, not headline architecture families:
CAL_P0 validation-only temperature-scaled P0;
A0 no EdgeContextAttention;
A1 no Sobel prior;
A2 no multi-scale context;
A3 no ECA;
A4 full P0.

PROFESSOR FAMILY COVERAGE DASHBOARD
===================================
Show inline before any training and in all reports:
family | representative models | role | actual parameter stratum | source/dependency | DirectML K-gates | status | primary-rank eligible | reason/limitation.

Families covered in AMD study:
- CNN: C0–C9.
- pure Transformer: T0.
- CNN–Transformer hybrid: H0/H1.
- graph vision: G0 only if pure PyTorch K-gates pass.
- proposed compact CNN: P0.
- optional frozen representation: F0/V0, if enabled.
- state-space/Mamba: scope note only, explicitly out of scope due to AMD-only hardware; no fake run.
- MLP/MetaFormer: do not force a >6M architecture into a lightweight study. State in coverage dashboard: `NOT INCLUDED IN PRIMARY STUDY — available standard representatives exceed predeclared lightweight budget`; list it in related work. If the supervisor explicitly mandates it, add PoolFormer-S12 as NON_LIGHT_CONTEXT_REFERENCE in a separate table, never primary ranking.

NON-TORCHVISION DIRECTML K-GATES
================================
For C7–C9, T0, H0–H1, G0, and optional F0/V0, run/display K1–K9 before enablement:
K1 verified original paper and reputable/official implementation;
K2 licence and pretrained-weight status recorded;
K3 no CUDA/Triton/xFormers/FlashAttention/Mamba/PyG CUDA custom operation;
K4 actual built model parameter/MAC count and stratum;
K5 DirectML build/forward;
K6 DirectML backward plus one real-batch optimization;
K7 state-dict serialization/load;
K8 fair data/preprocessing/recording protocol;
K9 stable execution and clear scientific purpose.

If a candidate fails: retain its named row, show FAILED/SKIPPED/NOT_AVAILABLE_LOCAL, exact error and next action. Never substitute silently.

Freeze selected source URL/commit if available, package/version, weight identifier, transform, trainability policy and model configuration hash before CP09.

PROPOSED P0: EDGEMIL-CA
=======================
Implement a serializable, testable, compact model:
- 224x224 RGB; Conv3x3 stride-2 + BN + Hardswish stem;
- four mobile inverted-residual/depthwise stages;
- GAP + dropout + 10-logit linear head;
- target <=3M measured params and <=0.50 GMACs batch 1, stated before test access.

At selected late stages use EdgeContextAttention(X):
1 fixed grouped depthwise Sobel x/y magnitude;
2 pointwise edge projection E to X channels;
3 U=Pointwise(Concat[DWConv3x3(X),DWConv3x3(X,dilation=2)]) to X channels;
4 r=Sigmoid(Conv1x1(Concat[X,E,U]));
5 F=r*E+(1-r)*U;
6 ECA=Sigmoid(Conv1d(GAP(F))) reshaped B,C,1,1;
7 Y=X+gamma*(ECA*F), gamma trainable scalar initialized exactly zero.

Test: shape alignment, bounded r/ECA, finite output/gradient, gamma state, save/load equivalence, actual parameter/MAC budget. Do not call blocks novel in code.

DATA AUDIT — BLOCKING
=====================
Before model training:
1 backup/read legacy sources; parse executable notebooks with nbformat/AST when possible; static-audit rendered HTML;
2 produce CODE_AUDIT_MATRIX and CHANGE_LEDGER;
3 discover splits, validate all ten classes; unknown image class => error;
4 build manifest with IDs, paths, split, labels, file/decode/dimension/mode, byte SHA-256, canonical RGB hash, pHash/dHash, duplicate group;
5 reject/log corrupt images; never placeholders;
6 detect byte/canonical duplicates and screen near duplicates using contact sheets;
7 confirmed same-label duplicate priority test > validation > train; remove lower-priority copies; quarantine conflicting labels;
8 hash final ordered IDs/groups and assert no confirmed duplicate group crosses splits;
9 state image-level cleanup does not prove vehicle/source/site/geographic/time independence.

FAIR TRAINING AND TEST LOCK
===========================
Primary RQ: does P0 differ from C3 MobileNetV3-Small in locked-test macro-F1 under the same cleaned protocol, and what is the measured efficiency trade-off?

Common primary protocol:
- same cleaned IDs/class order/224 input/conservative train-only augmentation/max epochs/early stop/seed policy/checkpoint monitor/test evaluator/timing method;
- augmentation only for training: crop-resize, horizontal flip, +/-10-degree rotation, mild translation and intensity variation; no vertical flips and no validation/test augmentation;
- AdamW and common initial LR/weight decay/label smoothing;
- official model-specific pretrained transforms are allowed; record them;
- if a model family needs a different LR, use the same predeclared development-only LR grid; record result before CP09; never select with test data;
- validation macro-F1 selects checkpoint; test is never called by training/model-selection functions.

CP09: display/freeze RQs, comparator, registry/statuses, C9/H1 choice, K-gates, split fingerprints, transforms, augmentation, seeds, budget, early stopping, metrics, calibration plan and config SHA-256. Set FINAL_CONFIGURATION_LOCKED=True. Test functions reject mismatch.

RUN SCHEDULE
============
QUICK: complete data audit/K-gates; train C0,C1,C3,C5,T0,H0,G0,P0 for one epoch if compatible; build/validate all other primary models; create all tables/figures/reports with preliminary placeholders; optional A0/A4 one epoch if feasible.

FULL: train every enabled C0–C9,T0,H0–H1,G0,P0. Train C3/P0 with seeds 42,123,2026; other enabled models seed 42 unless breadth flag enabled. Run A0–A4 and CAL_P0. Run F0/V0 only if explicitly enabled and K-gates pass. All failures remain visible.

METRICS, STATISTICS, EFFICIENCY, XAI
=====================================
Retain per completed test run: IDs, truth, prediction, logits/probabilities, checkpoint/run hashes, metrics and hardware metadata.

Compute accuracy, balanced accuracy, macro/micro/weighted P/R/F1, per-class sensitivity/specificity/F1, MCC, kappa, ROC-AUC/PR-AUC where valid, NLL, multiclass Brier, declared-bin ECE, entropy, count/normalized confusion matrices, reliability, risk-coverage and AURC.

Statistics: stratified seeded bootstrap 500 QUICK/2000 FULL; paired P0-C3 macro-F1 bootstrap CI; McNemar exact test; Holm correction; effects plus CI/p. Only identical test IDs. Separate seed variation from image bootstrap uncertainty.

Measure parameters, checkpoint size, THOP MAC/FLOP convention, batch-1/32 CPU/DirectML latency with warmups/repetitions/eval mode/median-IQR, throughput, seconds/epoch. DirectML timing may be approximate.

XAI: Grad-CAM for P0/C3/best completed non-proposed primary model on identical class-balanced IDs; P0 edge/reliability/ECA maps; confidence error galleries. Explain all maps are post-hoc, not localisation/causal/operational proof.

Synthetic perturbations after freeze: noise, blur, gamma/brightness, JPEG, small rotation. Call synthetic perturbation analysis only.

INLINE REPORTS, ARTIFACTS, CHECKPOINTS
========================================
Every key result appears inline immediately. Use one source object for inline display and saved output.

Implement `emit_status`, `emit_checkpoint`, `emit_table`, `emit_figure`, `emit_text`, `emit_metric_panel`, `register_artifact`, `assert_not_preliminary_for_manuscript`.

Save to:
lightweight_results/
  reports/professor_review.md
  reports/professor_review.pdf
  reports/llm_evidence_draft.md
  reports/journal_combined_results.md
  reports/journal_combined_results.pdf
  tables/results.xlsx, results.json, *.csv
  figures/png, figures/svg, figures/tiff
  models/
  logs/run_registry.json, environment.json, histories.json
  reproducibility/configuration.json, split_fingerprints.json,
  artifact_manifest.json, checksums.json
  CHANGE_LEDGER.md

Keep three separate result surfaces:
1 Primary AMD Compact Benchmark: completed C/T/H/G/P models only.
2 Optional Frozen-Probe Context: F0/V0 only, with full-backbone complexity/caution.
3 Calibration/Ablations: CAL_P0/A0–A4.

Mandatory CP00–CP19 cards: notebook integrity; environment; data/class; manifest; duplicates; split; preprocessing; model; P0 unit tests; lock; training; test; reconstruction; statistics; claims; XAI; efficiency; artifacts; release.

TARGET JOURNAL AND IEEE TWO-COLUMN PAGE BUDGET
================================================
At Phase 0 display `TARGET_JOURNAL = "IEEE Transactions on Artificial Intelligence (TAI) — candidate"` and `TARGET_PAGE_BUDGET = 12` in IEEE two-column format. This is a planning target, not a claim of fit or acceptance.

Before paper drafting, verify current scope, author instructions, indexing/quartile status, page charges and template directly from official journal sources. Do not rely on stale statements. If the method contribution is insufficient for TAI after adversarial review, label it `TARGET FIT RISK` and propose a human-supervisor decision between a more suitable IEEE venue and IEEE Access; never fabricate a journal recommendation.

Use this 12-page target allocation for the journal-facing combined report:
- Page 1: title, abstract, keywords, introduction/contributions.
- Pages 2–3: related work, data provenance/leakage protocol, method overview.
- Pages 4–5: EdgeMIL-CA architecture, training and evaluation protocol.
- Pages 6–8: combined main results and efficiency evidence.
- Pages 9–10: ablation, calibration, error/XAI and discussion.
- Page 11: limitations, ethics/dual-use/data/code statements, conclusion.
- Page 12: references or overflow controlled by the current journal template.

Never force 12 pages by padding prose. If current target requirements differ, update the page plan before manuscript generation and log the decision.

ENERGY, TIME, AND RESOURCE MEASUREMENT POLICY
===============================================
For every completed training run record:
- planned epochs, actual completed epochs, early-stop status and best epoch;
- total wall-clock training time, mean/median epoch time and validation time;
- batch size, effective batch size, seed, selected checkpoint and hardware/backend;
- batch-1 and batch-32 inference latency/throughput;
- parameter count, model file size, MAC/FLOP convention;
- peak memory only where safely measurable;
- measured energy in Wh/J for training and/or inference ONLY when a validated accessible system/GPU sensor or external power meter is detected.

Do not estimate energy from elapsed time, TDP, or a guessed GPU wattage. If a validated energy counter is not available on Windows/DirectML, write `N/A — no validated accessible energy counter` and explain this limitation. A transparent N/A is better than a fabricated energy claim.

Create two efficiency objects:
1. `training_resource_table`: model, family, seed, planned/actual epochs, best epoch, train time, epoch time, validation time, peak memory if available, energy if validated, status.
2. `inference_resource_table`: model, parameters, MACs/FLOPs, FP32 size, batch-1 latency, batch-32 latency, throughput, energy/inference if validated, timing/energy protocol status.

Use the same resource objects for inline tables, professor PDF/Markdown appendices, combined journal reports and figures. Never manually retype values.

IEEE 12-PAGE MAIN TABLE/FIGURE PLAN
====================================
Create concise journal-facing combined tables, not one table per model:

Table I — Dataset, leakage audit and final cleaned split counts.
Table II — Architecture and efficiency profile: model, family, params, MACs/FLOPs, FP32 size, actual epochs, total training time, epoch time, batch-1 latency, throughput, measured energy or N/A.
Table III — Main locked-test comparison: model, macro-F1 [95% CI], balanced accuracy, macro ROC-AUC, macro PR-AUC, ECE, Brier, status. Keep P0/C3 visually marked. Use only completed identical-ID primary models.
Table IV — EdgeMIL-CA ablation and calibration summary: A0–A4 and P0/CAL-P0; include N/A for not-run rows, never fake metrics.

The main paper normally uses Tables I–IV. Detailed per-class metrics, every individual history, all individual CMs/ROCs/PRs, complete run registry, all resource repetitions, and full status/error messages belong in the professor appendix/report—not the 12-page main manuscript.

Create journal main figures only as combined panels: architecture/workflow; combined performance CI; ROC/PR; calibration/risk; Pareto; selected CMs/errors; ablation. Consolidate subplots thoughtfully; each caption must be understandable independently.

COMBINED-RESULTS LOOP — MAIN JOURNAL STORY
============================================
The professor/reviewer needs full individual evidence; the journal manuscript needs concise combined evidence because page limits are real. Enforce BOTH outputs.

INDIVIDUAL EVIDENCE POLICY:
- Immediately after each completed model, display its status, training history, metrics panel, classification report, confusion matrix, ROC/PR, calibration/risk-coverage where valid, efficiency record, and selected XAI/error evidence INLINE in the notebook.
- Save each individual object under `figures/individual/`, `tables/individual/`, and append detailed per-model evidence to `reports/professor_review.md` and `reports/professor_review.pdf` appendix sections.
- Individual evidence remains available for professor/reviewer audit, but is NOT the default main-paper figure set.

COMBINATION GATE:
Run the combined-results loop only after every enabled primary registry row has a terminal status (COMPLETED, FAILED, SKIPPED, NOT_AVAILABLE_LOCAL, or NOT_RUN). Filter numerical comparison/ranking/statistics to COMPLETED runs with exactly the same locked-test fingerprint and ordered sample IDs. Never insert zero metrics for unavailable runs.

COMBINATION LOOP C0–C10:
C0 display the final model-status/coverage table including every configured model and exact reason for non-completion.
C1 construct one master long-form result object and one wide combined results DataFrame from retained prediction/evidence objects only.
C2 construct the main combined table: model, family, seed status, parameters, MACs, size, macro-F1 with 95% CI, balanced accuracy, macro ROC-AUC, macro PR-AUC, ECE, Brier, latency, throughput, run/test fingerprint, and status.
C3 produce a combined macro-F1/accuracy comparison with 95% CI; include only completed primary models and visually mark P0/C3.
C4 produce a single performance-efficiency Pareto plot: macro-F1 vs parameters; bubble size/annotation for batch-1 latency; use completed primary models only.
C5 produce a combined family/metric heatmap containing completed primary models and predeclared headline metrics.
C6 produce a combined ROC figure and combined PR figure: main journal panels show P0, C3, and the best completed non-proposed compact model; all completed models appear in a professor appendix figure if readable.
C7 produce combined calibration/reliability and risk-coverage panels for P0, CAL_P0, C3, and the strongest completed non-proposed compact baseline. Do not crowd all models into the main panel.
C8 produce combined ablation chart/table for A0–A4, only if full ablations completed; otherwise show NOT RUN with reason.
C9 produce a selected combined confusion-matrix panel for P0, C3, and the best completed non-proposed baseline; individual matrices remain in the appendix.
C10 create an evidence-linked combined Results narrative and ranking. Primary rank = predeclared macro-F1 on identical locked IDs; include CI and do not imply meaningful order where CIs/tests are inconclusive.

JOURNAL FIGURE POLICY:
Create `figures/journal_main/` containing only the manuscript-ready combined figures:
Fig_1_dataset_and_leakage_overview
Fig_2_EdgeMILCA_architecture
Fig_3_combined_performance_with_CI
Fig_4_combined_ROC_PR
Fig_5_calibration_and_risk_coverage
Fig_6_performance_efficiency_Pareto
Fig_7_selected_confusion_and_error_analysis
Fig_8_ablation_results

Each figure must be original, colour-blind-safe, captioned, high resolution (PNG >=300 dpi; SVG; TIFF where practical), and generated from registered source objects. Do not save an individual-model plot as a journal main figure unless it is part of a selected combined panel.

REPORT POLICY:
- `reports/professor_review.md` and `.pdf`: executive combined results first, then detailed individual model evidence in clearly labelled appendices/details sections.
- Create `reports/journal_combined_results.md` and `reports/journal_combined_results.pdf`: concise journal-facing results package containing ONLY combined main tables, combined figures, captions, evidence-linked narrative, model-status note, and limitations.
- Display final combined tables and journal-main figures INLINE at the end of the notebook after the model loop. The professor never needs to open a folder to understand the final ranking.
- Ensure all values in individual, combined, markdown, and PDF reports derive from the same master result object. Validate table/figure checksums and file readability.

Add CP19 COMBINED RESULTS PASS only if the combination gate passed, all completed rows share locked IDs, journal combined table/figures are present, unavailable models are status-only, and inline/saved evidence matches. CP19 may be WARN in QUICK mode and must never be described as final manuscript evidence.

PAPER LOOP
==========
After real FULL evidence only: verify relevant literature and Citation_IDs; build Evidence_ID objects; draft Methods -> Results -> Discussion -> Limitations -> Introduction -> Abstract; map every quantitative claim to evidence; explain architecture breadth is validation not novelty; include dataset licence/provenance/image-level dependence/AMD scope/dual-use limitations; run editor/method/statistics/efficiency/ethics/language reviewer passes; maximum 3 revision cycles.

Release labels only: RESEARCH NOTE, PROFESSOR REVIEW DRAFT, EVIDENCE-LINKED SUBMISSION DRAFT. Never say Q1 guaranteed/ready.

FINAL RESPONSE
==============
Return notebook path; proof legacy source untouched; exact AMD Boolean config; model-family dashboard; configured/enabled/completed/failed/skipped/tested/ranked counts; actual P0-C3 results or NOT RUN; blockers; artifact paths/validity; safe novelty wording; and run guide:
1 set dataset root;
2 RUN_QUICK=True/RUN_FULL=False and inspect checkpoints;
3 fresh output directory, then RUN_QUICK=False/RUN_FULL=True after supervisor approval.

Start now at Phase 0 self-audit and STATE 0 READ/STATE 1 PLAN. Do not train, test, or write result claims before audit/data checks and CP09 lock.
```

## Continuation command

```text
Continue from the latest valid checkpoint. First show AMD-only Boolean flags, run label, model-family dashboard, K-gate statuses, config hash, checkpoint ledger, output locations and blockers. Do not overwrite legacy code, attempt CUDA models, silently replace candidates, or use test results for design. Execute one smallest valid phase, save notebook, check it, render checkpoint evidence, and repair the smallest root cause if needed.
```
