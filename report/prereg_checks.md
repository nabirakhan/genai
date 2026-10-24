# Pre-registered validation checks

**Registered 2026-10-24**, before any classifier run exists (`results/classifier_runs.csv` is not yet written).

The locked design leaves two settings open. Each is settled by one check that compares the current setting
(baseline) with one variant, using a rule fixed here before any data are seen. Code: `src/analysis/checks.py`
(decision), `CHECKS` in `src/classify/queue.py` (runs), `src/classify/config.json` (the setting in force).

## Check 1: RandAugment magnitude (LAB_RUNBOOK S8a)

- **Setting:** magnitude of `FundusRandAugment` in conditions 5 and 6 (2 ops per image from Identity, ShearX/Y,
  TranslateX/Y, Brightness, Color, Contrast, Sharpness, AutoContrast; 31 magnitude bins).
- **Baseline:** magnitude 9 (about M = 3 on the RandAugment paper / timm 0–10 scale).
  **Variant:** magnitude 15 (about M = 5).
- **Why:** M = 3 is mild next to common ImageNet settings (M = 9); whether stronger augmentation helps here is unknown.
- **Runs (12):** condition 5 at 10% and 100%, seeds 0, 1, 2, per arm. Baseline: the main-grid runs
  `c5_f010_r0_s{0,1,2}`, `c5_f100_r0_s{0,1,2}`. Variant: `c5_f010_r0_s{0,1,2}_m15_val`, `c5_f100_r0_s{0,1,2}_m15_val`.

## Check 2: truncation ψ of the synthetic images (LAB_RUNBOOK S10a)

- **Setting:** which generated images condition 6 trains on.
- **Baseline:** conditional truncation ψ = 0.8 (`synth/f<ppp>`).
  **Variant:** ψ = 1, no truncation (`synth/f<ppp>_psi1`), from the same f010-ada and f100-ada generators and seeds.
- **Why:** ψ < 1 trades diversity for fidelity; with few real rare-grade images, diversity may matter more.
- **Runs (12):** condition 6 at ratio 1×, 10% and 100%, seeds 0, 1, 2, per arm, both arms at the magnitude
  adopted by check 1. Baseline: the main-grid runs `c6_f010_r1_s{0,1,2}`, `c6_f100_r1_s{0,1,2}`. Variant:
  `c6_f010_r1_s{0,1,2}_psi1_val`, `c6_f100_r1_s{0,1,2}_psi1_val`.

## Metrics

Per run, on the validation split, for the selected model (the epoch with the best val QWK, as in every run):
macro-F1 (`val_macro_f1`) and the recall of grades 3 and 4 (`val_recall_3`, `val_recall_4`).
Rare-grade recall = (recall_3 + recall_4) / 2. Baseline values come from the `val_*` columns of the main-grid
rows; variant runs are scored on validation only (one `dataset = val` row each in `results/checks/classifier_runs.csv`).

## Decision rule (verbatim)

> **PRE-REGISTERED RULE:** the variant is adopted only if, at BOTH fractions (10% and 100%), its mean val macro-F1
> over the 3 seeds exceeds the baseline mean by more than the baseline's seed standard deviation (ddof=1), AND its
> mean val rare-grade recall ((recall_3 + recall_4)/2) is not lower than the baseline's. Otherwise keep the baseline.

Applied per fraction: both conditions must hold at 10% and at 100%. "More than" is strict (a gain equal to the
standard deviation does not count); "not lower" admits a tie. The script refuses to decide unless all 12 runs are
present and the two arms differ in nothing but the tested setting.

## Order

1. Check 1 runs before the rest of S8 (S8a): its 6 baseline runs are S8 runs; the remaining S8 runs start only
   after the decision is committed.
2. Check 2 runs after S9 (synthetic images imported) and before S10 (S10a).

Enforced in code: the main queue holds every run until check 1 is applied and condition-6 runs until check 2
is applied; check 2 refuses to run or decide before check 1 is applied; every queue refuses to start if
`config.json` disagrees with an applied decision; an applied decision is never rewritten.

## Data use

Decisions use validation data only and are made before any aptos_test/IDRiD row is inspected. Variant runs are
never scored on APTOS test or IDRiD. Baseline runs are ordinary main-grid runs and are scored on both, but
`queue --checks` does not print those scores, and `results/classifier_runs.csv`, `results/preds/` and
`src.analysis.aggregate` stay unopened until the decision is committed. For check 2, condition 1–5 test results
exist by then (S8); they play no part in the rule, and no condition-6 aptos_test/IDRiD row is inspected first.

## Consequences

- **Variant adopted:** `checks <name> --apply` writes it to `src/classify/config.json` (`ra_magnitude` 15 or
  `synth_variant` "psi1") and moves the superseded runs (the 6 baseline runs, plus any other run of an affected
  condition already trained with the old setting) to `results/checks/superseded_runs.csv` and their predictions to
  `results/checks/preds_superseded/`. The main queue re-runs them under the same ids, so every reported condition-5
  and condition-6 number uses one setting. If ψ = 1 is adopted, condition 6 at 25% and 50% also needs ψ = 1 image sets.
- **Baseline kept:** nothing changes; the variant runs stay in `results/checks/` as the record.
- Either way the decision (rule text, per-fraction table, per-run values) is kept in
  `results/checks/decision_<name>.json` and reported.

## Known limitation

The validation split also selects the epoch in both arms, so validation scores are optimistic for both alike,
and three seeds give a coarse standard deviation. The rule is deliberately conservative: a variant must beat the
seed spread at both data sizes without losing rare-grade recall.
