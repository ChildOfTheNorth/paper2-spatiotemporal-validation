# Paper 2: dated analysis snapshot

Snapshot: **2026-10-08**, package version **v0.1.0**. Author: Petra Norlund.

This package preserves the supplied main analysis notebook, original standalone lag-basis validation, and subsequent basis/interval-calibration follow-up experiments. It is a snapshot of available work, not a claim that every analysis or manuscript result has been independently reproduced.

## Contents

| Path | Contents |
| --- | --- |
| `notebooks/paper2_main.ipynb` | Latest supplied main notebook; outputs cleared for browsing. |
| `notebooks/basis_validation_and_followups.ipynb` | Original Chunk 24 and follow-up Chunks 25–29. |
| `results/basis_flexibility_standalone_v1/` | Original 200-replication basis experiment: tables, figures, failures and settings. |
| `results/followup_basis_bootstrap_v1_pilot/` | Pilot follow-up: broad/two-peak targets; interval calibration, failures, figures and checkpoints. |
| `archive/` | Original supplied result ZIPs. The separately downloadable complete snapshot additionally includes byte-identical source notebooks with saved outputs. |
| `environment_recorded.json` | Software versions recorded by the saved simulation outputs. |
| `SHA256SUMS.json` | Hash and size of each snapshot file except this manifest itself. |

## Status of the results

The follow-up is **pilot**, with 20 shape replications, 20 bootstrap outer replications and 99 draws per outer replication. It must not be described as the completed larger calibration study. The original experiment and additions remain separately labelled.

The interval comparison is a conditional Gaussian **parametric ARMA(2,1) bootstrap**, with stationary initialization and parameter refitting. It preserves serial dependence through the fitted error model; it is not a nonparametric block bootstrap. Report coverage, width, failures and Monte Carlo uncertainty together. It does not establish calibration under arbitrary error misspecification.

## Reuse in Google Colab

1. Open `notebooks/basis_validation_and_followups.ipynb` in Colab.
2. Copy the two folders under `results/` into `MyDrive/Paper2_saved_results/`, retaining their folder names.
3. Run Chunk 25, then Chunks 28–29 to reconstruct summaries and figures from saved results. Chunk 24 is unnecessary for this route.
4. To regenerate pilot fits, run 25–29. For the predefined larger experiment, change `FU_MODE` to `"full"` in Chunk 25 and run 25–29. Full results go to a separate folder. This is computationally expensive.

The notebook checks environment/settings signatures. Restore the recorded versions when resuming existing checkpoints; do not bypass signature guards. `requirements-recorded.txt` records the producer versions (Python 3.13.16), but availability/installability of those versions was not independently verified here. The main notebook has additional dependencies in `requirements-main.txt`; its external data downloads are not frozen by this package. Its cells include exploratory iterations, so it is not asserted to be a clean restart-and-run-all pipeline.

## Verification and limitations

Run `python scripts/verify_snapshot.py` from the repository root. Packaging checks passed for archive integrity, all pilot manifest entries, the original-results provenance hash, and Python syntax in ordinary code cells. The statistical analyses were not rerun during packaging. No manuscript, review correspondence, or course/portfolio documents are included. No reuse licence is assigned; choose one before presenting this as open-source software.

## Dated repository snapshot

Repository: https://github.com/ChildOfTheNorth/paper2-spatiotemporal-validation

Use the publication commit permalink to identify the exact version. The repository contains the output-cleared notebooks and saved simulation results; the complete 37,403,347-byte downloadable snapshot additionally preserves the large archive of original source notebooks with their saved outputs. That complete ZIP has SHA-256 `c56014dd777c40ecd26eda488dba902f97ae28c185c68cde151c2d6aa6fd2d2d`.

`SHA256SUMS.json` in this repository verifies the files actually committed here, excluding itself. The repository commit permalink is a frozen version reference, not a DOI or a GitHub release. No DOI is claimed.
