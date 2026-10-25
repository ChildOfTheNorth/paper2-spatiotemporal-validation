# Post hoc DLNM robustness analysis — 25 October 2026

This directory adds the post hoc empirical distributed lag non-linear model (DLNM) sensitivity analysis for the revised SSTE manuscript.

## Files
- `Paper2_DLNM_US_multistart_FULL.ipynb`: Google Colab implementation of the 11-country, two-outcome comparison, sensitivity sweep and United States admissions multistart audit.
- `Paper2_DLNM_US_multistart_FULL.py`: standalone Python implementation of the same workflow.
- `VALIDATION_SUMMARY.md`: audit outcomes and limitations.

## Study and interpretation
- Weekly COVID-19 cases, admissions and occupancy; 11 countries and 1,882 usable country-weeks.
- Six-week linear-exposure DLNM compared against **archived** seven-coefficient primary lag profiles for 22 country–outcome pairs.
- Additional exposure nonlinearity, knot position, lag horizon and exploratory fixed-penalty sensitivity.
- Unpenalized reference estimates remain the scientific baseline. New refits are numerical reproducibility diagnostics, not replacements for archived coefficients.
- Lag maxima, cumulative associations and covariance intervals are retrospective conditional associations. They do **not** identify causal effects, individual clinical delays, calibrated forecasting uncertainty or operational lead times.

## Reproduction
Use Google Colab to open `Paper2_DLNM_US_multistart_FULL.ipynb` and run all cells with its full mode enabled. The script documents its arguments and can run as a standalone program. It downloads the historical pinned OWID input; software versions, source commit and fit diagnostics must be preserved with a release.

The code uses a Python cross-basis implementation analogous to DLNM methodology, **not** the R `dlnm` package. Penalized intervals and information criteria must be interpreted according to the implementation's stated approximations.

## Integrity and release status
The source notebook and script are available here. **The complete Colab results ZIP (approximately 10 MB), supplementary DOCX and checksum manifest are not yet committed to this branch.** The local validated archive contains 220 fitted model records and audit outputs; do not claim the public repository alone reproduces all reported results until the exact result archive is uploaded and independently checked.

For the original 8 October 2026 snapshot, use the repository history rather than overwriting its files. Verify published archive hashes against the local release manifest when depositing the remaining artifacts.

No Zenodo DOI or permanently archived public release has been minted by this update.
