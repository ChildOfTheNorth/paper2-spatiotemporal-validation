# Recovered OWID source and analytical-input verification

The original notebook downloaded `master/public/data/owid-covid-data.csv` three times without recording a retrieval timestamp or raw-file checksum. The original retrieval date remains unknown.

On 8 October 2026, the CSV was recovered from OWID commit `f0d53320b19703cf565940d8269181a8423d7032`, whose last CSV update was 19 August 2024 at 21:06:55 UTC. The commit date is **not** the original retrieval date. Exact URLs, hashes and recovery date are in `owid_provenance.json`.

The recovered file matches the saved notebook fingerprint (237 real countries, 395,311 country-date rows, dates 2020-01-01–2024-08-14). More decisively, rerunning the original coverage, Tier 1 selection, duplicate handling and weekly aggregation reproduces the archived **2,904 rows across 12 countries**, all three weekly indicators and their missingness, and the alignment audit. Numeric equality is checked with relative and absolute tolerances of 1e-12; this does not assert bytewise equality of the unrecovered original raw download. Model fits and simulations were not rerun for this verification.

## Reproduce

From the repository root, with Python 3.11 or later, pandas and NumPy installed:

```sh
python reproducibility/recover_owid_inputs.py
```

This downloads the commit-pinned CSV, verifies its SHA-256 before analysis, reconstructs the weekly inputs using code taken from original notebook cells 17–19 and 221–222, and compares them with the archived reference files. Use `--source /path/to/owid-covid-data.csv` to verify a local copy. Generated files go to `recovered_owid/` or a directory specified with `--output`.

Reference CSVs are extracted unchanged from `Paper2_spatial_kernel_extension_all_outputs.zip` supplied with the analysis. The codebook is recovered from the same pinned commit. OWID source attribution: https://github.com/owid/covid-19-data; licensing and underlying-provider terms: https://github.com/owid/covid-19-data/blob/f0d53320b19703cf565940d8269181a8423d7032/README.md#license.

The original code/results snapshot and its integrity manifest remain unchanged. This directory is a separately documented provenance addition.
