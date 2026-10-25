# DLNM reproducibility release candidate — 25 October 2026

This record describes the **complete, locally verified** supplementary DLNM analysis package prepared for the revised SSTE submission.

**Release asset pending publication:**
- Filename: `Paper2_DLNM_reproducibility_v20261025.zip`
- Size: 11,495,981 bytes
- SHA-256: `b0a078e4885b79544269c0e326e755eee095d5be77d647138149e90236ecc949`
- Contents: `Paper2_DLNM_outputs_20261025.zip` (10,276,172 bytes), standalone Python script, Colab notebook, archived-reference coefficient CSV, supplementary methods/results DOCX, `README.md`, `SHA256SUMS.txt`, and `verify_release.py`.
- Inner results archive SHA-256: `d3f269a6cd4cb1d510780874ade3819a854b229e204353c4bc77d67947437daa`.

## Verified scope

The complete Colab archive records 220 final fits across 22 country–outcome series and ten specifications. The matched six-week linear DLNM has 22/22 cumulative-direction agreements and 22/22 maximum positions within one week of the archived unrestricted reference; four cumulative differences exceed the exploratory absolute 0.10-unit threshold. Fresh unrestricted refits are not identical to all original archived coefficient values. These checks **do not validate** interval coverage or clinical delay interpretation.

## Public availability status

**This GitHub repository currently hosts the analysis source and the verification metadata, but does not host the complete binary results ZIP. No GitHub Release, Zenodo archive or DOI has been published for this package.**

For deposit, create a tagged GitHub Release from the exact `main` commit of the approved source tree, attach the ZIP above without modifying its contents, verify the downloaded SHA-256, then optionally archive the complete release in Zenodo. Update the manuscript's Data Availability text to include the exact published URL and DOI only after verification.

The original October 8 methodology snapshot is preserved in the repository history. Do not overwrite historical result files.
