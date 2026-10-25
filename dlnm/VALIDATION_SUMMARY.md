# Verification summary (full Colab run)

## Core comparisons
| Audit | Observed |
|---|---:|
| Model fits completed | 220 / 220 |
| Matched archived-reference country–outcome pairs | 22 |
| Peak lags within ±1 week (six-week linear DLNM) | 22 / 22 |
| Identical cumulative-association sign | 22 / 22 |
| Exact peak-lag matches | 18 / 22 |
| Median lag-profile correlation | 0.952 |
| Median absolute cumulative difference | approximately 0.031 standardized units |
| Differences exceeding descriptive 0.10 threshold | 4 / 22 |

The four larger cumulative differences are Malaysia occupancy (+0.148), Spain admissions (+0.122), United States admissions (−0.262), and United States occupancy (+0.204). These are descriptive comparisons, not formal joint tests.

## Numerical reproducibility
The archived seven-position original coefficients are the comparison anchor. Fresh state-space primary-model refits can converge to slightly different numerical solutions; the largest noted coefficient discrepancy in the Colab audit was approximately 0.054. A separate US admissions multistart audit improved the likelihood and showed the highest-likelihood recovered fit retained lag zero as the peak, but did not establish universal optimizer uniqueness.

## Extended specifications
Alternative knot placement, four- and eight-week lag horizons, nonlinear-exposure splines and exploratory fixed penalization were fitted. **Cumulative effects across unequal lag horizons are not directly comparable as estimates of the same cumulative window.** Nonlinear fits can improve AIC without stable magnitudes or calibrated inferential coverage.

## Critical caveats
This is post hoc sensitivity evidence, not the primary known-target simulation. It does not establish 95% coverage of DLNM intervals, clinically identified patient delays, spatial dependence, operational forecast lead time or causal mechanisms. The complete results archive, relevant CSVs, figures and environment/checksum record still require a verified public deposit; model counts above are sourced from the completed Colab run, not independently recomputed from public files in this branch.
