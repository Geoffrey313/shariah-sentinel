# Robustness Benchmark

This benchmark complements `phase5_injection`.

- `family1_correlated_gaussian.csv`: covariance-aware mean-shift benchmark.
- `family2_threshold_clustering.csv`: raw-data manipulation benchmark (`M1_v2`, `M2b` temporal spike, `M3_v2`, `M4_v2`, REM-style variants when available).
- `family3_adversarial_evasion.csv`: Shariah-targeted local evasion benchmark on the configured raw variables (`revtq`, `iditq`, `nopiq`, `xintq`, `niq`, `oibdpq`, `oancfq`, `ibq`, `atq`, `dlttq`, `dlcq`, `ltq`, `cheq`, `actq`, `lctq`, `rectq`, `invtq`, `xsgaq`, `ppentq`).
- `family4_anoshift.csv`: temporal transfer benchmark using a fixed global calibration on the honest reference sample, then near/far evaluation over time.
- `coverage_key_variables.csv`: current panel coverage of raw variables needed by the realistic manipulations.
- `benchmark_scoreboard.csv`: compact mean metrics by family and method.

The benchmark keeps `phase5_injection` as a faster z-score-level baseline.

Interpretation notes:
- `M2b` is a temporal spike / gap mechanism, not a literal ABN_CFO implementation.
- `ABN_PROD` and `ABN_DISX` probe REM-style manipulations that may bypass the Shariah-ratio layer and therefore stress non-SAC detector dimensions.
- `Family 3` does not test full general evasion over all accounting variables; it tests whether RED cases can be weakened through Shariah-targeted local raw-variable adjustments.
- `Family 4` is currently reported as a fixed-calibration temporal drift stress test rather than the stricter IID-only transfer protocol from the tex.

## Key Variable Coverage

| column | present_in_panel | n_nonnull | pct_nonnull |
| --- | --- | --- | --- |
| cogsq | True | 37447 | 64.53 |
| xrdq | True | 0 | 0.0 |
| xsgaq | True | 36200 | 62.38 |
| invtq | True | 39317 | 67.75 |
| oibdpq | True | 36420 | 62.76 |
| revtq | True | 50534 | 87.08 |
| dlttq | True | 48045 | 82.79 |
| dlcq | True | 50479 | 86.98 |
| cheq | True | 56881 | 98.02 |
| iditq | True | 44698 | 77.02 |

## Scoreboard

| family | method | entity_type | mean_auc | mean_detection_rate | mean_fpr | median_cost | p90_cost |
| --- | --- | --- | --- | --- | --- | --- | --- |
| family1 | correlated_gaussian | composite | 0.5563 | 0.0704 | 0.0931 |  |  |
| family2 | abn_disx_full | composite |  |  | 0.093 |  |  |
| family2 | abn_disx_full | detector |  |  | 0.1409 |  |  |
| family2 | abn_disx_partial | composite | 0.4994 | 0.072 | 0.0943 |  |  |
| family2 | abn_disx_partial | detector | 0.4951 | 0.1354 | 0.1411 |  |  |
| family2 | abn_prod_full | composite | 0.495 | 0.0642 | 0.0947 |  |  |
| family2 | abn_prod_full | detector | 0.4919 | 0.1325 | 0.1413 |  |  |
| family2 | benford_m3_v2 | composite | 0.7195 | 0.3162 | 0.0807 |  |  |
| family2 | benford_m3_v2 | detector | 0.5882 | 0.2791 | 0.1342 |  |  |
| family2 | interstatement_m4_v2 | composite | 0.6087 | 0.1071 | 0.0911 |  |  |
| family2 | interstatement_m4_v2 | detector | 0.5783 | 0.2192 | 0.1367 |  |  |
| family2 | m5_cod_break | composite | 0.5196 | 0.0951 | 0.0929 |  |  |
| family2 | m5_cod_break | detector | 0.5184 | 0.1472 | 0.1409 |  |  |
| family2 | m6_seasonal | composite | 0.4853 | 0.0803 | 0.0933 |  |  |
| family2 | m6_seasonal | detector | 0.5001 | 0.1315 | 0.1415 |  |  |
| family2 | temporal_spike_m2b | composite | 0.5451 | 0.1122 | 0.093 |  |  |
| family2 | temporal_spike_m2b | detector | 0.5131 | 0.1611 | 0.141 |  |  |
| family2 | threshold_clustering_m1_v2 | composite | 0.6136 | 0.1286 | 0.0776 |  |  |
| family2 | threshold_clustering_m1_v2 | detector | 0.5596 | 0.2124 | 0.1349 |  |  |
| family4 | anoshift_correlated_gaussian | composite | 0.4947 | 0.0376 | 0.0412 |  |  |
| family4 | anoshift_threshold_clustering_m1_v2 | composite |  |  | 0.093 |  |  |
| family3 | adversarial_evasion | row_attack |  | 0.34 |  | 0.0723 | 0.1537 |