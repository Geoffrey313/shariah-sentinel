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
| cogsq | True | 27468 | 64.06 |
| xrdq | True | 0 | 0.0 |
| xsgaq | True | 27123 | 63.26 |
| invtq | True | 24813 | 57.87 |
| oibdpq | True | 26416 | 61.61 |
| revtq | True | 36215 | 84.46 |
| dlttq | True | 32190 | 75.07 |
| dlcq | True | 35594 | 83.01 |
| cheq | True | 39670 | 92.52 |
| iditq | True | 27188 | 63.41 |

## Scoreboard

| family | method | entity_type | mean_auc | mean_detection_rate | mean_fpr | median_cost | p90_cost |
| --- | --- | --- | --- | --- | --- | --- | --- |
| family1 | correlated_gaussian | composite | 0.49 | 0.0398 | 0.0997 |  |  |
| family2 | abn_disx_full | composite |  |  | 0.0997 |  |  |
| family2 | abn_disx_full | detector |  |  | 0.1706 |  |  |
| family2 | abn_disx_partial | composite | 0.4916 | 0.0754 | 0.1017 |  |  |
| family2 | abn_disx_partial | detector | 0.4907 | 0.1546 | 0.1718 |  |  |
| family2 | abn_prod_full | composite | 0.4877 | 0.064 | 0.1019 |  |  |
| family2 | abn_prod_full | detector | 0.4882 | 0.1523 | 0.1716 |  |  |
| family2 | benford_m3_v2 | composite | 0.7154 | 0.2599 | 0.0924 |  |  |
| family2 | benford_m3_v2 | detector | 0.5948 | 0.308 | 0.1627 |  |  |
| family2 | interstatement_m4_v2 | composite | 0.6159 | 0.1316 | 0.0992 |  |  |
| family2 | interstatement_m4_v2 | detector | 0.5826 | 0.2774 | 0.1655 |  |  |
| family2 | m5_cod_break | composite | 0.5301 | 0.106 | 0.0998 |  |  |
| family2 | m5_cod_break | detector | 0.5026 | 0.164 | 0.1707 |  |  |
| family2 | m6_seasonal | composite | 0.489 | 0.0674 | 0.1013 |  |  |
| family2 | m6_seasonal | detector | 0.4997 | 0.1664 | 0.1725 |  |  |
| family2 | temporal_spike_m2b | composite | 0.5407 | 0.1185 | 0.0995 |  |  |
| family2 | temporal_spike_m2b | detector | 0.4931 | 0.1873 | 0.1714 |  |  |
| family2 | threshold_clustering_m1_v2 | composite | 0.5522 | 0.0779 | 0.0963 |  |  |
| family2 | threshold_clustering_m1_v2 | detector | 0.5469 | 0.207 | 0.1636 |  |  |
| family4 | anoshift_correlated_gaussian | composite | 0.5159 | 0.0607 | 0.0363 |  |  |
| family4 | anoshift_threshold_clustering_m1_v2 | composite |  |  | 0.0997 |  |  |
| family3 | adversarial_evasion | row_attack |  | 0.34 |  | 0.0605 | 0.1425 |