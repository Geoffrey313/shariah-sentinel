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
| cogsq | True | 12034 | 62.37 |
| xrdq | True | 0 | 0.0 |
| xsgaq | True | 11980 | 62.09 |
| invtq | True | 11725 | 60.77 |
| oibdpq | True | 11955 | 61.96 |
| revtq | True | 17204 | 89.16 |
| dlttq | True | 14118 | 73.17 |
| dlcq | True | 14531 | 75.31 |
| cheq | True | 18975 | 98.34 |
| iditq | True | 7607 | 39.42 |

## Scoreboard

| family | method | entity_type | mean_auc | mean_detection_rate | mean_fpr | median_cost | p90_cost |
| --- | --- | --- | --- | --- | --- | --- | --- |
| family1 | correlated_gaussian | composite | 0.5502 | 0.0413 | 0.0942 |  |  |
| family2 | abn_disx_full | composite |  |  | 0.0941 |  |  |
| family2 | abn_disx_full | detector |  |  | 0.1369 |  |  |
| family2 | abn_disx_partial | composite | 0.485 | 0.0486 | 0.097 |  |  |
| family2 | abn_disx_partial | detector | 0.4883 | 0.1165 | 0.1382 |  |  |
| family2 | abn_prod_full | composite | 0.4803 | 0.0451 | 0.0971 |  |  |
| family2 | abn_prod_full | detector | 0.482 | 0.11 | 0.1385 |  |  |
| family2 | benford_m3_v2 | composite | 0.7906 | 0.3787 | 0.0772 |  |  |
| family2 | benford_m3_v2 | detector | 0.669 | 0.3487 | 0.1253 |  |  |
| family2 | interstatement_m4_v2 | composite | 0.6554 | 0.1644 | 0.0943 |  |  |
| family2 | interstatement_m4_v2 | detector | 0.6016 | 0.2537 | 0.1338 |  |  |
| family2 | m5_cod_break | composite | 0.5024 | 0.0766 | 0.0941 |  |  |
| family2 | m5_cod_break | detector | 0.5192 | 0.1311 | 0.1369 |  |  |
| family2 | m6_seasonal | composite | 0.4953 | 0.0915 | 0.0947 |  |  |
| family2 | m6_seasonal | detector | 0.4997 | 0.1245 | 0.1379 |  |  |
| family2 | temporal_spike_m2b | composite | 0.5155 | 0.0776 | 0.0945 |  |  |
| family2 | temporal_spike_m2b | detector | 0.5083 | 0.1312 | 0.1373 |  |  |
| family2 | threshold_clustering_m1_v2 | composite | 0.6557 | 0.128 | 0.0754 |  |  |
| family2 | threshold_clustering_m1_v2 | detector | 0.6004 | 0.2193 | 0.1276 |  |  |
| family4 | anoshift_correlated_gaussian | composite | 0.5181 | 0.0549 | 0.0416 |  |  |
| family4 | anoshift_threshold_clustering_m1_v2 | composite |  |  | 0.0941 |  |  |
| family3 | adversarial_evasion | row_attack |  | 0.3778 |  | 0.0473 | 0.1157 |