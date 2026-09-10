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
| cogsq | True | 4578 | 39.81 |
| xrdq | True | 0 | 0.0 |
| xsgaq | True | 4517 | 39.28 |
| invtq | True | 4977 | 43.28 |
| oibdpq | True | 4560 | 39.66 |
| revtq | True | 10563 | 91.86 |
| dlttq | True | 8604 | 74.82 |
| dlcq | True | 8870 | 77.14 |
| cheq | True | 11277 | 98.07 |
| iditq | True | 7603 | 66.12 |

## Scoreboard

| family | method | entity_type | mean_auc | mean_detection_rate | mean_fpr | median_cost | p90_cost |
| --- | --- | --- | --- | --- | --- | --- | --- |
| family1 | correlated_gaussian | composite | 0.4515 | 0.0584 | 0.1892 |  |  |
| family2 | abn_disx_full | composite |  |  | 0.1892 |  |  |
| family2 | abn_disx_full | detector |  |  | 0.1733 |  |  |
| family2 | abn_disx_partial | composite | 0.4866 | 0.1364 | 0.1935 |  |  |
| family2 | abn_disx_partial | detector | 0.4892 | 0.155 | 0.1744 |  |  |
| family2 | abn_prod_full | composite | 0.4784 | 0.1135 | 0.194 |  |  |
| family2 | abn_prod_full | detector | 0.4913 | 0.1492 | 0.1744 |  |  |
| family2 | benford_m3_v2 | composite | 0.7463 | 0.4696 | 0.1724 |  |  |
| family2 | benford_m3_v2 | detector | 0.6266 | 0.3894 | 0.1641 |  |  |
| family2 | interstatement_m4_v2 | composite | 0.6104 | 0.2434 | 0.1813 |  |  |
| family2 | interstatement_m4_v2 | detector | 0.5821 | 0.2636 | 0.1691 |  |  |
| family2 | m5_cod_break | composite | 0.5466 | 0.1558 | 0.1891 |  |  |
| family2 | m5_cod_break | detector | 0.5333 | 0.2002 | 0.1732 |  |  |
| family2 | m6_seasonal | composite | 0.4333 | 0.1341 | 0.1897 |  |  |
| family2 | m6_seasonal | detector | 0.4724 | 0.1322 | 0.174 |  |  |
| family2 | temporal_spike_m2b | composite | 0.5032 | 0.1858 | 0.189 |  |  |
| family2 | temporal_spike_m2b | detector | 0.4967 | 0.1618 | 0.1737 |  |  |
| family2 | threshold_clustering_m1_v2 | composite | 0.5704 | 0.1571 | 0.1698 |  |  |
| family2 | threshold_clustering_m1_v2 | detector | 0.553 | 0.2506 | 0.1718 |  |  |
| family4 | anoshift_correlated_gaussian | composite | 0.5155 | 0.0866 | 0.0507 |  |  |
| family4 | anoshift_threshold_clustering_m1_v2 | composite |  |  | 0.1892 |  |  |
| family3 | adversarial_evasion | row_attack |  | 0.24 |  | 0.0479 | 0.1035 |