"""Pydantic configuration for the scoring analysis pipeline.

Every threshold used by Phase 0..7 lives here — the modules must not hardcode
numerical constants of their own. Each field documents the framework section
(``docs/math-framework/framework.md``) it derives from so a reviewer can trace
a value back to the methodology.

The settings are grouped by concern:

- :class:`ReferenceSampleSettings` — Phase 0 (C / C_s / peer groups).
- :class:`DetectorPreconditionSettings` — Phase 3 row-level preconditions
  per detector, framework §9.2.
- :class:`PanelSchemaSettings` — column names the pipeline expects on the
  panel parquet (explicit so that a rename in ``panel_creation`` surfaces as
  a config change rather than a silent mismatch).
- :class:`OutputLayoutSettings` — where JSON / parquet / CSV deliverables
  are written.
- :class:`AnalysisSettings` — top-level composite, the only object callers
  normally instantiate.
"""
from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field, model_validator


# ─────────────────────────────────────────────────────────────────────────────
# Panel schema
# ─────────────────────────────────────────────────────────────────────────────
class PanelSchemaSettings(BaseModel):
    """Column names read from the panel parquet.

    Set once in ``panel_creation`` and mirrored here so that any rename is an
    explicit config edit.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    firm_id: str = Field(default="gvkey", description="Firm identifier column.")
    quarter: str = Field(
        default="datacqtr",
        description="Calendar quarter label (format ``YYYY-Qn``).",
    )
    fiscal_quarter_number: str = Field(
        default="fqtr",
        description="Fiscal quarter number 1..4.",
    )
    sac_shariah: str = Field(
        default="sac_shariah",
        description="Legacy MYS authority label — 1 if SC Malaysia flagged "
        "the firm as Shariah-compliant for that row, 0 otherwise. Panels "
        "built after the UAE slice carry the generic tri-state column "
        "instead (see ``compliance_shariah``); Phase 0 prefers that column "
        "when present and falls back to this one, which keeps every "
        "historical MYS artefact readable.",
    )
    compliance_shariah: str = Field(
        default="compliance_shariah",
        description="Generic tri-state authority label — 1 = compliant per "
        "the country authority list, 0 = explicitly non-compliant (only "
        "negative-list authorities like SC Malaysia emit this), NA = "
        "unknown / not covered (positive-only lists like DFM never emit "
        "0). Reference sample C selects strictly ``== 1``, never ``!= 0``.",
    )
    methodology_key: str = Field(
        default="methodology_key",
        description="Methodology identifier stamped by the panel builder "
        "(``sac_my``, ``dfm_uae`` — mirrors "
        "``server/panel/methodology.py``). Drives per-country threshold "
        "resolution in Phase 0 / Phase 7 / detectors; absent on legacy "
        "MYS panels, which resolve to SAC.",
    )
    sector: str = Field(
        default="sector",
        description="Sector label carried from SC Malaysia listing; used for "
        "``C_s`` stratification and z7 peer grouping.",
    )
    synthetic_row: str = Field(
        default="_synthetic_row",
        description="1 for rows added by quarterly grid expansion (no raw "
        "Compustat observation).",
    )
    balance_sheet_any_ffilled: str = Field(
        default="bs_any_ffilled",
        description="1 if any of ``{atq, dlttq, dlcq, ltq, cheq}`` was "
        "forward-filled on this row.",
    )
    clean_sample: str = Field(
        default="clean_sample",
        description="Convenience flag written by ``add_clean_sample_flag``: "
        "non-synthetic, no bs ffill, ratios within sanity bounds.",
    )


# ─────────────────────────────────────────────────────────────────────────────
# Phase 0 — reference sample curation
# ─────────────────────────────────────────────────────────────────────────────
class ReferenceSampleSettings(BaseModel):
    """Inclusion rules and size thresholds for the honest reference samples.

    The framework (§9.2 D3, §9.2 D5, §9.2 D7) requires three reference
    populations:

    - ``C`` — global presumed-honest sample, used for D3 empirical PIT and for
      the non-parametric bootstrap (framework §3.3).
    - ``C_s`` — sector-stratified slices of ``C``, used for D5 cross-statement
      residual standardisation and D7 peer moments.
    - Peer groups — the subset of ``C`` sharing a firm's peer key, used for
      D7 Hotelling's T² test.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    require_sac_compliant: bool = Field(
        default=True,
        description="Include a row in ``C`` only when ``sac_shariah == 1``. "
        "The SC Malaysia label is the authority ground truth.",
    )
    require_clean_sample: bool = Field(
        default=True,
        description="Additionally require ``clean_sample == 1`` (no synthetic "
        "row, no ffilled balance sheet, ratios within sanity bounds). This "
        "replaces the ``not delisted ∧ clean_sample == 1`` clause of "
        "analysis_plan_v2.md — no delisting flag exists today.",
    )
    require_ratio_compliance: bool = Field(
        default=True,
        description="When True, exclude from C any SAC-labeled row whose "
        "recomputed ratios exceed the official SAC thresholds "
        "(ratio_debt_adj > 0.33, ratio_cash_adj > 0.33, "
        "ratio_income > 0.05). Tightens the null distribution by "
        "removing the ~13% of C rows that violate thresholds due to "
        "data-source divergence between Compustat and Bursa Malaysia. "
        "Enabled via --strict-compliance.",
    )

    min_global_size: int = Field(
        default=30,
        ge=1,
        description="Minimum ``|C|``. Framework §9.2 D3 requires at least 30 "
        "rows for a stable empirical PIT [Thomas 1989].",
    )
    min_sector_size: int = Field(
        default=27,
        ge=1,
        description="Minimum ``|C_s|`` per sector. Framework §9.2 D5 requires "
        "``|C_s| ≥ 3q`` for stable covariance estimation. With the enriched "
        "z5 detector we now use ``q = 9`` cross-statement relations, hence 27.",
    )
    peer_group_min_size: int = Field(
        default=9,
        ge=1,
        description="Minimum ``N_peer``. Framework §9.2 D7 recommends "
        "``N_peer ≥ 3p`` where ``p`` is the number of ratios in the peer "
        "Mahalanobis distance. Default ``p = 3`` (debt/cash/income SAC "
        "ratios) gives 9.",
    )

    peer_group_column: str = Field(
        default="sector",
        description="Panel column used to group firms into peer sets for "
        "D7. Currently the SC Malaysia sector label.",
    )
    sector_column: str = Field(
        default="sector",
        description="Panel column used to stratify ``C`` into ``C_s``.",
    )


# ─────────────────────────────────────────────────────────────────────────────
# Phase 3 — per-detector preconditions (framework §9.2)
# ─────────────────────────────────────────────────────────────────────────────
class BenfordPreconditionSettings(BaseModel):
    """z1 Benford precondition — framework §9.2 D1.

    Supports two calibration paths:

    - ``chi2_asymptotic`` — original asymptotic ``χ²₈`` PIT. Requires
      ``N_fig ≥ min_figures`` (109 for the standard ``E_d ≥ 5`` rule).
    - ``monte_carlo`` — Monte-Carlo null: draw multinomial samples from
      the Benford probabilities at the same ``N_fig``, build an empirical
      CDF, PIT through that. Drops the ``E_d ≥ 5`` requirement → can score
      rows with as few as ``min_figures_mc`` figures.
    - ``auto`` — picks MC when ``N_fig < chi2_minimum_figures`` and χ²
      otherwise. Lets a single run score small and large windows uniformly.

    Sprint 1 originally found 0% z1 coverage because the 4-quarter × 22-
    column pool never reached 109 figures. Sprint 4 confirmed no other
    detector catches ``round_number`` manipulation well (z2 peaks at 64%
    @ δ=2). ``auto`` mode + an 8-quarter window brings z1 online.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    min_figures: int = Field(
        default=109,
        ge=1,
        description="Pooled-figure floor under ``chi2_asymptotic`` — derived "
        "from the ``E_d ≥ 5`` rule with ``d = 9`` binding: "
        "``109 = ceil(5 / log10(10/9))``.",
    )
    min_figures_mc: int = Field(
        default=30,
        ge=9,
        description="Pooled-figure floor under ``monte_carlo``. MC drops the "
        "``E_d ≥ 5`` constraint but still needs enough digits for the Benford "
        "departure to be distinguishable from sampling noise. 30 is a common "
        "practical floor for digit tests; the config can go as low as 9 "
        "(one per digit category).",
    )
    pooling_window_quarters: int = Field(
        default=8,
        ge=1,
        description="Rolling window over which figures are pooled for one "
        "firm-quarter score. Widened from 4 → 8 after Sprint 1 — 4 × 22 "
        "columns max = 88 figures < 109 floor (structural dead-zone). "
        "8 × 22 = 176 max, ~60–120 typical.",
    )
    calibration_mode: str = Field(
        default="auto",
        description="``chi2_asymptotic`` / ``monte_carlo`` / ``auto`` "
        "(see class docstring).",
    )
    chi2_minimum_figures: int = Field(
        default=109,
        ge=1,
        description="In ``auto`` mode, rows with ``N_fig ≥ this`` use χ² "
        "and rows below use MC. Kept separate from ``min_figures`` so the "
        "two thresholds can move independently.",
    )
    monte_carlo_replicates: int = Field(
        default=20_000,
        ge=1_000,
        description="Number of multinomial draws used to build the MC null "
        "for each unique ``N_fig``. The CDF is cached per ``N_fig`` so this "
        "cost is amortised across all rows sharing that figure count.",
    )
    monte_carlo_seed: int = Field(
        default=2,
        ge=0,
        description="Seed for MC draws — distinct from bootstrap / injection "
        "seeds so the three sources of randomness are independent.",
    )

    @model_validator(mode="after")
    def _check_mode(self) -> "BenfordPreconditionSettings":
        allowed = {"chi2_asymptotic", "monte_carlo", "auto"}
        if self.calibration_mode not in allowed:
            raise ValueError(
                f"calibration_mode {self.calibration_mode!r} not in {allowed}."
            )
        if self.min_figures_mc > self.min_figures:
            raise ValueError(
                f"min_figures_mc ({self.min_figures_mc}) should be ≤ min_figures "
                f"({self.min_figures}); MC relaxes the asymptotic floor."
            )
        return self


class ZipfPreconditionSettings(BaseModel):
    """z2 Zipf precondition — framework §9.2 D2."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    min_points: int = Field(
        default=8,
        ge=3,
        description="Minimum positive magnitudes required to fit the rank-size "
        "OLS regression (``n - 2`` residual degrees of freedom).",
    )
    pooling_window_quarters: int = Field(
        default=4,
        ge=1,
        description="Rolling firm-quarter window for pooled magnitudes.",
    )


class MScorePreconditionSettings(BaseModel):
    """z3 M-Score precondition — framework §9.2 D3."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    min_reference_size: int = Field(
        default=30,
        ge=1,
        description="Sector-level minimum ``|C|`` required for the empirical "
        "PIT (Thomas 1989).",
    )
    require_same_sector_reference: bool = Field(
        default=True,
        description="Framework §9.2 D3 requires ``C`` to be sectorially "
        "homogeneous — mark rows of sectors with ``|C_s| < min_reference_size``"
        " as invalid.",
    )


class ThresholdProximityPreconditionSettings(BaseModel):
    """z4 threshold-proximity precondition — framework §9.2 D4.

    The minimum history length follows the §9.2 D4 power formula::

        Q_min ≈ (z_{1-α} + z_{1-β})^2 / (3 (1 - δ_m)^2)

    All three parameters are exposed so the minimum moves with the declared
    power target rather than being baked in.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    significance_alpha: float = Field(
        default=0.05,
        gt=0.0,
        lt=1.0,
        description="Type-I error ``α`` used in the power computation.",
    )
    power_target: float = Field(
        default=0.80,
        gt=0.0,
        lt=1.0,
        description="Target power ``1 - β``.",
    )
    detectable_relative_shift: float = Field(
        default=0.05,
        gt=0.0,
        lt=1.0,
        description="Minimum detectable relative shift ``δ_m`` (fraction of "
        "the threshold ``τ`` the manipulator stays below).",
    )
    absolute_floor: int = Field(
        default=4,
        ge=1,
        description="Hard floor applied regardless of the power formula — the "
        "OLS fit in d4 cannot start with fewer quarters than this.",
    )


class CrossStatementPreconditionSettings(BaseModel):
    """z5 cross-statement precondition — framework §9.2 D5."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    residual_count: int = Field(
        default=9,
        ge=1,
        description="Number of cross-statement residuals ``q`` (R1..R5 and "
        "their active proxies as "
        "instantiated in ``detectors/d5_coherence.py``).",
    )
    sector_reference_multiplier: int = Field(
        default=3,
        ge=1,
        description="Multiplier ``m`` in ``|C_s| ≥ m q``. Framework §9.2 D5 "
        "uses 3 for stable covariance estimation.",
    )

    @property
    def min_sector_reference(self) -> int:
        """``|C_s|`` floor derived from ``m q``."""
        return self.sector_reference_multiplier * self.residual_count


class TemporalPreconditionSettings(BaseModel):
    """z6 temporal-consistency precondition — framework §9.2 D6."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    min_history_quarters: int = Field(
        default=8,
        ge=2,
        description="Number of past quarters required to estimate moving "
        "mean/std of Δm_i. Framework §9.2 D6 proposes 8.",
    )


class PeerPreconditionSettings(BaseModel):
    """z7 peer-consistency precondition — framework §9.2 D7."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    ratio_count: int = Field(
        default=3,
        ge=1,
        description="Number of ratios ``p`` entering the peer Mahalanobis "
        "distance (debt / cash / income).",
    )
    peer_size_multiplier: int = Field(
        default=3,
        ge=1,
        description="Multiplier in ``N_peer ≥ m p``. Framework §9.2 D7 uses 3 "
        "so the ``χ²_p`` approximation of Hotelling T² is acceptable.",
    )

    @property
    def min_peer_size(self) -> int:
        """``N_peer`` floor derived from ``m p``."""
        return self.peer_size_multiplier * self.ratio_count


class DetectorPreconditionSettings(BaseModel):
    """Row-level preconditions for detectors z1..z7, framework §9.2."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    benford: BenfordPreconditionSettings = Field(default_factory=BenfordPreconditionSettings)
    zipf: ZipfPreconditionSettings = Field(default_factory=ZipfPreconditionSettings)
    mscore: MScorePreconditionSettings = Field(default_factory=MScorePreconditionSettings)
    threshold: ThresholdProximityPreconditionSettings = Field(
        default_factory=ThresholdProximityPreconditionSettings
    )
    cross_statement: CrossStatementPreconditionSettings = Field(
        default_factory=CrossStatementPreconditionSettings
    )
    temporal: TemporalPreconditionSettings = Field(default_factory=TemporalPreconditionSettings)
    peer: PeerPreconditionSettings = Field(default_factory=PeerPreconditionSettings)

    benford_value_columns: tuple[str, ...] = Field(
        default=(
            "revtq", "iditq", "nopiq", "xintq", "xintdy", "niq", "oibdpq",
            "atq", "dlttq", "dlcq", "ltq", "cheq", "chq", "chsq",
            "actq", "lctq", "rectq", "invtq", "rectrq", "dd1q", "dlsq", "notesq",
        ),
        description="Monetary columns pooled for Benford / Zipf. Must match "
        "``detectors/d1_benford.DEFAULT_BENFORD_VALUE_COLUMNS``.",
    )


# ─────────────────────────────────────────────────────────────────────────────
# Output layout
# ─────────────────────────────────────────────────────────────────────────────
class OutputLayoutSettings(BaseModel):
    """Where Sprint 1 deliverables are written.

    Mirrors the ``panel_creation`` convention: one folder per country under
    ``outputs/`` and one subfolder per analysis phase.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    root: Path = Field(
        default=Path("outputs"),
        description="Pipeline-wide output root.",
    )
    country_code_lower: str = Field(
        default="mys",
        description="Lower-case country code subfolder (matches panel layout).",
    )
    panel_relative: Path = Field(
        default=Path("panel/{country}/compustat_quarterly.parquet"),
        description="Panel parquet relative to ``root``. ``{country}`` is "
        "replaced with ``country_code_lower``.",
    )
    scores_relative: Path = Field(
        default=Path("scores/{country}"),
        description="Root of scoring deliverables. Each phase writes a "
        "subfolder under this path.",
    )
    phase0_subdir: str = Field(default="phase0_reference_sample")
    phase3_subdir: str = Field(default="phase3_detector_qc")
    zscores_subdir: str = Field(default="zscores")
    phase1_subdir: str = Field(default="phase1_null_calibration")
    phase2_subdir: str = Field(default="phase2_dependence")
    phase4_subdir: str = Field(default="phase4_composites")
    phase4b_subdir: str = Field(default="phase4b_confidence")
    phase5_subdir: str = Field(default="phase5_injection")
    phase6_subdir: str = Field(default="phase6_robustness")
    phase7_subdir: str = Field(default="phase7_fdr")
    robustness_benchmark_subdir: str = Field(default="robustness_benchmark")
    figures_subdir: str = Field(default="figures")
    dashboard_subdir: str = Field(default="dashboard")

    def panel_path(self) -> Path:
        """Resolve the panel parquet path for the current country."""
        return self.root / Path(str(self.panel_relative).format(country=self.country_code_lower))

    def phase0_dir(self) -> Path:
        """Folder for Phase 0 deliverables."""
        return self.root / Path(str(self.scores_relative).format(country=self.country_code_lower)) / self.phase0_subdir

    def phase3_dir(self) -> Path:
        """Folder for Phase 3 deliverables."""
        return self.root / Path(str(self.scores_relative).format(country=self.country_code_lower)) / self.phase3_subdir

    def zscores_dir(self) -> Path:
        """Folder for cached per-detector z-score parquet."""
        return self.root / Path(str(self.scores_relative).format(country=self.country_code_lower)) / self.zscores_subdir

    def phase1_dir(self) -> Path:
        """Folder for Phase 1 (null calibration) deliverables."""
        return self.root / Path(str(self.scores_relative).format(country=self.country_code_lower)) / self.phase1_subdir

    def phase2_dir(self) -> Path:
        """Folder for Phase 2 (dependence) deliverables."""
        return self.root / Path(str(self.scores_relative).format(country=self.country_code_lower)) / self.phase2_subdir

    def phase4_dir(self) -> Path:
        """Folder for Phase 4 (composites + bootstrap) deliverables."""
        return self.root / Path(str(self.scores_relative).format(country=self.country_code_lower)) / self.phase4_subdir

    def phase4b_dir(self) -> Path:
        """Folder for Phase 4b (confidence qualifier κ) deliverables."""
        return self.root / Path(str(self.scores_relative).format(country=self.country_code_lower)) / self.phase4b_subdir

    def phase5_dir(self) -> Path:
        """Folder for Phase 5 (injection study) deliverables."""
        return self.root / Path(str(self.scores_relative).format(country=self.country_code_lower)) / self.phase5_subdir

    def phase6_dir(self) -> Path:
        """Folder for Phase 6 (robustness) deliverables."""
        return self.root / Path(str(self.scores_relative).format(country=self.country_code_lower)) / self.phase6_subdir

    def phase7_dir(self) -> Path:
        """Folder for Phase 7 (FDR) deliverables."""
        return self.root / Path(str(self.scores_relative).format(country=self.country_code_lower)) / self.phase7_subdir

    def robustness_benchmark_dir(self) -> Path:
        """Folder for cross-family robustness benchmark deliverables."""
        return self.root / Path(str(self.scores_relative).format(country=self.country_code_lower)) / self.robustness_benchmark_subdir

    def figures_dir(self) -> Path:
        """Folder for paper figures (PNG / PDF)."""
        return self.root / Path(str(self.scores_relative).format(country=self.country_code_lower)) / self.figures_subdir

    def dashboard_dir(self) -> Path:
        """Folder for the analyst dashboard HTML bundle."""
        return self.root / Path(str(self.scores_relative).format(country=self.country_code_lower)) / self.dashboard_subdir


# ─────────────────────────────────────────────────────────────────────────────
# Sprint 2 — Phase 1 null calibration
# ─────────────────────────────────────────────────────────────────────────────
class CalibrationSettings(BaseModel):
    """Phase 1 — per-detector null calibration on reference sample ``C``.

    The framework (§9.1) claims each ``z_j`` is marginally ``N(0, 1)`` under
    ``H_0`` once the PIT is correctly applied. Phase 1 verifies this
    empirically on ``C`` and flags any detector whose parametric null is
    rejected so Phase 4 can fall back to non-parametric bootstrap on that
    detector.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    target_mean: float = Field(
        default=0.0,
        description="Expected ``E[z_j]`` under ``H_0`` (framework §9.1).",
    )
    target_std: float = Field(
        default=1.0,
        gt=0.0,
        description="Expected ``sd[z_j]`` under ``H_0``.",
    )
    ks_alpha: float = Field(
        default=0.01,
        gt=0.0,
        lt=1.0,
        description="Significance level for the KS vs ``N(0, 1)`` test. A "
        "detector with p-value below this is flagged as mis-calibrated and "
        "forced to non-parametric bootstrap in Phase 4.",
    )
    anderson_alpha: float = Field(
        default=0.01,
        gt=0.0,
        lt=1.0,
        description="Significance level for the Anderson-Darling test.",
    )
    bias_tolerance: float = Field(
        default=0.1,
        ge=0.0,
        description="Maximum ``|mean - target_mean|`` accepted before the "
        "detector is flagged as biased.",
    )
    variance_ratio_tolerance: float = Field(
        default=0.2,
        ge=0.0,
        description="Maximum ``|variance - target_std²|`` / ``target_std²`` "
        "accepted before the detector is flagged as mis-scaled.",
    )
    qq_quantiles: int = Field(
        default=99,
        ge=10,
        description="Number of evenly-spaced quantiles stored for QQ-plot "
        "reconstruction in the deliverable JSON.",
    )
    min_subsample_size: int = Field(
        default=30,
        ge=1,
        description="Minimum ``|C|`` on a year/sector subsample before its "
        "moments are reported. Matches the D3 floor.",
    )
    year_column: str = Field(
        default="fyearq",
        description="Panel column used for year-level sub-sample calibration. "
        "``fyearq`` is preferred over calendar year derived from ``datadate`` "
        "to stay aligned with Compustat fiscal reporting.",
    )


# ─────────────────────────────────────────────────────────────────────────────
# Sprint 2 — Phase 2 Σ̂ estimation
# ─────────────────────────────────────────────────────────────────────────────
class DependenceSettings(BaseModel):
    """Phase 2 — covariance estimation and dependence structure.

    Framework §4.2 designates Ledoit-Wolf shrinkage as the production ``Σ̂``;
    MCD is kept as a robustness sensitivity check. PCA flags quasi-singular
    covariance structures that would destabilise ``Z²_Mah`` (§4.1).
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    min_observations: int = Field(
        default=30,
        ge=1,
        description="Minimum rows with a complete z-vector required to "
        "estimate ``Σ̂``. Matches the D3 floor — if ``C`` has fewer complete "
        "rows, Phase 2 aborts.",
    )
    shrinkage_block_if_lt: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Block the run if the Ledoit-Wolf shrinkage intensity "
        "``λ`` falls below this. ``0`` = never block (pure diagnostic).",
    )
    pca_variance_cutoff: float = Field(
        default=0.95,
        gt=0.0,
        lt=1.0,
        description="Variance share used to report effective dimensionality "
        "``K_eff`` — the smallest number of PCs whose cumulative variance "
        "share exceeds this cutoff.",
    )
    k_eff_warn_below: int = Field(
        default=5,
        ge=1,
        description="Threshold below which ``K_eff`` triggers a singularity "
        "warning (``Σ̂`` close to collinear).",
    )
    mcd_support_fraction: float | None = Field(
        default=None,
        description="Support fraction for the MCD robust covariance. "
        "``None`` delegates to sklearn's default ``(n + p + 1) / 2 / n``, "
        "which is the robustness/efficiency sweet spot recommended by "
        "[Rousseeuw & Van Driessen 1999].",
    )
    partial_corr_controls: tuple[str, ...] = Field(
        default=("sector", "fyearq"),
        description="Panel columns regressed out before partial-correlation "
        "estimation — reveals residual detector dependence net of sector/year"
        " confounds (framework §4.2 mentions this explicitly).",
    )


# ─────────────────────────────────────────────────────────────────────────────
# Sprint 2 — z-score cache
# ─────────────────────────────────────────────────────────────────────────────
class DetectorMergeRule(BaseModel):
    """One post-computation merge of correlated detectors into a single column.

    Sprint 2 + Sprint 4 findings identified z5 (cross-statement coherence) and
    z7 (peer consistency) as structurally collinear (ρ=0.82 Pearson, 0.83
    partial). Both are Mahalanobis-style scores on sector-relative ratio
    vectors. Merging them avoids double-counting the shared direction in the
    joint bootstrap and composites.

    The rule is defined by an output column name, a tuple of input detector
    names, and a reduction method:

    - ``max`` — element-wise max, picks whichever input is more anomalous on
      each row. Simple, robust to NaN, preserves anomaly direction.
    - ``mean`` — element-wise mean, appropriate for near-identical signals.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    output_name: str = Field(description="Column name of the merged detector.")
    inputs: tuple[str, ...] = Field(
        min_length=2,
        description="Raw detector names consumed by the merge.",
    )
    method: str = Field(
        default="max",
        description="Reduction: ``'max'`` or ``'mean'``.",
    )
    drop_inputs: bool = Field(
        default=True,
        description="If ``True``, the raw inputs are removed from the active "
        "battery after the merge. Set to ``False`` to keep them side-by-side "
        "for diagnostics (at the cost of a larger battery).",
    )
    empirical_pit_on_c: bool = Field(
        default=True,
        description="If ``True``, the merged column is passed through "
        "``pit_empirical(z, z_on_C)`` so it self-calibrates to mean ≈ 0 on "
        "the honest sample. Mirrors the z1 / z4 post-PIT step — the raw "
        "reduction (``max`` in particular) is right-skewed and its null on "
        "``C`` is not N(0, 1) by default.",
    )

    @model_validator(mode="after")
    def _check_method(self) -> "DetectorMergeRule":
        allowed = {"max", "mean"}
        if self.method not in allowed:
            raise ValueError(f"method {self.method!r} not in {allowed}.")
        return self


class ZScoreSettings(BaseModel):
    """Configuration for the per-detector z-score cache.

    The cache is produced once per panel build and consumed by both Phase 1
    (calibration) and Phase 2 (dependence). Keeping it separate avoids
    re-running every detector on every downstream phase.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    filename: str = Field(
        default="zscores.parquet",
        description="File name for the cached z-score parquet.",
    )
    temporal_mode: str = Field(
        default="diagonal",
        description="Mode passed to ``detect_temporal`` — ``'diagonal'`` is "
        "the phase-1 ``sum(Z_i²)`` variant (framework §9.2 D6, solution 2).",
    )
    include_detectors: tuple[str, ...] = Field(
        default=("z1", "z2", "z3", "z4", "z5", "z6", "z7", "z8"),
        description="Raw detectors to run. Ordered identically in all "
        "downstream matrices (Σ̂, correlation heatmaps). Merges (see "
        "``merge_rules``) may replace some of these in the active battery.",
    )
    merge_rules: tuple[DetectorMergeRule, ...] = Field(
        default=(
            DetectorMergeRule(
                output_name="z57",
                inputs=("z5", "z7"),
                method="max",
                drop_inputs=True,
            ),
        ),
        description="Post-computation merges applied to the raw z-score "
        "frame. Default merges z5+z7 → z57 (max) — see Sprint 2/4 findings "
        "on the structural collinearity of peer-style detectors.",
    )

    def active_detectors(self) -> tuple[str, ...]:
        """Effective detector battery after merges.

        Order: surviving raw detectors (in original order) followed by the
        merge outputs (in their registered order). This keeps the ordering
        stable across runs and matches how ``compute_zscores`` assembles
        the output frame.
        """
        dropped: set[str] = set()
        for rule in self.merge_rules:
            if rule.drop_inputs:
                dropped.update(rule.inputs)
        surviving = tuple(d for d in self.include_detectors if d not in dropped)
        merged = tuple(rule.output_name for rule in self.merge_rules)
        return surviving + merged


# ─────────────────────────────────────────────────────────────────────────────
# Sprint 3 — Phase 4 composites + joint bootstrap
# ─────────────────────────────────────────────────────────────────────────────
class CompositeSettings(BaseModel):
    """Phase 4 — composite aggregation across detectors (framework §3–§6).

    The four composites live side-by-side so each question they answer can
    be reported independently:

    - ``Z+``              — §3.2 truncated sum (sensitive to any single
      strong signal, diluted by silent detectors).
    - ``Z+_renorm``       — §6.2 dilution-free intensity on active set.
    - ``B_A``             — §6.4 breadth (share of detectors that fired).
    - ``Z²_Mah``          — §4.1 covariance-aware joint anomaly.
    - ``T_IUT``           — §5.2 unanimity test.

    All five share the same detector ordering and the same Σ̂ (when needed),
    so one joint bootstrap simultaneously calibrates all of them.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    weighting_mode: str = Field(
        default="uniform",
        description="Weight scheme for ``Z+`` / ``Z+_renorm``. ``'uniform'`` "
        "sets ``w_j = 1 / k`` on the detectors that produced any finite "
        "score on ``C``. Other modes to be added once Rémi commits to a "
        "calibration strategy (framework §8 task 4).",
    )
    active_set_threshold: float = Field(
        default=0.0,
        description="Z-score above which a detector is counted as active in "
        "``A = {j : z_j > active_set_threshold}`` (framework §6.2). The "
        "framework uses 0; kept configurable in case a stricter lower bound "
        "is needed to suppress noise.",
    )
    min_active_for_renorm: int = Field(
        default=1,
        ge=1,
        description="Minimum ``|A|`` required for ``Z+_renorm`` to be "
        "defined. Rows with no active detector return NaN.",
    )
    min_active_for_iut: int = Field(
        default=1,
        ge=1,
        description="Minimum number of detectors with a finite score "
        "required for ``T_IUT = min_j z_j`` to be defined.",
    )
    softmax_gamma: float = Field(
        default=1.0,
        gt=0.0,
        description="Temperature parameter ``γ`` used by the softmax-based "
        "composite extensions. The same value is used for observed scores, "
        "bootstrap nulls, and injection studies.",
    )


class BootstrapSettings(BaseModel):
    """Phase 4 joint bootstrap — framework §3.3, §5.3, §6.3.

    One bootstrap simultaneously feeds the null distributions of all five
    composites. The parametric variant assumes joint normality and uses
    ``Σ̂_LW`` from Phase 2; the non-parametric variant resamples complete
    z-vectors from ``C`` and therefore makes no distributional assumption.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    n_replicates: int = Field(
        default=10_000,
        ge=100,
        description="Number of bootstrap replicates ``B``. The plan of "
        "record uses 10,000; lower only for smoke tests.",
    )
    parametric: bool = Field(
        default=True,
        description="If ``True``, simulate ``z^(b) ∼ N(0, Σ̂_LW)``.",
    )
    non_parametric: bool = Field(
        default=True,
        description="If ``True``, resample complete z-vectors from ``C``.",
    )
    random_seed: int = Field(
        default=0,
        ge=0,
        description="Seed for reproducibility.",
    )
    pvalue_floor_eps: float = Field(
        default=1e-12,
        gt=0.0,
        description="Numerical floor for bootstrap p-values "
        "``p = (1 + Σ 1{sim ≥ obs}) / (B + 1)`` — kept for logging only, "
        "the formula itself already bounds ``p ≥ 1/(B+1)``.",
    )
    primary_mode: str = Field(
        default="non_parametric",
        description="Which bootstrap variant drives the persisted p-values. "
        "``'non_parametric'`` is safer when the parametric-null calibration "
        "is rejected (Phase 1 forced-bootstrap list is non-empty).",
    )


class ConfidenceSettings(BaseModel):
    """Phase 4b — confidence qualifier κ (framework §A.3).

    The qualifier turns the *shape* of the active-set z-score vector into a
    tag ``κ ∈ {targeted, mixed, systematic}``. Low entropy on the active
    set means one detector dominates (targeted manipulation); high entropy
    means the signal is spread across many detectors (systematic
    manipulation). Rémi has flagged the exact ``S_F*`` definition as TBD —
    the implementation uses the proposal in ``analysis_plan_v2.md`` and is
    easy to swap once the final definition lands.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    min_active_for_kappa: int = Field(
        default=2,
        ge=2,
        description="Minimum ``|A|`` for ``S_F*`` / ``κ`` to be defined. "
        "With a single active detector the entropy is trivially 0 and the "
        "tag uninformative — the row is better reported as ``'single'`` "
        "(see :attr:`label_single`).",
    )
    targeted_quantile_cutoff: float = Field(
        default=1.0 / 3.0,
        gt=0.0,
        lt=1.0,
        description="Rows with ``S_F*`` below this quantile of the ``C`` "
        "distribution are labelled ``'targeted'``.",
    )
    systematic_quantile_cutoff: float = Field(
        default=2.0 / 3.0,
        gt=0.0,
        lt=1.0,
        description="Rows with ``S_F*`` above this quantile of the ``C`` "
        "distribution are labelled ``'systematic'``. Between the two "
        "quantiles: ``'mixed'``.",
    )
    label_targeted: str = Field(default="targeted")
    label_mixed: str = Field(default="mixed")
    label_systematic: str = Field(default="systematic")
    label_single: str = Field(default="single")
    label_inactive: str = Field(default="inactive")

    @model_validator(mode="after")
    def _check_quantile_order(self) -> "ConfidenceSettings":
        if self.targeted_quantile_cutoff >= self.systematic_quantile_cutoff:
            raise ValueError(
                f"targeted_quantile_cutoff ({self.targeted_quantile_cutoff}) "
                f"must be < systematic_quantile_cutoff "
                f"({self.systematic_quantile_cutoff})."
            )
        return self


# ─────────────────────────────────────────────────────────────────────────────
# Sprint 4 — Phase 5 injection study
# ─────────────────────────────────────────────────────────────────────────────
class ArchetypeSettings(BaseModel):
    """One injection archetype.

    An archetype names a manipulation pattern and the detectors it is
    expected to affect. Phase 5 (first ship) injects the ``δ`` signal
    directly into the z-scores of ``affected_detectors`` on the sampled
    honest rows — this validates the framework's theoretical-power claims
    (§5.5) without the combinatorial cost of re-running every detector on
    a perturbed panel.

    Raw-data-level injection (modify Compustat fields → rebuild panel →
    rerun detectors) is a follow-up extension and only needs this
    registry to be augmented with field-level perturbation rules.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    name: str = Field(description="Human-readable archetype identifier.")
    description: str = Field(description="One-sentence summary for reports.")
    affected_detectors: tuple[str, ...] = Field(
        description="Detectors whose z-score the injection lifts by ``δ``. "
        "Framework §9.2 defines each detector's sensitivity target — "
        "``scale_shift`` propagates across the ratio family (z3–z7), "
        "``round_number`` is isolated to digit-based detectors (z1–z2), etc.",
    )


def _default_archetypes() -> tuple[ArchetypeSettings, ...]:
    """Archetype roster described in ``docs/scores/analysis_plan_v2.md``."""
    return (
        ArchetypeSettings(
            name="scale_shift",
            description="Multiply a raw monetary input (e.g. dlttq) by "
                        "(1 - δ) to reduce a Sharia ratio. Propagates across "
                        "the ratio family.",
            affected_detectors=("z3", "z4", "z6", "z7"),
        ),
        ArchetypeSettings(
            name="threshold_clustering",
            description="Pin a ratio just below the SAC threshold. By "
                        "construction, z4 (threshold proximity) is the only "
                        "primary target; z7 can pick up the peer-divergence.",
            affected_detectors=("z4", "z7"),
        ),
        ArchetypeSettings(
            name="round_number",
            description="Snap monetary figures to round multiples. Affects "
                        "the digit-based detectors exclusively.",
            affected_detectors=("z1", "z2"),
        ),
        ArchetypeSettings(
            name="temporal_smoothing",
            description="Replace each quarter's value with a moving average. "
                        "Dampens Δm_i → z6 is the primary target.",
            affected_detectors=("z6",),
        ),
        ArchetypeSettings(
            name="peer_gaming",
            description="Move a firm's ratio toward the sector median. "
                        "Collapses the peer Mahalanobis distance → z7 primary; "
                        "z3/z4 can follow when dynamics change.",
            affected_detectors=("z3", "z4", "z7"),
        ),
        ArchetypeSettings(
            name="systematic",
            description="Uniform positive shift on every detector — the "
                        "benchmark archetype from framework §5.5 used to "
                        "verify the theoretical T_IUT power table.",
            affected_detectors=("z1", "z2", "z3", "z4", "z5", "z6", "z7"),
        ),
    )


class InjectionSettings(BaseModel):
    """Phase 5 — power analysis via controlled injection.

    Framework §5.5 gives a theoretical power table for ``T_IUT`` (α=0.05,
    k=7, uniform ``δ`` on all detectors): ``π(1)=0.01, π(2)=0.53,
    π(3)=0.97``. Phase 5 validates that table empirically via the
    ``systematic`` archetype, while the five other archetypes
    characterise per-detector specificity.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    n_inject: int = Field(
        default=500,
        ge=10,
        description="Number of honest rows perturbed per "
        "(archetype, δ) cell. Plan of record uses 500.",
    )
    delta_grid: tuple[float, ...] = Field(
        default=(0.5, 1.0, 1.5, 2.0, 3.0),
        description="Grid of δ magnitudes (unit: z-score shift). The "
        "framework §5.5 table is defined at δ ∈ {1, 1.5, 2, 3}; 0.5 is "
        "added to detect early-onset sensitivity.",
    )
    alpha_grid: tuple[float, ...] = Field(
        default=(0.05, 0.01, 0.001),
        description="Significance levels reported in the power curves.",
    )
    random_seed: int = Field(
        default=1,
        ge=0,
        description="Seed for injection sampling — distinct from "
        "``BootstrapSettings.random_seed`` so the two bases of randomness "
        "are decoupled.",
    )
    power_target: float = Field(
        default=0.80,
        gt=0.0,
        lt=1.0,
        description="Target power used to derive ``δ*`` (minimum detectable "
        "effect): the smallest ``δ`` whose empirical detection rate ≥ "
        "``power_target`` at the primary ``α``.",
    )
    primary_alpha_for_mde: float = Field(
        default=0.05,
        gt=0.0,
        lt=1.0,
        description="Primary α used for the MDE table. Must be in "
        "``alpha_grid``.",
    )
    archetypes: tuple[ArchetypeSettings, ...] = Field(
        default_factory=_default_archetypes,
        description="Active archetype roster.",
    )
    only_active_detectors: bool = Field(
        default=True,
        description="Restrict affected detectors to those retained by "
        "Phase 2 (finite scores on C). Avoids injecting into detectors "
        "that are structurally silent (z1 under the current Benford "
        "config).",
    )

    @model_validator(mode="after")
    def _validate_alpha_alignment(self) -> "InjectionSettings":
        if self.primary_alpha_for_mde not in self.alpha_grid:
            raise ValueError(
                f"primary_alpha_for_mde ({self.primary_alpha_for_mde}) "
                f"must be one of alpha_grid ({self.alpha_grid})."
            )
        return self


# ─────────────────────────────────────────────────────────────────────────────
# Sprint 5 — Phase 6 robustness / external validation
# ─────────────────────────────────────────────────────────────────────────────
class RobustnessSettings(BaseModel):
    """Phase 6 — external validation and sensitivity checks.

    Four diagnostics, each addressing a distinct failure mode:

    - **Delisting proxy** — firms whose last observed quarter falls well
      before the panel's max quarter are a proxy for "delisted". Compare
      their composites in the trailing ``delisting_trailing_quarters``
      against their own earlier history — manipulation should (if the
      detectors work) elevate the late-period composites.
    - **Sector false-positive rate** — RED rate per sector on
      ``sac_shariah == 1`` rows. A single sector capturing a large share of
      the RED-in-C rows indicates a business-model mismatch (e.g. Financial
      Services). The plan of record flagged Financial Services in
      particular; this check scales.
    - **Integrity sensitivity** — rank the composites once on the full
      panel and once on ``bs_any_ffilled == 0`` (strict integrity). Large
      rank shifts indicate the composites rely on forward-filled cells.
    - **Detector / composite stability** — lag-1 autocorrelation of each
      composite within a firm. Near-zero means quarter-to-quarter noise;
      near-one means persistent signal (a chronically flagged firm).
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    delisting_trailing_quarters: int = Field(
        default=4,
        ge=1,
        description="Trailing quarters counted as the 'late period' when "
        "comparing a firm's own early-vs-late composites.",
    )
    delisting_cutoff_share: float = Field(
        default=0.2,
        gt=0.0,
        lt=1.0,
        description="Fraction of firms treated as the 'delisted proxy' "
        "cohort — those whose ``last_quarter_rank`` falls in the bottom "
        "``cutoff_share`` of the firms ordered by last observation quarter. "
        "No delisting flag exists on the panel; this is a heuristic.",
    )
    strict_integrity_column: str = Field(
        default="bs_any_ffilled",
        description="Panel column whose zero value marks the strict-"
        "integrity subset (no forward-filled balance-sheet input).",
    )
    red_threshold: float = Field(
        default=0.01,
        gt=0.0,
        lt=1.0,
        description="Threshold used to count RED rows per sector in the "
        "sector-false-positive table (matches framework §A.3).",
    )
    stability_min_quarters: int = Field(
        default=8,
        ge=2,
        description="Minimum quarters per firm before the lag-1 auto"
        "correlation of a composite is reported.",
    )


class RobustnessBenchmarkSettings(BaseModel):
    """Cross-family robustness benchmark settings."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    methods: tuple[str, ...] = Field(
        default=(
            "correlated_gaussian",
            "threshold_clustering_m1_v2",
            "adversarial_evasion",
            "anoshift",
        ),
        description="Enabled benchmark methods.",
    )
    realistic_methods: tuple[str, ...] = Field(
        default=(
            "threshold_clustering_m1_v2",
            "temporal_spike_m2b",
            "benford_m3_v2",
            "interstatement_m4_v2",
            "m5_cod_break",
            "m6_seasonal",
            "abn_disx_full",
            "abn_disx_partial",
            "abn_prod_full",
        ),
        description="Realistic raw-data manipulation methods evaluated inside family 2.",
    )
    rho_grid: tuple[float, ...] = Field(
        default=(0.01, 0.05, 0.10),
        description="Contamination-rate grid for row-targeted methods.",
    )
    delta_grid: tuple[float, ...] = Field(
        default=(0.05, 0.10, 0.15, 0.20, 0.30),
        description="Perturbation-magnitude grid for benchmark methods.",
    )
    alpha: float = Field(
        default=0.05,
        gt=0.0,
        lt=1.0,
        description="Primary alpha level used throughout the benchmark.",
    )
    primary_composite: str = Field(
        default="p_z_mahalanobis_sq",
        description="Primary composite p-value column for adversarial evasion.",
    )
    anoshift_splits: dict[str, tuple[int, int]] = Field(
        default={
            "iid": (2006, 2012),
            "near": (2013, 2017),
            "far": (2020, 2025),
        },
        description="Chronological split definition for AnoShift.",
    )
    anoshift_min_complete_rows: int = Field(
        default=30,
        ge=1,
        description="Minimum number of complete C rows required for an AnoShift transfer calibration window.",
    )
    anoshift_adaptive_start_years: tuple[int, ...] = Field(
        default=(),
        description=(
            "Optional explicit start years for adaptive AnoShift transfer windows. "
            "When empty, the benchmark derives a small grid automatically from the available C years."
        ),
    )
    bootstrap_replicates: int = Field(
        default=500,
        ge=100,
        description="Bootstrap replicates used inside repeated benchmark reruns.",
    )
    random_seed: int = Field(
        default=7,
        ge=0,
        description="Seed used for benchmark randomness.",
    )
    adversarial_max_rows: int = Field(
        default=5,
        ge=1,
        description="Maximum number of RED rows targeted in the adversarial method.",
    )
    adversarial_target_strategy: str = Field(
        default="most_extreme",
        description=(
            "How Family 3 selects RED rows to attack: "
            "`most_extreme` (smallest p-values first) or "
            "`closest_to_threshold` (RED rows closest to alpha first)."
        ),
    )
    adversarial_target_dominant_detectors: tuple[str, ...] = Field(
        default=(),
        description=(
            "Optional whitelist of dominant baseline detectors for Family 3 row selection. "
            "When non-empty, only RED rows whose dominant detector belongs to this set are targeted."
        ),
    )
    adversarial_maxiter: int = Field(
        default=20,
        ge=1,
        description="Maximum epsilon-box trials in the adversarial PGD search.",
    )
    workers: int = Field(
        default=1,
        ge=1,
        description="Process-pool size for scoring contamination cells "
        "(Families 1 & 2). 1 = serial (default; unchanged behaviour). Job "
        "construction stays sequential — only the per-cell scoring fans out — "
        "and the baseline null/sigma is passed as an explicit override so "
        "pooled workers reproduce the serial result bit-for-bit. Cap BLAS "
        "threads per worker (OMP/MKL_NUM_THREADS) and bound total CPU via a "
        "cgroup so N_workers × threads cannot oversubscribe the host.",
    )
    pgd_iterations: int = Field(
        default=24,
        ge=1,
        description="Projected-gradient iterations for the family 3 adversarial benchmark.",
    )
    pgd_step_size: float = Field(
        default=0.10,
        gt=0.0,
        description="PGD step size in raw-column normalized units.",
    )
    pgd_eps_grid: tuple[float, ...] = Field(
        default=(0.05, 0.10, 0.20, 0.35, 0.50, 0.75, 1.00),
        description="Relative L-infinity box radii explored by the family 3 PGD search.",
    )
    pgd_penalty_lambda: float = Field(
        default=0.05,
        ge=0.0,
        description="Quadratic proximity penalty used in the torch PGD surrogate loss.",
    )
    pgd_temperature: float = Field(
        default=0.02,
        gt=0.0,
        description="Softplus temperature used to smooth threshold exceedances in the PGD surrogate.",
    )
    pgd_tolerance: float = Field(
        default=1e-4,
        gt=0.0,
        description="Early-stop tolerance on primary p-value improvement in PGD.",
    )
    pgd_prox_lambda: float = Field(
        default=1e-4,
        ge=0.0,
        description=(
            "Proximity regularisation weight (λ_prox) in the softmax_general surrogate. "
            "Keeps the optimisation numerically stable without dominating the SC penalty. "
            "Set to 0 to disable."
        ),
    )

    @model_validator(mode="after")
    def _validate_anoshift_splits(self) -> "RobustnessBenchmarkSettings":
        required = {"iid", "near", "far"}
        missing = required - set(self.anoshift_splits)
        if missing:
            raise ValueError(f"anoshift_splits missing required keys: {sorted(missing)}")
        allowed_target_strategies = {"most_extreme", "closest_to_threshold"}
        if self.adversarial_target_strategy not in allowed_target_strategies:
            raise ValueError(
                "adversarial_target_strategy "
                f"{self.adversarial_target_strategy!r} not in {sorted(allowed_target_strategies)}"
            )
        if any(not str(v).strip() for v in self.adversarial_target_dominant_detectors):
            raise ValueError("adversarial_target_dominant_detectors cannot contain empty values.")
        if any(int(y) <= 0 for y in self.anoshift_adaptive_start_years):
            raise ValueError("anoshift_adaptive_start_years must contain positive years only.")
        return self


# ─────────────────────────────────────────────────────────────────────────────
# Sprint 5 — Phase 7 multiple-testing correction
# ─────────────────────────────────────────────────────────────────────────────

    # ── Family-3 counterfactual (XAI) knobs — from dev/remi ──────────────
    family3_max_rows: int = Field(
        default=5,
        ge=1,
        description="Maximum number of candidate rows targeted in Family 3.",
    )
    family3_modes: tuple[str, ...] = Field(
        default=("global_per_row",),
        description=(
            "Active Family-3 counterfactual modes. "
            "Supported values: `global_per_row`, `global_cohort`, `local_firm_single`."
        ),
    )
    family3_modifiable_columns: tuple[str, ...] = Field(
        default=(
            "revtq",
            "iditq",
            "nopiq",
            "xintq",
            "niq",
            "oibdpq",
            "oancfq",
            "ibq",
            "atq",
            "dlttq",
            "dlcq",
            "ltq",
            "cheq",
            "actq",
            "lctq",
            "rectq",
            "invtq",
            "xsgaq",
            "ppentq",
        ),
        description=(
            "Raw accounting columns that Family 3 exact may modify. Defaults "
            "to the union of the raw Compustat inputs used by detectors z3 to "
            "z8 and by the canonical Sharia ratios consumed by z4, z6, and z7. "
            "Exogenous macro connectors such as sukuk_ratio_t or "
            "islamic_cash_ratio_t are intentionally excluded."
        ),
    )
    family3_target_scores: tuple[str, ...] = Field(
        default=(
            "z_plus",
            "z_plus_renorm",
            "breadth",
            "z_mahalanobis_sq",
            "t_iut",
        ),
        description=(
            "Composite target score names optimized one-by-one by the Family-3 XAI runs. "
            "Family 3 loss is defined only on composite scores, not on individual detector PIT scores."
        ),
    )
    family3_directions: tuple[str, ...] = Field(
        default=("to_green", "to_red"),
        description="Target directions for Family 3. Supported values: `to_green`, `to_red`.",
    )
    family3_p_red: float = Field(default=0.01, gt=0.0, lt=1.0, description="Family-3 RED cutoff.")
    family3_p_orange: float = Field(default=0.05, gt=0.0, lt=1.0, description="Family-3 ORANGE upper cutoff.")
    family3_z_target_green: float = Field(
        default=1.64,
        gt=0.0,
        description="One-sided z target used when pushing ORANGE rows toward GREEN.",
    )
    family3_z_target_red: float = Field(
        default=2.33,
        gt=0.0,
        description="One-sided z target used when pushing ORANGE rows toward RED.",
    )
    family3_lambda_l1: float = Field(
        default=0.05,
        ge=0.0,
        description="L1 regularisation weight used in the exact Family-3 XAI loss.",
    )
    family3_lambda_l2: float = Field(
        default=0.0,
        ge=0.0,
        description="Optional L2 regularisation weight used in the exact Family-3 XAI loss.",
    )
    family3_loss_name: str = Field(
        default="hinge",
        description=(
            "Loss shape used by the exact Family-3 loop. Supported values: "
            "`hinge_squared`, `hinge`, `mse`."
        ),
    )
    family3_cohort_loss_mode: str = Field(
        default="mean",
        description=(
            "How the exact Family-3 cohort objective aggregates row losses. "
            "Supported values: `mean`, `min`."
        ),
    )
    family3_optimizer: str = Field(
        default="torch_adam",
        description=(
            "Optimizer used by the canonical Family-3 counterfactual loop. "
            "The public default is `torch_adam`; legacy exact optimizers "
            "`finite_diff_adam` and `spsa_sign` remain available for local "
            "diagnostics."
        ),
    )
    family3_epochs: int = Field(
        default=12,
        ge=1,
        description="Number of PGD epochs for the exact Family-3 XAI loop.",
    )
    family3_fd_step: float = Field(
        default=0.02,
        gt=0.0,
        description="Finite-difference step in normalized variable space for Family 3.",
    )
    family3_max_eps: float = Field(
        default=1.0,
        gt=0.0,
        description="Maximum absolute normalized perturbation allowed in Family 3.",
    )
    family3_scale_floor_quantile: float = Field(
        default=0.05,
        gt=0.0,
        lt=1.0,
        description=(
            "Lower quantile of each modifiable column's nonzero absolute values, taken "
            "across the full panel, used as that column's normalization floor "
            "(`scale = max(|baseline|, floor)`). Per-column rather than a single constant "
            "so the floor stays meaningful in each column's own units: a firm whose value "
            "for a naturally small-magnitude column (e.g. an interest-expense line) sits "
            "near zero should not get an artificially 'cheap' large relative move just "
            "because it shares a constant floor with a naturally large-magnitude column "
            "(e.g. total assets)."
        ),
    )
    family3_early_stop_loss: float | None = Field(
        default=None,
        ge=0.0,
        description="Optional early-stop threshold on the exact Family-3 total loss.",
    )
    family3_early_stop_patience: int = Field(
        default=0,
        ge=0,
        description=(
            "Optional early-stop patience on exact Family-3 epochs without improvement. "
            "Set to 0 to disable patience-based stopping."
        ),
    )
    family3_frozen_detectors_during_optimization: tuple[str, ...] = Field(
        default=("z1", "z2"),
        description=(
            "Optional raw detectors to keep fixed at their baseline values during exact "
            "Family-3 optimisation epochs. `z1` (Benford-style digit anomalies) and `z2` "
            "(Zipf-style rank-size anomalies) are frozen by default because both are "
            "natural-data-distribution checks: they are not methodologically meaningful "
            "on values we are synthetically perturbing ourselves, and freezing them "
            "avoids the optimizer being spuriously helped or penalised by digit/rank "
            "artefacts of the search path rather than genuine manipulation signal. A "
            "full exact rescoring is still run at the end of the trajectory for final "
            "validation."
        ),
    )
    family3_strict_benford_freeze: bool = Field(
        default=True,
        description=(
            "When True, fail fast if Benford-linked raw columns overlap the Family 3 "
            "modifiable columns while `z1` is not frozen during optimization. This keeps "
            "the default counterfactual setup from optimizing against self-manipulated "
            "Benford signals."
        ),
    )
    family3_adam_beta1: float = Field(
        default=0.9,
        gt=0.0,
        lt=1.0,
        description="Beta1 used by the finite-difference Adam optimizer for Family 3.",
    )
    family3_adam_beta2: float = Field(
        default=0.999,
        gt=0.0,
        lt=1.0,
        description="Beta2 used by the finite-difference Adam optimizer for Family 3.",
    )
    family3_adam_eps: float = Field(
        default=1e-8,
        gt=0.0,
        description="Numerical epsilon used by the finite-difference Adam optimizer for Family 3.",
    )
    family3_torch_surrogate_supported_columns: tuple[str, ...] = Field(
        default=(
            "revtq",
            "iditq",
            "nopiq",
            "xintq",
            "niq",
            "oibdpq",
            "oancfq",
            "ibq",
            "atq",
            "dlttq",
            "dlcq",
            "ltq",
            "cheq",
            "actq",
            "lctq",
            "rectq",
            "invtq",
            "xsgaq",
            "ppentq",
        ),
        description=(
            "Raw columns supported by the Torch Family-3 surrogate. The "
            "current branch covers the 19-variable z3→z8 attack space used by "
            "the main Family 3 paper runs. When "
            "`family3_optimizer='torch_adam'`, configured modifiable columns "
            "must stay within this subset."
        ),
    )
    family3_torch_surrogate_device: str = Field(
        default="auto",
        description=(
            "Device policy for the Torch Family-3 surrogate. Supported values: "
            "`auto` (prefer GPU when available) or `cpu`."
        ),
    )
    family3_torch_surrogate_penalty_lambda: float = Field(
        default=1.0,
        ge=0.0,
        description="Penalty weight used by the Torch Family-3 surrogate loss.",
    )
    family3_torch_surrogate_temperature: float = Field(
        default=0.1,
        gt=0.0,
        description="Soft surrogate temperature used by the Torch Family-3 loss.",
    )
    family3_torch_surrogate_prox_lambda: float = Field(
        default=1e-4,
        ge=0.0,
        description="Proximity regularisation weight used by the Torch Family-3 surrogate loss.",
    )
    family3_torch_objective_mode: str = Field(
        default="hybrid_exact_like",
        description=(
            "Objective used by the experimental torch Family-3 branch. "
            "Supported values: `hybrid_exact_like`, `surrogate`."
        ),
    )
    family3_torch_aux_surrogate_weight: float = Field(
        default=0.1,
        ge=0.0,
        description=(
            "Auxiliary smooth threshold-pressure weight added on top of the hybrid "
            "exact-like torch objective."
        ),
    )
    family3_torch_exact_replay_mode: str = Field(
        default="full",
        description=(
            "How the experimental torch Family-3 path replays the authoritative exact evaluator "
            "over the torch trajectory. Supported values: `full`, `stride`, `best_only`."
        ),
    )
    family3_torch_exact_replay_stride: int = Field(
        default=5,
        ge=1,
        description=(
            "When `family3_torch_exact_replay_mode='stride'`, replay every Nth torch epoch exactly, "
            "plus the baseline/final/best-surrogate states."
        ),
    )
    family3_plateau_shrink_patience: int = Field(
        default=2,
        ge=0,
        description=(
            "After this many non-improving exact Family 3 epochs, shrink the optimizer step size. "
            "Set to 0 to disable adaptive shrink."
        ),
    )
    family3_plateau_shrink_factor: float = Field(
        default=0.5,
        gt=0.0,
        le=1.0,
        description=(
            "Multiplicative factor applied to the exact Family 3 step size after a plateau. "
            "Values below 1.0 shrink the step."
        ),
    )
    family3_min_step_size: float = Field(
        default=0.002,
        ge=0.0,
        description="Lower bound for the adaptive exact Family 3 step size.",
    )
    family3_restore_best_on_shrink: bool = Field(
        default=True,
        description="When shrinking the exact Family 3 step size, restart from the best state seen so far.",
    )
    family3_reset_moments_on_shrink: bool = Field(
        default=True,
        description="When shrinking finite-difference Adam, reset first/second moments to stabilise the new phase.",
    )
    family3_local_single_case: tuple[str, str] | None = Field(
        default=None,
        description="Unused by public modes. Kept for backward compatibility with legacy scripts.",
    )
    family3_local_cases: tuple[tuple[str, str], ...] = Field(
        default=(),
        description="Unused by public modes. Kept for backward compatibility with legacy scripts.",
    )
    family3_local_firm_case: str | None = Field(
        default=None,
        description="Optional `firm_id` case used by the `local_firm_single` Family-3 mode.",
    )
    family3_firm_aggregation: str = Field(
        default="min_pvalue",
        description=(
            "How `local_firm_single` / future firm-level Family 3 modes collapse "
            "multiple firm-quarter target p-values into one firm-level target. "
            "Supported values: `min_pvalue`."
        ),
    )

class FDRSettings(BaseModel):
    """Phase 7 — Benjamini-Hochberg FDR control.

    Framework §7 downgraded this from high priority because ``T_IUT``
    produces tiny p-values under independence (``≤ 10^{-9}``) when it
    rejects. Still useful for ``Z+`` / ``Z+_renorm`` / ``Z²_Mah``, and for
    the firm-level aggregation that prevents a single persistent outlier
    from counting as many "independent" discoveries.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    q_levels: tuple[float, ...] = Field(
        default=(0.05, 0.01),
        description="FDR thresholds reported in the Phase 7 summary.",
    )
    composites: tuple[str, ...] = Field(
        default=(
            "p_z_plus", "p_z_plus_renorm", "p_z_mahalanobis_sq", "p_t_iut",
            "p_z_plus_softmax", "p_z_plus_orth",
        ),
        description="Composite p-value columns to BH-correct. Breadth is "
        "discrete on 7 levels and not meaningfully correctable.",
    )
    firm_aggregation: str = Field(
        default="min_pvalue",
        description="How to collapse each firm's row-level p-values into a "
        "single firm-level p before applying BH. ``'min_pvalue'`` keeps the "
        "worst (most significant) row — conservative per framework §7.",
    )


# ─────────────────────────────────────────────────────────────────────────────
# Figures (paper-ready matplotlib)
# ─────────────────────────────────────────────────────────────────────────────
class FigureSettings(BaseModel):
    """Rendering settings for the paper figures.

    The figures module (``scores.figures``) reads the parquet/CSV outputs
    of the phases and writes publication-ready PNG/PDF plots under
    :attr:`output_directory`. All visual parameters — colour scheme, DPI,
    figure size — live here so a reviewer can confirm that nothing was
    baked into an individual plot.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    file_format: str = Field(
        default="png",
        description="Output format — ``'png'`` for draft, ``'pdf'`` for "
        "publication vector output, or ``'both'`` to emit both.",
    )
    dpi: int = Field(
        default=150,
        ge=72,
        description="PNG rendering DPI. 150 is sharp on screen, 300 for print.",
    )
    figure_width_inches: float = Field(default=7.0, gt=0.0)
    figure_height_inches: float = Field(default=4.5, gt=0.0)
    colour_green: str = Field(default="#2ca02c")
    colour_amber: str = Field(default="#ff7f0e")
    colour_red: str = Field(default="#d62728")
    colour_grey: str = Field(default="#8c8c8c")
    top_n_firms: int = Field(
        default=15,
        ge=1,
        description="How many firms to name on the top-N bar and scatter "
        "labels.",
    )
    red_threshold: float = Field(
        default=0.01,
        gt=0.0,
        lt=1.0,
        description="`min p` threshold for the RED verdict in the figures "
        "(matches framework §A.3).",
    )
    amber_threshold: float = Field(
        default=0.05,
        gt=0.0,
        lt=1.0,
        description="`min p` threshold for AMBER.",
    )

    @model_validator(mode="after")
    def _check_thresholds(self) -> "FigureSettings":
        if self.red_threshold >= self.amber_threshold:
            raise ValueError(
                f"red_threshold ({self.red_threshold}) must be < "
                f"amber_threshold ({self.amber_threshold})."
            )
        allowed = {"png", "pdf", "both"}
        if self.file_format not in allowed:
            raise ValueError(
                f"file_format {self.file_format!r} not in {allowed}."
            )
        return self


# ─────────────────────────────────────────────────────────────────────────────
# Dashboard (analyst-facing HTML)
# ─────────────────────────────────────────────────────────────────────────────
class DashboardSettings(BaseModel):
    """Rendering settings for the self-contained Plotly dashboard.

    The dashboard module (``scores.dashboard``) produces a single HTML
    file that can be opened locally — no server, no Python runtime. It
    wraps the same phase-4/6/7 outputs the paper figures read.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    filename: str = Field(
        default="triage.html",
        description="Name of the dashboard HTML file.",
    )
    triage_top_n: int = Field(
        default=50,
        ge=1,
        description="Number of firms shown in the triage table (sorted by "
        "the primary composite's firm-level BH q-value).",
    )
    heatmap_top_n: int = Field(
        default=40,
        ge=1,
        description="Number of firms shown in the panel heatmap. Ranked by "
        "share of RED quarters to surface chronic suspects.",
    )
    primary_composite_q: str = Field(
        default="p_z_mahalanobis_sq",
        description="Which firm-level BH q-value column drives the default "
        "triage sort. ``p_z_mahalanobis_sq`` is the headline composite "
        "(Sprint 4 finding).",
    )
    title: str = Field(
        default="Shariah-compliance anomaly triage",
        description="Title shown at the top of the HTML file.",
    )
    embed_plotly_js: bool = Field(
        default=True,
        description="If ``True``, the Plotly JS library is bundled into the "
        "HTML so the file opens offline. ``False`` loads it from a CDN "
        "(smaller file but needs internet).",
    )


# ─────────────────────────────────────────────────────────────────────────────
# Top-level composite
# ─────────────────────────────────────────────────────────────────────────────
class AnalysisSettings(BaseModel):
    """Top-level configuration for the scoring analysis pipeline."""

    model_config = ConfigDict(extra="forbid")

    exclude_sectors: tuple[str, ...] = Field(
        default=(),
        description="Sector names to exclude from the entire scoring "
        "analysis. Empty tuple = no exclusion (default). The filter is "
        "applied once at the start of the orchestrator before Phase 0; "
        "everything downstream (C, detectors, composites, bootstrap, "
        "FDR) automatically adjusts. Example: ``('Financial Services',)``.",
    )
    compliant_only: bool = Field(
        default=False,
        description="When True, restrict verdict reporting (Phase 7 FDR, "
        "qualitative analysis, headline numbers) to firm-quarters whose "
        "recomputed ratios satisfy all SAC thresholds (debt < 33%%, "
        "cash < 33%%, income < 5%%), regardless of authority label. "
        "This includes non-SAC firms that look compliant in the data "
        "and excludes SAC firms that violate in the data. Detectors "
        "and composites are still computed on the full panel. "
        "Enabled via --compliant-only.",
    )
    run_robustness: bool = Field(
        default=False,
        description="When True, run the cross-family robustness benchmark "
        "(adversarial evasion, AnoShift, realistic manipulations, "
        "correlated Gaussian injection) after Phase 7. "
        "Enabled via --robustness.",
    )
    panel_schema: PanelSchemaSettings = Field(default_factory=PanelSchemaSettings)
    reference_sample: ReferenceSampleSettings = Field(default_factory=ReferenceSampleSettings)
    detector_preconditions: DetectorPreconditionSettings = Field(
        default_factory=DetectorPreconditionSettings
    )
    zscores: ZScoreSettings = Field(default_factory=ZScoreSettings)
    calibration: CalibrationSettings = Field(default_factory=CalibrationSettings)
    dependence: DependenceSettings = Field(default_factory=DependenceSettings)
    composites: CompositeSettings = Field(default_factory=CompositeSettings)
    bootstrap: BootstrapSettings = Field(default_factory=BootstrapSettings)
    confidence: ConfidenceSettings = Field(default_factory=ConfidenceSettings)
    injection: InjectionSettings = Field(default_factory=InjectionSettings)
    robustness: RobustnessSettings = Field(default_factory=RobustnessSettings)
    robustness_benchmark: RobustnessBenchmarkSettings = Field(
        default_factory=RobustnessBenchmarkSettings
    )
    fdr: FDRSettings = Field(default_factory=FDRSettings)
    figures: FigureSettings = Field(default_factory=FigureSettings)
    dashboard: DashboardSettings = Field(default_factory=DashboardSettings)
    output_layout: OutputLayoutSettings = Field(default_factory=OutputLayoutSettings)

    @model_validator(mode="after")
    def _check_cross_section_consistency(self) -> "AnalysisSettings":
        """Guard against contradictions between §9.2 D5 and Phase 0."""
        ref_min = self.reference_sample.min_sector_size
        d5_min = self.detector_preconditions.cross_statement.min_sector_reference
        if ref_min < d5_min:
            raise ValueError(
                f"reference_sample.min_sector_size ({ref_min}) must be ≥ the D5 "
                f"floor (m q = {d5_min}); Phase 3 masks would be stricter than "
                f"the sample Phase 0 curated."
            )
        return self
