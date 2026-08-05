"""Configuration for the contamination framework.

Every threshold, contamination rate, mechanism parameter, and
evaluation grid lives here — no magic numbers in the module code.
"""
from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


class MechanismSettings(BaseModel):
    """Parameters for one contamination mechanism."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    name: str = Field(description="Mechanism identifier (M1–M4).")
    description: str = Field(description="One-line summary.")
    enabled: bool = Field(default=True)


class M1Settings(MechanismSettings):
    """M1 — threshold optimisation."""

    name: str = "M1_threshold"
    description: str = "Pin ratio just below the SAC cap."
    band_fraction: float = Field(
        default=0.10,
        gt=0.0, lt=1.0,
        description="Width of the just-below-threshold band as a "
        "fraction of the threshold (ε = band_fraction × τ).",
    )


class M2Settings(MechanismSettings):
    """M2 — temporal smoothing."""

    name: str = "M2_smoothing"
    description: str = "Replace quarterly values with a moving average."
    window: int = Field(
        default=4, ge=2,
        description="Smoothing window in quarters.",
    )


class M3Settings(MechanismSettings):
    """M3 — Benford distortion (round-number preference)."""

    name: str = "M3_benford"
    description: str = "Snap monetary figures to round multiples."
    rounding_magnitude: int = Field(
        default=3, ge=1,
        description="Power of 10 to round to (3 = nearest 1000).",
    )
    fraction_of_fields: float = Field(
        default=0.5, gt=0.0, le=1.0,
        description="Fraction of monetary fields modified per row.",
    )


class M4Settings(MechanismSettings):
    """M4 — cross-statement inconsistency."""

    name: str = "M4_inconsistency"
    description: str = (
        "Reduce debt without proportionally reducing interest expense."
    )
    debt_reduction: float = Field(
        default=0.50, gt=0.0, lt=1.0,
        description="Fraction by which dlttq is reduced (δ_m).",
    )


class M5Settings(MechanismSettings):
    """M5 — cost-of-debt break (targets z8).

    Reduce dlttq in only 1-2 quarters per firm (not all),
    creating a sudden CoD spike that z8 is designed to catch.
    """

    name: str = "M5_cod_break"
    description: str = (
        "Reduce debt in 1-2 quarters only, creating a CoD spike."
    )
    debt_reduction: float = Field(
        default=0.50, gt=0.0, lt=1.0,
        description="Fraction by which dlttq is reduced in the target quarters.",
    )
    n_quarters: int = Field(
        default=2, ge=1, le=4,
        description="Number of quarters per firm to perturb.",
    )


class M6Settings(MechanismSettings):
    """M6 — seasonal manipulation (targets z9).

    Reduce debt ratios only at fiscal year-end quarters,
    leave interim quarters unchanged. Mimics review-date
    management.
    """

    name: str = "M6_seasonal"
    description: str = (
        "Reduce debt at year-end only, leave interim unchanged."
    )
    debt_reduction: float = Field(
        default=0.30, gt=0.0, lt=1.0,
        description="Fraction by which dlttq is reduced at year-end.",
    )


class ContaminationSettings(BaseModel):
    """Top-level configuration for the contamination study."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    contamination_rates: tuple[float, ...] = Field(
        default=(0.01, 0.05, 0.10),
        description="Fraction of observations contaminated per cell.",
    )
    h0_sample_size: int = Field(
        default=100_000, ge=100,
        description="Target number of clean observations in each "
        "benchmark dataset. In panel mode, whole firms are sampled "
        "until this target is reached. Set high to use all of C.",
    )
    random_seed: int = Field(default=42, ge=0)
    alpha_grid: tuple[float, ...] = Field(
        default=(0.05, 0.01, 0.001),
        description="Significance levels for precision/recall.",
    )
    m1: M1Settings = Field(default_factory=M1Settings)
    m2: M2Settings = Field(default_factory=M2Settings)
    m3: M3Settings = Field(default_factory=M3Settings)
    m4: M4Settings = Field(default_factory=M4Settings)
    m5: M5Settings = Field(default_factory=M5Settings)
    m6: M6Settings = Field(default_factory=M6Settings)

    monetary_columns: tuple[str, ...] = Field(
        default=(
            "revtq", "atq", "dlttq", "dlcq", "ltq", "cheq",
            "iditq", "nopiq", "xintq", "oibdpq",
        ),
        description="Columns eligible for Benford distortion (M3).",
    )
    ratio_definitions: dict[str, dict[str, str]] = Field(
        default={
            "ratio_debt_adj": {"num": "dlttq", "den": "atq", "threshold": "0.33"},
            "ratio_cash_adj": {"num": "cheq", "den": "atq", "threshold": "0.33"},
            "ratio_income": {"num": "iditq", "den": "revtq", "threshold": "0.05"},
        },
        description="Sharia ratios: numerator, denominator, SAC cap.",
    )
