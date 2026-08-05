"""Per-country Sharia-screening methodology registry for the quant pipeline.

Single source of the compliance thresholds and label semantics consumed by
the panel builder (``ratios.py``), the threshold-aware detectors
(``d4_proximity``, ``d9_seasonal_gap``) and the scoring phases
(``phase0_reference_sample``, ``run_analysis``). None of those modules may
hardcode a threshold of their own.

The ``methodology_key`` values mirror ``ComplianceMethodology.key`` in the
product config (``server/src/config.py`` — ``sac_my`` /
``dfm_uae``). The quant pipeline runs standalone (no src import,
so the ``bible`` conda env stays dependency-light); keep the two registries
aligned when adding a country.

Threshold routing at scoring time is **panel-driven**: the builder stamps a
``methodology_key`` column on every non-MYS panel, and downstream consumers
call :func:`thresholds_for_panel` for all canonical cutoffs or
:func:`screening_thresholds_for_panel` for the active ratios an authority
actually screens. A panel without the column (every MYS artefact ever built)
resolves to SAC — this fallback is what keeps the MYS pipeline byte-identical.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass

import pandas as pd

log = logging.getLogger(__name__)

# Column stamped on non-MYS panels by the builder; consumed by
# ``thresholds_for_panel`` and the tri-state label logic.
METHODOLOGY_COLUMN: str = "methodology_key"

# Generic tri-state compliance label (1 = compliant per authority list,
# 0 = explicitly non-compliant — only exists for negative-list authorities
# like SC Malaysia — NA = unknown / not covered by the authority list).
COMPLIANCE_COLUMN: str = "compliance_shariah"


@dataclass(frozen=True)
class CountryCompliancePolicy:
    """One country's screening authority: thresholds + list semantics."""

    country_key: str        # canonical CLI / filename key ("MYS", "UAE")
    compustat_fic: str      # Compustat ``fic`` value ("MYS", "ARE")
    currency: str           # local reporting currency ("MYR", "AED")
    methodology_key: str    # mirrors src ComplianceMethodology.key
    authority_label: str    # display label for reports/logs
    threshold_debt: float
    threshold_cash: float
    threshold_income: float
    # True when the authority publishes ONLY the compliant securities
    # (DFM): an unmatched firm is unknown (NA), never non-compliant (0).
    positive_only: bool

    # Whether the country's sukuk connector is trusted for the DEBT screen.
    # Set False when only a COUNTRY_TOTAL / non-corporate-scope sukuk proxy
    # exists: applying an all-issuer sukuk share (mostly sovereign) to
    # corporate ``dlttq`` massively over-states compliance (corporate sukuk is
    # a far smaller share). With False the debt adjustment degrades to the RAW
    # (conservative) ratio; the cash adjustment is unaffected. Defaults True
    # (MYS/UAE/SAU corporate-scope series).
    apply_sukuk_adjustment: bool = True

    # Whether the authority applies a cash / liquid-assets screen at all.
    # Some regimes have a genuine two-ratio methodology. Indonesia's DES, for
    # example, screens debt and non-permissible income but has no cash cap; this
    # flag prevents a placeholder cap from becoming an accidental filter.
    cash_screen_enabled: bool = True

    def ratio_thresholds(self) -> dict[str, float]:
        """All canonical ratio-column → threshold values for this authority."""
        return {
            "ratio_debt_adj": self.threshold_debt,
            "ratio_cash_adj": self.threshold_cash,
            "ratio_income": self.threshold_income,
        }

    def screening_thresholds(self) -> dict[str, float]:
        """Ratio-column → threshold map for actual compliance screening."""
        thresholds = {
            "ratio_debt_adj": self.threshold_debt,
            "ratio_income": self.threshold_income,
        }
        if self.cash_screen_enabled:
            thresholds["ratio_cash_adj"] = self.threshold_cash
        return thresholds


SAC_MY = CountryCompliancePolicy(
    country_key="MYS",
    compustat_fic="MYS",
    currency="MYR",
    methodology_key="sac_my",
    authority_label="SC Malaysia",
    threshold_debt=0.33,
    threshold_cash=0.33,
    threshold_income=0.05,
    positive_only=False,
)

DFM_UAE = CountryCompliancePolicy(
    country_key="UAE",
    compustat_fic="ARE",
    currency="AED",
    methodology_key="dfm_uae",
    authority_label="DFM",
    threshold_debt=0.30,
    threshold_cash=0.30,
    threshold_income=0.10,
    positive_only=True,
)

BOUBYAN_SAU = CountryCompliancePolicy(
    country_key="SAU",
    compustat_fic="SAU",
    currency="SAR",
    methodology_key="boubyan_sau",
    authority_label="Boubyan Capital (Saudi)",
    # Boubyan publishes only compliant constituents (positive list).
    # Saudi screening tracks the MSCI/S&P Islamic band (33.33% debt / 33.33%
    # cash on total assets, non-permissible income ≤5%) per the internal
    # "Shariah-Compliant Lists — Screening Standards" reference.
    threshold_debt=0.3333,
    threshold_cash=0.3333,
    threshold_income=0.05,
    positive_only=True,
)

PSX_PAK = CountryCompliancePolicy(
    country_key="PAK",
    compustat_fic="PAK",
    currency="PKR",
    methodology_key="psx_pk",
    authority_label="PSX / KMI (SECP approved)",
    # KMI screening (SECP): debt / total assets < 33% (revised 2026; was 37%),
    # non-compliant investments < 33%, non-permissible income < 5%. The panel's
    # three-ratio screen is a simplification of the full 5-screen KMI test —
    # the anchor uses the published final_shariah_status verdict regardless.
    threshold_debt=0.33,
    threshold_cash=0.33,
    threshold_income=0.05,
    # NEGATIVE-determinable list (like SC Malaysia): PSX publishes explicit
    # non-compliant verdicts, so an unlisted firm is unknown but a listed
    # Non-Compliant is a real 0.
    positive_only=False,
)

DES_IDN = CountryCompliancePolicy(
    country_key="IDN",
    compustat_fic="IDN",
    currency="IDR",
    methodology_key="des_idn",
    authority_label="OJK (Daftar Efek Syariah)",
    # Indonesia's DES screen (OJK, POJK 35/2017) is a TWO-ratio test that
    # differs from the GCC/MSCI bands: interest-based debt / total assets
    # <= 45% and non-permissible income / revenue <= 10%. There is NO cash
    # screen — a genuine cross-jurisdiction heterogeneity, not an omission.
    threshold_debt=0.45,
    threshold_cash=1.00,
    threshold_income=0.10,
    # OJK publishes the DES as the list of Shariah-compliant securities only
    # (positive-only, like DFM/Boubyan): an unmatched firm is unknown.
    positive_only=True,
    # Indonesia has a genuine CORPORATE-scope sukuk series (OJK "obligasi
    # korporasi" sukuk vs conventional), R_t low (~3-11%); apply it like
    # MYS/PAK. Cash connector (Islamic deposit share ~6-8%) also applied.
    apply_sukuk_adjustment=True,
    cash_screen_enabled=False,
)

POLICIES: dict[str, CountryCompliancePolicy] = {
    SAC_MY.country_key: SAC_MY,
    DFM_UAE.country_key: DFM_UAE,
    BOUBYAN_SAU.country_key: BOUBYAN_SAU,
    PSX_PAK.country_key: PSX_PAK,
    DES_IDN.country_key: DES_IDN,
}

_BY_METHODOLOGY: dict[str, CountryCompliancePolicy] = {
    p.methodology_key: p for p in POLICIES.values()
}

# Fallback for panels without a methodology_key column — every MYS artefact.
DEFAULT_POLICY: CountryCompliancePolicy = SAC_MY


def get_policy(country_code: str) -> CountryCompliancePolicy:
    """Resolve a policy from a CLI-style country code (case-insensitive).

    Accepts the canonical key ("MYS", "UAE") and the Compustat fic
    ("ARE" → UAE) so callers can pass whichever identifier they hold.
    """
    key = str(country_code).strip().upper()
    if key in POLICIES:
        return POLICIES[key]
    for policy in POLICIES.values():
        if policy.compustat_fic == key:
            return policy
    raise KeyError(
        f"No compliance policy registered for country {country_code!r}; "
        f"known: {sorted(POLICIES)} (see server/panel/methodology.py)."
    )


def policy_for_methodology(methodology_key: str) -> CountryCompliancePolicy:
    """Resolve a policy from a ``methodology_key`` panel value."""
    try:
        return _BY_METHODOLOGY[str(methodology_key)]
    except KeyError as exc:
        raise KeyError(
            f"Unknown methodology_key {methodology_key!r}; "
            f"known: {sorted(_BY_METHODOLOGY)}."
        ) from exc


def policy_for_panel(df: pd.DataFrame) -> CountryCompliancePolicy:
    """Resolve the policy a panel was built under.

    Reads the ``methodology_key`` column stamped by the builder; a panel
    without it (or with only-null values) is a legacy MYS artefact and
    resolves to :data:`DEFAULT_POLICY`. Mixed-methodology panels are not
    supported — the first non-null key wins and a warning is logged if
    several distinct keys coexist.
    """
    if METHODOLOGY_COLUMN not in df.columns:
        return DEFAULT_POLICY
    keys = df[METHODOLOGY_COLUMN].dropna().unique()
    if len(keys) == 0:
        return DEFAULT_POLICY
    if len(keys) > 1:
        log.warning(
            "policy_for_panel: %d distinct methodology keys %s in one panel; "
            "using %r.",
            len(keys), sorted(map(str, keys)), str(keys[0]),
        )
    return policy_for_methodology(str(keys[0]))


def thresholds_for_panel(df: pd.DataFrame) -> dict[str, float]:
    """All canonical ratio-column thresholds for the panel's methodology.

    Use :func:`screening_thresholds_for_panel` when defining the compliant
    population, because some authorities do not screen every canonical ratio.
    """
    return policy_for_panel(df).ratio_thresholds()


def screening_thresholds_for_panel(df: pd.DataFrame) -> dict[str, float]:
    """Active screening thresholds for the panel's methodology.

    Unlike :func:`thresholds_for_panel`, this omits ratio columns that the
    authority does not screen. Use it when defining the compliant population.
    """
    return policy_for_panel(df).screening_thresholds()
