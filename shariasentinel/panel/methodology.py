"""Per-country Sharia-screening methodology registry for the quant pipeline.

Single source of the compliance thresholds and label semantics consumed by
the panel builder (``ratios.py``), the threshold-aware detectors
(``d4_proximity``, ``d9_seasonal_gap``) and the scoring phases
(``phase0_reference_sample``, ``run_analysis``). None of those modules may
hardcode a threshold of their own.

The ``methodology_key`` values mirror ``ComplianceMethodology.key`` in the
product config (``server/shariasentinel/config.py`` — ``sac_my`` /
``dfm_uae``). The quant pipeline runs standalone (no shariasentinel import,
so the ``bible`` conda env stays dependency-light); keep the two registries
aligned when adding a country.

Threshold routing at scoring time is **panel-driven**: the builder stamps a
``methodology_key`` column on every non-MYS panel, and downstream consumers
call :func:`thresholds_for_panel` to resolve the right cutoffs. A panel
without the column (every MYS artefact ever built) resolves to SAC — this
fallback is what keeps the MYS pipeline byte-identical.
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
    methodology_key: str    # mirrors shariasentinel ComplianceMethodology.key
    authority_label: str    # display label for reports/logs
    threshold_debt: float
    threshold_cash: float
    threshold_income: float
    # True when the authority publishes ONLY the compliant securities
    # (DFM): an unmatched firm is unknown (NA), never non-compliant (0).
    positive_only: bool

    # Whether the country's sukuk connector is trusted for the DEBT screen.
    # Set False when only a COUNTRY_TOTAL / non-corporate-scope sukuk proxy
    # exists (Qatar, Kuwait): applying an all-issuer sukuk share (~20-40%,
    # mostly sovereign) to corporate ``dlttq`` massively over-states
    # compliance (corporate sukuk ≈ 1%). With False the debt adjustment
    # degrades to the RAW (conservative) ratio; the cash adjustment is
    # unaffected. Defaults True (MYS/UAE/SAU corporate-scope series).
    apply_sukuk_adjustment: bool = True

    def ratio_thresholds(self) -> dict[str, float]:
        """Canonical ratio-column → threshold map used by detectors/phases."""
        return {
            "ratio_debt_adj": self.threshold_debt,
            "ratio_cash_adj": self.threshold_cash,
            "ratio_income": self.threshold_income,
        }


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

QSE_QAT = CountryCompliancePolicy(
    country_key="QAT",
    compustat_fic="QAT",
    currency="QAR",
    methodology_key="qse_qat",
    authority_label="QSE (Al Rayan Islamic Index)",
    # QSE publishes only the compliant constituents (positive list, like DFM);
    # thresholds follow the MSCI/S&P Islamic proxy the QSE index tracks
    # (debt ≤33%, cash ≤33%, non-permissible income ≤5%) per the internal
    # "Shariah-Compliant Lists — Screening Standards" reference.
    threshold_debt=0.33,
    threshold_cash=0.33,
    threshold_income=0.05,
    positive_only=True,
    # Qatar has no corporate-scope sukuk series — only a country-total proxy
    # (~20%, mostly sovereign). Do NOT apply it to the corporate debt screen
    # (it over-states compliance for ~29 firms); degrade to raw/conservative.
    apply_sukuk_adjustment=False,
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

BOUBYAN_KWT = CountryCompliancePolicy(
    country_key="KWT",
    compustat_fic="KWT",
    currency="KWD",
    methodology_key="boubyan_kwt",
    authority_label="Boubyan Capital (Kuwait)",
    # Kuwait AAOIFI-inspired GCC practice: 30% debt / 30% cash / 5% income.
    threshold_debt=0.30,
    threshold_cash=0.30,
    threshold_income=0.05,
    positive_only=True,
    # Kuwait has only a flat flow-average sukuk proxy (~40%, non-corporate) —
    # do NOT apply it to the debt screen (same rationale as Qatar).
    apply_sukuk_adjustment=False,
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
    # screen, so the cash cap is set non-binding (1.0) — a genuine
    # cross-jurisdiction heterogeneity, not an omission.
    threshold_debt=0.45,
    threshold_cash=1.00,
    threshold_income=0.10,
    # OJK publishes the DES as the list of Shariah-compliant securities only
    # (positive-only, like DFM/QSE/Boubyan): an unmatched firm is unknown.
    positive_only=True,
    # Indonesia has a genuine CORPORATE-scope sukuk series (OJK "obligasi
    # korporasi" sukuk vs conventional), R_t low (~3-11%); apply it like
    # MYS/PAK. Cash connector (Islamic deposit share ~6-8%) also applied.
    apply_sukuk_adjustment=True,
)

POLICIES: dict[str, CountryCompliancePolicy] = {
    SAC_MY.country_key: SAC_MY,
    DFM_UAE.country_key: DFM_UAE,
    QSE_QAT.country_key: QSE_QAT,
    BOUBYAN_SAU.country_key: BOUBYAN_SAU,
    PSX_PAK.country_key: PSX_PAK,
    BOUBYAN_KWT.country_key: BOUBYAN_KWT,
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
    """Ratio-column → threshold map for the panel's methodology.

    The one accessor detectors and scoring phases should use instead of
    hardcoding SAC numbers.
    """
    return policy_for_panel(df).ratio_thresholds()
