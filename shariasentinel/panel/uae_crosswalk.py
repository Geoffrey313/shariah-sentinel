"""DFM stock_code → Compustat gvkey crosswalk (UAE), by company name.

Direct identifier joins are impossible: the DFM list is keyed by exchange
symbols (``AIRARABIA``, ``AAN``, …) and NO Compustat column (``tic``,
``gvkey``, ``conm``, ``conml``, ``isin``, ``sedol``) intersects them —
0 / 64 verified 2026-07-14 (the vendor ``raw_UAE`` file is column-shifted;
its ``tic`` column actually holds gvkeys). The crosswalk therefore matches
**normalized company names** (DFM ``company_name`` ↔ Compustat
``conml``/``conm``) and is completed by a hand-curated override table
(``data/raw/dfm_uae/crosswalk_overrides.csv``) so that 100 % of the DFM
securities end up decided: matched to a gvkey, or explicitly excluded as
"not in Compustat".

Outputs (written by ``scripts/build_uae_crosswalk.py``):

- ``data_clean/uae_dfm_gvkey_crosswalk.csv`` — one row per DFM
  stock_code: ``stock_code, company_name_dfm, gvkey, conm, method,
  score, comment``.
- ``data_clean/uae_dfm_gvkey_crosswalk_report.md`` — nominal report of
  every decision for review.

Acceptance (plan §3b): ≥ 80 % matched automatically, 100 % decided after
curation, zero false positives on a manual sample of 10.
"""
from __future__ import annotations

import logging
import re
from difflib import SequenceMatcher

import pandas as pd

from shariasentinel import config as cfg

log = logging.getLogger(__name__)

# Methods recorded in the crosswalk (stable strings — tests rely on them).
METHOD_EXACT = "exact_name"
METHOD_FUZZY = "fuzzy_name"
METHOD_TOKEN = "distinctive_token"
METHOD_OVERRIDE = "override"
METHOD_EXCLUDED = "excluded"      # curated: security has no Compustat gvkey
METHOD_UNMATCHED = "unmatched"    # needs curation

# Auto-acceptance rules for fuzzy matches. A candidate is accepted when its
# score clears MIN_AUTO_SCORE and beats the runner-up by MIN_SCORE_GAP —
# both guards exist to keep the false-positive rate at zero.
MIN_AUTO_SCORE = 0.90
MIN_SCORE_GAP = 0.04
# Pass 2b (distinctive-token) acceptance: a firm sharing a RARE, non-generic
# token with a single candidate is a confident match even when the gap guard
# rejects it (markets with many "National X" / "Al X" names). The token must
# appear in at most this many Compustat names to count as distinctive.
MIN_TOKEN_SCORE = 0.55
MAX_TOKEN_DF = 3

# Tokens too common to disambiguate on (a match resting only on these is not
# trustworthy). Distinct from _LEGAL_TOKENS (legal forms) — these are generic
# business/geography words.
_GENERIC_TOKENS = {
    "SAUDI", "ARABIA", "ARABIAN", "NATIONAL", "INTERNATIONAL", "GROUP",
    "HOLDING", "INVESTMENT", "DEVELOPMENT", "INDUSTRIAL", "INDUSTRIES",
    "REAL", "ESTATE", "SERVICES", "SERVICE", "TRADING", "GENERAL", "GULF",
    "MIDDLE", "EAST", "UNITED", "FIRST", "BANK", "INSURANCE", "COOPERATIVE",
    "FINANCIAL", "FINANCE", "MEDICAL", "FOOD", "FOODS", "POWER", "WATER",
    "CEMENT", "PETROCHEMICAL", "MANUFACTURING", "MARKETING", "TELECOMMUNICATIONS",
    "COMMUNICATIONS", "COMPANY", "AND", "FOR", "OF",
}

# Legal-form tokens stripped during normalization. Deliberately excludes
# business words like HOLDING / GROUP / NATIONAL that distinguish firms.
_LEGAL_TOKENS = {
    "PJSC", "PSC", "PLC", "LLC", "CO", "COMPANY", "CORP", "CORPORATION",
    "INC", "LTD", "LIMITED", "SAOG", "KSC", "KSCP", "KSCC", "BSC", "QSC",
    "PJS", "SAL", "SAK", "SAKC",
}

# Common abbreviation expansions (applied token-wise after uppercasing).
# Compustat heavily abbreviates; expanding both sides to a canonical token
# turns many near-misses into exact matches.
_TOKEN_EXPANSIONS = {
    "INTL": "INTERNATIONAL",
    "INTRNL": "INTERNATIONAL",
    "NATL": "NATIONAL",
    "GRP": "GROUP",
    "INV": "INVESTMENT",
    "INVT": "INVESTMENT",
    "INVTMENT": "INVESTMENT",
    "INVTMENTS": "INVESTMENT",
    "INVEST": "INVESTMENT",
    "DEV": "DEVELOPMENT",
    "DEVLPMT": "DEVELOPMENT",
    "BK": "BANK",
    "INS": "INSURANCE",
    "REINS": "REINSURANCE",
    "TELECOMM": "TELECOMMUNICATIONS",
    "TELECOM": "TELECOMMUNICATIONS",
    "MFG": "MANUFACTURING",
    "MANUFACT": "MANUFACTURING",
    "INDL": "INDUSTRIAL",
    "INDUS": "INDUSTRIAL",
    "PETROCHEM": "PETROCHEMICAL",
    "PETROL": "PETROLEUM",
    "SVCS": "SERVICES",
    "SVC": "SERVICE",
    "HLDG": "HOLDING",
    "HLDGS": "HOLDING",
    "HLDING": "HOLDING",
    "MKTG": "MARKETING",
    "PROV": "PROVINCE",
    "CEMNT": "CEMENT",
    "TRANSPRT": "TRANSPORT",
    "AGRIC": "AGRICULTURAL",
    "PHARM": "PHARMACEUTICAL",
    "COOP": "COOPERATIVE",
    "ELECT": "ELECTRIC",
    "&": "AND",
}

# Arabic article and English article — dropped as standalone tokens so
# "AL JAZIRA" and "ALJAZIRA" reconcile (the glued form is handled in
# ``_tokens_match``).
_ARTICLE_TOKENS = {"AL", "THE"}


def normalize_name(name: object) -> str:
    """Uppercase, strip punctuation/parentheticals/legal suffixes, expand
    abbreviations, drop stray single letters (``P J S C`` remnants)."""
    if name is None or (isinstance(name, float) and pd.isna(name)):
        return ""
    s = str(name).upper()
    # Parentheticals are alternate labels or legal forms — "(Tabreed)",
    # "(Aman)", "(P.J.S.C.)" — never the identity itself.
    s = re.sub(r"\([^)]*\)", " ", s)
    # Kill periods outright so "P.J.S.C." collapses to the PJSC token
    # instead of exploding into single letters.
    s = s.replace(".", "")
    s = s.replace("&", " AND ")
    s = re.sub(r"[^A-Z0-9 ]+", " ", s)
    tokens = [
        _TOKEN_EXPANSIONS.get(t, t)
        for t in s.split()
        if t not in _LEGAL_TOKENS and t not in _ARTICLE_TOKENS and len(t) > 1
    ]
    return " ".join(tokens)


def _strip_article(t: str) -> str:
    """Drop a glued Arabic ``AL`` prefix so ``ALJAZIRA`` ≈ ``JAZIRA``,
    ``ALINMA`` ≈ ``INMA`` (matched against a spaced ``AL JAZIRA`` whose
    standalone ``AL`` was already dropped in ``normalize_name``). Only when
    the remainder stays a real token (≥4 chars) to avoid mangling."""
    if t.startswith("AL") and len(t) >= 6:
        return t[2:]
    return t


def _tokens_match(a: str, b: str) -> bool:
    """Tokens match when equal, one is a ≥4-char prefix of the other
    (absorbs Compustat truncations ``INVESTME`` → INVESTMENT and
    singular/plural drift), or they agree once a glued Arabic ``AL`` article
    is removed (``ALINMA`` ↔ ``INMA``)."""
    if a == b:
        return True
    a2, b2 = _strip_article(a), _strip_article(b)
    if a2 == b2:
        return True
    if min(len(a2), len(b2)) < 4:
        return False
    return a2.startswith(b2) or b2.startswith(a2)


def _pair_score(a: str, b: str) -> float:
    """Blend of character similarity and prefix-aware token overlap."""
    if not a or not b:
        return 0.0
    seq = SequenceMatcher(None, a, b).ratio()
    ta, tb = set(a.split()), set(b.split())
    if not ta or not tb:
        return 0.0
    matched_a = sum(1 for t in ta if any(_tokens_match(t, u) for u in tb))
    matched_b = sum(1 for u in tb if any(_tokens_match(u, t) for t in ta))
    overlap = min(matched_a, matched_b)
    jac = overlap / (len(ta) + len(tb) - overlap)
    return 0.4 * seq + 0.6 * jac


def compustat_name_table(df_comp_raw: pd.DataFrame) -> pd.DataFrame:
    """One row per valid gvkey with its best display + normalized names.

    ``df_comp_raw`` is the raw Compustat CSV frame (string dtypes). Rows
    whose gvkey is non-numeric or all-zero (the 43 column-shifted Falcon
    rows) are dropped — same guard as the panel loader.
    """
    gv = df_comp_raw["gvkey"].astype(str).str.strip()
    valid = gv.str.fullmatch(r"\d+") & (gv.str.strip("0") != "")
    sub = df_comp_raw.loc[valid, ["gvkey", "conm"]].copy()
    if "conml" in df_comp_raw.columns:
        sub["conml"] = df_comp_raw.loc[valid, "conml"]
    else:
        sub["conml"] = pd.NA
    sub["gvkey"] = sub["gvkey"].astype(int)
    sub = sub.drop_duplicates(subset=["gvkey"]).reset_index(drop=True)
    sub["norm_conm"] = sub["conm"].map(normalize_name)
    sub["norm_conml"] = sub["conml"].map(normalize_name)
    return sub


def load_overrides() -> pd.DataFrame:
    """Load the manual override table (may be absent → empty frame).

    Columns: ``stock_code, gvkey, comment``. An empty ``gvkey`` marks the
    security as decided-out ("not in Compustat").
    """
    path = cfg.uae_crosswalk_overrides_csv()
    if not path.exists():
        return pd.DataFrame(columns=["stock_code", "gvkey", "comment"])
    df = pd.read_csv(path, dtype=str)
    df["stock_code"] = df["stock_code"].astype(str).str.strip()
    return df


def build_crosswalk(
    df_dfm: pd.DataFrame,
    df_comp_names: pd.DataFrame,
    overrides: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Match every distinct DFM security to a gvkey (or record why not).

    Args:
        df_dfm: Output of ``load_dfm_long`` (needs ``stock_code``,
            ``company_name``).
        df_comp_names: Output of :func:`compustat_name_table`.
        overrides: Manual override table (see :func:`load_overrides`).

    Returns:
        One row per DFM ``stock_code`` with columns ``stock_code,
        company_name_dfm, gvkey (nullable Int64), conm, method, score,
        comment``, sorted by stock_code.
    """
    overrides = overrides if overrides is not None else load_overrides()
    ov_by_code = {r["stock_code"]: r for _, r in overrides.iterrows()}

    dfm_firms = (
        df_dfm.sort_values("date_effective")
        .drop_duplicates("stock_code", keep="last")[["stock_code", "company_name"]]
        .reset_index(drop=True)
    )
    dfm_firms["norm_name"] = dfm_firms["company_name"].map(normalize_name)

    # ── Distinctive-token index (for Pass 2b) ────────────────────────────
    # Canonical token = article-stripped, so ALJAZIRA and JAZIRA share one.
    # A token is "distinctive" when it is non-generic and appears in at most
    # MAX_TOKEN_DF Compustat names — a shared distinctive token is a strong,
    # low-collision signal even when several names look alike overall.
    from collections import Counter as _Counter

    comp_ctoks: dict[int, set[str]] = {}
    token_df: _Counter = _Counter()
    for _, r in df_comp_names.iterrows():
        toks = {
            _strip_article(t)
            for t in f"{r['norm_conm'] or ''} {r['norm_conml'] or ''}".split()
        }
        comp_ctoks[int(r["gvkey"])] = toks
        for t in toks:
            token_df[t] += 1

    def _distinctive(t: str) -> bool:
        return len(t) >= 4 and t not in _GENERIC_TOKENS and token_df.get(t, 0) <= MAX_TOKEN_DF

    used_gvkeys: set[int] = set()
    rows: list[dict] = []

    # Pass 0 — overrides claim their gvkeys first so name matching cannot
    # steal a hand-assigned firm.
    for code, ov in ov_by_code.items():
        gv_raw = str(ov.get("gvkey") or "").strip()
        if gv_raw and gv_raw.lower() != "nan":
            used_gvkeys.add(int(gv_raw))

    for _, firm in dfm_firms.iterrows():
        code = firm["stock_code"]
        base = {
            "stock_code": code,
            "company_name_dfm": firm["company_name"],
            "gvkey": pd.NA,
            "conm": pd.NA,
            "method": METHOD_UNMATCHED,
            "score": pd.NA,
            "comment": pd.NA,
        }

        ov = ov_by_code.get(code)
        if ov is not None:
            gv_raw = str(ov.get("gvkey") or "").strip()
            if gv_raw and gv_raw.lower() != "nan":
                gv = int(gv_raw)
                hit = df_comp_names.loc[df_comp_names["gvkey"] == gv]
                base.update(
                    gvkey=gv,
                    conm=hit["conm"].iloc[0] if not hit.empty else pd.NA,
                    method=METHOD_OVERRIDE,
                    score=1.0,
                    comment=ov.get("comment"),
                )
            else:
                base.update(method=METHOD_EXCLUDED, comment=ov.get("comment"))
            rows.append(base)
            continue

        # Pass 1 — exact normalized-name equality (conml preferred).
        cand = df_comp_names.loc[
            ~df_comp_names["gvkey"].isin(used_gvkeys)
            & (
                (df_comp_names["norm_conml"] == firm["norm_name"])
                | (df_comp_names["norm_conm"] == firm["norm_name"])
            )
        ]
        if len(cand) == 1:
            hit = cand.iloc[0]
            used_gvkeys.add(int(hit["gvkey"]))
            base.update(
                gvkey=int(hit["gvkey"]), conm=hit["conm"],
                method=METHOD_EXACT, score=1.0,
            )
            rows.append(base)
            continue

        # Pass 2 — fuzzy best-candidate with runner-up gap guard.
        pool = df_comp_names.loc[~df_comp_names["gvkey"].isin(used_gvkeys)]
        scored = [
            (
                max(
                    _pair_score(firm["norm_name"], r["norm_conml"]),
                    _pair_score(firm["norm_name"], r["norm_conm"]),
                ),
                int(r["gvkey"]),
                r["conm"],
            )
            for _, r in pool.iterrows()
        ]
        scored.sort(reverse=True)
        if scored:
            best_score, best_gv, best_conm = scored[0]
            second = scored[1][0] if len(scored) > 1 else 0.0
            if best_score >= MIN_AUTO_SCORE and (best_score - second) >= MIN_SCORE_GAP:
                used_gvkeys.add(best_gv)
                base.update(
                    gvkey=best_gv, conm=best_conm,
                    method=METHOD_FUZZY, score=round(best_score, 4),
                )
            else:
                # Pass 2b — distinctive-token match. Among candidates sharing
                # a rare, non-generic token with the firm, accept the best if
                # it is confident and unambiguous. Solves the "many similar
                # names" case (e.g. NATIONAL INDUSTRIALIZATION → the sole firm
                # with the rare token INDUSTRIALIZATION) that the gap guard
                # rejects. ``scored`` is already sorted best-first.
                firm_ctoks = {_strip_article(t) for t in firm["norm_name"].split()}
                firm_dist = {t for t in firm_ctoks if _distinctive(t)}
                tok_cands = (
                    [(sc, gv, cn) for sc, gv, cn in scored
                     if firm_dist & comp_ctoks.get(gv, set())]
                    if firm_dist else []
                )
                tok_second = tok_cands[1][0] if len(tok_cands) > 1 else 0.0
                if (tok_cands and tok_cands[0][0] >= MIN_TOKEN_SCORE
                        and (len(tok_cands) == 1
                             or (tok_cands[0][0] - tok_second) >= MIN_SCORE_GAP)):
                    sc, gv, cn = tok_cands[0]
                    used_gvkeys.add(gv)
                    shared = sorted(firm_dist & comp_ctoks.get(gv, set()))
                    base.update(
                        gvkey=gv, conm=cn, method=METHOD_TOKEN,
                        score=round(sc, 4),
                        comment=f"distinctive token(s) {shared}",
                    )
                else:
                    base.update(
                        score=round(best_score, 4),
                        comment=f"best candidate gvkey={best_gv} conm={best_conm!r} "
                                f"(score {best_score:.3f}, runner-up {second:.3f})",
                    )
        rows.append(base)

    out = pd.DataFrame(rows).sort_values("stock_code").reset_index(drop=True)
    out["gvkey"] = out["gvkey"].astype("Int64")

    n = len(out)
    n_auto = int(out["method"].isin([METHOD_EXACT, METHOD_FUZZY, METHOD_TOKEN]).sum())
    n_decided = int((out["method"] != METHOD_UNMATCHED).sum())
    log.info(
        "uae_crosswalk: %d securities — %d auto-matched (%.0f%%), "
        "%d decided (%.0f%%), %d awaiting curation.",
        n, n_auto, 100 * n_auto / max(n, 1),
        n_decided, 100 * n_decided / max(n, 1),
        n - n_decided,
    )
    return out


def load_crosswalk(require_fully_decided: bool = True, policy=None) -> pd.DataFrame:
    """Read the built crosswalk for the panel build.

    ``policy`` selects the country's crosswalk file; ``None`` keeps the
    historical UAE default (byte-identical). Raises if the file is missing
    or (by default) still has unmatched securities — the panel must not be
    built on a half-curated table.
    """
    country_label = "UAE" if policy is None else policy.country_key
    authority_label = "DFM" if policy is None else policy.authority_label
    if policy is None or policy.country_key == "UAE":
        path = cfg.uae_crosswalk_csv()
        builder = "python -m scripts.build_uae_crosswalk"
    else:
        path = cfg.crosswalk_csv(policy.country_key, policy.methodology_key)
        builder = f"python -m scripts.build_crosswalk --country {policy.country_key}"
    if not path.exists():
        raise FileNotFoundError(
            f"crosswalk not found at {path}; run `{builder}` first."
        )
    df = pd.read_csv(path, dtype={"stock_code": str})
    df["gvkey"] = pd.to_numeric(df["gvkey"], errors="coerce").astype("Int64")
    undecided = df[df["method"] == METHOD_UNMATCHED]
    if require_fully_decided and not undecided.empty:
        raise ValueError(
            f"{country_label} crosswalk ({authority_label}) has "
            f"{len(undecided)} unmatched securities "
            f"({', '.join(undecided['stock_code'].head(10))}…); complete "
            f"the manual overrides for that country and rebuild with `{builder}`."
        )
    return df
