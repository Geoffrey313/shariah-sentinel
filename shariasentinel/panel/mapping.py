
from __future__ import annotations

import logging
import re
from typing import Optional, Tuple

import pandas as pd

from collections import defaultdict

log = logging.getLogger(__name__)


def _build_keyword_index(keyword_series: pd.Series) -> dict[str, set[int]]:
    """Map each keyword to the set of row indices containing it."""
    index = defaultdict(set)
    for idx, kw_set in keyword_series.items():
        for kw in kw_set:
            index[kw].add(idx)
    return index

def extract_isin_3_7(isin: object) -> Tuple[Optional[str], str]:
    """
    Strict extraction: take ISIN[3:7] (0-based) after uppercasing.
    Returns (code_zfill4, status).

    status:
      - ok
      - isin_missing
      - isin_non_my
      - needs_flex   (digits present but not 4-digit clean -> use flexible extractor)
    """
    if isin is None or (isinstance(isin, float) and pd.isna(isin)):
        return None, "isin_missing"
    s = str(isin).strip().upper()
    if len(s) < 7:
        return None, "isin_missing"
    if not s.startswith("MY"):
        return None, "isin_non_my"
    chunk = s[3:7]
    if not re.fullmatch(r"\d{4}", chunk):
        return None, "needs_flex"
    return chunk, "ok"

def extract_isin_flexible(isin: object) -> Tuple[Optional[str], str]:
    """
    Flexible extraction: take consecutive digits starting at position 3,
    cast to int, then zfill(4). Handles Compustat patterns:
      - MYL8419OO003  -> 8419  (standard)
      - MYL03006O003  -> 3006  (30xx codes, leading zero in Compustat)
      - MYQ0329OO001  -> 0329  (ACE/LEAP Market)

    Note on MYL0xxxx: Compustat encodes recent 30xx Bursa codes with a
    spurious leading zero (MYL03006 instead of MYL3006). This is a
    Compustat encoding inconsistency, not a Bursa standard.

    Returns (code_zfill4, status).
    status:
      - ok
      - isin_missing
      - isin_too_short
      - isin_non_my
      - no_digits_at_3
      - out_of_range
    """
    if isin is None or (isinstance(isin, float) and pd.isna(isin)):
        return None, "isin_missing"
    s = str(isin).strip().upper()
    if len(s) < 7:
        return None, "isin_too_short"
    if not s.startswith("MY"):
        return None, "isin_non_my"

    m = re.match(r"^(\d+)", s[3:])
    if not m:
        return None, "no_digits_at_3"

    digits = m.group(1)

    try:
        code_int = int(digits)

    except ValueError:
        return None, "digits_not_int"

    if code_int < 0 or code_int > 99999:
        return None, "out_of_range"

    return str(code_int).zfill(4), "ok"



def build_mapping_compustat(df_comp: pd.DataFrame) -> pd.DataFrame:
    """
    One row per Compustat gvkey with ISIN-derived Bursa code.
    We keep gvkey as an internal Compustat identifier but we DO NOT compare it
    to Bursa stock codes. The join key is bursa_code derived from ISIN.

    Output columns:
      gvkey, conm, isin,
      bursa_code_3_7, status_3_7,
      bursa_code_flex, status_flex
    """
    u = (df_comp.sort_values("datadate")
                .drop_duplicates("gvkey", keep="last")[["gvkey", "conm", "isin"]]
                .copy())

    strict = u["isin"].apply(extract_isin_3_7)
    u["bursa_code_3_7"] = strict.apply(lambda x: x[0])
    u["status_3_7"]     = strict.apply(lambda x: x[1])

    flex = u["isin"].apply(extract_isin_flexible)
    u["bursa_code_flex"] = flex.apply(lambda x: x[0])
    u["status_flex"]     = flex.apply(lambda x: x[1])
    
    return u


def intersection_stats(
    df_sc: pd.DataFrame,
    mapping: pd.DataFrame,
    df_comp_pre_pydantic: pd.DataFrame = None,
) -> dict:
    sc_codes = set(df_sc["stock_code"].astype(int).astype(str).str.zfill(4).unique())

    comp_codes_3_7  = set(mapping["bursa_code_3_7"].dropna().unique())
    comp_codes_flex = set(mapping["bursa_code_flex"].dropna().unique())

    inter_3_7  = sc_codes & comp_codes_3_7
    inter_flex = sc_codes & comp_codes_flex

    missing = sorted(sc_codes - comp_codes_flex)

    status_flex_counts = mapping["status_flex"].value_counts().to_dict()
    status_3_7_counts  = mapping["status_3_7"].value_counts().to_dict()

    result = {
        "sc_unique_stock_codes": len(sc_codes),
        "compustat_unique_isin_3_7":         len(comp_codes_3_7),
        "intersection_isin_3_7__stock_code": len(inter_3_7),
        "coverage_sc_by_isin_3_7_pct":       round(100 * len(inter_3_7)  / max(len(sc_codes), 1), 2),
        "coverage_comp_by_sc_pct_3_7":       round(100 * len(inter_3_7)  / max(len(comp_codes_3_7), 1), 2),
        "isin_status_3_7": status_3_7_counts,
        "compustat_unique_isin_flexible":              len(comp_codes_flex),
        "intersection_isin_flexible__stock_code":      len(inter_flex),
        "coverage_sc_by_isin_flexible_pct":            round(100 * len(inter_flex) / max(len(sc_codes), 1), 2),
        "coverage_comp_by_sc_pct_flex":                round(100 * len(inter_flex) / max(len(comp_codes_flex), 1), 2),
        "isin_status_flex": status_flex_counts,
        "sc_codes_not_in_compustat":   missing,
        "sc_codes_not_in_compustat_n": len(missing),
    }

    # Diagnostic enrichi si df_comp_pre_pydantic fourni
    if df_comp_pre_pydantic is not None:
        result["sc_codes_not_in_compustat_diagnosed"] = diagnose_missing_sc_codes(
            missing, df_comp_pre_pydantic, df_sc
        )

    return result



def _name_keywords(name: str) -> set[str]:
    """Tokenize company name into significant keywords."""
    stopwords = {
        "bhd", "berhad", "sdn", "holdings", "holding", "group",
        "ltd", "limited", "corp", "corporation", "co", "company",
        "industries", "indl", "mfg", "the", "and", "teknologi",
        "technology", "resources", "capital", "international",
    }
    return {
        w for w in re.sub(r"[^a-z0-9\s]", " ", str(name).lower()).split()
        if w not in stopwords and len(w) >= 3
    }


def keyword_overlap(name_sc: str, name_comp: str) -> int:
    """Number of significant keywords shared between two company names."""
    return len(_name_keywords(name_sc) & _name_keywords(name_comp))


# ISIN non-MY connus comme foreign listings sur Bursa (codes 5xxx)
_FOREIGN_LISTING_CODES = {"5150", "5172", "5298", "5340"}


def diagnose_missing_sc_codes(
    missing_codes: list,
    df_comp_pre_pydantic: pd.DataFrame,
    df_sc: pd.DataFrame,
) -> list:
    """
    For each SC code absent from Compustat (after flexible ISIN extraction),
    produce a structured diagnostic.

    Categories:
      - foreign_listing
      - short_code
      - code_30xx_myq
      - code_30xx_name_match
      - code_30xx_missing
      - other_code_2xxx
      - unknown
    """
    # Full Compustat universe before Pydantic filtering
    comp_isin_all = (
        df_comp_pre_pydantic
        .drop_duplicates("gvkey", keep="last")[["gvkey", "conm", "isin"]]
        .copy()
        .reset_index(drop=True)
    )

    # Precompute keywords once for all Compustat names
    comp_isin_all["_kw"] = comp_isin_all["conm"].apply(_name_keywords)

    kw_index = _build_keyword_index(comp_isin_all["_kw"])

    # SC Malaysia names for each missing code
    sc_names = (
        df_sc[df_sc["stock_code"].astype(str).str.zfill(4).isin(missing_codes)]
        .drop_duplicates("stock_code")[["stock_code", "company_name"]]
        .assign(stock_code=lambda x: x["stock_code"].astype(str).str.zfill(4))
        .set_index("stock_code")["company_name"]
        .to_dict()
    )

    # MYQ codes in Compustat (positions 3:7)
    myq_isin = comp_isin_all[
        comp_isin_all["isin"].fillna("").str.startswith("MYQ")
    ].copy()
    myq_isin["myq_code"] = myq_isin["isin"].str[3:7]
    myq_by_code = myq_isin.set_index("myq_code")[["gvkey", "conm", "isin"]].to_dict("index")

    results = []

    for code in sorted(missing_codes):
        code_int = int(code)
        sc_name = sc_names.get(code, "")
        sc_kw = _name_keywords(sc_name)

        entry = {
            "bursa_code": code,
            "sc_name": sc_name,
            "diagnostic": "",
            "detail": "",
            "gvkey_compustat": None,
            "conm_compustat": None,
            "isin_compustat": None,
        }

        if code in _FOREIGN_LISTING_CODES:
            entry["diagnostic"] = "foreign_listing"
            entry["detail"] = (
                "Entreprise étrangère listée à Bursa Malaysia. "
                "ISIN non-MYL — non joinable via ISIN Malaysia."
            )

        elif code_int < 1000:
            entry["diagnostic"] = "short_code"
            entry["detail"] = (
                "Code Bursa < 1000 : small cap historique ou marché LEAP. "
                "Absent de la couverture Compustat standard."
            )

        elif code_int == 2607:
            entry["diagnostic"] = "other_code_2xxx"
            entry["detail"] = (
                "Code 2xxx isolé — probable incorporation étrangère "
                "(ex: UK) listée à Bursa. ISIN non-MYL attendu."
            )

        elif 3000 <= code_int <= 3099:
            # First: MYQ fallback
            if code in myq_by_code:
                m = myq_by_code[code]
                entry["diagnostic"] = "code_30xx_myq"
                entry["detail"] = (
                    f"Présent dans Compustat sous ISIN MYQ ({m['isin']}) "
                    f"— non joinable via extracteur ISIN flex."
                )
                entry["gvkey_compustat"] = int(m["gvkey"])
                entry["conm_compustat"] = m["conm"]
                entry["isin_compustat"] = m["isin"]

            else:
                # Name-based fallback using precomputed keyword sets
                candidates = []
                if sc_kw:
                    candidate_idx = set()
                    for kw in sc_kw:
                        candidate_idx.update(kw_index.get(kw, set()))

                    if candidate_idx:
                        sub = comp_isin_all.loc[sorted(candidate_idx), ["gvkey", "conm", "isin", "_kw"]].copy()
                        sub["kw_overlap"] = sub["_kw"].apply(lambda kw: len(sc_kw & kw))
                        sub = sub[sub["kw_overlap"] >= 2].copy()

                        if not sub.empty:
                            candidates = (
                                sub.assign(gvkey=sub["gvkey"].astype(int))
                                .sort_values("kw_overlap", ascending=False)
                                .drop(columns=["_kw"])
                                .to_dict("records")
                            )

                if candidates:
                    best = candidates[0]
                    entry["diagnostic"] = "code_30xx_name_match"
                    entry["detail"] = (
                        f"Non trouvé via ISIN mais candidat Compustat identifié par nom "
                        f"(overlap={best['kw_overlap']} mots). ISIN Compustat : {best['isin']}. "
                        f"Jointure ISIN impossible — gvkey fourni pour référence manuelle."
                    )
                    entry["gvkey_compustat"] = best["gvkey"]
                    entry["conm_compustat"] = best["conm"]
                    entry["isin_compustat"] = best["isin"]
                    if len(candidates) > 1:
                        entry["name_match_other_candidates"] = candidates[1:4]
                else:
                    entry["diagnostic"] = "code_30xx_missing"
                    entry["detail"] = (
                        "Absent de Compustat via ISIN et aucun candidat trouvé par nom. "
                        "IPO récente probable ou non couverte par Compustat Global."
                    )

        else:
            entry["diagnostic"] = "unknown"
            entry["detail"] = "Non catégorisé — investigation manuelle requise."

        results.append(entry)

    return results