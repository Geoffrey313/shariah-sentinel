# Reproduction data

Two things are shipped in this directory — enough to regenerate every reported
result without any licensed vendor data:

- **`raw/macro_connectors/`** — the Islamic-correction macro series (sukuk share
  in corporate debt, Islamic cash/deposit share), small annual tables drawn from
  public official sources (each row cites its source).
- **`scores/<country>/`** — the **de-identified derived outputs** of the frozen
  run: detector z-scores, composite statistics with their p- and q-values,
  verdicts, and the reduced result tables. Firm keys are replaced by stable
  synthetic ids, company names dropped, and **no licensed Compustat field is
  present**.

Run `python reproduce.py results` to regenerate the paper's numbers
(`numbers_manifest.json`) and figure data from these — no panel needed.

**Not shipped: the reconstructed input panels.** They are built from S&P Global
Compustat Global, whose license does not permit redistributing the underlying
vendor fields, and the panels still carry those fields. They are needed only to
re-run the pipeline from scratch (`pipeline`, `score`, `benchmark`,
`family3-*`), or to redraw the single leverage U-shape figure (which reads a
per-firm debt ratio); obtain them on request (below).

## How to obtain the panels

The reconstructed panels can be shared on a motivated request: state who you
are and the intended research use, and send it to the corresponding author
(contact withheld for single-blind review). Data will be shared to the extent
the S&P Compustat license allows.

## Expected layout

Place one panel per country under `data/`, using the two-digit lowercase
country codes `mys`, `idn`, `pak`, `sau`, `uae`:

```
data/
  panel/<country>/compustat_quarterly.parquet
      the reconstructed quarterly panel; `pipeline` regenerates every run
      output from it
  scores/<country>/phase0_reference_sample/panel_with_split.parquet
      the same panel carrying the reference-sample split column, read directly
      by `score`, `benchmark`, and `family3-*`
```

The run outputs under `data/scores/<country>/` are **shipped here in
de-identified form** (synthetic firm ids, no Compustat fields), so the reported
results reproduce without the panels. Placing the panels above lets you instead
re-run the full pipeline and regenerate those outputs from scratch.

## Panel schema

Format: Apache Parquet. One row = one firm (`gvkey`) x one Compustat quarter
(`datacqtr`). Quarters use the standard `YYYY-Qn` string. Monetary fields are
in the firm's reporting currency (`curcdq`), in millions, following Compustat
conventions. All financial fields are nullable.

**Identifiers and keys**

| Column | Type | Meaning |
|---|---|---|
| `gvkey` | int | Compustat firm key |
| `datadate` | date | fiscal period-end date |
| `datacqtr` | str | calendar quarter, `YYYY-Qn` |
| `fyearq` | int | fiscal year |
| `fqtr` | int | fiscal quarter (1 to 4) |
| `conm` | str | company name |
| `isin`, `sedol` | str | security identifiers (optional) |
| `fic` | str | country of incorporation |
| `curcdq` | str | reporting currency |

**Balance sheet (stock variables)**

`atq` total assets; `actq` current assets; `lctq` current liabilities; `ltq`
total liabilities; `cheq`/`chq`/`chsq` cash and short-term investments;
`rectq`/`rectrq` receivables; `invtq` inventories; `ppentq` net property, plant
and equipment; `dlttq` long-term debt; `dlcq` debt in current liabilities;
`dd1q` long-term debt due in one year; `dlsq` short-term borrowings; `notesq`
notes payable; `ltmibq` total liabilities plus minority interest.

**Income statement (flow variables)**

`revtq` revenue; `cogsq` cost of goods sold; `gpq` gross profit; `xsgaq` SG&A;
`xrdq` R&D; `dpq` depreciation; `oibdpq` operating income before depreciation
and amortization; `xintq`/`xintdy` interest expense (quarterly, annual
fallback); `nopiq` non-operating income; `niq`/`ibq` net income and income
before extraordinary items.

**Cash flow and derived**

`oancfy` operating cash flow year-to-date cumulative; `oancfq` operating cash
flow per quarter, derived from `oancfy` by de-cumulation within each fiscal
year. Annual-only flow variables are disaggregated to quarters by seasonal
share; stock variables use point-in-time logic. This disaggregation is applied
upstream during panel reconstruction and is already reflected in the shipped
columns.

**Reference-sample split** (in `panel_with_split.parquet` only)

| Column | Type | Meaning |
|---|---|---|
| `_split` | str | reference-sample membership; the clean reference sample used to calibrate each authority's bootstrap null is read from this column |

## Macro connectors (Islamic corrections)

The Islamic-correction ratios (sukuk share, Islamic cash share) read small
annual connector tables from `data/raw/macro_connectors/`:

```
data/raw/macro_connectors/
  sukuk_annual.csv                 global fallback
  islamic_cash_annual.csv          global fallback
  <country>/sukuk_annual.csv       per-country override (optional)
  <country>/islamic_cash_annual.csv
```

Each is a small annual table keyed by year (and country); when a connector is
absent, the corresponding adjusted ratio is left missing and a warning is
logged. These tables are **shipped in this repository**: they are aggregate
macro series from public official sources (central banks, market authorities),
and every row carries a `*_source` column documenting its origin.

## De-identified score bundle (`data/scores/<country>/`)

This is the shipped, license-safe form of the frozen run outputs. Every firm key
(`gvkey`) is replaced by a **stable synthetic integer** (consistent across all
files so firm- and row-level artifacts still join); company names and security
identifiers are removed; and no licensed Compustat monetary field appears. It
contains the detector z-scores (`zscores/`), the composite statistics and their
p-/q-values (`phase4_composites/`, `phase7_fdr/`), the confidence qualifier
(`phase4b_confidence/`), the null-calibration and dependence diagnostics, the
power/robustness studies, the qualitative cross-sections, and the paper-ready
result tables. `reproduce.py results` reads only this bundle.

## Provenance

Source: S&P Global Compustat Global, quarterly fundamentals. Sample: five
Sharia-screening regimes (Malaysia, Indonesia, Pakistan, Saudi Arabia, United
Arab Emirates). The panels are reconstructed upstream (ratio construction, annual-to-quarterly
disaggregation, integrity filtering, reference-sample split); the raw vendor
extract and the panel-construction code are not part of this repository, which
reproduces the scoring and robustness results from the shipped panels.
