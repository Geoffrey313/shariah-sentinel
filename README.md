# ShariaSentinel — reproducibility repository

Minimal code + reconstructed data to **reproduce the empirical results** of the
ShariaSentinel anomaly-screening paper. This repository is reproduction-only: it
contains **no** application/serving code, **no** unit tests, and **no** raw
vendor data — only the reconstructed per-country panels and the scoring /
robustness pipeline needed to regenerate the reported numbers.

## Layout

```
main.py                       single argparse entry point (run one step or all)
requirements.txt
shariasentinel/               the pipeline package
  common/  panel/             ratio construction, per-country methodology
  detectors/                  z1–z9 anomaly detectors (+ PIT calibration)
  scoring/                    composites, bootstrap nulls, phase runners, config
  robustness/                 robustness benchmark (families 1–4)
    family3_counterfactual/   counterfactual + SAC-projected PGD evasion
data/
  scores/<country>/phase0_reference_sample/panel_with_split.parquet
                              reconstructed panels (mys, uae, qat, sau, pak, kwt)
```

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

Runs on CPU. `torch` uses CUDA automatically if available (optional).

## Usage

Every analysis is a subcommand of `main.py`. The shared option is
`--country {mys,uae,qat,sau,pak,kwt}` (default `mys`). All results are written
under `data/scores/<country>/` (git-ignored — only the input panels are
versioned). Run `python main.py <command> --help` for the full option list.

### Quick reference

| Command | Reproduces | Typical runtime |
|---|---|---|
| `score` | detector z-scores (z1–z9 → z57) | seconds–minutes |
| `benchmark` | robustness families 1–4 (incl. AnoShift) | minutes–hours |
| `family3-cf` | counterfactual RED→GREEN flip rate + variable composition | ~1 h at 500 rows |
| `family3-pgd` | SAC-projected PGD adversarial-evasion result | minutes |
| `all` | `score` then `benchmark` | as above |

### `score` — detector z-scores

Recomputes all detector z-scores from the reconstructed panel (the input to
every downstream table).

```bash
python main.py score --country mys
```

Writes `data/scores/<country>/z_scores.parquet`.

### `benchmark` — robustness families 1–4

Runs the full robustness benchmark: Family 1 (correlated-Gaussian
contamination), Family 2 (realistic manipulation mechanisms), Family 3
(adversarial), and Family 4 (AnoShift temporal robustness). Slow.

```bash
python main.py benchmark --country mys            # resumes from checkpoints
python main.py benchmark --country mys --no-resume  # recompute every family
```

Writes tables + figures under `data/scores/<country>/robustness_benchmark/`.

### `family3-cf` — counterfactual (RED → GREEN)

Finds the smallest accounting move that flips a near-boundary RED
firm-quarter to GREEN, and reports the flip rate and per-variable
perturbation composition (paper §"Adversarial counterfactual robustness").

```bash
# headline paper run: 500 near-boundary RED rows, 19-field attack surface
python main.py family3-cf --country mys --rows 500 --near-boundary
```

Options:
- `--rows N` — number of firm-quarters to attack (default 500).
- `--near-boundary` — select the top-N RED rows closest to the threshold
  (deterministic); omit to use the package's seeded eligible-pool sample.
- `--target {z_plus_renorm,z_plus,breadth,z_mahalanobis_sq,t_iut}` — target
  composite (default `z_plus_renorm`, the renormalised truncated sum).
- `--direction {to_green,to_red}` — flip direction (default `to_green`).

Prints the flip rate and writes the per-row results CSV under
`data/scores/<country>/robustness_benchmark/`.

### `family3-pgd` — SAC-projected PGD evasion

Runs the projected-gradient adversarial-evasion attack on the highest-risk
RED rows, projecting every step back onto the SAC ratio constraints, and
reports the evasion rate at `eps in {0, 0.05}` (paper §"Adversarial evasion").

```bash
python main.py family3-pgd --country mys --rows 5
```

Options: `--rows N` — number of highest-risk RED rows to attack (default 5).
Prints the evasion rate per epsilon level.

### `all` — score + benchmark end-to-end

```bash
python main.py all --country mys
```

## Notes on reproducibility

- The reconstructed panels already carry the reference-sample split
  (`_split`), so `score` recomputes all detector z-scores directly from them;
  no raw Compustat data is required.
- Randomised steps (bootstrap nulls, Family-3 candidate sampling) are seeded via
  the settings in `shariasentinel/scoring/config.py`, so runs are deterministic.
