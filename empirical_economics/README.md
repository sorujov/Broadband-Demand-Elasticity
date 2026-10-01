# Empirical Economics submission (September 2026)

Replication package for "What can country panels tell us about broadband price responsiveness?
Evidence from the European Union and the Eastern Partnership, 2010–2024".

## Run

```bash
pip install -r code/requirements.txt
python code/run_all.py      # ~3 min; rebuilds the panel, all estimates, tables, figures and Online Resource 1
```

`run_all.py` copies the inputs into a temporary `_work/` folder, runs the scripts in this order and
copies the outputs back:

| Step | Script | Output |
|---|---|---|
| 1 | `build_data_v2.py` | analysis panel (`results/analysis_panel_v2.csv`, imputation counts) |
| 2 | `analysis.py` | main estimates, Tables 3–5 and 7, S1–S4 → `results/results.json` |
| 3 | `analysis2.py` | long differences, decomposition, timing placebo, night lights, sub-periods |
| 4 | `analysis3_ivfd.py` | first-difference Hausman-type IV |
| 5 | `analysis4_referee.py` | FD with country effects, group-specific year effects, mean group |
| 6 | `simulate.py` | Monte Carlo with no price effect (Fig. S1) |
| 7 | `figures.py`, `make_tables.py`, `make_esm.py` | `paper/figures`, `paper/tables`, `paper/ESM_1.tex` |
| 8 | `verify_numbers.py` | checks every number in the manuscript text against `results.json` |

Helper modules: `common.py` (fixed-effects algebra), `infer.py` (wild cluster restricted bootstrap,
randomisation inference), `dyn.py` (Arellano–Bond GMM). Data-preparation helpers not in the default run:
`ntl_extract.py` (night-light sums from the public rasters), `es_parse.py` and `regional.py`
(Eurostat JSON → `data/eurostat/*.csv`).

## Contents

- `data/raw/itu_prices.xlsx` – ITU ICT Price Baskets 2008–2025; `data/raw/ntl_country.csv` – night-light sums;
  `data/wbcache/` – World Bank API responses (13 July 2026 release); `data/eurostat/` – regional extracts
- `results/results.json` – every number in the article; `results/analysis_panel_v2.csv` – the analysis panel
- `paper/` – `manuscript.tex`/`.pdf`, `ESM_1.tex`/`.pdf` (Online Resource 1), tables, figures, bibliography
- `REVISION_NOTES.md` – changes relative to the earlier version and data corrections
