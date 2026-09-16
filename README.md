# What can country panels tell us about broadband price responsiveness?

**Evidence from the European Union and the Eastern Partnership, 2010–2024**

Samir Orujov (ADA University; ICTA; CERGE-EI) · Ilgar Ismayilov (ICTA; Charles University) · Jeyhun Huseynzade (ICTA)

Replication package for the article submitted to *Empirical Economics* (September 2026).

## Summary

A two-way fixed-effects regression of log fixed-broadband subscriptions on log ITU entry-level prices
for 27 EU and six Eastern Partnership (EaP) countries gives a sizeable negative coefficient whose
year-specific values rise towards zero. A simulation with no price effect reproduces this pattern, and
it disappears once country trends are allowed or the model is estimated in differences. In the EU the
association is a tightly bounded zero (first differences, 90% CI [−0.04, 0.02]). In the EaP it rests on
movements common to the six countries and vanishes with EaP-specific year effects. Country panels of
this kind do not identify a price response.

## Repository layout

| Folder | Contents |
|---|---|
| `empirical_economics/` | **Current replication package** — code, data, results, manuscript and Online Resource 1 |
| `empirical_economics/code/` | Pipeline; `run_all.py` reproduces everything |
| `empirical_economics/data/` | ITU price workbook, World Bank API responses, Eurostat regional extracts, night-light sums |
| `empirical_economics/results/` | `results.json` (every number in the article) and the analysis panel |
| `empirical_economics/paper/` | LaTeX source, tables, figures, PDFs |
| `archive/` | Earlier version of the analysis (2025–2026), kept for the record; superseded |

## Reproducing the results

```bash
cd empirical_economics
pip install -r code/requirements.txt
python code/run_all.py          # about 3 minutes; rewrites results/, paper/tables, paper/figures
cd paper && pdflatex manuscript && bibtex manuscript && pdflatex manuscript && pdflatex manuscript
```

`run_all.py` ends with `verify_numbers.py`, which checks every decimal number in the manuscript text
against `results/results.json`; the only unmatched values are estimates quoted from other studies.
Set `REFRESH_WB=1` to download the World Bank series again instead of using the cached responses.
The night-light sums are produced by `code/ntl_extract.py` from the public harmonised DMSP/VIIRS rasters
(Li et al. 2020, figshare) and Natural Earth country boundaries, which are not redistributed here.

## Data sources

- ITU, ICT Price Baskets, 2008–2025 (fixed-broadband and mobile-broadband baskets)
- World Bank, World Development Indicators and Worldwide Governance Indicators (API, 13 July 2026)
- Eurostat, `isoc_r_broad_h` (household broadband access) and `nama_10r_2gdp` (regional GDP)
- Li, X., Zhou, Y., Zhao, M. and Zhao, X. (2020), A harmonized global nighttime light dataset 1992–2018, *Scientific Data* 7, 168 (2024 update)

Third-party data remain subject to their providers' terms of use.

## License

Code: MIT (see `LICENSE`). Derived data and results: CC BY 4.0.

## Citation

See `CITATION.cff`. An archived version of each release is available on Zenodo.
