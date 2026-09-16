# Empirical Economics submission (September 2026)

Revised and reframed version of the broadband price paper after rejections at the
Journal of Regulatory Economics and Information Economics and Policy.

- `paper/manuscript.pdf` – submission manuscript (with authors); `manuscript_anonymous.pdf` – blind copy
- `paper/ESM_1.pdf` – Online Resource 1; `paper/cover_letter.pdf`
- `REVISION_NOTES.md` – what changed, data corrections, point-by-point mapping to earlier reviews, checklist
- `code/` – full pipeline (see REVISION_NOTES §6 for the run order); scripts expect to run from a
  working directory containing `panel.pkl`, `results/`, `raw/`, `eurostat/`, `paper/`
- `results/results.json` – every number in the paper; `analysis_panel_v2.csv` – the analysis panel
- `data/` – ITU price workbook, cached World Bank API responses (13 July 2026 release),
  Eurostat regional extracts, country night-light sums

The older `manuscript/` and `springer/` folders are kept unchanged for the record.
