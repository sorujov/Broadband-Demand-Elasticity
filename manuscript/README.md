# Submission Build Instructions

This folder contains the LaTeX source for the manuscript. It is organized to compile cleanly with relative paths and Elsevier's `elsarticle` class.

## Structure
- paper.tex — main manuscript
- sections/ — section files (abstract, introduction, literature, data, methodology, results, discussion, conclusion)
- tables/ — regression tables (table1_baseline, table2_covid, table3_price_robustness, table4_placebo)
- styles/ — Elsevier document class and bibliography style files
- figures/ — publication figures (accessed via `\graphicspath{{figures/}}`)
- references.bib — bibliography database

## Compile
Run from this folder:

```bash
pdflatex -interaction=nonstopmode paper.tex
bibtex paper
pdflatex -interaction=nonstopmode paper.tex
pdflatex -interaction=nonstopmode paper.tex
```

## Notes
- `\graphicspath{{figures/}}` ensures figures load from `manuscript/figures/`.
- All section inputs use `\input{sections/<name>}`.
- All table inputs use `\input{tables/<name>}`.
- Ensure the `figures/` folder is included when sharing source.
