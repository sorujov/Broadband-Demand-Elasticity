"""Reproduce every table, figure and number of the Empirical Economics submission.

Run from the `empirical_economics/` folder:   python code/run_all.py
The scripts use a flat working layout (raw/, eurostat/, results/, paper/, figs/).
This runner builds that layout in `_work/`, runs the pipeline in order and copies
the outputs back to `results/` and `paper/`.  Runtime: about 5-10 minutes.

Inputs shipped with the package:
  data/raw/itu_prices.xlsx        ITU ICT Price Baskets 2008-2025
  data/wbcache/*.json             World Bank API responses (WDI/WGI, 13 July 2026 release)
  data/raw/ntl_country.csv        country sums of harmonised night-time lights (Li et al. 2020);
                                  produced by code/ntl_extract.py from the public rasters
  data/eurostat/*.csv             Eurostat isoc_r_broad_h and nama_10r_2gdp extracts
Set REFRESH_WB=1 to re-download the World Bank series instead of using the cache.
"""
import os, shutil, subprocess, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
W = ROOT / '_work'
STEPS = ['build_data_v2.py', 'analysis.py', 'analysis2.py', 'analysis3_ivfd.py',
         'analysis4_referee.py', 'simulate.py', 'figures.py', 'make_tables.py',
         'make_esm.py', 'verify_numbers.py']

def main():
    if W.exists():
        shutil.rmtree(W)
    (W / 'raw').mkdir(parents=True)
    for f in (ROOT / 'data/raw').iterdir():
        shutil.copy(f, W / 'raw' / f.name)
    if os.environ.get('REFRESH_WB') != '1':
        shutil.copytree(ROOT / 'data/wbcache', W / 'raw/wbcache')
    shutil.copytree(ROOT / 'data/eurostat', W / 'eurostat')
    (W / 'results').mkdir()
    (W / 'figs').mkdir()
    (W / 'paper/tables').mkdir(parents=True)
    shutil.copy(ROOT / 'paper/manuscript.tex', W / 'paper/manuscript.tex')
    for f in (ROOT / 'code').glob('*.py'):
        shutil.copy(f, W / f.name)
    env = dict(os.environ, MPLBACKEND='Agg')
    for s in STEPS:
        print(f'--- {s}', flush=True)
        subprocess.run([sys.executable, s], cwd=W, env=env, check=True)
    for f in (W / 'results').iterdir():
        shutil.copy(f, ROOT / 'results' / f.name)
    for f in (W / 'figs').iterdir():
        shutil.copy(f, ROOT / 'paper/figures' / f.name)
    for f in (W / 'paper/tables').iterdir():
        shutil.copy(f, ROOT / 'paper/tables' / f.name)
    shutil.copy(W / 'paper/ESM_1.tex', ROOT / 'paper/ESM_1.tex')
    print('Done. Compile paper/manuscript.tex and paper/ESM_1.tex with pdflatex + bibtex.')

if __name__ == '__main__':
    main()
