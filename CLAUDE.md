# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Econometric analysis of broadband price elasticity of demand comparing EU (27 countries) and Eastern Partnership (6 countries) using panel data methods (2010-2024). The project implements OLS, fixed effects, and IV/2SLS estimation with robustness checks. Target publication: Telecommunications Policy journal.

## Common Commands

```bash
# Activate virtual environment (Windows)
.venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Run full pipeline
python code/main.py

# Run data preparation only
python code/data_preparation/02_prepare_data.py

# Run analysis only (skip data collection/preparation)
python code/main.py --analysis-only

# Run individual analysis scripts
python code/analysis/02_main_analysis.py
python code/analysis/05_panel_iv_estimation.py
```

## Data Pipeline

### Stage 1: Data Collection (`code/data_collection/`)
- `step1_download_itu.py`: Downloads ITU telecommunications data
- `step2_download_worldbank.py`: Downloads World Bank indicators
- `step3_process_raw_data.py`: Cleans raw data files
- `step4_merge_datasets.py`: Merges ITU + World Bank
- Output: `data/processed/data_merged_with_series.xlsx`

### Stage 2: Data Preparation (`code/data_preparation/`)
- `02_prepare_data.py`: Creates analysis-ready dataset
  - Applies column mappings from `config.py`
  - Missing data: Forward fill + linear interpolation
  - Creates log transformations
  - Creates regional indicators and interactions
  - Creates lagged variables for IV
- Output: `data/processed/analysis_ready_data.csv`

### Stage 3: Analysis (`code/analysis/`)
- `02_main_analysis.py`: Main OLS/FE specifications
- `05_panel_iv_estimation.py`: IV/2SLS estimation for price endogeneity
- Output: `results/` subdirectories

## Central Configuration

`code/utils/config.py` contains all project configuration:

### Paths
- `DATA_MERGED_FILE`: Input from data collection
- `ANALYSIS_READY_FILE`: Output from data preparation
- `RESULTS_DIR`, `FIGURES_DIR`: Output directories

### Variables
- `PRIMARY_DV = 'log_fixed_broadband_subs'`: Fixed broadband subscriptions per 100 inhabitants
- `PRIMARY_PRICE = 'log_fixed_broad_price'`: GNI-adjusted broadband price
- `ROBUSTNESS_DVS`: Alternative DVs (internet_users_pct, int_bandwidth)
- `COLUMN_MAPPINGS`: ITU series names -> standardized names
- `LOG_TRANSFORM_VARS`: Variables to log-transform

### Country Lists
- `EU_COUNTRIES`: 27 EU member states (ISO3 codes)
- `EAP_COUNTRIES`: 6 Eastern Partnership countries (ISO3 codes)

## Methodology

### Dependent Variables
- **Primary**: `log_fixed_broadband_subs` (subscriptions per 100 inhabitants)
- **Robustness**: `log_internet_users_pct`, `log_int_bandwidth`

### Price Variable
- **Primary**: `log_fixed_broad_price` (% of GNI per capita - measures affordability)
- **Robustness**: USD prices, PPP-adjusted prices

### Identification Strategy
- Two-way fixed effects (country + year)
- Regional heterogeneity via `Price × EaP` interaction
- IV/2SLS addressing price endogeneity (instruments: mobile prices, regulatory quality, lagged prices)

### Panel Structure
- Countries: 33 (27 EU + 6 EaP)
- Years: 2010-2024
- Standard errors: Clustered by country

## Key Dependencies
- **linearmodels**: Panel data models (PanelOLS, IV2SLS)
- **statsmodels**: OLS, diagnostics
- **pandas**: Data manipulation
- **numpy**: Numerical operations

## Optional Tools
- `quick_method/method_diagnostic.py`: Automated method selection guidance
