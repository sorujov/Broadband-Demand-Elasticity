# Data Collection Module

This folder contains all scripts for downloading and preparing raw data from ITU and World Bank APIs.

## 📁 File Structure

```
data_collection/
├── run_data_collection.py          # 🚀 MASTER ORCHESTRATOR - Run this!
├── step1_download_itu.py            # Download ITU telecommunications data
├── step2_download_worldbank.py      # Download World Bank economic indicators  
├── step3_process_raw_data.py        # Process raw data, preserve metadata
├── step4_merge_datasets.py          # Merge ITU + World Bank into analysis-ready format
└── README.md                        # This file
```

## 🚀 Quick Start

### Run Complete Pipeline
```bash
python code/data_collection/run_data_collection.py
```

This will:
1. Download 6 ITU indicators (fixed/mobile broadband, internet users, bandwidth)
2. Download 22 World Bank indicators (GDP, education, R&D, etc.)
3. Process raw data and preserve series metadata
4. Merge datasets into `data/processed/data_merged_with_series.csv`

**Expected Runtime:** ~3-5 minutes  
**Output:** 495 observations (33 countries × 15 years: 2010-2024)

---

## 📋 Advanced Usage

### Skip Downloads (Use Existing Data)
```bash
python code/data_collection/run_data_collection.py --skip-download
```
Only runs processing and merge steps. Useful when you already have raw data.

### Download Only ITU Data
```bash
python code/data_collection/run_data_collection.py --itu-only
```

### Download Only World Bank Data
```bash
python code/data_collection/run_data_collection.py --wb-only
```

### Show Detailed Output
```bash
python code/data_collection/run_data_collection.py --verbose
```
Displays full output from each step (useful for debugging).

---

## 📊 Individual Scripts

### Step 1: Download ITU Data
```bash
python code/data_collection/step1_download_itu.py
```

#### ITU Data Structure & API Usage

**Data Sources:**

ITU provides telecommunications data through two sources:
1. **Excel File** (NEW - Primary Source): Official price baskets with complete historical data
   - URL: https://www.itu.int/en/ITU-D/Statistics/Documents/publications/prices2024/ITU_ICTPriceBaskets_2008-2024.xlsx
   - Contains: Fixed-broadband and Mobile-broadband baskets
   - Format: Wide format (years as columns), reshaped to long format
   - Coverage: 2010-2024 with USD, GNI%, and PPP prices
   - **Advantage**: 98.4% PPP coverage (487/495 obs) vs API's 42.4% (196 obs)

2. **DataHub API** (Secondary Source): Other telecommunications indicators
   - URL: https://api.datahub.itu.int/v2
   - Used for: Subscriptions, internet users, bandwidth
   - Each indicator has Code ID and multiple series

**Why We Use Excel for Prices:**

The ITU API only provides PPP price data from 2018+, creating a gap for 2010-2017. The official Excel file contains the complete historical series with 98.4% PPP coverage across all years.

**Discovery Process:**

1. **Price Data (Excel)**:
   - Download Excel file with all price baskets
   - Filter to Fixed-broadband and Mobile-broadband baskets
   - Filter to our 33 countries and 2010-2024 period
   - Reshape from wide (years as columns) to long format
   - Pivot units (USD/GNI%/PPP) to separate columns

2. **Other Indicators (API)**:
   - Get Catalog: `https://api.datahub.itu.int/v2/dictionaries/getcategories`
   - Download Data: `https://api.datahub.itu.int/v2/data/download/byid/{CODE_ID}/iscollection/false`
   - Returns ZIP file with CSV containing all series

**What We Download:**

| Indicator | Source | Coverage | Data Structure |
|-----------|--------|----------|----------------|
| **Fixed broadband prices** | **Excel** | **495 obs (2010-2024)** | **USD, GNI%, PPP as columns** |
| **Mobile broadband prices** | **Excel** | **394 obs (2010-2024)** | **USD, GNI%, PPP as columns** |
| Fixed broadband subscriptions | API (19303) | 990 obs | 2 series |
| Mobile subscriptions | API (178) | 984 obs | 2 series |
| Internet users % | API (11624) | 484 obs | 1 series |
| International bandwidth | API (242) | 1145 obs | 4 series |

**Total Downloaded:** 6 indicators (2 from Excel with 3 price types each, 4 from API with 9 series)

**Key Improvement (Dec 2025):**
- **Before**: API-only approach, PPP data missing for 2010-2017 (64.7% missing)
- **After**: Excel file provides official PPP data for full period (only 1.6% missing!)
- **Result**: Fixed-broadband PPP coverage improved from 196 obs → 487 obs (+148%)

**Output:** 6 CSV files in `data/raw/itu_*.csv`
- `itu_fixed_broad_price.csv`: 495 obs with price_usd, price_gni_pct, price_ppp columns
- `itu_mobile_broad_price.csv`: 394 obs with price_usd, price_gni_pct, price_ppp columns
- `itu_fixed_broadband_subs.csv`: 990 obs (2 series)
- `itu_mobile_subs.csv`: 984 obs (2 series)
- `itu_internet_users_pct.csv`: 484 obs (1 series)
- `itu_int_bandwidth.csv`: 1145 obs (4 series)

### Step 2: Download World Bank Data
```bash
python code/data_collection/step2_download_worldbank.py
```

**Downloads:** 22 indicators including:
- GDP per capita, growth rate
- Education (secondary completion, tertiary enrollment)
- R&D expenditure
- ICT exports
- Secure internet servers
- Regulatory quality
- Demographics (urban population)

**Output:** `data/raw/worldbank_data.csv` (495 rows × 21 columns)

**Note:** 3 indicators may fail (discontinued by World Bank API):
- ease_of_doing_business_score
- time_required_start_business  
- cost_business_startup

This is expected and does not affect the analysis.

### Step 3: Process Raw Data
```bash
python code/data_collection/step3_process_raw_data.py
```

#### Processing ITU Multi-Format Data

**Challenge:** ITU data comes in two different formats:
1. **Price data (Excel)**: Already in wide format with USD/GNI%/PPP as separate columns
2. **Non-price data (API)**: Long format with multiple series per indicator

**Example Raw Structures:**

*Price Data (Excel):*
```csv
country_iso3,dataYear,price_usd,price_gni_pct,price_ppp,seriesCode
ARM,2018,10.33,2.93,25.62,fixed_broadband
```

*Non-Price Data (API):*
```csv
country,year,seriesCode,seriesName,seriesUnits,dataValue
ARM,2018,i4213tfbb,Fixed broadband subscriptions,per 100 people,15.2
```

**Processing Steps:**

1. **Detect Format**: Check if file has `price_usd`, `price_gni_pct`, `price_ppp` columns
   - If yes → Price data (already wide, 3 columns per indicator)
   - If no → Non-price data (needs pivoting)

2. **Price Data Processing**:
   - Rename columns to standard format: `fixed_broad_price_usd`, `fixed_broad_price_gni_pct`, `fixed_broad_price_ppp`
   - Each price type becomes a separate variable
   - Coverage reporting for each price type

3. **Non-Price Data Processing**:
   - Identify all unique series codes
   - Create variable names: `indicator_seriescode` (e.g., `fixed_broadband_subs_i4213tfbb`)
   - Preserve metadata (series codes, units)

4. **Document in Catalog**: Create `data_catalog.xlsx`
   - ITU Excel: 6 price variables (2 indicators × 3 price types)
   - ITU API: 9 series variables
   - World Bank: 19 indicators
   - **Total: 34 variables documented**

**Why This Matters:**
- Handles both price (wide) and non-price (long) formats correctly
- Analyst can choose which price type to use (USD, GNI%, or PPP)
- Metadata ensures correct interpretation
- Series codes enable traceability to source

**Output:** 
- 6 processed ITU files in `data/interim/itu_*_processed.xlsx`
  - 2 price files (already wide with 3 price columns each)
  - 4 non-price files (long format with series codes)
- `data/interim/worldbank_processed.xlsx`
- `data/interim/data_catalog.xlsx` (34 variables documented)

### Step 4: Merge Datasets
```bash
python code/data_collection/step4_merge_datasets.py
```

#### Merging Multi-Format ITU Data with World Bank

**Conversion to Wide Format:**

*Price data (Excel)* - Already wide, just select columns:
```
country,year,fixed_broad_price_usd,fixed_broad_price_gni_pct,fixed_broad_price_ppp
ARM,2018,10.33,2.93,25.62
```

*Non-price data (API)* - Convert from long to wide:
```
Before (long):
ARM,2018,fixed_broadband_subs_i4213tfbb,15.2
ARM,2018,fixed_broadband_subs_i992b,15.2

After (wide):
country,year,fixed_broadband_subs_i4213tfbb,fixed_broadband_subs_i992b
ARM,2018,15.2,15.2
```

**Merge Process:**

1. **Separate Price and Non-Price Data**: Detect format by checking for price columns
2. **Process Price Data**: Select country, year, and 3 price columns (already wide)
3. **Process Non-Price Data**: Pivot 9 series into 9 columns  
4. **Merge ITU Data**: Combine all 15 ITU variables (6 price + 9 series)
5. **Convert World Bank to Wide**: Pivot 19 indicators into 19 columns
6. **Final Merge**: Inner join on country-year (all 495 obs match)
7. **Preserve NAs**: Missing values kept (not imputed)

**Why Multiple Price Types Matter:**

For regression analysis, you can now choose:
- **USD prices** - Nominal prices in US dollars
- **GNI-adjusted** - Affordability measure (% of income)
- **PPP-adjusted** - Purchasing power parity prices (recommended for cross-country comparison)

Each price type answers different research questions about price elasticity.

**Output:**
- `data/processed/data_merged_with_series.xlsx` (495 obs × 37 vars)
  - 6 price variables (USD/GNI%/PPP for fixed & mobile) + 9 API series + 19 WB indicators + 3 metadata
- `data/processed/itu_series_reference.xlsx` (series guide for variable selection)
- `data/processed/missing_data_report.xlsx` (coverage statistics)

---

## 📈 Output Summary

After running the complete pipeline, you'll have:

### Raw Data (`data/raw/`) - CSV Format
- 6 ITU CSV files (4,992 total rows)
  - 2 price files: 495 + 394 = 889 rows
  - 4 other files: 990 + 984 + 484 + 1145 = 3,603 rows
- 1 World Bank CSV file (495 rows × 21 columns)

### Interim Data (`data/interim/`) - Excel Format
- 6 processed ITU files (.xlsx)
- 1 processed World Bank file (long format, .xlsx)
- 1 data catalog (variable documentation, .xlsx)

### Final Data (`data/processed/`) - Excel Format
- **`data_merged_with_series.xlsx`** ← Main analysis file
  - 495 observations (33 countries × 15 years: 2010-2024)
  - 37 variables (6 ITU price + 9 ITU series + 19 World Bank + 3 metadata)
  - Regional split: EU (405 obs), EaP (90 obs)
- **`itu_series_reference.xlsx`** ← Series selection guide
  - Lists all 15 ITU variables with codes, units, descriptions
  - Use this to choose which price type for regression
- **`missing_data_report.xlsx`** ← Data coverage report
  - Shows missing data % for each variable
  - Fixed-broadband PPP: 98.4% coverage (487/495 obs, 2010-2024)

---

## 🔍 Key Variables in Final Dataset

### Dependent Variable
- `internet_users_pct_i99H` - Internet users (% of population)

### Independent Variables (Price) - Choose One:
- `fixed_broad_price_usd` - Fixed broadband price in USD (495 obs, 2010-2024)
- `fixed_broad_price_gni_pct` - Price as % GNI per capita (494 obs, affordability measure)
- `fixed_broad_price_ppp` - **Price in PPP$ (487 obs, purchasing power adjusted)** ← Recommended

**Why Multiple Price Types?**
- **USD** - Nominal prices, good for cross-country comparison
- **GNI%** - Affordability (price relative to income), captures ability to pay
- **PPP** - Real prices adjusted for purchasing power, controls for cost of living differences
- **Coverage**: All three types have 98%+ coverage across 2010-2024 period

### Controls
- `gdp_per_capita` - GDP per capita (constant 2015 USD)
- `gdp_growth` - GDP growth (annual %)
- `education_secondary_pct` - Secondary education completion rate
- `rd_expenditure` - R&D expenditure (% GDP)
- `secure_internet_servers` - Secure servers (per 1M people)
- `regulatory_quality` - Regulatory quality index
- `urban_population_pct` - Urban population (%)
- And 12 more...

---

## 🛠️ Troubleshooting

### Issue: World Bank API errors for 3 indicators
**Solution:** Expected behavior. Those indicators were discontinued. Script continues with 19 other indicators.

### Issue: ITU download slow
**Solution:** ITU API sometimes has rate limits. Script includes retry logic. If it fails, wait 5 minutes and run again.

### Issue: Missing data in output
**Solution:** Normal. 1,646 missing values documented in `missing_data_report.xlsx`. These are preserved (not imputed) for analysis. Most complete variable: fixed_broad_price_ppp with 98.4% coverage.

---

## ⏭️ Next Steps

After data collection, run data preparation:

```bash
python code/data_preparation/01_analysis.py     # Exploratory data analysis
python code/data_preparation/02_prepare_data.py # Create analysis-ready dataset
```

Or run the complete analysis pipeline:
```bash
python code/main.py
```

---

## 📝 Notes

- **Countries:** 27 EU + 6 EaP (Armenia, Azerbaijan, Belarus, Georgia, Moldova, Ukraine)
- **Time Period:** 2010-2024 (15 years)
- **Missing Data:** Preserved, not imputed. Fixed-broadband PPP has 98.4% coverage (487/495 obs).
- **ITU Data Sources:** Excel file for prices (2010-2024 with full PPP coverage), API for other indicators
- **Key Improvement:** Switched from API to Excel for price data, improving PPP coverage from 42.4% to 98.4%
- **Data Structure:** Price data has USD/GNI%/PPP as separate columns; other indicators have multiple series codes

---

**Last Updated:** December 11, 2025  
**Author:** Samir Orujov
