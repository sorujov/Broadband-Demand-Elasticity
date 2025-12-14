# code/utils/config.py
"""
================================================================================
Project Configuration
================================================================================
Central configuration file for the broadband elasticity project.
Contains paths, country lists (ISO3 codes), and analysis parameters.
================================================================================
"""

import os
from pathlib import Path

# ============================================================================
# PROJECT PATHS
# ============================================================================

# Get project root directory (go up from code/utils/ to project root)
PROJECT_ROOT = Path(__file__).parent.parent.parent
DATA_DIR = PROJECT_ROOT / 'data'
CODE_DIR = PROJECT_ROOT / 'code'
RESULTS_DIR = PROJECT_ROOT / 'results'
DOCS_DIR = PROJECT_ROOT / 'docs'

# Data subdirectories
DATA_RAW = DATA_DIR / 'raw'
DATA_INTERIM = DATA_DIR / 'interim'
DATA_PROCESSED = DATA_DIR / 'processed'

# Results subdirectories
RESULTS_REGRESSION = RESULTS_DIR / 'regression_output'

# Create directories if they don't exist
for path in [DATA_RAW, DATA_INTERIM, DATA_PROCESSED, 
             RESULTS_REGRESSION]:
    path.mkdir(parents=True, exist_ok=True)

# ============================================================================
# COUNTRY CLASSIFICATIONS (ISO3 CODES)
# ============================================================================

# European Union countries (27 members) - ISO3 codes
EU_COUNTRIES = [
    'AUT',  # Austria
    'BEL',  # Belgium
    'BGR',  # Bulgaria
    'CYP',  # Cyprus
    'CZE',  # Czech Republic
    'DEU',  # Germany
    'DNK',  # Denmark
    'ESP',  # Spain
    'EST',  # Estonia
    'FIN',  # Finland
    'FRA',  # France
    'GRC',  # Greece
    'HRV',  # Croatia
    'HUN',  # Hungary
    'IRL',  # Ireland
    'ITA',  # Italy
    'LTU',  # Lithuania
    'LUX',  # Luxembourg
    'LVA',  # Latvia
    'MLT',  # Malta
    'NLD',  # Netherlands
    'POL',  # Poland
    'PRT',  # Portugal
    'ROU',  # Romania
    'SVK',  # Slovakia
    'SVN',  # Slovenia
    'SWE',  # Sweden
]

# Eastern Partnership countries (6 members) - ISO3 codes
EAP_COUNTRIES = [
    'ARM',  # Armenia
    'AZE',  # Azerbaijan
    'BLR',  # Belarus
    'GEO',  # Georgia
    'MDA',  # Moldova
    'UKR',  # Ukraine
]

# All countries in analysis
ALL_COUNTRIES = EU_COUNTRIES + EAP_COUNTRIES

# Country name mappings (ISO3 to full names)
COUNTRY_NAMES = {
    # EU Countries
    'AUT': 'Austria',
    'BEL': 'Belgium',
    'BGR': 'Bulgaria',
    'CYP': 'Cyprus',
    'CZE': 'Czech Republic',
    'DEU': 'Germany',
    'DNK': 'Denmark',
    'ESP': 'Spain',
    'EST': 'Estonia',
    'FIN': 'Finland',
    'FRA': 'France',
    'GRC': 'Greece',
    'HRV': 'Croatia',
    'HUN': 'Hungary',
    'IRL': 'Ireland',
    'ITA': 'Italy',
    'LTU': 'Lithuania',
    'LUX': 'Luxembourg',
    'LVA': 'Latvia',
    'MLT': 'Malta',
    'NLD': 'Netherlands',
    'POL': 'Poland',
    'PRT': 'Portugal',
    'ROU': 'Romania',
    'SVK': 'Slovakia',
    'SVN': 'Slovenia',
    'SWE': 'Sweden',
    # EaP Countries
    'ARM': 'Armenia',
    'AZE': 'Azerbaijan',
    'BLR': 'Belarus',
    'GEO': 'Georgia',
    'MDA': 'Moldova',
    'UKR': 'Ukraine',
}

# Reverse mapping (full name to ISO3)
COUNTRY_CODES = {v: k for k, v in COUNTRY_NAMES.items()}

# ============================================================================
# TIME PERIOD
# ============================================================================

START_YEAR = 2010  # Price data available from 2010
END_YEAR = 2024    # Most recent complete year
YEARS = list(range(START_YEAR, END_YEAR + 1))

# ============================================================================
# DATA PIPELINE CONFIGURATION
# ============================================================================

# Input/output files
DATA_MERGED_FILE = DATA_PROCESSED / 'data_merged_with_series.xlsx'
ANALYSIS_READY_FILE = DATA_PROCESSED / 'analysis_ready_data.csv'

# Column mappings: ITU raw series names -> standardized names
COLUMN_MAPPINGS = {
    # Demand measures (potential DVs)
    'fixed_broadband_subs_i4213tfbb': 'fixed_broadband_subs',  # per 100 inhabitants
    'fixed_broadband_subs_i992b': 'fixed_broadband_subs_alt',  # alternative series
    'internet_users_pct_i99H': 'internet_users_pct',           # % of population
    'int_bandwidth_i4214': 'int_bandwidth',                    # Gbit/s (primary)
    'int_bandwidth_i994': 'int_bandwidth_alt',                 # alternative series
    'mobile_subs_i271': 'mobile_subs',                         # per 100 inhabitants

    # Price measures
    'fixed_broad_price_gni_pct': 'fixed_broad_price',          # % of GNI (primary)
    'fixed_broad_price_usd': 'fixed_broad_price_usd',          # USD
    'fixed_broad_price_ppp': 'fixed_broad_price_ppp',          # PPP-adjusted
    'mobile_broad_price_gni_pct': 'mobile_broad_price',        # % of GNI (primary)
    'mobile_broad_price_usd': 'mobile_broad_price_usd',        # USD
    'mobile_broad_price_ppp': 'mobile_broad_price_ppp',        # PPP-adjusted
}

# ============================================================================
# DEPENDENT VARIABLE DEFINITIONS
# ============================================================================

# PRIMARY DV: Fixed broadband subscriptions per 100 inhabitants
# This is the most direct measure of broadband demand
PRIMARY_DV = 'log_fixed_broadband_subs'
PRIMARY_DV_RAW = 'fixed_broadband_subs'

# ROBUSTNESS DVs: Alternative demand measures
ROBUSTNESS_DVS = {
    'internet_users': {
        'log': 'log_internet_users_pct',
        'raw': 'internet_users_pct',
        'description': 'Internet users as % of population'
    },
    'bandwidth': {
        'log': 'log_int_bandwidth',
        'raw': 'int_bandwidth',
        'description': 'International Internet bandwidth (Gbit/s)'
    },
}

# ============================================================================
# PRICE VARIABLE DEFINITIONS
# ============================================================================

# PRIMARY price variable: GNI-adjusted (economically meaningful for affordability)
PRIMARY_PRICE = 'log_fixed_broad_price'
PRIMARY_PRICE_RAW = 'fixed_broad_price'

# Alternative price measures for robustness
ROBUSTNESS_PRICES = {
    'usd': 'log_fixed_broad_price_usd',
    'ppp': 'log_fixed_broad_price_ppp',
}

# ============================================================================
# CONTROL VARIABLES
# ============================================================================

# Control variables - Economic
ECONOMIC_CONTROLS = [
    'log_gdp_per_capita',
    'gdp_growth',
    'log_population',
    'log_population_density',
    'urban_population_pct'
]

# Control variables - Infrastructure
INFRASTRUCTURE_CONTROLS = [
    'log_mobile_subs',
    'log_secure_internet_servers'
]

# Control variables - Institutional
INSTITUTIONAL_CONTROLS = [
    'regulatory_quality_estimate',
]

# Control variables - Human capital
HUMAN_CAPITAL_CONTROLS = [
    'education_tertiary_pct',
]

# Instrumental variables for IV estimation
INSTRUMENTS = [
    'log_mobile_broad_price',      # Mobile price as supply shifter
    'regulatory_quality_estimate',  # Regulatory quality
]

# All control variables
ALL_CONTROLS = (ECONOMIC_CONTROLS + INFRASTRUCTURE_CONTROLS +
                INSTITUTIONAL_CONTROLS + HUMAN_CAPITAL_CONTROLS)

# Variables to log-transform
LOG_TRANSFORM_VARS = [
    'fixed_broadband_subs',
    'internet_users_pct',
    'int_bandwidth',
    'mobile_subs',
    'fixed_broad_price',
    'fixed_broad_price_usd',
    'fixed_broad_price_ppp',
    'mobile_broad_price',
    'gdp_per_capita',
    'population',
    'population_density',
    'secure_internet_servers',
]

# ============================================================================
# WORLD BANK INDICATOR CODES
# ============================================================================

WB_INDICATORS = {
    # Economic indicators
    'gdp_per_capita': 'NY.GDP.PCAP.CD',
    'gdp_per_capita_constant': 'NY.GDP.PCAP.KD',
    'gdp_growth': 'NY.GDP.MKTP.KD.ZG',
    'inflation_gdp_deflator': 'NY.GDP.DEFL.KD.ZG',

    # Population and demographics
    'population': 'SP.POP.TOTL',
    'population_density': 'EN.POP.DNST',
    'urban_population_pct': 'SP.URB.TOTL.IN.ZS',
    'population_ages_15_64': 'SP.POP.1564.TO.ZS',

    # Education
    'education_tertiary_pct': 'SE.TER.ENRR',
    'education_secondary_pct': 'SE.SEC.CUAT.UP.ZS',
    'labor_force_advanced_education': 'SL.TLF.ADVN.ZS',

    # Infrastructure
    'secure_internet_servers': 'IT.NET.SECR.P6',
    'electric_power_consumption': 'EG.USE.ELEC.KH.PC',
    'access_to_electricity': 'EG.ELC.ACCS.ZS',

    # Institutional
    'regulatory_quality_estimate': 'RQ.EST',
    'ease_of_doing_business_score': 'IC.BUS.EASE.XQ',
    'time_required_start_business': 'IC.REG.DURS',
    'cost_business_startup': 'IC.REG.COST.PC.ZS',

    # Technology and trade
    'high_tech_exports': 'TX.VAL.TECH.CD',
    'ict_goods_exports': 'TX.VAL.ICTG.ZS.UN',
    'research_development_expenditure': 'GB.XPD.RSDV.GD.ZS',

    # Labor
    'wage_salaried_workers': 'SL.EMP.WORK.ZS',
}

# ============================================================================
# ITU INDICATOR CODES
# ============================================================================

ITU_INDICATORS = {
    'int_band_use': '242',  # International Internet bandwidth
    'fixed_broad_basket_USD': '34620',  # Fixed broadband basket price
    'mobile_broad_basket_USD': '34608',  # Mobile broadband basket price
    'fixed_broadband_subs': '29',  # Fixed broadband subscriptions per 100
    'internet_users_pct': '39',  # Individuals using Internet (%)
    'mobile_subs': '26',  # Mobile cellular subscriptions per 100
}

# ============================================================================
# ANALYSIS PARAMETERS
# ============================================================================

# Minimum observations required per country
MIN_OBS_PER_COUNTRY = 10

# Significance levels
ALPHA_LEVELS = [0.01, 0.05, 0.10]

# Random seed for reproducibility
RANDOM_SEED = 42

# ============================================================================
# VISUALIZATION SETTINGS
# ============================================================================

# Figure sizes (width, height in inches)
FIGSIZE_SINGLE = (10, 6)
FIGSIZE_DOUBLE = (12, 5)
FIGSIZE_MULTI = (15, 10)

# Colors by region
REGION_COLORS = {
    'EU': '#1f77b4',   # Blue
    'EaP': '#ff7f0e',  # Orange
}

# DPI for saving figures
DPI = 300

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def get_region(country_code):
    """Return region (EU or EaP) for a given country code."""
    if country_code in EU_COUNTRIES:
        return 'EU'
    elif country_code in EAP_COUNTRIES:
        return 'EaP'
    else:
        return 'Other'

def get_country_name(country_code):
    """Return full country name for a given ISO3 code."""
    return COUNTRY_NAMES.get(country_code, country_code)

def get_country_code(country_name):
    """Return ISO3 code for a given country name."""
    return COUNTRY_CODES.get(country_name, country_name)

def print_config_summary():
    """Print configuration summary."""
    print("="*80)
    print("PROJECT CONFIGURATION SUMMARY")
    print("="*80)
    print(f"\nProject root: {PROJECT_ROOT}")
    print(f"\nCountries: {len(ALL_COUNTRIES)} ({len(EU_COUNTRIES)} EU + {len(EAP_COUNTRIES)} EaP)")
    print(f"Time period: {START_YEAR}-{END_YEAR} ({len(YEARS)} years)")
    print(f"\nDependent variables: {len(DEPENDENT_VARS)}")
    print(f"Control variables: {len(ALL_CONTROLS)}")
    print(f"Instruments: {len(INSTRUMENTS)}")
    print(f"\nWorld Bank indicators: {len(WB_INDICATORS)}")
    print(f"ITU indicators: {len(ITU_INDICATORS)}")
    print("="*80)


if __name__ == "__main__":
    print_config_summary()
