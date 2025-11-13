"""
================================================================================
World Bank Data Collection Script
================================================================================
Purpose: Download economic and social indicators from World Bank API
Author: Samir Orujov
Date: November 13, 2025

Data Sources:
- World Bank Open Data API
- Indicators: GDP, population, education, institutions, infrastructure
================================================================================
"""

import wbgapi as wb
import pandas as pd
import numpy as np
from pathlib import Path
import time
from datetime import datetime
import sys

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))
from utils.config import DATA_RAW, EU_COUNTRIES, EAP_COUNTRIES, START_YEAR, END_YEAR

class WorldBankDownloader:
    """Download data from World Bank API using wbgapi."""

    def __init__(self):
        self.countries = EU_COUNTRIES + EAP_COUNTRIES
        self.time_range = range(START_YEAR, END_YEAR + 1)

        # World Bank indicator codes
        self.indicators = {
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

    def download_indicator(self, indicator_code, indicator_name):
        """
        Download a single indicator from World Bank.

        Args:
            indicator_code: World Bank indicator code
            indicator_name: Friendly name for the indicator
        """
        print(f"  Downloading {indicator_name} ({indicator_code})...", end=' ')

        try:
            # Download data using wbgapi
            data = wb.data.DataFrame(
                indicator_code,
                self.countries,
                time=self.time_range,
                labels=False,
                skipBlanks=False
            )

            # Reshape from wide to long format
            data = data.reset_index()
            data = data.melt(
                id_vars=['economy'], 
                var_name='year', 
                value_name=indicator_name
            )
            data = data.rename(columns={'economy': 'country'})
            data['year'] = data['year'].astype(int)

            print(f"✓ ({len(data)} obs)")
            return data

        except Exception as e:
            print(f"✗ Error: {str(e)}")
            return None

    def download_all_indicators(self):
        """Download all World Bank indicators."""
        print("="*80)
        print("WORLD BANK DATA DOWNLOAD")
        print("="*80)
        print(f"\nCountries: {len(self.countries)}")
        print(f"Period: {START_YEAR}-{END_YEAR}")
        print(f"Indicators: {len(self.indicators)}\n")

        # Download first indicator to establish base dataframe
        first_indicator = list(self.indicators.items())[0]
        df_all = self.download_indicator(first_indicator[1], first_indicator[0])
        time.sleep(0.5)

        # Download and merge remaining indicators
        for name, code in list(self.indicators.items())[1:]:
            df_temp = self.download_indicator(code, name)
            time.sleep(0.5)  # Rate limiting

            if df_temp is not None:
                df_all = df_all.merge(
                    df_temp, 
                    on=['country', 'year'], 
                    how='outer'
                )

        return df_all

    def save_data(self, df, filename='worldbank_data.csv'):
        """Save downloaded data to CSV."""
        output_path = DATA_RAW / filename
        df.to_csv(output_path, index=False)
        print(f"\n✓ Data saved: {output_path}")
        print(f"  - Shape: {df.shape[0]} rows × {df.shape[1]} columns")
        print(f"  - Countries: {df['country'].nunique()}")
        print(f"  - Years: {df['year'].min()}-{df['year'].max()}")

        # Show missing data summary
        missing = df.isnull().sum()
        missing_pct = (missing / len(df) * 100).round(1)
        missing_summary = pd.DataFrame({
            'Missing': missing[missing > 0],
            'Pct': missing_pct[missing > 0]
        }).sort_values('Pct', ascending=False)

        if len(missing_summary) > 0:
            print(f"\n  Missing data summary (top 10):")
            print(missing_summary.head(10).to_string())

        return output_path


def main():
    """Main execution function."""
    print("="*80)
    print("WORLD BANK DATA COLLECTION SCRIPT")
    print("="*80)
    print(f"Execution time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"\nUsing wbgapi version: {wb.__version__}")

    try:
        # Initialize downloader
        downloader = WorldBankDownloader()

        # Download all indicators
        df = downloader.download_all_indicators()

        # Save to file
        if df is not None:
            downloader.save_data(df)

            print("\n" + "="*80)
            print("DOWNLOAD COMPLETE ✓")
            print("="*80)
        else:
            print("\n✗ Download failed")

    except Exception as e:
        print(f"\n✗ Error: {str(e)}")
        print("\nTroubleshooting:")
        print("  1. Install wbgapi: pip install wbgapi")
        print("  2. Check internet connection")
        print("  3. Verify World Bank API is accessible")


if __name__ == "__main__":
    main()
