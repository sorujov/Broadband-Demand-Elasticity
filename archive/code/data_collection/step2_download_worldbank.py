# code/data_collection/02_download_worldbank_data.py
"""
================================================================================
World Bank Data Collection Script (FIXED VERSION)
================================================================================
Purpose: Download economic and social indicators from World Bank API
Author: Samir Orujov
Date: November 13, 2025

FIXED: Now uses ISO3 country codes instead of full names
Data Sources: World Bank Open Data API
================================================================================
"""

import wbgapi as wb
import pandas as pd
import numpy as np
from pathlib import Path
import time
from datetime import datetime
import sys

# Add project root to path for imports
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

# Import from code.utils.config
try:
    from code.utils.config import (
        DATA_RAW, EU_COUNTRIES, EAP_COUNTRIES, 
        START_YEAR, END_YEAR, WB_INDICATORS
    )
except ImportError:
    # Fallback for direct execution
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from utils.config import (
        DATA_RAW, EU_COUNTRIES, EAP_COUNTRIES, 
        START_YEAR, END_YEAR, WB_INDICATORS
    )

class WorldBankDownloader:
    """Download data from World Bank API using wbgapi."""

    def __init__(self):
        # Use ISO3 codes directly (fixed!)
        self.countries = EU_COUNTRIES + EAP_COUNTRIES
        self.time_range = range(START_YEAR, END_YEAR + 1)
        self.indicators = WB_INDICATORS

    def download_indicator(self, indicator_code, indicator_name):
        """
        Download a single indicator from World Bank.

        Args:
            indicator_code: World Bank indicator code
            indicator_name: Friendly name for the indicator
        """
        print(f"  Downloading {indicator_name} ({indicator_code})...", end=' ')

        try:
            # Download data using wbgapi with ISO3 codes
            data = wb.data.DataFrame(
                indicator_code,
                self.countries,  # Now using ISO3 codes!
                time=self.time_range,
                labels=False,
                skipBlanks=False,
                numericTimeKeys=True  # Important: use numeric year keys
            )

            # Reshape from wide to long format
            data = data.reset_index()

            # Check if we have year columns
            year_cols = [col for col in data.columns if col != 'economy' and str(col).startswith('YR')]
            if year_cols:
                # Year columns are like 'YR2010', 'YR2011', etc.
                data = data.melt(
                    id_vars=['economy'], 
                    value_vars=year_cols,
                    var_name='year', 
                    value_name=indicator_name
                )
                # Clean year column (remove 'YR' prefix)
                data['year'] = data['year'].str.replace('YR', '').astype(int)
            else:
                # Direct year columns
                data = data.melt(
                    id_vars=['economy'], 
                    var_name='year', 
                    value_name=indicator_name
                )
                data['year'] = pd.to_numeric(data['year'], errors='coerce')

            # Rename economy to country
            data = data.rename(columns={'economy': 'country'})

            # Filter to our year range
            data = data[data['year'].between(START_YEAR, END_YEAR)]

            print(f"[OK] ({len(data)} obs)")
            return data

        except Exception as e:
            print(f"[ERROR] Error: {str(e)}")
            return None

    def download_all_indicators(self):
        """Download all World Bank indicators."""
        print("="*80)
        print("WORLD BANK DATA DOWNLOAD")
        print("="*80)
        print(f"\nCountries: {len(self.countries)} (ISO3 codes)")
        print(f"  EU: {EU_COUNTRIES[:5]}... ({len(EU_COUNTRIES)} total)")
        print(f"  EaP: {EAP_COUNTRIES}")
        print(f"Period: {START_YEAR}-{END_YEAR}")
        print(f"Indicators: {len(self.indicators)}\n")

        # Track successful downloads
        successful = []
        failed = []

        # Download first indicator to establish base dataframe
        first_indicator = list(self.indicators.items())[0]
        df_all = self.download_indicator(first_indicator[1], first_indicator[0])

        if df_all is not None:
            successful.append(first_indicator[0])
        else:
            failed.append(first_indicator[0])

        time.sleep(0.5)

        # Download and merge remaining indicators
        for name, code in list(self.indicators.items())[1:]:
            df_temp = self.download_indicator(code, name)
            time.sleep(0.5)  # Rate limiting

            if df_temp is not None:
                if df_all is not None:
                    df_all = df_all.merge(
                        df_temp, 
                        on=['country', 'year'], 
                        how='outer'
                    )
                else:
                    df_all = df_temp
                successful.append(name)
            else:
                failed.append(name)

        # Print download summary
        print("\n" + "="*80)
        print("DOWNLOAD SUMMARY")
        print("="*80)
        print(f"\nSuccessful: {len(successful)}/{len(self.indicators)} indicators")
        if failed:
            print(f"Failed: {len(failed)} indicators")
            print(f"  {', '.join(failed[:5])}" + ("..." if len(failed) > 5 else ""))

        return df_all

    def save_data(self, df, filename='worldbank_data.csv'):
        """Save downloaded data to CSV."""
        if df is None or len(df) == 0:
            print("\n[ERROR] No data to save")
            return None

        output_path = DATA_RAW / filename
        df.to_csv(output_path, index=False)

        print("\n" + "="*80)
        print("SAVING DATA")
        print("="*80)
        print(f"\n[OK] Data saved: {output_path}")
        print(f"  - Shape: {df.shape[0]} rows × {df.shape[1]} columns")
        print(f"  - Countries: {df['country'].nunique()}")
        print(f"  - Years: {df['year'].min()}-{df['year'].max()}")

        # Show missing data summary
        print("\n" + "="*80)
        print("MISSING DATA SUMMARY")
        print("="*80)

        missing = df.isnull().sum()
        missing_pct = (missing / len(df) * 100).round(1)
        missing_summary = pd.DataFrame({
            'Variable': missing.index,
            'Missing': missing.values,
            'Pct': missing_pct.values
        })
        missing_summary = missing_summary[missing_summary['Missing'] > 0].sort_values('Pct', ascending=False)

        if len(missing_summary) > 0:
            print(f"\nVariables with missing data (top 10):")
            print(missing_summary.head(10).to_string(index=False))
        else:
            print("\n[OK] No missing data!")

        return output_path


def main():
    """Main execution function."""
    print("="*80)
    print("WORLD BANK DATA COLLECTION SCRIPT (FIXED)")
    print("="*80)
    print(f"Execution time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Using wbgapi version: {wb.__version__}")

    try:
        # Initialize downloader
        downloader = WorldBankDownloader()

        # Download all indicators
        df = downloader.download_all_indicators()

        # Save to file
        if df is not None:
            downloader.save_data(df)

            print("\n" + "="*80)
            print("DOWNLOAD COMPLETE [OK]")
            print("="*80)
            print("\nNext step: Run 03_merge_data.py to merge with ITU data")
        else:
            print("\n[ERROR] Download failed - no data retrieved")
            sys.exit(1)

    except Exception as e:
        print(f"\n[ERROR] Error: {str(e)}")
        print("\nTroubleshooting:")
        print("  1. Ensure config.py has ISO3 country codes (AUT, BEL, etc.)")
        print("  2. Check: pip install wbgapi --upgrade")
        print("  3. Check internet connection")
        print("  4. Verify World Bank API is accessible: https://api.worldbank.org")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
