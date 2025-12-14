# code/data_collection/01.5_process_data.py
"""
================================================================================
Data Processing Script (NEW)
================================================================================
Purpose: Process downloaded ITU and World Bank data to preserve series metadata
Author: Samir Orujov
Date: November 14, 2025

This script runs AFTER download scripts and BEFORE merge script.
It ensures that series identifiers (seriesCode, seriesUnits, indicator codes)
are preserved for later analysis.
================================================================================
"""

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
import sys

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

try:
    from code.utils.config import (
        DATA_RAW, DATA_INTERIM, EU_COUNTRIES, EAP_COUNTRIES,
        START_YEAR, END_YEAR, COUNTRY_NAMES, WB_INDICATORS
    )
except ImportError:
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from utils.config import (
        DATA_RAW, DATA_INTERIM, EU_COUNTRIES, EAP_COUNTRIES,
        START_YEAR, END_YEAR, COUNTRY_NAMES, WB_INDICATORS
    )


class DataProcessor:
    """Process ITU and World Bank data to preserve metadata."""

    def __init__(self):
        self.itu_files = {
            'fixed_broad_price': DATA_RAW / 'itu_fixed_broad_price.csv',
            'mobile_broad_price': DATA_RAW / 'itu_mobile_broad_price.csv',
            'fixed_broadband_subs': DATA_RAW / 'itu_fixed_broadband_subs.csv',
            'internet_users_pct': DATA_RAW / 'itu_internet_users_pct.csv',
            'int_bandwidth': DATA_RAW / 'itu_int_bandwidth.csv',
            'mobile_subs': DATA_RAW / 'itu_mobile_subs.csv',
        }

        self.wb_file = DATA_RAW / 'worldbank_data.csv'

        # Ensure interim directory exists
        DATA_INTERIM.mkdir(parents=True, exist_ok=True)

    def process_itu_data(self):
        """
        Process ITU data files to preserve series metadata.
        Price data now comes with USD, GNI%, PPP as separate columns.
        Other indicators have single dataValue column.
        """
        print("="*80)
        print("PROCESSING ITU DATA")
        print("="*80)

        processed_files = []

        for var_name, filepath in self.itu_files.items():
            if not filepath.exists():
                print(f"\n[!] Skipping {var_name}: File not found")
                continue

            print(f"\nProcessing: {var_name}")
            print("-"*80)

            # Load data
            df = pd.read_csv(filepath)
            print(f"  Loaded: {df.shape[0]} rows × {df.shape[1]} columns")

            # Check if this is price data (has separate price columns)
            if 'price_usd' in df.columns:
                # Price data from Excel - has USD, GNI%, PPP columns
                print(f"  Format: Multi-column price data (USD, GNI%, PPP)")
                
                # Rename columns to standard format
                df_processed = df[[
                    'country_iso3', 'dataYear', 
                    'price_usd', 'price_gni_pct', 'price_ppp',
                    'seriesCode', 'seriesUnits', 'dataSource'
                ]].copy()
                
                df_processed.columns = [
                    'country', 'year',
                    f'{var_name}_usd', f'{var_name}_gni_pct', f'{var_name}_ppp',
                    'series_code', 'series_units', 'data_source'
                ]
                
                # Add variable name
                df_processed['variable'] = var_name
                
                # Show coverage
                print(f"  Coverage:")
                print(f"    USD: {df_processed[f'{var_name}_usd'].notna().sum()}/{len(df_processed)} ({df_processed[f'{var_name}_usd'].notna().sum()/len(df_processed)*100:.1f}%)")
                print(f"    GNI%: {df_processed[f'{var_name}_gni_pct'].notna().sum()}/{len(df_processed)} ({df_processed[f'{var_name}_gni_pct'].notna().sum()/len(df_processed)*100:.1f}%)")
                print(f"    PPP: {df_processed[f'{var_name}_ppp'].notna().sum()}/{len(df_processed)} ({df_processed[f'{var_name}_ppp'].notna().sum()/len(df_processed)*100:.1f}%)")
                
            else:
                # Non-price data from API - has single dataValue column
                print(f"  Format: Single-column data")
                
                # Check for multiple series
                if 'seriesCode' in df.columns:
                    series_info = df.groupby(['seriesCode']).size().reset_index(name='count')
                    print(f"  Found {len(series_info)} unique series:")
                    for idx, row in series_info.iterrows():
                        print(f"    - {row['seriesCode']}: {row['count']} obs")
                
                # Rename columns to standard format
                df_processed = df[[
                    'country_iso3', 'dataYear', 'dataValue',
                    'seriesCode', 'seriesUnits'
                ]].copy()
                
                df_processed.columns = [
                    'country', 'year', var_name,
                    'series_code', 'series_units'
                ]
                
                # Add variable name
                df_processed['variable'] = var_name
                df_processed['data_source'] = 'ITU API'

            # Save processed data
            output_file = DATA_INTERIM / f'itu_{var_name}_processed.xlsx'
            df_processed.to_excel(output_file, index=False, engine='openpyxl')

            print(f"  [OK] Saved: {output_file.name}")
            print(f"    {df_processed.shape[0]} rows × {df_processed.shape[1]} columns")

            processed_files.append(var_name)

        print(f"\n[OK] Processed {len(processed_files)} ITU indicators")
        return processed_files

    def process_worldbank_data(self):
        """
        Process World Bank data to preserve indicator codes.
        """
        print("\n" + "="*80)
        print("PROCESSING WORLD BANK DATA")
        print("="*80)

        if not self.wb_file.exists():
            print(f"\n[ERROR] File not found: {self.wb_file}")
            return False

        # Load data
        df = pd.read_csv(self.wb_file)
        print(f"\nLoaded: {df.shape[0]} rows × {df.shape[1]} columns")

        # Convert to long format with indicator metadata
        id_vars = ['country', 'year']
        value_vars = [col for col in df.columns if col not in id_vars]

        print(f"Found {len(value_vars)} indicators")

        # Create reverse mapping: variable_name -> indicator_code
        wb_codes = {v: k for k, v in WB_INDICATORS.items()}

        # Reshape to long format
        df_long = df.melt(
            id_vars=id_vars,
            value_vars=value_vars,
            var_name='variable',
            value_name='value'
        )

        # Add indicator codes
        df_long['indicator_code'] = df_long['variable'].map(wb_codes)

        # Save processed data
        output_file = DATA_INTERIM / 'worldbank_processed.xlsx'
        df_long.to_excel(output_file, index=False, engine='openpyxl')

        print(f"\n[OK] Saved: {output_file.name}")
        print(f"  {df_long.shape[0]} rows × {df_long.shape[1]} columns")

        # Show sample
        print("\nSample data:")
        print(df_long[['country', 'year', 'variable', 'indicator_code', 'value']].head(3).to_string(index=False))

        return True

    def create_metadata_catalog(self):
        """
        Create a catalog documenting all variables and their series.
        """
        print("\n" + "="*80)
        print("CREATING METADATA CATALOG")
        print("="*80)

        catalog = []

        # ITU data
        for var_name, filepath in self.itu_files.items():
            if not filepath.exists():
                continue

            df = pd.read_csv(filepath)
            
            # Check if this is price data
            if 'price_usd' in df.columns:
                # Price data - add entries for each price type
                for price_type, col_name in [('USD', 'price_usd'), ('GNI%', 'price_gni_pct'), ('PPP', 'price_ppp')]:
                    catalog.append({
                        'data_source': 'ITU Excel',
                        'variable': f'{var_name}_{price_type.lower()}',
                        'series_code': df['seriesCode'].iloc[0] if 'seriesCode' in df.columns else 'N/A',
                        'series_name': f'{var_name} ({price_type})',
                        'units': price_type,
                        'observations': df[col_name].notna().sum(),
                        'countries': df['country_iso3'].nunique(),
                        'year_min': int(df['dataYear'].min()),
                        'year_max': int(df['dataYear'].max()),
                        'value_min': round(df[col_name].min(), 2) if df[col_name].notna().any() else np.nan,
                        'value_max': round(df[col_name].max(), 2) if df[col_name].notna().any() else np.nan
                    })
            else:
                # Non-price data
                series_info = df.groupby(['seriesCode', 'seriesUnits']).agg({
                    'dataValue': ['count', 'min', 'max'],
                    'dataYear': ['min', 'max'],
                    'country_iso3': 'nunique'
                }).reset_index()

                for idx, row in series_info.iterrows():
                    catalog.append({
                        'data_source': 'ITU API',
                        'variable': var_name,
                        'series_code': row['seriesCode'],
                        'series_name': row['seriesCode'],
                        'units': row['seriesUnits'],
                        'observations': row[('dataValue', 'count')],
                        'countries': row[('country_iso3', 'nunique')],
                        'year_min': int(row[('dataYear', 'min')]),
                        'year_max': int(row[('dataYear', 'max')]),
                        'value_min': round(row[('dataValue', 'min')], 2) if not np.isnan(row[('dataValue', 'min')]) else np.nan,
                        'value_max': round(row[('dataValue', 'max')], 2) if not np.isnan(row[('dataValue', 'max')]) else np.nan
                    })

        # World Bank data
        if self.wb_file.exists():
            df_wb = pd.read_csv(self.wb_file)
            wb_codes = {v: k for k, v in WB_INDICATORS.items()}

            for col in df_wb.columns:
                if col not in ['country', 'year']:
                    indicator_code = wb_codes.get(col, 'N/A')

                    catalog.append({
                        'data_source': 'World Bank',
                        'variable': col,
                        'series_code': indicator_code,
                        'series_name': col,
                        'units': 'Various (see WB documentation)',
                        'observations': df_wb[col].notna().sum(),
                        'countries': df_wb['country'].nunique(),
                        'year_min': int(df_wb['year'].min()),
                        'year_max': int(df_wb['year'].max()),
                        'value_min': round(df_wb[col].min(), 2) if df_wb[col].notna().any() else np.nan,
                        'value_max': round(df_wb[col].max(), 2) if df_wb[col].notna().any() else np.nan
                    })

        # Save catalog
        catalog_df = pd.DataFrame(catalog)
        catalog_file = DATA_INTERIM / 'data_catalog.xlsx'
        catalog_df.to_excel(catalog_file, index=False, engine='openpyxl')

        print(f"\n[OK] Saved catalog: {catalog_file.name}")
        print(f"  {len(catalog_df)} variables documented")

        # Print summary
        print("\nCatalog summary:")
        print(catalog_df.groupby('data_source').size())

        return catalog_df


def main():
    """Main execution function."""
    print("="*80)
    print("DATA PROCESSING SCRIPT")
    print("="*80)
    print(f"Execution time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"\nThis script preserves series metadata (codes, units, definitions)")
    print(f"for proper variable selection in analysis.\n")

    try:
        processor = DataProcessor()

        # Process ITU data
        itu_processed = processor.process_itu_data()

        # Process World Bank data
        wb_processed = processor.process_worldbank_data()

        # Create metadata catalog
        catalog = processor.create_metadata_catalog()

        print("\n" + "="*80)
        print("PROCESSING COMPLETE [OK]")
        print("="*80)
        print(f"\nProcessed files saved to: {DATA_INTERIM}")
        print(f"\nNext step: Run 03_merge_data.py (updated version)")

    except Exception as e:
        print(f"\n[ERROR] Error: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
