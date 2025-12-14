# code/data_collection/step1_download_itu.py
"""
================================================================================
ITU Data Collection Script
================================================================================
Purpose: Download telecommunications data from ITU Excel file and API
Author: Samir Orujov
Date: December 11, 2025

Data Sources:
- ITU Excel File: Price baskets (Fixed/Mobile broadband) with USD, GNI%, PPP
- ITU DataHub API: Subscriptions, internet users, bandwidth
================================================================================
"""

import requests
import pandas as pd
from pathlib import Path
import zipfile
from io import BytesIO, StringIO
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
        START_YEAR, END_YEAR, COUNTRY_NAMES
    )
except ImportError:
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from utils.config import (
        DATA_RAW, EU_COUNTRIES, EAP_COUNTRIES, 
        START_YEAR, END_YEAR, COUNTRY_NAMES
    )


class ITUDataDownloader:
    """Download data from ITU Excel file and API."""

    def __init__(self):
        self.api_base_url = "https://api.datahub.itu.int/v2"
        self.excel_url = "https://www.itu.int/en/ITU-D/Statistics/Documents/publications/prices2024/ITU_ICTPriceBaskets_2008-2024.xlsx"
        
        self.country_iso3_codes = EU_COUNTRIES + EAP_COUNTRIES
        self.country_names = [COUNTRY_NAMES[code] for code in self.country_iso3_codes]
        
        print(f"Initialized with {len(self.country_iso3_codes)} countries")
        
        self.target_indicators = [
            {
                'name': 'fixed_broad_price',
                'source': 'excel',
                'basket': 'Fixed-broadband basket',
                'description': 'Fixed-broadband prices (USD, GNI%, PPP)'
            },
            {
                'name': 'mobile_broad_price',
                'source': 'excel',
                'basket': 'Data-only mobile-broadband basket',
                'description': 'Mobile-broadband prices (USD, GNI%, PPP)'
            },
            {
                'name': 'fixed_broadband_subs',
                'source': 'api',
                'code_id': 19303,
                'description': 'Fixed-broadband subscriptions'
            },
            {
                'name': 'mobile_subs',
                'source': 'api',
                'code_id': 178,
                'description': 'Mobile-cellular subscriptions'
            },
            {
                'name': 'internet_users_pct',
                'source': 'api',
                'code_id': 11624,
                'description': 'Individuals using the Internet (%)'
            },
            {
                'name': 'int_bandwidth',
                'source': 'api',
                'code_id': 242,
                'description': 'International bandwidth (Mbit/s)'
            }
        ]
    
    def download_excel_prices(self, basket_name, indicator_name):
        """Download price data from ITU Excel file."""
        print(f"  -> Downloading Excel file...")
        
        try:
            response = requests.get(self.excel_url, timeout=60)
            
            if response.status_code != 200:
                print(f"  [ERROR] HTTP {response.status_code}")
                return None
            
            # Read Excel file
            excel_file = BytesIO(response.content)
            df = pd.read_excel(excel_file, sheet_name='economies_2008-2024')
            
            # Filter to specific basket
            df = df[df['basket_combined_simplified'] == basket_name].copy()
            
            # Filter to our countries
            df = df[df['IsoCode'].isin(self.country_iso3_codes)].copy()
            
            # Reshape from wide to long format
            year_cols = [col for col in df.columns if isinstance(col, int)]
            df_long = df.melt(
                id_vars=['IsoCode', 'Economy', 'Unit'],
                value_vars=year_cols,
                var_name='dataYear',
                value_name='dataValue'
            )
            
            # Filter to our year range
            df_long = df_long[
                (df_long['dataYear'] >= START_YEAR) & 
                (df_long['dataYear'] <= END_YEAR)
            ].copy()
            
            # Pivot to have USD, GNI, PPP as separate columns
            df_wide = df_long.pivot_table(
                index=['IsoCode', 'Economy', 'dataYear'],
                columns='Unit',
                values='dataValue',
                aggfunc='first'
            ).reset_index()
            
            # Rename columns
            df_wide = df_wide.rename(columns={
                'IsoCode': 'country_iso3',
                'Economy': 'entityName',
                'GNIpc': 'price_gni_pct',
                'PPP': 'price_ppp',
                'USD': 'price_usd'
            })
            
            # Add metadata
            df_wide['seriesCode'] = basket_name.replace(' basket', '').replace('-', '_').lower()
            df_wide['seriesUnits'] = 'Multiple (USD, GNI%, PPP)'
            df_wide['dataSource'] = 'ITU Excel'
            
            # Save to file
            output_file = DATA_RAW / f'itu_{indicator_name}.csv'
            df_wide.to_csv(output_file, index=False, encoding='utf-8-sig')
            
            # Print coverage
            usd_cov = df_wide['price_usd'].notna().sum()
            gni_cov = df_wide['price_gni_pct'].notna().sum()
            ppp_cov = df_wide['price_ppp'].notna().sum()
            
            print(f"  [OK] {len(df_wide)} obs, {df_wide['country_iso3'].nunique()} countries")
            print(f"  -> Coverage: USD {usd_cov}, GNI% {gni_cov}, PPP {ppp_cov}")
            
            return df_wide
                
        except Exception as e:
            print(f"  [ERROR] {str(e)[:100]}")
            return None
    
    def download_api_indicator(self, code_id, indicator_name):
        """Download a single indicator from ITU DataHub API."""
        url = f"{self.api_base_url}/data/download/byid/{code_id}/iscollection/false"
        
        print(f"  -> Downloading from API (code {code_id})...")
        
        try:
            response = requests.get(url, timeout=60)
            
            if response.status_code != 200:
                print(f"  [ERROR] HTTP {response.status_code}")
                return None
            
            # Handle ZIP file
            if response.content[:2] == b'PK':
                with zipfile.ZipFile(BytesIO(response.content)) as z:
                    csv_files = [f for f in z.namelist() if f.endswith('.csv')]
                    if csv_files:
                        with z.open(csv_files[0]) as csv_file:
                            df = pd.read_csv(csv_file, encoding='utf-8')
                    else:
                        print("  [ERROR] No CSV in ZIP")
                        return None
            else:
                df = pd.read_csv(StringIO(response.text), encoding='utf-8')
            
            # Add ISO3 mapping
            name_to_iso3 = {v: k for k, v in COUNTRY_NAMES.items()}
            df['country_iso3'] = df['entityName'].map(name_to_iso3)
            
            # Filter by countries and years
            df = df[df['entityName'].isin(self.country_names)]
            df = df[(df['dataYear'] >= START_YEAR) & (df['dataYear'] <= END_YEAR)]
            
            # Save to file
            output_file = DATA_RAW / f'itu_{indicator_name}.csv'
            df.to_csv(output_file, index=False, encoding='utf-8-sig')
            
            print(f"  [OK] {len(df)} obs, {df['country_iso3'].nunique()} countries")
            
            return df
                
        except Exception as e:
            print(f"  [ERROR] {str(e)[:100]}")
            return None
    
    def download_all_indicators(self):
        """Download all required ITU indicators."""
        print("=" * 80)
        print("ITU DATA DOWNLOAD")
        print("=" * 80)
        print(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Countries: {len(self.country_iso3_codes)}")
        print(f"Period: {START_YEAR}-{END_YEAR}")
        print(f"Indicators: {len(self.target_indicators)}\n")
        
        DATA_RAW.mkdir(parents=True, exist_ok=True)
        
        results = {}
        successful = 0
        
        for i, indicator in enumerate(self.target_indicators, 1):
            print(f"[{i}/{len(self.target_indicators)}] {indicator['description']}")
            print("-" * 80)
            
            if indicator['source'] == 'excel':
                df = self.download_excel_prices(
                    basket_name=indicator['basket'],
                    indicator_name=indicator['name']
                )
            else:
                df = self.download_api_indicator(
                    code_id=indicator['code_id'],
                    indicator_name=indicator['name']
                )
            
            results[indicator['name']] = df
            if df is not None:
                successful += 1
            
            print()
            time.sleep(1)
        
        # Summary
        print("=" * 80)
        print("DOWNLOAD COMPLETE")
        print("=" * 80)
        print(f"Success: {successful}/{len(self.target_indicators)}\n")
        
        for indicator in self.target_indicators:
            if results[indicator['name']] is not None:
                df = results[indicator['name']]
                print(f"  [OK] {indicator['name']}: {len(df)} rows, {df['country_iso3'].nunique()} countries")
            else:
                print(f"  [FAILED] {indicator['name']}: FAILED")
        
        self._create_metadata(results)
        
        print(f"\nFiles saved to: {DATA_RAW}")
        return results
    
    def _create_metadata(self, results):
        """Create metadata file documenting the download."""
        metadata = []
        
        for indicator in self.target_indicators:
            df = results[indicator['name']]
            metadata.append({
                'variable_name': indicator['name'],
                'description': indicator['description'],
                'source': indicator['source'],
                'download_date': datetime.now().strftime('%Y-%m-%d'),
                'rows': len(df) if df is not None else 0,
                'countries': df['country_iso3'].nunique() if df is not None else 0,
                'status': 'Success' if df is not None else 'Failed'
            })
        
        metadata_df = pd.DataFrame(metadata)
        metadata_file = DATA_RAW / 'itu_download_metadata.csv'
        metadata_df.to_csv(metadata_file, index=False)


def main():
    """Main execution function."""
    downloader = ITUDataDownloader()
    results = downloader.download_all_indicators()
    print("\n[OK] Script complete")


if __name__ == "__main__":
    main()
