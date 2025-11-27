# code/data_collection/01_download_itu_data.py
"""
================================================================================
ITU Data Collection Script
================================================================================
Purpose: Download telecommunications data from ITU DataHub
Author: Samir Orujov
Date: November 13, 2025

Data Sources:
- ITU DataHub API: https://datahub.itu.int/
- Indicators: Broadband prices, subscriptions, bandwidth usage
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
    # Fallback for direct execution
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from utils.config import (
        DATA_RAW, EU_COUNTRIES, EAP_COUNTRIES, 
        START_YEAR, END_YEAR, COUNTRY_NAMES
    )


class ITUDataDownloader:
    """Download data from ITU DataHub using the v2 API."""

    def __init__(self):
        self.base_url = "https://api.datahub.itu.int/v2"
        
        # Convert ISO3 codes to full country names for ITU filtering
        self.country_iso3_codes = EU_COUNTRIES + EAP_COUNTRIES
        self.country_names = [COUNTRY_NAMES[code] for code in self.country_iso3_codes]
        
        print(f"Initialized with {len(self.country_iso3_codes)} countries (ISO3 codes)")
        print(f"Converted to {len(self.country_names)} country names for ITU filtering")
        
        # Direct code IDs based on ITU catalog analysis
        self.target_indicators = [
            {
                'name': 'fixed_broadband_subs',
                'code_id': 19303,
                'is_collection': False,
                'description': 'Fixed-broadband subscriptions'
            },
            {
                'name': 'mobile_subs',
                'code_id': 178,
                'is_collection': False,
                'description': 'Mobile-cellular subscriptions'
            },
            {
                'name': 'internet_users_pct',
                'code_id': 11624,
                'is_collection': False,
                'description': 'Individuals using the Internet'
            },
            {
                'name': 'int_bandwidth',
                'code_id': 242,
                'is_collection': False,
                'description': 'International bandwidth usage'
            },
            {
                'name': 'fixed_broad_price',
                'code_id': 34616,
                'is_collection': False,
                'description': 'Fixed-broadband Internet basket'
            },
            {
                'name': 'mobile_broad_price',
                'code_id': 34617,
                'is_collection': False,
                'description': 'Data-only mobile broadband basket'
            }
        ]
    
    def download_indicator(self, code_id, is_collection, indicator_name):
        """
        Download a single indicator from ITU DataHub API.
        Handles both ZIP and plain CSV responses.
        """
        url = f"{self.base_url}/data/download/byid/{code_id}/iscollection/{str(is_collection).lower()}"
        
        print(f"  -> Downloading from API...")
        
        try:
            response = requests.get(url, timeout=60)
            
            if response.status_code == 200:
                
                # Check if response is a ZIP file (starts with 'PK' magic bytes)
                if response.content[:2] == b'PK':
                    # Handle ZIP file
                    try:
                        with zipfile.ZipFile(BytesIO(response.content)) as z:
                            # Find CSV files in the archive
                            csv_files = [f for f in z.namelist() if f.endswith('.csv')]
                            
                            if csv_files:
                                with z.open(csv_files[0]) as csv_file:
                                    df = pd.read_csv(csv_file, encoding='utf-8')
                                print(f"  [OK] Downloaded {len(df)} rows from ZIP")
                            else:
                                print("  [ERROR] No CSV found in ZIP archive")
                                return None
                    except Exception as e:
                        print(f"  [ERROR] Error extracting ZIP: {str(e)[:50]}")
                        return None
                        
                else:
                    # Handle plain CSV with multiple encoding strategies
                    try:
                        df = pd.read_csv(StringIO(response.text), encoding='utf-8')
                        print(f"  [OK] Downloaded {len(df)} rows (plain CSV)")
                    except:
                        try:
                            df = pd.read_csv(StringIO(response.content.decode('utf-8-sig')))
                            print(f"  [OK] Downloaded {len(df)} rows (plain CSV)")
                        except:
                            try:
                                df = pd.read_csv(StringIO(response.content.decode('latin-1')))
                                print(f"  [OK] Downloaded {len(df)} rows (plain CSV)")
                            except Exception as e:
                                print(f"  [ERROR] Parse error: {str(e)[:50]}")
                                return None
                
                # Add ISO3 code mapping column
                # ITU uses full country names, we need ISO3 for consistency
                country_col = None
                for col in ['entityName', 'Economy', 'Country', 'economy', 'country']:
                    if col in df.columns:
                        country_col = col
                        break
                
                if country_col:
                    # Create reverse mapping (name -> ISO3)
                    name_to_iso3 = {v: k for k, v in COUNTRY_NAMES.items()}
                    
                    # Add ISO3 column
                    df['country_iso3'] = df[country_col].map(name_to_iso3)
                    
                    # Filter by countries (using country names from ITU)
                    original_len = len(df)
                    df = df[df[country_col].isin(self.country_names)]
                    
                    if len(df) > 0:
                        print(f"  -> Filtered by country: {original_len} -> {len(df)} records")
                        print(f"  -> Countries found: {df[country_col].nunique()} unique")
                    else:
                        print(f"  ⚠ No matching countries found!")
                        print(f"  ℹ Sample countries in data: {df[country_col].head(5).tolist()[:3]}")
                else:
                    print(f"  ⚠ No country column found in data!")
                    print(f"  ℹ Available columns: {', '.join(df.columns.tolist()[:5])}")
                
                # Filter by years
                year_col = None
                for col in ['dataYear', 'Year', 'year']:
                    if col in df.columns:
                        year_col = col
                        break
                
                if year_col:
                    df[year_col] = pd.to_numeric(df[year_col], errors='coerce')
                    original_len = len(df)
                    df = df[(df[year_col] >= START_YEAR) & (df[year_col] <= END_YEAR)]
                    if len(df) > 0:
                        print(f"  -> Filtered by year ({START_YEAR}-{END_YEAR}): {original_len} -> {len(df)} records")
                
                # Save to file
                if len(df) > 0:
                    output_file = DATA_RAW / f'itu_{indicator_name}.csv'
                    df.to_csv(output_file, index=False, encoding='utf-8-sig')
                    print(f"  [OK] Saved to: {output_file.name}")
                    
                    # Show preview
                    print(f"\n  Preview (first 3 rows, key columns):")
                    preview_cols = [col for col in ['country_iso3', 'entityName', 'dataYear', 'dataValue'] if col in df.columns]
                    if preview_cols:
                        print(df[preview_cols].head(3).to_string(index=False))
                    
                    return df
                else:
                    print("  ⚠ No matching data after filtering")
                    return None
                    
            elif response.status_code == 404:
                print(f"  [ERROR] Not found (HTTP 404)")
                return None
            elif response.status_code == 500:
                print(f"  [ERROR] Server error (HTTP 500)")
                return None
            else:
                print(f"  [ERROR] HTTP status: {response.status_code}")
                return None
                
        except requests.exceptions.Timeout:
            print(f"  [ERROR] Request timeout")
            return None
        except Exception as e:
            print(f"  [ERROR] Error: {str(e)[:80]}")
            return None
    
    def download_all_indicators(self):
        """Download all required ITU indicators."""
        print("=" * 80)
        print("ITU DATA DOWNLOAD")
        print("=" * 80)
        print(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Output directory: {DATA_RAW}")
        print(f"Countries (ISO3): {len(self.country_iso3_codes)} ({', '.join(self.country_iso3_codes[:5])}...)")
        print(f"Period: {START_YEAR}-{END_YEAR}")
        print(f"Total indicators: {len(self.target_indicators)}\n")
        
        # Ensure output directory exists
        DATA_RAW.mkdir(parents=True, exist_ok=True)
        
        results = {}
        successful_downloads = 0
        failed_downloads = 0
        
        # Download each indicator
        for i, indicator in enumerate(self.target_indicators, 1):
            print("-" * 80)
            print(f"[{i}/{len(self.target_indicators)}] {indicator['description']}")
            print("-" * 80)
            print(f"  Code ID: {indicator['code_id']}")
            print(f"  Variable name: {indicator['name']}")
            
            # Download data
            df = self.download_indicator(
                code_id=indicator['code_id'],
                is_collection=indicator['is_collection'],
                indicator_name=indicator['name']
            )
            
            # Store result
            if df is not None:
                results[indicator['name']] = df
                successful_downloads += 1
            else:
                results[indicator['name']] = None
                failed_downloads += 1
            
            print()
            
            # Rate limiting - wait 2 seconds between requests
            if i < len(self.target_indicators):
                time.sleep(2)
        
        # Summary
        print("=" * 80)
        print("DOWNLOAD COMPLETE")
        print("=" * 80)
        print(f"[OK] Successfully downloaded: {successful_downloads}/{len(self.target_indicators)}")
        
        if successful_downloads > 0:
            print("\nDownloaded files:")
            for indicator in self.target_indicators:
                if results[indicator['name']] is not None:
                    filename = f"itu_{indicator['name']}.csv"
                    row_count = len(results[indicator['name']])
                    unique_countries = results[indicator['name']]['country_iso3'].nunique() if 'country_iso3' in results[indicator['name']].columns else 'N/A'
                    print(f"  [OK] {filename} - {row_count} rows, {unique_countries} countries")
        
        if failed_downloads > 0:
            print(f"\n[ERROR] Failed to download: {failed_downloads} indicators")
            print("\nFailed indicators:")
            for indicator in self.target_indicators:
                if results[indicator['name']] is None:
                    print(f"  • {indicator['description']} (Code: {indicator['code_id']})")
            print("\nYou can try manual download from: https://datahub.itu.int/")
        
        # Create metadata file
        self._create_metadata(results)
        
        print("\n" + "=" * 80)
        print(f"All files saved to: {DATA_RAW}")
        print("=" * 80)
        
        return results
    
    def _create_metadata(self, results):
        """Create metadata file documenting the download."""
        metadata = []
        
        for indicator in self.target_indicators:
            if results[indicator['name']] is not None:
                row_count = len(results[indicator['name']])
                unique_countries = results[indicator['name']]['country_iso3'].nunique() if 'country_iso3' in results[indicator['name']].columns else None
            else:
                row_count = None
                unique_countries = None
            
            status = "Success" if results[indicator['name']] is not None else "Failed"
            
            metadata.append({
                'variable_name': indicator['name'],
                'code_id': indicator['code_id'],
                'description': indicator['description'],
                'download_date': datetime.now().strftime('%Y-%m-%d'),
                'rows_downloaded': row_count,
                'unique_countries': unique_countries,
                'status': status
            })
        
        metadata_df = pd.DataFrame(metadata)
        metadata_file = DATA_RAW / 'itu_download_metadata.csv'
        metadata_df.to_csv(metadata_file, index=False)
        print(f"\n[OK] Metadata saved to: {metadata_file.name}")


def main():
    """Main execution function."""
    print("=" * 80)
    print("ITU DATA COLLECTION SCRIPT")
    print("=" * 80)
    print(f"Execution time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    
    # Initialize downloader
    downloader = ITUDataDownloader()
    
    # Download data from ITU
    results = downloader.download_all_indicators()
    
    print("\n" + "=" * 80)
    print("SCRIPT COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    main()
