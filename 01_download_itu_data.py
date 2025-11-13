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
import numpy as np
from pathlib import Path
import json
import time
from datetime import datetime
import sys

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))
from utils.config import DATA_RAW, EU_COUNTRIES, EAP_COUNTRIES, START_YEAR, END_YEAR

class ITUDataDownloader:
    """Download data from ITU DataHub."""

    def __init__(self):
        self.base_url = "https://api.datahub.itu.int/v2"
        self.countries = EU_COUNTRIES + EAP_COUNTRIES
        self.indicators = {
            'int_band_use': '242',  # International Internet bandwidth
            'fixed_broad_basket_USD': '34620',  # Fixed broadband basket price
            'mobile_broad_basket_USD': '34608',  # Mobile broadband basket price
            'fixed_broadband_subs': '29',  # Fixed broadband subscriptions per 100
            'internet_users_pct': '39',  # Individuals using Internet (%)
            'mobile_subs': '26',  # Mobile cellular subscriptions per 100
        }

    def download_indicator(self, indicator_id, indicator_name):
        """
        Download a single indicator from ITU.

        Note: ITU DataHub API requires authentication for bulk downloads.
        This function demonstrates the structure. For actual use:
        1. Register at https://datahub.itu.int/
        2. Get API key
        3. Add authentication headers
        """
        print(f"\nDownloading {indicator_name} (ID: {indicator_id})...")

        # Method 1: Direct API call (requires authentication)
        # url = f"{self.base_url}/data/download/byid/{indicator_id}"
        # headers = {'Authorization': 'Bearer YOUR_API_KEY'}
        # response = requests.get(url, headers=headers)

        # Method 2: Manual download instructions
        print(f"  → ITU data requires manual download from:")
        print(f"     https://datahub.itu.int/query/")
        print(f"  → Select indicator ID: {indicator_id}")
        print(f"  → Select countries: {', '.join(self.countries[:5])}...")
        print(f"  → Select years: {START_YEAR}-{END_YEAR}")
        print(f"  → Download as CSV")
        print(f"  → Save to: data/raw/itu_{indicator_name}.csv")

        return None

    def download_all_indicators(self):
        """Download all required ITU indicators."""
        print("="*80)
        print("ITU DATA DOWNLOAD")
        print("="*80)
        print(f"\nCountries: {len(self.countries)}")
        print(f"Period: {START_YEAR}-{END_YEAR}")
        print(f"Indicators: {len(self.indicators)}")

        for name, id_code in self.indicators.items():
            self.download_indicator(id_code, name)
            time.sleep(1)  # Rate limiting

        print("\n" + "="*80)
        print("ITU DOWNLOAD INSTRUCTIONS COMPLETE")
        print("="*80)
        print("\nAlternative: Use pre-downloaded ITU data")
        print("  - If you have ITU subscription, download via official portal")
        print("  - Save CSVs to data/raw/ folder")
        print("  - Run data_preparation script to process")

    def create_sample_data(self):
        """
        Create sample data structure for testing.
        Use this if ITU data is not yet available.
        """
        print("\n" + "="*80)
        print("CREATING SAMPLE DATA STRUCTURE")
        print("="*80)

        # Create sample dataframe
        years = list(range(START_YEAR, END_YEAR + 1))
        countries = self.countries

        data = []
        for country in countries:
            for year in years:
                row = {
                    'country': country,
                    'year': year,
                    'int_band_use': np.random.randint(100000, 2000000),
                    'fixed_broad_basket_USD': np.random.uniform(5, 50),
                    'mobile_broad_basket_USD': np.random.uniform(3, 30),
                    'fixed_broadband_subs': np.random.uniform(10, 45),
                    'internet_users_pct': np.random.uniform(40, 95),
                    'mobile_subs': np.random.uniform(80, 150)
                }
                data.append(row)

        df = pd.DataFrame(data)

        # Save sample data
        output_file = DATA_RAW / 'itu_sample_data.csv'
        df.to_csv(output_file, index=False)

        print(f"\n✓ Sample data created: {output_file}")
        print(f"  - {len(df)} observations")
        print(f"  - {len(countries)} countries")
        print(f"  - {len(years)} years")

        return df


def main():
    """Main execution function."""
    print("="*80)
    print("ITU DATA COLLECTION SCRIPT")
    print("="*80)
    print(f"Execution time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    # Initialize downloader
    downloader = ITUDataDownloader()

    # Option 1: Show download instructions
    print("\nOption 1: Manual download from ITU DataHub")
    downloader.download_all_indicators()

    # Option 2: Create sample data for testing
    print("\n\nOption 2: Create sample data for testing")
    response = input("Create sample data? (y/n): ").lower()
    if response == 'y':
        downloader.create_sample_data()

    print("\n" + "="*80)
    print("SCRIPT COMPLETE")
    print("="*80)


if __name__ == "__main__":
    main()
