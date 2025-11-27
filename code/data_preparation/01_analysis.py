
# code/data_analysis/03.5_explore_data.py

"""
================================================================================
Exploratory Data Analysis (EDA) - Missing Patterns & Structure
================================================================================
Purpose: Analyze raw merged data BEFORE processing/imputation
Author: Samir Orujov
Date: November 20, 2025

This script runs AFTER merge (03_merge_data.py) and BEFORE processing (04_prepare_data_FIXED.py)

Outputs:
1. Missing value statistics and patterns
2. Dendrogram of variable correlations (missingness)
3. Panel completeness analysis
4. Visualizations for paper methodology section
================================================================================
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import missingno as msno
from pathlib import Path
import sys

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

try:
    from code.utils.config import DATA_PROCESSED
except ImportError:
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from utils.config import DATA_PROCESSED

class DataAnalyzer:
    def __init__(self, input_file=None):
        """Initialize with merged data file."""
        if input_file is None:
            self.input_file = DATA_PROCESSED / 'data_merged_with_series.csv'
        else:
            self.input_file = input_file
        
        self.output_dir = DATA_PROCESSED / 'eda_outputs'
        self.output_dir.mkdir(exist_ok=True)
        
        self.df = None
        self.df_panel = None  # Panel-indexed version
    
    def load_and_prepare(self):
        """Load data and prepare panel structure."""
        print("="*80)
        print("LOADING MERGED DATA FOR EDA")
        print("="*80)
        
        self.df = pd.read_csv(self.input_file)
        print(f"\n[OK] Loaded: {self.df.shape[0]} rows × {self.df.shape[1]} columns")
        print(f"  Countries: {self.df['country'].nunique()}")
        print(f"  Years: {self.df['year'].min()}-{self.df['year'].max()}")
        
        # Create panel-indexed version for analysis
        self.df_panel = self.df.set_index(['country', 'year']).sort_index()
        
        # Select only numeric columns for missingness analysis
        numeric_cols = self.df_panel.select_dtypes(include=[np.number]).columns
        self.df_panel = self.df_panel[numeric_cols]
        
        print(f"\n[OK] Panel structure created: {len(numeric_cols)} numeric variables")
        
        return self.df_panel

    def analyze_missing_values(self):
        """Compute missing value statistics."""
        print("\n" + "="*80)
        print("MISSING VALUE STATISTICS")
        print("="*80)
        
        # Overall statistics
        missing_stats = pd.DataFrame({
            'Missing Count': self.df_panel.isnull().sum(),
            'Missing Percentage': (self.df_panel.isnull().sum() * 100 / len(self.df_panel)).round(2),
            'Total Observations': len(self.df_panel),
            'Available Observations': self.df_panel.count()
        }).sort_values('Missing Percentage', ascending=False)
        
        # Filter to show only variables with missing data
        missing_stats = missing_stats[missing_stats['Missing Count'] > 0]
        
        # Save to CSV
        output_file = self.output_dir / 'missing_value_statistics.csv'
        missing_stats.to_csv(output_file)
        print(f"\n[OK] Saved: {output_file}")
        
        # By year
        missing_by_year = self.df_panel.groupby(level='year').apply(
            lambda x: x.isnull().sum()
        )
        missing_by_year.to_csv(self.output_dir / 'missing_by_year.csv')
        
        # By country
        missing_by_country = self.df_panel.groupby(level='country').apply(
            lambda x: x.isnull().sum()
        )
        missing_by_country.to_csv(self.output_dir / 'missing_by_country.csv')
        
        print(f"[OK] Saved: missing_by_year.csv")
        print(f"[OK] Saved: missing_by_country.csv")
        
        return missing_stats, missing_by_year, missing_by_country

    def plot_missing_patterns(self):
        """Visualize missing data patterns."""
        print("\n" + "="*80)
        print("GENERATING MISSING PATTERN HEATMAP")
        print("="*80)
        
        fig, ax = plt.subplots(figsize=(20, 80))
        
        # Sort by country and year
        sorted_df = self.df_panel.sort_index(level=['country', 'year'])
        
        sns.heatmap(
            sorted_df.isnull(), 
            cbar=False,
            cmap='viridis',
            ax=ax,
            yticklabels=False  # Too many to show
        )
        
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right', fontsize=9)
        ax.set_title('Missing Value Patterns (Yellow = Missing, Purple = Present)', 
                     fontsize=14, pad=20)
        ax.set_ylabel('Country-Year Observations')
        
        plt.tight_layout()
        
        output_file = self.output_dir / 'missing_patterns_heatmap.png'
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"\n[OK] Saved: {output_file}")

    def plot_dendrogram(self):
        """Generate dendrogram of missing value correlations."""
        print("\n" + "="*80)
        print("GENERATING MISSING VALUE DENDROGRAM")
        print("="*80)
        
        # Use missingno's dendrogram
        fig = msno.dendrogram(
            self.df_panel,
            orientation='left',
            figsize=(15, 10),
            fontsize=10,
            method='ward'
        )
        
        plt.title('Hierarchical Clustering of Variables by Missingness Correlation',
                  fontsize=12, pad=20)
        plt.tight_layout()
        
        output_file = self.output_dir / 'missing_dendrogram.png'
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"\n[OK] Saved: {output_file}")
        print("\nInterpretation:")
        print("  - Variables that cluster together have similar missing patterns")
        print("  - Use this to identify series that can substitute for each other")
        print("  - Document in paper methodology section")

    def panel_completeness(self):
        """Analyze panel structure completeness."""
        print("\n" + "="*80)
        print("PANEL STRUCTURE ANALYSIS")
        print("="*80)
        
        countries = self.df['country'].unique()
        years = self.df['year'].unique()
        
        theoretical_max = len(countries) * len(years)
        actual_obs = len(self.df)
        
        completeness = (actual_obs / theoretical_max) * 100
        
        print(f"\nNumber of countries: {len(countries)}")
        print(f"Time period: {years.min()}-{years.max()}")
        print(f"Number of years: {len(years)}")
        print(f"Theoretical maximum observations: {theoretical_max:,}")
        print(f"Actual observations: {actual_obs:,}")
        print(f"Panel completeness: {completeness:.1f}%")
        
        # Missing country-year combinations
        full_index = pd.MultiIndex.from_product(
            [countries, years],
            names=['country', 'year']
        )
        current_index = pd.MultiIndex.from_frame(
            self.df[['country', 'year']]
        )
        
        missing_combos = full_index.difference(current_index)
        
        if len(missing_combos) > 0:
            print(f"\n⚠ Missing country-year combinations: {len(missing_combos)}")
            print("\nFirst 10 missing combinations:")
            for combo in list(missing_combos)[:10]:
                print(f"  - {combo[0]}, {combo[1]}")
        
        # Save panel structure summary
        structure_summary = pd.DataFrame({
            'Metric': [
                'Countries',
                'Years',
                'Time Span',
                'Theoretical Max Obs',
                'Actual Obs',
                'Panel Completeness (%)',
                'Missing Combinations'
            ],
            'Value': [
                len(countries),
                len(years),
                f"{years.min()}-{years.max()}",
                theoretical_max,
                actual_obs,
                f"{completeness:.1f}",
                len(missing_combos)
            ]
        })
        
        structure_summary.to_csv(
            self.output_dir / 'panel_structure_summary.csv',
            index=False
        )
        print(f"\n[OK] Saved: panel_structure_summary.csv")

    def main(self):
        """Execute full EDA workflow."""
        print("="*80)
        print("EXPLORATORY DATA ANALYSIS - MISSING PATTERNS")
        print("="*80)
        
        # Load data
        self.load_and_prepare()
        
        # Analyze missing values
        missing_stats, missing_by_year, missing_by_country = self.analyze_missing_values()
        
        # Panel completeness
        self.panel_completeness()
        
        # Visualizations
        self.plot_missing_patterns()
        self.plot_dendrogram()
        
        print("\n" + "="*80)
        print("EDA COMPLETE [OK]")
        print("="*80)
        print(f"\nAll outputs saved to: {self.output_dir}")
        print("\nUse these outputs to:")
        print("  1. Select which ITU series to use (based on coverage)")
        print("  2. Justify imputation strategies in your paper")
        print("  3. Identify potential instruments with good coverage")
        print("  4. Document data limitations in methodology section")
        
        return missing_stats, missing_by_year, missing_by_country

# Main execution
if __name__ == "__main__":
    analyzer = DataAnalyzer()
    missing_stats, missing_by_year, missing_by_country = analyzer.main()
