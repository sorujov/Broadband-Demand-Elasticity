# code/analysis/05_descriptive_stats.py
"""
================================================================================
Descriptive Statistics Script
================================================================================
Purpose: Generate comprehensive descriptive statistics and initial visualizations
Author: Samir Orujov
Date: November 13, 2025
================================================================================
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from datetime import datetime
import sys

# Add project root to path for imports
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

# Import from code.utils.config
try:
    from code.utils.config import (
        DATA_PROCESSED, RESULTS_TABLES, FIGURES_DESC,
        EU_COUNTRIES, EAP_COUNTRIES, START_YEAR, END_YEAR
    )
except ImportError:
    # Fallback for direct execution
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from utils.config import (
        DATA_PROCESSED, RESULTS_TABLES, FIGURES_DESC,
        EU_COUNTRIES, EAP_COUNTRIES, START_YEAR, END_YEAR
    )

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (10, 6)
plt.rcParams['font.size'] = 10

class DescriptiveAnalysis:
    """Generate descriptive statistics and visualizations."""

    def __init__(self):
        self.data_file = DATA_PROCESSED / 'broadband_analysis_clean.csv'
        self.output_dir = RESULTS_TABLES
        self.figures_dir = FIGURES_DESC

    def load_data(self):
        """Load clean dataset."""
        print("="*80)
        print("LOADING CLEAN DATASET")
        print("="*80)

        if not self.data_file.exists():
            print(f"\n✗ File not found: {self.data_file}")
            print("\nPlease run: python 04_prepare_data.py first")
            return None

        df = pd.read_csv(self.data_file)
        print(f"\n✓ Loaded: {df.shape[0]} rows × {df.shape[1]} columns")
        print(f"  Countries: {df['country'].nunique()}")
        print(f"  Years: {df['year'].min()}-{df['year'].max()}")
        print(f"  Regions: {df['region'].value_counts().to_dict()}")

        return df

    def summary_statistics(self, df):
        """Generate summary statistics by region."""
        print("\n" + "="*80)
        print("SUMMARY STATISTICS BY REGION")
        print("="*80)

        # Key variables for analysis
        key_vars = [col for col in [
            'price_fixed_bb', 'bandwidth_use', 'bb_subs_per100',
            'gdp_per_capita', 'internet_users_pct', 'population',
            'log_price', 'log_bandwidth_use', 'log_bb_subs', 'log_gdp_pc'
        ] if col in df.columns]

        if not key_vars:
            print("\n⚠ No key variables found")
            return None

        # Overall statistics
        print("\n--- OVERALL STATISTICS ---")
        overall_stats = df[key_vars].describe()
        print(overall_stats.round(2))

        # By region
        print("\n--- BY REGION ---")
        regional_stats = df.groupby('region')[key_vars].describe()
        print(regional_stats.round(2))

        # Save to CSV
        overall_file = self.output_dir / 'descriptive_stats_overall.csv'
        regional_file = self.output_dir / 'descriptive_stats_by_region.csv'

        overall_stats.to_csv(overall_file)
        regional_stats.to_csv(regional_file)

        print(f"\n✓ Saved: {overall_file}")
        print(f"✓ Saved: {regional_file}")

        return overall_stats, regional_stats

    def correlation_analysis(self, df):
        """Compute correlation matrices."""
        print("\n" + "="*80)
        print("CORRELATION ANALYSIS")
        print("="*80)

        # Variables for correlation
        corr_vars = [col for col in [
            'log_price', 'log_bandwidth_use', 'log_bb_subs', 'log_gdp_pc',
            'internet_users_pct', 'gdp_growth'
        ] if col in df.columns]

        if len(corr_vars) < 2:
            print("\n⚠ Not enough variables for correlation")
            return None

        # Overall correlation
        corr_matrix = df[corr_vars].corr()
        print("\nCorrelation Matrix:")
        print(corr_matrix.round(3))

        # Save correlation matrix
        corr_file = self.output_dir / 'correlation_matrix.csv'
        corr_matrix.to_csv(corr_file)
        print(f"\n✓ Saved: {corr_file}")

        # Plot correlation heatmap
        plt.figure(figsize=(10, 8))
        sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm', 
                   center=0, square=True, linewidths=1)
        plt.title('Correlation Matrix of Key Variables', fontsize=14, fontweight='bold')
        plt.tight_layout()

        heatmap_file = self.figures_dir / 'correlation_heatmap.png'
        plt.savefig(heatmap_file, dpi=300, bbox_inches='tight')
        print(f"✓ Saved: {heatmap_file}")
        plt.close()

        return corr_matrix

    def regional_comparison(self, df):
        """Compare key variables across regions."""
        print("\n" + "="*80)
        print("REGIONAL COMPARISON")
        print("="*80)

        # Variables to compare
        compare_vars = [col for col in [
            'price_fixed_bb', 'bandwidth_use', 'bb_subs_per100', 
            'gdp_per_capita', 'internet_users_pct'
        ] if col in df.columns]

        if not compare_vars:
            print("\n⚠ No variables to compare")
            return None

        # Compute means by region
        comparison = df.groupby('region')[compare_vars].mean()

        # Add difference and percentage difference
        if 'EU' in comparison.index and 'EaP' in comparison.index:
            comparison.loc['Difference'] = comparison.loc['EaP'] - comparison.loc['EU']
            comparison.loc['Pct_Diff'] = (comparison.loc['Difference'] / comparison.loc['EU'] * 100)

        print("\nRegional Means:")
        print(comparison.round(2))

        # Save comparison
        comparison_file = self.output_dir / 'regional_comparison.csv'
        comparison.to_csv(comparison_file)
        print(f"\n✓ Saved: {comparison_file}")

        return comparison

    def time_trends(self, df):
        """Plot time trends for key variables."""
        print("\n" + "="*80)
        print("TIME TRENDS VISUALIZATION")
        print("="*80)

        # Variables to plot
        plot_vars = {
            'price_fixed_bb': 'Fixed Broadband Price (USD)',
            'bb_subs_per100': 'Broadband Subscriptions per 100',
            'bandwidth_use': 'Internet Bandwidth (Gbit/s)',
            'gdp_per_capita': 'GDP per Capita (USD)'
        }

        # Filter to available variables
        plot_vars = {k: v for k, v in plot_vars.items() if k in df.columns}

        if not plot_vars:
            print("\n⚠ No variables to plot")
            return None

        # Create 2x2 subplot
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        axes = axes.flatten()

        for idx, (var, title) in enumerate(plot_vars.items()):
            if idx >= 4:
                break

            # Calculate means by region and year
            trend_data = df.groupby(['year', 'region'])[var].mean().reset_index()

            # Plot
            for region in ['EU', 'EaP']:
                data = trend_data[trend_data['region'] == region]
                axes[idx].plot(data['year'], data[var], marker='o', 
                             label=region, linewidth=2, markersize=4)

            axes[idx].set_xlabel('Year', fontsize=10)
            axes[idx].set_ylabel(title, fontsize=10)
            axes[idx].set_title(title, fontsize=11, fontweight='bold')
            axes[idx].legend()
            axes[idx].grid(True, alpha=0.3)

        # Remove empty subplots
        for idx in range(len(plot_vars), 4):
            fig.delaxes(axes[idx])

        plt.suptitle('Time Trends: EU vs Eastern Partnership', 
                    fontsize=14, fontweight='bold', y=1.00)
        plt.tight_layout()

        trends_file = self.figures_dir / 'time_trends_by_region.png'
        plt.savefig(trends_file, dpi=300, bbox_inches='tight')
        print(f"\n✓ Saved: {trends_file}")
        plt.close()

        return trend_data

    def scatter_plots(self, df):
        """Create scatter plots for price-demand relationship."""
        print("\n" + "="*80)
        print("PRICE-DEMAND SCATTER PLOTS")
        print("="*80)

        # Check if log variables exist
        if 'log_price' not in df.columns or 'log_bandwidth_use' not in df.columns:
            print("\n⚠ Log variables not found, skipping scatter plots")
            return None

        # Create scatter plot
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

        # Plot 1: Log-log (for elasticity visualization)
        for region in ['EU', 'EaP']:
            data = df[df['region'] == region]
            ax1.scatter(data['log_price'], data['log_bandwidth_use'], 
                       alpha=0.5, label=region, s=30)

        ax1.set_xlabel('Log(Price)', fontsize=11)
        ax1.set_ylabel('Log(Bandwidth Use)', fontsize=11)
        ax1.set_title('Price-Demand Relationship (Log-Log)', 
                     fontsize=12, fontweight='bold')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # Plot 2: Original scale
        if 'price_fixed_bb' in df.columns and 'bandwidth_use' in df.columns:
            for region in ['EU', 'EaP']:
                data = df[df['region'] == region]
                ax2.scatter(data['price_fixed_bb'], data['bandwidth_use'], 
                           alpha=0.5, label=region, s=30)

            ax2.set_xlabel('Price (USD)', fontsize=11)
            ax2.set_ylabel('Bandwidth Use (Gbit/s)', fontsize=11)
            ax2.set_title('Price-Demand Relationship (Original Scale)', 
                         fontsize=12, fontweight='bold')
            ax2.legend()
            ax2.grid(True, alpha=0.3)

        plt.tight_layout()

        scatter_file = self.figures_dir / 'price_demand_scatter.png'
        plt.savefig(scatter_file, dpi=300, bbox_inches='tight')
        print(f"\n✓ Saved: {scatter_file}")
        plt.close()

        return True

    def data_quality_report(self, df):
        """Generate data quality report."""
        print("\n" + "="*80)
        print("DATA QUALITY REPORT")
        print("="*80)

        # Missing data by variable
        missing = df.isnull().sum()
        missing = missing[missing > 0].sort_values(ascending=False)

        if len(missing) > 0:
            print("\nVariables with missing data:")
            missing_pct = (missing / len(df) * 100).round(2)
            for var, count in missing.items():
                print(f"  {var}: {count} ({missing_pct[var]}%)")
        else:
            print("\n✓ No missing data!")

        # Observations by country
        print("\n\nObservations by country:")
        country_counts = df.groupby('country').size().sort_values(ascending=False)
        print(country_counts)

        # Balanced panel check
        expected_obs = len(df['country'].unique()) * len(df['year'].unique())
        actual_obs = len(df)
        balance_pct = (actual_obs / expected_obs * 100)

        print(f"\n\nPanel balance:")
        print(f"  Expected: {expected_obs} observations")
        print(f"  Actual: {actual_obs} observations")
        print(f"  Balance: {balance_pct:.1f}%")

        # Save quality report
        quality_report = {
            'Total_Observations': len(df),
            'Countries': df['country'].nunique(),
            'Years': f"{df['year'].min()}-{df['year'].max()}",
            'Missing_Variables': len(missing),
            'Panel_Balance_Pct': f"{balance_pct:.1f}%"
        }

        quality_file = self.output_dir / 'data_quality_report.csv'
        pd.DataFrame([quality_report]).to_csv(quality_file, index=False)
        print(f"\n✓ Saved: {quality_file}")

        return quality_report


def main():
    """Main execution function."""
    print("="*80)
    print("DESCRIPTIVE STATISTICS SCRIPT")
    print("="*80)
    print(f"Execution time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    try:
        # Initialize analyzer
        analyzer = DescriptiveAnalysis()

        # Load data
        df = analyzer.load_data()
        if df is None:
            return

        # Generate summary statistics
        analyzer.summary_statistics(df)

        # Correlation analysis
        analyzer.correlation_analysis(df)

        # Regional comparison
        analyzer.regional_comparison(df)

        # Time trends
        analyzer.time_trends(df)

        # Scatter plots
        analyzer.scatter_plots(df)

        # Data quality report
        analyzer.data_quality_report(df)

        print("\n" + "="*80)
        print("DESCRIPTIVE ANALYSIS COMPLETE ✓")
        print("="*80)
        print(f"\nResults saved to:")
        print(f"  Tables: {RESULTS_TABLES}")
        print(f"  Figures: {FIGURES_DESC}")
        print("\nNext step: Run baseline regression analysis")

    except Exception as e:
        print(f"\n✗ Error: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
