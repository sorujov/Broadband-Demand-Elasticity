# code/analysis/07_create_publication_outputs.py

"""
================================================================================
Generate Publication-Ready Tables and Figures
================================================================================
Purpose: Create professional outputs for Telecommunications Policy submission
Author: Samir Orujov
Date: December 11, 2025

Generates:
1. Table 1: Descriptive Statistics with Missing Data Summary
2. Table 2: Missing Data Pattern Analysis
3. Table 3: Main Results - Price Elasticity Estimates (MI)
4. Table 4: Robustness Checks - Method Comparison
5. Figure 1: Missing Data Patterns Visualization
6. Figure 2: Comparison of Imputation Methods
7. Supplementary tables for appendix
================================================================================
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import sys

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

try:
    from code.utils.config import DATA_PROCESSED, EU_COUNTRIES, EAP_COUNTRIES
except ImportError:
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from utils.config import DATA_PROCESSED, EU_COUNTRIES, EAP_COUNTRIES


class PublicationOutputs:
    """Generate publication-ready tables and figures."""
    
    def __init__(self):
        """Initialize paths."""
        self.data_dir = DATA_PROCESSED
        self.missing_dir = DATA_PROCESSED / 'missing_data_analysis'
        self.results_dir = DATA_PROCESSED / 'mi_regression_results'
        self.output_dir = DATA_PROCESSED / 'publication_outputs'
        self.output_dir.mkdir(exist_ok=True)
        
        # Set publication style
        plt.style.use('seaborn-v0_8-paper')
        sns.set_palette("Set2")
    
    def create_table1_descriptives(self):
        """
        Table 1: Descriptive Statistics with Missing Data
        
        Layout:
        Variable | N | Mean | SD | Min | Max | Missing (%)
        """
        print("="*80)
        print("CREATING TABLE 1: DESCRIPTIVE STATISTICS")
        print("="*80)
        
        # Load original data
        df = pd.read_excel(self.data_dir / 'data_merged_with_series.xlsx', 
                          engine='openpyxl')
        
        # Key variables
        key_vars = {
            'Fixed Broadband Subs (per 100)': 'fixed_broadband_subs_i4213tfbb',
            'Internet Users (%)': 'internet_users_pct_i99H',
            'Mobile Subs (per 100)': 'mobile_subs_i271',
            'Fixed Broadband Price (% GNI)': 'fixed_broad_price_i154_FBB_ts_GNI',
            'Mobile Broadband Price (% GNI)': 'mobile_broad_price_i271mb_ts_GNI',
            'GDP per capita (USD)': 'gdp_per_capita',
            'Population (millions)': 'population',
            'Urban Population (%)': 'urban_population_pct',
            'Regulatory Quality': 'regulatory_quality_estimate'
        }
        
        # Create descriptive table
        desc_data = []
        
        for label, varname in key_vars.items():
            if varname in df.columns:
                series = df[varname]
                
                row = {
                    'Variable': label,
                    'N': series.notna().sum(),
                    'Mean': series.mean(),
                    'SD': series.std(),
                    'Min': series.min(),
                    'Max': series.max(),
                    'Missing (%)': (series.isna().sum() / len(series) * 100)
                }
                
                # Special formatting for population (convert to millions)
                if varname == 'population':
                    row['Mean'] = row['Mean'] / 1e6
                    row['SD'] = row['SD'] / 1e6
                    row['Min'] = row['Min'] / 1e6
                    row['Max'] = row['Max'] / 1e6
                
                desc_data.append(row)
        
        table1 = pd.DataFrame(desc_data)
        
        # Format numbers
        table1['Mean'] = table1['Mean'].round(2)
        table1['SD'] = table1['SD'].round(2)
        table1['Min'] = table1['Min'].round(2)
        table1['Max'] = table1['Max'].round(2)
        table1['Missing (%)'] = table1['Missing (%)'].round(1)
        
        # Save
        table1.to_excel(self.output_dir / 'Table1_Descriptives.xlsx', 
                       index=False, engine='openpyxl')
        
        # Also save LaTeX version
        latex_str = table1.to_latex(index=False, float_format="%.2f")
        with open(self.output_dir / 'Table1_Descriptives.tex', 'w') as f:
            f.write(latex_str)
        
        print("\n[OK] Created Table 1: Descriptive Statistics")
        print(f"  Variables: {len(table1)}")
        print(f"  Saved: Excel + LaTeX")
        
        return table1
    
    def create_table2_missing_patterns(self):
        """
        Table 2: Missing Data Patterns by Region and Time
        
        Shows missingness for key variables across EU/EaP and time periods
        """
        print("\n" + "="*80)
        print("CREATING TABLE 2: MISSING DATA PATTERNS")
        print("="*80)
        
        # Load data
        df = pd.read_excel(self.data_dir / 'data_merged_with_series.xlsx',
                          engine='openpyxl')
        
        # Add region
        df['Region'] = df['country'].apply(
            lambda x: 'EU' if x in EU_COUNTRIES else 
                     ('EaP' if x in EAP_COUNTRIES else 'Other')
        )
        
        # Add time period
        df['Period'] = pd.cut(df['year'], 
                             bins=[2009, 2015, 2020, 2024],
                             labels=['2010-2015', '2016-2020', '2021-2023'])
        
        # Key variables
        key_vars = {
            'Price': 'fixed_broad_price_i154_FBB_ts_GNI',
            'Subscriptions': 'fixed_broadband_subs_i4213tfbb',
            'Users': 'internet_users_pct_i99H'
        }
        
        # Create pattern table
        pattern_data = []
        
        for region in ['EU', 'EaP']:
            for period in ['2010-2015', '2016-2020', '2021-2023']:
                mask = (df['Region'] == region) & (df['Period'] == period)
                subset = df[mask]
                
                if len(subset) > 0:
                    row = {
                        'Region': region,
                        'Period': period,
                        'N': len(subset)
                    }
                    
                    for label, varname in key_vars.items():
                        if varname in subset.columns:
                            missing_pct = (subset[varname].isna().sum() / len(subset) * 100)
                            row[f'{label} Missing (%)'] = missing_pct
                    
                    pattern_data.append(row)
        
        table2 = pd.DataFrame(pattern_data)
        
        # Format
        for col in table2.columns:
            if 'Missing' in col:
                table2[col] = table2[col].round(1)
        
        # Save
        table2.to_excel(self.output_dir / 'Table2_MissingPatterns.xlsx',
                       index=False, engine='openpyxl')
        
        latex_str = table2.to_latex(index=False, float_format="%.1f")
        with open(self.output_dir / 'Table2_MissingPatterns.tex', 'w') as f:
            f.write(latex_str)
        
        print("\n[OK] Created Table 2: Missing Data Patterns")
        print(f"  Rows: {len(table2)}")
        
        return table2
    
    def create_table3_main_results(self):
        """
        Table 3: Main Regression Results (Multiple Imputation)
        
        Shows pooled estimates from MI analysis
        """
        print("\n" + "="*80)
        print("CREATING TABLE 3: MAIN RESULTS (MI)")
        print("="*80)
        
        # Load pooled results
        try:
            baseline = pd.read_excel(
                self.results_dir / 'pooled_results_baseline.xlsx',
                engine='openpyxl'
            )
            twoway = pd.read_excel(
                self.results_dir / 'pooled_results_twoway.xlsx',
                engine='openpyxl'
            )
            
            # Combine
            table3 = pd.concat([baseline, twoway], ignore_index=True)
            
            # Clean up
            table3 = table3[[
                'Specification', 'Price Elasticity', 'Std. Error',
                't-statistic', 'p-value', 'CI Lower', 'CI Upper', 'FMI'
            ]]
            
            # Rename for clarity
            table3.columns = [
                'Model', 'β (Price)', 'SE', 't-stat', 'p-value',
                '95% CI Lower', '95% CI Upper', 'FMI'
            ]
            
            # Format
            table3['β (Price)'] = table3['β (Price)'].round(4)
            table3['SE'] = table3['SE'].round(4)
            table3['t-stat'] = table3['t-stat'].round(2)
            table3['p-value'] = table3['p-value'].round(4)
            table3['95% CI Lower'] = table3['95% CI Lower'].round(4)
            table3['95% CI Upper'] = table3['95% CI Upper'].round(4)
            table3['FMI'] = table3['FMI'].round(3)
            
            # Save
            table3.to_excel(self.output_dir / 'Table3_MainResults.xlsx',
                           index=False, engine='openpyxl')
            
            latex_str = table3.to_latex(index=False, float_format="%.4f")
            with open(self.output_dir / 'Table3_MainResults.tex', 'w') as f:
                f.write(latex_str)
            
            print("\n[OK] Created Table 3: Main Results")
            print(f"  Models: {len(table3)}")
            
            return table3
            
        except FileNotFoundError:
            print("\nâš  Warning: Pooled results not found")
            print("  Run 06_regression_with_multiple_imputation.py first")
            return None
    
    def create_table4_robustness(self):
        """
        Table 4: Robustness Checks - Method Comparison
        
        Compares MI, Forward Fill, and Listwise Deletion
        """
        print("\n" + "="*80)
        print("CREATING TABLE 4: ROBUSTNESS CHECKS")
        print("="*80)
        
        # Load comparison data
        try:
            comparison = pd.read_excel(
                self.missing_dir / '04_imputation_method_comparison.xlsx',
                engine='openpyxl'
            )
            
            # Select relevant methods
            methods_to_show = ['mice', 'forward_fill', 'listwise']
            table4 = comparison[comparison['method'].isin(methods_to_show)].copy()
            
            # Rename methods
            table4['method'] = table4['method'].map({
                'mice': 'Multiple Imputation (MICE)',
                'forward_fill': 'Forward Fill (LOCF)',
                'listwise': 'Listwise Deletion'
            })
            
            # Rename columns
            table4 = table4.rename(columns={
                'method': 'Method',
                'n_obs': 'N',
                'pct_obs_retained': '% Obs Retained',
                'mean_bias': 'Mean Bias (%)',
                'std_ratio': 'SD Ratio',
                'corr_preservation': 'Corr. Distance'
            })
            
            # Select columns
            table4 = table4[[
                'Method', 'N', '% Obs Retained', 
                'Mean Bias (%)', 'SD Ratio', 'Corr. Distance'
            ]]
            
            # Format
            table4['% Obs Retained'] = table4['% Obs Retained'].round(1)
            table4['Mean Bias (%)'] = table4['Mean Bias (%)'].round(2)
            table4['SD Ratio'] = table4['SD Ratio'].round(3)
            table4['Corr. Distance'] = table4['Corr. Distance'].round(3)
            
            # Save
            table4.to_excel(self.output_dir / 'Table4_Robustness.xlsx',
                           index=False, engine='openpyxl')
            
            latex_str = table4.to_latex(index=False, float_format="%.3f")
            with open(self.output_dir / 'Table4_Robustness.tex', 'w') as f:
                f.write(latex_str)
            
            print("\n[OK] Created Table 4: Robustness Checks")
            
            return table4
            
        except FileNotFoundError:
            print("\nâš  Warning: Comparison file not found")
            print("  Run 03_rigorous_missing_analysis.py first")
            return None
    
    def create_figure1_missing_visualization(self):
        """
        Figure 1: Missing Data Patterns Visualization
        
        Panel A: Missing percentage by variable
        Panel B: Missing patterns over time
        """
        print("\n" + "="*80)
        print("CREATING FIGURE 1: MISSING DATA VISUALIZATION")
        print("="*80)
        
        # Load data
        df = pd.read_excel(self.data_dir / 'data_merged_with_series.xlsx',
                          engine='openpyxl')
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # Panel A: Bar chart of missing percentages
        key_vars = {
            'Price': 'fixed_broad_price_i154_FBB_ts_GNI',
            'Subscriptions': 'fixed_broadband_subs_i4213tfbb',
            'Users': 'internet_users_pct_i99H',
            'GDP': 'gdp_per_capita',
            'Population': 'population'
        }
        
        missing_pcts = []
        var_labels = []
        
        for label, varname in key_vars.items():
            if varname in df.columns:
                pct = (df[varname].isna().sum() / len(df) * 100)
                missing_pcts.append(pct)
                var_labels.append(label)
        
        axes[0].barh(var_labels, missing_pcts, color='steelblue')
        axes[0].set_xlabel('Missing (%)', fontsize=11)
        axes[0].set_title('Panel A: Missing Data by Variable', fontsize=12, fontweight='bold')
        axes[0].grid(axis='x', alpha=0.3)
        
        # Panel B: Time series of missingness
        missing_by_year = df.groupby('year').apply(
            lambda x: pd.Series({
                'Price': (x['fixed_broad_price_i154_FBB_ts_GNI'].isna().sum() / len(x) * 100) if 'fixed_broad_price_i154_FBB_ts_GNI' in x.columns else 0,
                'Subs': (x['fixed_broadband_subs_i4213tfbb'].isna().sum() / len(x) * 100) if 'fixed_broadband_subs_i4213tfbb' in x.columns else 0
            })
        )
        
        missing_by_year.plot(ax=axes[1], marker='o', linewidth=2)
        axes[1].set_xlabel('Year', fontsize=11)
        axes[1].set_ylabel('Missing (%)', fontsize=11)
        axes[1].set_title('Panel B: Missing Data Over Time', fontsize=12, fontweight='bold')
        axes[1].legend(['Price', 'Subscriptions'], loc='best')
        axes[1].grid(alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'Figure1_MissingPatterns.png', 
                   dpi=300, bbox_inches='tight')
        plt.close()
        
        print("\n[OK] Created Figure 1: Missing Data Visualization")
        
        return fig
    
    def create_figure2_method_comparison(self):
        """
        Figure 2: Comparison of Imputation Methods
        
        Shows distribution preservation across methods
        """
        print("\n" + "="*80)
        print("CREATING FIGURE 2: METHOD COMPARISON")
        print("="*80)
        
        try:
            comparison = pd.read_excel(
                self.missing_dir / '04_imputation_method_comparison.xlsx',
                engine='openpyxl'
            )
            
            fig, axes = plt.subplots(1, 2, figsize=(12, 5))
            
            # Filter to key methods
            methods = ['listwise', 'forward_fill', 'mice']
            comp_subset = comparison[comparison['method'].isin(methods)].copy()
            
            # Rename
            comp_subset['method'] = comp_subset['method'].map({
                'listwise': 'Listwise\nDeletion',
                'forward_fill': 'Forward\nFill',
                'mice': 'Multiple\nImputation'
            })
            
            # Panel A: Mean bias
            if 'mean_bias' in comp_subset.columns:
                axes[0].bar(comp_subset['method'], comp_subset['mean_bias'], 
                           color=['#e74c3c', '#f39c12', '#2ecc71'])
                axes[0].axhline(0, color='black', linestyle='--', linewidth=1)
                axes[0].set_ylabel('Mean Bias (%)', fontsize=11)
                axes[0].set_title('Panel A: Distribution Preservation\n(Lower is Better)', 
                                 fontsize=12, fontweight='bold')
                axes[0].grid(axis='y', alpha=0.3)
            
            # Panel B: Sample size retained
            if 'pct_obs_retained' in comp_subset.columns:
                axes[1].bar(comp_subset['method'], comp_subset['pct_obs_retained'],
                           color=['#e74c3c', '#f39c12', '#2ecc71'])
                axes[1].set_ylabel('Observations Retained (%)', fontsize=11)
                axes[1].set_title('Panel B: Sample Size\n(Higher is Better)', 
                                 fontsize=12, fontweight='bold')
                axes[1].set_ylim([0, 105])
                axes[1].grid(axis='y', alpha=0.3)
            
            plt.tight_layout()
            plt.savefig(self.output_dir / 'Figure2_MethodComparison.png',
                       dpi=300, bbox_inches='tight')
            plt.close()
            
            print("\n[OK] Created Figure 2: Method Comparison")
            
            return fig
            
        except FileNotFoundError:
            print("\nâš  Warning: Comparison file not found")
            return None
    
    def generate_all_outputs(self):
        """Generate all publication-ready outputs."""
        print("="*80)
        print("GENERATING ALL PUBLICATION OUTPUTS")
        print("="*80)
        
        # Tables
        table1 = self.create_table1_descriptives()
        table2 = self.create_table2_missing_patterns()
        table3 = self.create_table3_main_results()
        table4 = self.create_table4_robustness()
        
        # Figures
        fig1 = self.create_figure1_missing_visualization()
        fig2 = self.create_figure2_method_comparison()
        
        print("\n" + "="*80)
        print("ALL OUTPUTS GENERATED")
        print("="*80)
        print(f"\nSaved to: {self.output_dir}")
        print("\nFiles created:")
        print("  Tables (Excel + LaTeX):")
        print("    - Table1_Descriptives")
        print("    - Table2_MissingPatterns")
        print("    - Table3_MainResults")
        print("    - Table4_Robustness")
        print("\n  Figures (PNG, 300 DPI):")
        print("    - Figure1_MissingPatterns")
        print("    - Figure2_MethodComparison")
        
        print("\n✅ Ready for submission to Telecommunications Policy!")
        
        return self


# Main execution
if __name__ == "__main__":
    generator = PublicationOutputs()
    generator.generate_all_outputs()
