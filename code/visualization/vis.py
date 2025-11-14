# figure_generation.py
"""
================================================================================
Figure Generation Script for Academic Paper
================================================================================
Purpose: Create Figure 1 (Time Trends) and Figure 2 (Price-Demand Scatter)
Author: Samir Orujov
Date: November 13, 2025
================================================================================
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Set style for publication-quality figures
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman']
plt.rcParams['font.size'] = 11
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['axes.titlesize'] = 13
plt.rcParams['xtick.labelsize'] = 10
plt.rcParams['ytick.labelsize'] = 10
plt.rcParams['legend.fontsize'] = 10
plt.rcParams['figure.dpi'] = 300

# Paths
DATA_PATH = Path('data/processed/broadband_analysis_clean.csv')
OUTPUT_DIR = Path('figures/descriptive')
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

def load_data():
    """Load cleaned analysis data."""
    print("Loading data...")
    df = pd.read_csv(DATA_PATH)
    print(f"✓ Loaded {len(df)} observations")
    print(f"  Countries: {df['country'].nunique()}")
    print(f"  Years: {df['year'].min()}-{df['year'].max()}")
    print(f"  Regions: {df['region'].value_counts().to_dict()}")
    return df

def create_time_trends_figure(df):
    """
    Create Figure 1: Time trends of key variables by region (2010-2023)
    Shows 4 panels: (a) prices, (b) bandwidth, (c) subscriptions, (d) GDP
    """
    print("\nCreating Figure 1: Time Trends by Region...")
    
    # Aggregate by year and region
    trends = df.groupby(['year', 'region']).agg({
        'fixed_broad_price': 'mean',
        'int_bandwidth': 'mean',
        'bb_subs_per100': 'mean',
        'gdp_per_capita': 'mean'
    }).reset_index()
    
    # Create figure with 4 subplots (2x2)
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle('Evolution of Key Variables by Region (2010-2023)', 
                 fontsize=14, fontweight='bold', y=0.995)
    
    # Color palette
    colors = {'EU': '#2E86AB', 'EaP': '#A23B72'}
    
    # Panel (a): Broadband Prices
    ax = axes[0, 0]
    for region in ['EU', 'EaP']:
        data = trends[trends['region'] == region]
        ax.plot(data['year'], data['fixed_broad_price'], 
               marker='o', linewidth=2.5, markersize=6,
               label=region, color=colors[region])
    ax.set_xlabel('Year', fontweight='bold')
    ax.set_ylabel('Price (USD/month)', fontweight='bold')
    ax.set_title('(a) Fixed Broadband Prices', fontweight='bold', loc='left')
    ax.legend(loc='best', frameon=True, shadow=True)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(2009.5, 2023.5)
    
    # Add trend annotation
    eu_start = trends[(trends['region']=='EU') & (trends['year']==2010)]['fixed_broad_price'].values[0]
    eu_end = trends[(trends['region']=='EU') & (trends['year']==2023)]['fixed_broad_price'].values[0]
    eu_change = ((eu_end - eu_start) / eu_start) * 100
    
    eap_start = trends[(trends['region']=='EaP') & (trends['year']==2010)]['fixed_broad_price'].values[0]
    eap_end = trends[(trends['region']=='EaP') & (trends['year']==2023)]['fixed_broad_price'].values[0]
    eap_change = ((eap_end - eap_start) / eap_start) * 100
    
    ax.text(0.05, 0.95, f'EU: {eu_change:.0f}% decline\nEaP: {eap_change:.0f}% decline',
           transform=ax.transAxes, fontsize=9, verticalalignment='top',
           bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    # Panel (b): International Bandwidth Usage
    ax = axes[0, 1]
    for region in ['EU', 'EaP']:
        data = trends[trends['region'] == region]
        ax.plot(data['year'], data['int_bandwidth'], 
               marker='s', linewidth=2.5, markersize=6,
               label=region, color=colors[region])
    ax.set_xlabel('Year', fontweight='bold')
    ax.set_ylabel('Bandwidth (Gbit/s)', fontweight='bold')
    ax.set_title('(b) International Bandwidth Usage', fontweight='bold', loc='left')
    ax.legend(loc='best', frameon=True, shadow=True)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(2009.5, 2023.5)
    
    # Add growth annotation
    eu_bw_start = trends[(trends['region']=='EU') & (trends['year']==2010)]['int_bandwidth'].values[0]
    eu_bw_end = trends[(trends['region']=='EU') & (trends['year']==2023)]['int_bandwidth'].values[0]
    eu_bw_growth = eu_bw_end / eu_bw_start
    
    eap_bw_start = trends[(trends['region']=='EaP') & (trends['year']==2010)]['int_bandwidth'].values[0]
    eap_bw_end = trends[(trends['region']=='EaP') & (trends['year']==2023)]['int_bandwidth'].values[0]
    eap_bw_growth = eap_bw_end / eap_bw_start
    
    ax.text(0.05, 0.95, f'EU: {eu_bw_growth:.1f}× growth\nEaP: {eap_bw_growth:.1f}× growth',
           transform=ax.transAxes, fontsize=9, verticalalignment='top',
           bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.5))
    
    # Highlight COVID-19 period
    ax.axvspan(2020, 2021, alpha=0.15, color='red', label='COVID-19')
    
    # Panel (c): Fixed Broadband Subscriptions
    ax = axes[1, 0]
    for region in ['EU', 'EaP']:
        data = trends[trends['region'] == region]
        ax.plot(data['year'], data['bb_subs_per100'], 
               marker='^', linewidth=2.5, markersize=6,
               label=region, color=colors[region])
    ax.set_xlabel('Year', fontweight='bold')
    ax.set_ylabel('Subscriptions per 100 inhabitants', fontweight='bold')
    ax.set_title('(c) Fixed Broadband Penetration', fontweight='bold', loc='left')
    ax.legend(loc='best', frameon=True, shadow=True)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(2009.5, 2023.5)
    
    # Add gap annotation
    gap_2010 = trends[(trends['region']=='EU') & (trends['year']==2010)]['bb_subs_per100'].values[0] - \
               trends[(trends['region']=='EaP') & (trends['year']==2010)]['bb_subs_per100'].values[0]
    gap_2023 = trends[(trends['region']=='EU') & (trends['year']==2023)]['bb_subs_per100'].values[0] - \
               trends[(trends['region']=='EaP') & (trends['year']==2023)]['bb_subs_per100'].values[0]
    
    ax.text(0.05, 0.95, f'Gap 2010: {gap_2010:.1f} pp\nGap 2023: {gap_2023:.1f} pp',
           transform=ax.transAxes, fontsize=9, verticalalignment='top',
           bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.5))
    
    # Panel (d): GDP per Capita
    ax = axes[1, 1]
    for region in ['EU', 'EaP']:
        data = trends[trends['region'] == region]
        ax.plot(data['year'], data['gdp_per_capita'], 
               marker='d', linewidth=2.5, markersize=6,
               label=region, color=colors[region])
    ax.set_xlabel('Year', fontweight='bold')
    ax.set_ylabel('GDP per capita (USD)', fontweight='bold')
    ax.set_title('(d) Economic Development', fontweight='bold', loc='left')
    ax.legend(loc='best', frameon=True, shadow=True)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(2009.5, 2023.5)
    
    # Format y-axis with thousands separator
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{int(x):,}'))
    
    # Add income ratio annotation
    eu_gdp_2023 = trends[(trends['region']=='EU') & (trends['year']==2023)]['gdp_per_capita'].values[0]
    eap_gdp_2023 = trends[(trends['region']=='EaP') & (trends['year']==2023)]['gdp_per_capita'].values[0]
    ratio = eu_gdp_2023 / eap_gdp_2023
    
    ax.text(0.05, 0.95, f'EU/EaP ratio: {ratio:.1f}×',
           transform=ax.transAxes, fontsize=9, verticalalignment='top',
           bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.5))
    
    plt.tight_layout()
    
    # Save figure
    output_path = OUTPUT_DIR / 'time_trends_by_region.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {output_path}")
    
    # Also save as PDF for LaTeX
    output_path_pdf = OUTPUT_DIR / 'time_trends_by_region.pdf'
    plt.savefig(output_path_pdf, dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {output_path_pdf}")
    
    plt.show()
    
    return fig

def create_price_demand_scatter(df):
    """
    Create Figure 2: Price-Demand relationship (log-log scale)
    Scatter plot showing negative correlation with separate markers for EU/EaP
    """
    print("\nCreating Figure 2: Price-Demand Scatter...")
    
    # Remove missing values
    plot_df = df[['country', 'year', 'region', 'fixed_broad_price', 
                   'int_bandwidth', 'bb_subs_per100']].dropna()
    
    # Create figure
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle('Price-Demand Relationship (Log-Log Scale)', 
                 fontsize=14, fontweight='bold')
    
    # Color and marker settings
    colors = {'EU': '#2E86AB', 'EaP': '#A23B72'}
    markers = {'EU': 'o', 'EaP': '^'}
    
    # Panel (a): Price vs Bandwidth Usage
    ax = axes[0]
    for region in ['EU', 'EaP']:
        data = plot_df[plot_df['region'] == region]
        ax.scatter(data['fixed_broad_price'], data['int_bandwidth'],
                  c=colors[region], marker=markers[region], 
                  s=60, alpha=0.6, edgecolors='black', linewidth=0.5,
                  label=region)
    
    # Add regression lines
    from scipy import stats
    for region in ['EU', 'EaP']:
        data = plot_df[plot_df['region'] == region]
        x = np.log(data['fixed_broad_price'])
        y = np.log(data['int_bandwidth'])
        
        # Remove infinite values
        mask = np.isfinite(x) & np.isfinite(y)
        x_clean = x[mask]
        y_clean = y[mask]
        
        if len(x_clean) > 10:
            slope, intercept, r_value, p_value, std_err = stats.linregress(x_clean, y_clean)
            
            # Create prediction line
            x_range = np.linspace(x_clean.min(), x_clean.max(), 100)
            y_pred = intercept + slope * x_range
            
            ax.plot(np.exp(x_range), np.exp(y_pred), 
                   '--', color=colors[region], linewidth=2, alpha=0.8)
            
            # Add elasticity annotation
            ax.text(0.95, 0.95 if region == 'EU' else 0.88, 
                   f'{region}: elasticity = {slope:.2f}\n(R² = {r_value**2:.3f})',
                   transform=ax.transAxes, fontsize=9,
                   verticalalignment='top', horizontalalignment='right',
                   bbox=dict(boxstyle='round', facecolor=colors[region], 
                            alpha=0.2, edgecolor=colors[region]))
    
    ax.set_xlabel('Fixed Broadband Price (USD/month)', fontweight='bold')
    ax.set_ylabel('International Bandwidth (Gbit/s)', fontweight='bold')
    ax.set_title('(a) Price vs. Bandwidth Usage', fontweight='bold', loc='left')
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.legend(loc='lower left', frameon=True, shadow=True)
    ax.grid(True, alpha=0.3, which='both', linestyle=':')
    
    # Panel (b): Price vs Subscriptions
    ax = axes[1]
    for region in ['EU', 'EaP']:
        data = plot_df[plot_df['region'] == region]
        ax.scatter(data['fixed_broad_price'], data['bb_subs_per100'],
                  c=colors[region], marker=markers[region], 
                  s=60, alpha=0.6, edgecolors='black', linewidth=0.5,
                  label=region)
    
    # Add regression lines
    for region in ['EU', 'EaP']:
        data = plot_df[plot_df['region'] == region]
        x = np.log(data['fixed_broad_price'])
        y = np.log(data['bb_subs_per100'])
        
        # Remove infinite values
        mask = np.isfinite(x) & np.isfinite(y)
        x_clean = x[mask]
        y_clean = y[mask]
        
        if len(x_clean) > 10:
            slope, intercept, r_value, p_value, std_err = stats.linregress(x_clean, y_clean)
            
            # Create prediction line
            x_range = np.linspace(x_clean.min(), x_clean.max(), 100)
            y_pred = intercept + slope * x_range
            
            ax.plot(np.exp(x_range), np.exp(y_pred), 
                   '--', color=colors[region], linewidth=2, alpha=0.8)
            
            # Add elasticity annotation
            ax.text(0.95, 0.95 if region == 'EU' else 0.88, 
                   f'{region}: elasticity = {slope:.2f}\n(R² = {r_value**2:.3f})',
                   transform=ax.transAxes, fontsize=9,
                   verticalalignment='top', horizontalalignment='right',
                   bbox=dict(boxstyle='round', facecolor=colors[region], 
                            alpha=0.2, edgecolor=colors[region]))
    
    ax.set_xlabel('Fixed Broadband Price (USD/month)', fontweight='bold')
    ax.set_ylabel('Fixed Broadband Subscriptions per 100', fontweight='bold')
    ax.set_title('(b) Price vs. Penetration Rate', fontweight='bold', loc='left')
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.legend(loc='lower left', frameon=True, shadow=True)
    ax.grid(True, alpha=0.3, which='both', linestyle=':')
    
    plt.tight_layout()
    
    # Save figure
    output_path = OUTPUT_DIR / 'price_demand_scatter.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {output_path}")
    
    # Also save as PDF
    output_path_pdf = OUTPUT_DIR / 'price_demand_scatter.pdf'
    plt.savefig(output_path_pdf, dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {output_path_pdf}")
    
    plt.show()
    
    return fig

def main():
    """Main execution function."""
    print("="*80)
    print("FIGURE GENERATION FOR ACADEMIC PAPER")
    print("="*80)
    print(f"Output directory: {OUTPUT_DIR}")
    
    # Load data
    df = load_data()
    
    # Create figures
    fig1 = create_time_trends_figure(df)
    fig2 = create_price_demand_scatter(df)
    
    print("\n" + "="*80)
    print("✓ FIGURE GENERATION COMPLETE")
    print("="*80)
    print(f"\nGenerated files:")
    print(f"  1. time_trends_by_region.png (Figure 1)")
    print(f"  2. time_trends_by_region.pdf (Figure 1 - LaTeX)")
    print(f"  3. price_demand_scatter.png (Figure 2)")
    print(f"  4. price_demand_scatter.pdf (Figure 2 - LaTeX)")
    print(f"\nLocation: {OUTPUT_DIR}/")
    print("\nYou can now uncomment the figure code in your LaTeX document!")

if __name__ == "__main__":
    main()
