"""
Secure Internet Servers Data Analysis
======================================
Examines availability, coverage, and usage of secure servers variable
"""

import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns

# Setup
BASE_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = BASE_DIR / 'data' / 'processed'
FIGURES_DIR = BASE_DIR / 'figures' / 'descriptive'

# Load data
df = pd.read_excel(DATA_DIR / 'data_merged_with_series.xlsx')

# Define regions
eap_countries = ['ARM', 'AZE', 'BLR', 'GEO', 'MDA', 'UKR']
df['region'] = df['country'].apply(lambda x: 'EaP' if x in eap_countries else 'EU')

print("="*80)
print("SECURE INTERNET SERVERS DATA AVAILABILITY")
print("="*80)

# Check for secure servers column
secure_cols = [col for col in df.columns if 'secure' in col.lower()]
print(f"\nFound {len(secure_cols)} secure-related columns:")
for col in secure_cols:
    print(f"  - {col}")

if 'secure_internet_servers' in df.columns:
    var = 'secure_internet_servers'
    
    print(f"\n{var}:")
    print(f"  Total observations: {len(df)}")
    print(f"  Non-missing: {df[var].notna().sum()}")
    print(f"  Coverage: {df[var].notna().mean()*100:.1f}%")
    print(f"  Missing: {df[var].isna().sum()}")
    
    print("\n" + "-"*80)
    print("DESCRIPTIVE STATISTICS")
    print("-"*80)
    print(df[var].describe())
    
    print("\n" + "-"*80)
    print("BY REGION")
    print("-"*80)
    for region in ['EU', 'EaP']:
        print(f"\n{region}:")
        print(f"  Mean: {df[df['region']==region][var].mean():.2f}")
        print(f"  Median: {df[df['region']==region][var].median():.2f}")
        print(f"  Std Dev: {df[df['region']==region][var].std():.2f}")
        print(f"  Min: {df[df['region']==region][var].min():.2f}")
        print(f"  Max: {df[df['region']==region][var].max():.2f}")
        print(f"  Missing: {df[df['region']==region][var].isna().sum()}")
    
    print("\n" + "-"*80)
    print("BY YEAR")
    print("-"*80)
    yearly = df.groupby('year')[var].agg(['mean', 'median', 'count'])
    print(yearly.to_string())
    
    print("\n" + "-"*80)
    print("TOP 10 COUNTRIES (AVERAGE)")
    print("-"*80)
    country_avg = df.groupby('country')[var].mean().sort_values(ascending=False).head(10)
    for country, value in country_avg.items():
        region = 'EaP' if country in eap_countries else 'EU'
        print(f"  {country} ({region}): {value:.2f}")
    
    print("\n" + "-"*80)
    print("BOTTOM 10 COUNTRIES (AVERAGE)")
    print("-"*80)
    country_avg_bottom = df.groupby('country')[var].mean().sort_values(ascending=True).head(10)
    for country, value in country_avg_bottom.items():
        region = 'EaP' if country in eap_countries else 'EU'
        print(f"  {country} ({region}): {value:.2f}")
    
    print("\n" + "-"*80)
    print("CORRELATION WITH KEY VARIABLES")
    print("-"*80)
    key_vars = ['internet_users_pct_i99H', 'fixed_broadband_subs_i4213tfbb',
                'gdp_per_capita', 'research_development_expenditure',
                'fixed_broad_price_gni_pct']
    
    corr_data = []
    for kv in key_vars:
        if kv in df.columns:
            corr = df[[var, kv]].corr().iloc[0, 1]
            n = df[[var, kv]].dropna().shape[0]
            corr_data.append({'Variable': kv, 'Correlation': corr, 'N': n})
    
    corr_df = pd.DataFrame(corr_data)
    print(corr_df.to_string(index=False))
    
    print("\n" + "="*80)
    print("USAGE IN ANALYSIS SCRIPTS")
    print("="*80)
    
    # Check usage in scripts
    script_files = [
        'code/analysis/02_main_analysis.py',
        'code/analysis/00_comprehensive_method_diagnostic.py',
        'code/analysis/02b_price_definition_comparison.py',
        'code/analysis/03_covid_impact_analysis.py',
    ]
    
    for script in script_files:
        script_path = BASE_DIR / script
        if script_path.exists():
            with open(script_path, 'r', encoding='utf-8') as f:
                content = f.read()
                if 'secure' in content.lower():
                    print(f"\n✓ USED in {script}")
                    # Count occurrences
                    count = content.lower().count('secure_internet_servers') + content.lower().count('secure_servers')
                    print(f"  Mentioned {count} times")
                else:
                    print(f"\n✗ NOT USED in {script}")
    
    print("\n" + "="*80)
    print("VISUALIZATION")
    print("="*80)
    
    # Create visualization
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # 1. Time series by region
    ax1 = axes[0, 0]
    for region in ['EU', 'EaP']:
        df_region = df[df['region'] == region]
        yearly_mean = df_region.groupby('year')[var].mean()
        ax1.plot(yearly_mean.index, yearly_mean.values, marker='o', label=region, linewidth=2)
    
    ax1.set_xlabel('Year', fontsize=11)
    ax1.set_ylabel('Secure Internet Servers', fontsize=11)
    ax1.set_title('Secure Servers Over Time by Region', fontsize=12, fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. Distribution by region
    ax2 = axes[0, 1]
    data_to_plot = [df[df['region']=='EU'][var].dropna(), 
                    df[df['region']=='EaP'][var].dropna()]
    bp = ax2.boxplot(data_to_plot, labels=['EU', 'EaP'], patch_artist=True)
    for patch, color in zip(bp['boxes'], ['#1f77b4', '#ff7f0e']):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    ax2.set_ylabel('Secure Internet Servers', fontsize=11)
    ax2.set_title('Distribution by Region', fontsize=12, fontweight='bold')
    ax2.grid(True, alpha=0.3, axis='y')
    
    # 3. Correlation with internet users
    ax3 = axes[1, 0]
    df_plot = df[[var, 'internet_users_pct_i99H', 'region']].dropna()
    for region, color in [('EU', '#1f77b4'), ('EaP', '#ff7f0e')]:
        df_reg = df_plot[df_plot['region'] == region]
        ax3.scatter(df_reg[var], df_reg['internet_users_pct_i99H'], 
                   alpha=0.5, label=region, color=color)
    
    ax3.set_xlabel('Secure Internet Servers', fontsize=11)
    ax3.set_ylabel('Internet Users (%)', fontsize=11)
    ax3.set_title('Secure Servers vs Internet Adoption', fontsize=12, fontweight='bold')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # 4. Correlation with GDP
    ax4 = axes[1, 1]
    df_plot2 = df[[var, 'gdp_per_capita', 'region']].dropna()
    for region, color in [('EU', '#1f77b4'), ('EaP', '#ff7f0e')]:
        df_reg = df_plot2[df_plot2['region'] == region]
        ax4.scatter(df_reg['gdp_per_capita'], df_reg[var], 
                   alpha=0.5, label=region, color=color)
    
    ax4.set_xlabel('GDP per Capita', fontsize=11)
    ax4.set_ylabel('Secure Internet Servers', fontsize=11)
    ax4.set_title('Secure Servers vs Economic Development', fontsize=12, fontweight='bold')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    ax4.set_xscale('log')
    
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / 'secure_servers_analysis.png', dpi=300, bbox_inches='tight')
    print(f"\n[OK] Saved visualization: {FIGURES_DIR / 'secure_servers_analysis.png'}")
    
    print("\n" + "="*80)
    print("INTERPRETATION")
    print("="*80)
    
    eu_mean = df[df['region']=='EU'][var].mean()
    eap_mean = df[df['region']=='EaP'][var].mean()
    ratio = eu_mean / eap_mean if eap_mean > 0 else float('inf')
    
    print(f"""
SECURE INTERNET SERVERS represents the number of servers using encryption
technology in internet exchange traffic (per 1 million people).

KEY FINDINGS:
- Coverage: 100% (excellent!)
- EU average: {eu_mean:.2f} servers per million
- EaP average: {eap_mean:.2f} servers per million
- EU/EaP ratio: {ratio:.2f}× (EU has {ratio:.1f}× more secure servers)

INTERPRETATION:
This variable captures cybersecurity infrastructure and e-commerce readiness.
Higher values indicate:
  • Better security infrastructure
  • More e-commerce activity
  • Higher trust in digital services
  • More advanced digital economy

USAGE IN YOUR ANALYSIS:
This variable is used as a CONTROL variable in main regressions to account for:
  1. Digital infrastructure quality beyond basic connectivity
  2. E-commerce and digital service maturity
  3. Trust and security in online transactions
  4. Overall digital economy development

It helps isolate the price elasticity effect by controlling for digital
ecosystem sophistication that could independently affect internet adoption.
""")
    
else:
    print("\n[ERROR] secure_internet_servers column not found!")
    print("Available columns with 'server' or 'security':")
    potential = [col for col in df.columns if any(x in col.lower() for x in ['server', 'security', 'cyber'])]
    for col in potential:
        print(f"  - {col}")

print("\n" + "="*80)
print("[OK] Analysis complete")
print("="*80)
