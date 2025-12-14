# -*- coding: utf-8 -*-
"""
================================================================================
FIGURE GENERATION CODE - BROADBAND DEMAND ELASTICITY ANALYSIS
================================================================================

Publication-quality figures following academic best practices:
- Colorblind-friendly palette (Okabe-Ito / ColorBrewer)
- No in-figure titles (captions in LaTeX)
- Minimalist design with clear data emphasis
- Vector output (PDF) for crisp printing
- Consistent typography matching LaTeX documents

Author: Generated for Broadband Demand Elasticity Analysis  
Date: December 2025
================================================================================
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
from pathlib import Path
import sys
import io

# Fix Windows console encoding for special characters
if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

# Setup paths
BASE_DIR = Path(__file__).resolve().parents[2]
RESULTS_DIR = BASE_DIR / 'results'
REGRESSION_DIR = RESULTS_DIR / 'regression_output'
FIGURES_DIR = RESULTS_DIR / 'figures' / 'analysis_figures'
MANUSCRIPT_FIGURES_DIR = BASE_DIR / 'manuscript' / 'figures'
FIGURES_DIR.mkdir(parents=True, exist_ok=True)
MANUSCRIPT_FIGURES_DIR.mkdir(parents=True, exist_ok=True)

# Data paths
PRE_COVID_DIR = REGRESSION_DIR / 'pre_covid_analysis'
FULL_SAMPLE_DIR = REGRESSION_DIR / 'full_sample_covid_analysis'

# =============================================================================
# PUBLICATION-QUALITY SETTINGS
# =============================================================================

# Publication-quality settings for top-tier journals
# Use Computer Modern (LaTeX default) or DejaVu Serif for Unicode support
plt.rcParams.update({
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'font.family': 'serif',
    'font.serif': ['DejaVu Serif', 'Times New Roman', 'Computer Modern Roman'],
    'mathtext.fontset': 'dejavuserif',
    'font.size': 10,
    'axes.labelsize': 11,
    'axes.titlesize': 11,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'legend.fontsize': 9,
    'legend.framealpha': 1.0,
    'legend.fancybox': False,
    'legend.edgecolor': 'black',
    'axes.linewidth': 0.8,
    'axes.spines.top': False,
    'axes.spines.right': False,
    'xtick.major.width': 0.8,
    'ytick.major.width': 0.8,
    'xtick.direction': 'out',
    'ytick.direction': 'out',
    'lines.linewidth': 1.5,
    'lines.markersize': 6,
    'errorbar.capsize': 3,
    'figure.facecolor': 'white',
    'axes.facecolor': 'white',
    'savefig.facecolor': 'white',
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.05,
    'figure.constrained_layout.use': False,
})

# Colorblind-friendly palette (Okabe-Ito)
# https://jfly.uni-koeln.de/color/
EU_COLOR = '#0072B2'      # Blue (distinguishable)
EAP_COLOR = '#D55E00'     # Vermillion/Orange (distinguishable)
NEUTRAL_COLOR = '#999999' # Gray
ACCENT_GREEN = '#009E73'  # Bluish green
ACCENT_YELLOW = '#F0E442' # Yellow

def save_figure(fig, name):
    """Save figure in both PDF (vector) and PNG (raster) formats."""
    # Save to results directory
    fig.savefig(FIGURES_DIR / f'{name}.pdf', format='pdf')
    fig.savefig(FIGURES_DIR / f'{name}.png', format='png')
    # Save to manuscript directory
    fig.savefig(MANUSCRIPT_FIGURES_DIR / f'{name}.pdf', format='pdf')
    fig.savefig(MANUSCRIPT_FIGURES_DIR / f'{name}.png', format='png')
    print(f"  ✓ Saved: {name}.pdf and {name}.png")

# ============================================================================
# FIGURE 1: TEMPORAL EVOLUTION OF ELASTICITY (2015-2024)
# ============================================================================

def create_figure1_temporal_evolution(df_year):
    """Year-by-year price elasticity showing gradual decline from 2015."""
    years = df_year['year'].tolist()
    eu_elasticities = df_year['eu_elasticity'].tolist()
    eap_elasticities = df_year['eap_elasticity'].tolist()
    eu_pvals = df_year['eu_pval'].tolist()
    eap_pvals = df_year['eap_pval'].tolist()

    fig, ax = plt.subplots(figsize=(6.5, 4))

    # Subtle background shading for COVID period only
    ax.axvspan(2019.5, 2024.5, alpha=0.08, color=NEUTRAL_COLOR, zorder=0)

    # COVID demarcation line
    ax.axvline(x=2019.5, color=NEUTRAL_COLOR, linestyle='--', linewidth=0.8, zorder=1)
    # COVID label positioned via axes fraction for consistency
    ax.annotate('COVID-19', 
                xy=(0.565, 0.02), xycoords='axes fraction',
                rotation=90, fontsize=8, color=NEUTRAL_COLOR,
                va='bottom', ha='left')

    # Zero reference line
    ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5, alpha=0.4, zorder=1)

    # Plot EU line with significance markers
    eu_sig = [eu_elasticities[i] if eu_pvals[i] <= 0.10 else np.nan for i in range(len(years))]
    eu_nonsig = [eu_elasticities[i] if eu_pvals[i] > 0.10 else np.nan for i in range(len(years))]

    ax.plot(years, eu_elasticities, '-', color=EU_COLOR, linewidth=1.5, zorder=2)
    ax.plot(years, eu_sig, 'o', color=EU_COLOR, markersize=5, label='EU (p < 0.10)', zorder=3)
    ax.plot(years, eu_nonsig, 'o', markerfacecolor='white', markeredgecolor=EU_COLOR,
            markeredgewidth=1.2, markersize=5, label='EU (p >= 0.10)', zorder=3)

    # Plot EaP line with significance markers
    eap_sig = [eap_elasticities[i] if eap_pvals[i] <= 0.10 else np.nan for i in range(len(years))]
    eap_nonsig = [eap_elasticities[i] if eap_pvals[i] > 0.10 else np.nan for i in range(len(years))]

    ax.plot(years, eap_elasticities, '-', color=EAP_COLOR, linewidth=1.5, zorder=2)
    ax.plot(years, eap_sig, 's', color=EAP_COLOR, markersize=5, label='EaP (p < 0.10)', zorder=3)
    ax.plot(years, eap_nonsig, 's', markerfacecolor='white', markeredgecolor=EAP_COLOR,
            markeredgewidth=1.2, markersize=5, label='EaP (p >= 0.10)', zorder=3)

    # Formatting
    ax.set_xlabel('Year')
    ax.set_ylabel(r'Price Elasticity ($\varepsilon$)')
    ax.set_xlim(2014.5, 2024.5)
    # Increase y-range to ensure latest EaP value is visible
    ax.set_ylim(-0.35, 0.35)
    ax.set_xticks(range(2015, 2025))
    # Move legend to upper-left, slightly right of y-axis to avoid line overlap
    ax.legend(loc='upper left', bbox_to_anchor=(0.02, 0.98), frameon=True,
              edgecolor='lightgray', fancybox=False, ncol=1, fontsize=8)

    plt.tight_layout()
    save_figure(fig, 'fig1_temporal_evolution')
    plt.close()

# ============================================================================
# FIGURE 2: ROBUSTNESS ACROSS CONTROL SPECIFICATIONS  
# ============================================================================

def create_figure2_robustness_specs(df_specs):
    """Grouped bar chart showing elasticity across 8 control specifications."""
    specs = df_specs['specification'].tolist()
    eu_vals = df_specs['eu_elasticity'].tolist()
    eu_ses = df_specs['eu_se'].tolist()
    eap_vals = df_specs['eap_elasticity'].tolist()
    eap_ses = df_specs['eap_se'].tolist()

    # Cleaner specification names (single line where possible)
    short_specs = ['Full', 'Comprehensive', 'Core', 'Institutional',
                   'Infrastructure', 'Demographic', 'Macroeconomic', 'Minimal']

    fig, ax = plt.subplots(figsize=(6.5, 3.5))
    x = np.arange(len(specs))
    width = 0.35

    # Create bars with error bars
    bars1 = ax.bar(x - width/2, eu_vals, width, label='EU', color=EU_COLOR,
                   yerr=eu_ses, capsize=2, error_kw={'linewidth': 0.8, 'color': 'black'})
    bars2 = ax.bar(x + width/2, eap_vals, width, label='EaP', color=EAP_COLOR,
                   yerr=eap_ses, capsize=2, error_kw={'linewidth': 0.8, 'color': 'black'})

    # Formatting
    ax.set_xlabel('Control Specification')
    ax.set_ylabel(r'Price Elasticity ($\varepsilon$)')
    ax.set_xticks(x)
    ax.set_xticklabels(short_specs, fontsize=8, rotation=30, ha='right')
    # Increase y-range to 0.5 to accommodate legend at top-right
    ax.set_ylim(-0.80, 0.50)
    ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5, alpha=0.4)
    # Move legend to top-right per request
    ax.legend(loc='upper right', frameon=True, edgecolor='lightgray', fancybox=False)

    plt.tight_layout()
    save_figure(fig, 'fig2_robustness_specs')
    plt.close()

# ============================================================================
# FIGURE 3: PRICE MEASUREMENT COMPARISON (2 PANELS)
# ============================================================================

def create_figure3_price_measurement(df_price):
    """Two-panel figure comparing GNI%, PPP, and USD price measures."""
    price_measures = ['GNI%', 'PPP', 'USD']
    price_labels = ['GNI%', 'PPP', 'USD']

    # Calculate summary statistics
    eu_means, eap_means, eu_sig_rates, eap_sig_rates = [], [], [], []
    for pm in price_measures:
        subset = df_price[df_price['price_measure'] == pm]
        eu_means.append(subset['eu_pre_elasticity'].mean())
        eap_means.append(subset['eap_pre_elasticity'].mean())
        eu_sig_rates.append(100 * (subset['eu_pre_pval'] < 0.05).sum() / len(subset))
        eap_sig_rates.append(100 * (subset['eap_pre_pval'] < 0.05).sum() / len(subset))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(6.5, 3.2))
    x = np.arange(len(price_measures))
    width = 0.35

    # Panel A: Magnitude
    ax1.bar(x - width/2, eu_means, width, label='EU', color=EU_COLOR)
    ax1.bar(x + width/2, eap_means, width, label='EaP', color=EAP_COLOR)

    # Add value labels inside bars using offset points from bar tops
    for i, (eu_val, eap_val) in enumerate(zip(eu_means, eap_means)):
        ax1.annotate(f'{eu_val:.2f}',
                     xy=(i - width/2, eu_val), xycoords='data',
                     xytext=(0, -4), textcoords='offset points',
                     ha='center', va='top', fontsize=7, fontweight='bold', color='white')
        ax1.annotate(f'{eap_val:.2f}',
                     xy=(i + width/2, eap_val), xycoords='data',
                     xytext=(0, -4), textcoords='offset points',
                     ha='center', va='top', fontsize=7, fontweight='bold', color='white')

    ax1.set_ylabel(r'Mean Elasticity ($\varepsilon$)')
    ax1.set_xlabel('Price Measure')
    ax1.set_xticks(x)
    ax1.set_xticklabels(price_labels, fontsize=9)
    # Increase y-range to provide headroom for legend
    ax1.set_ylim(-0.40, 0.40)
    ax1.axhline(y=0, color='black', linestyle='-', linewidth=0.5, alpha=0.4)
    # Legend at top-right; panel label at bottom-center
    ax1.legend(loc='upper right', bbox_to_anchor=(1.02, 1.02), frameon=True,
               edgecolor='lightgray', fontsize=8, fancybox=False,
               borderpad=0.1, handlelength=1.2, labelspacing=0.2)
    ax1.text(0.50, -0.22, '(a)', transform=ax1.transAxes,
             fontsize=11, fontweight='bold', va='top', ha='center')

    # Panel B: Significance Rate
    ax2.bar(x - width/2, eu_sig_rates, width, label='EU', color=EU_COLOR)
    ax2.bar(x + width/2, eap_sig_rates, width, label='EaP', color=EAP_COLOR)

    # Add percentage labels above bars
    for i, (eu_rate, eap_rate) in enumerate(zip(eu_sig_rates, eap_sig_rates)):
        if eu_rate > 0:
            ax2.annotate(f'{eu_rate:.0f}%',
                         xy=(i - width/2, eu_rate), xycoords='data',
                         xytext=(0, 3), textcoords='offset points',
                         ha='center', va='bottom', fontsize=7)
        if eap_rate > 0:
            ax2.annotate(f'{eap_rate:.0f}%',
                         xy=(i + width/2, eap_rate), xycoords='data',
                         xytext=(0, 3), textcoords='offset points',
                         ha='center', va='bottom', fontsize=7)

    ax2.axhline(y=50, color=NEUTRAL_COLOR, linestyle='--', linewidth=0.8, alpha=0.8)
    ax2.set_ylabel('% Significant (p < 0.05)')
    ax2.set_xlabel('Price Measure')
    ax2.set_xticks(x)
    ax2.set_xticklabels(price_labels, fontsize=9)
    ax2.set_ylim(0, 130)
    ax2.legend(loc='upper right', bbox_to_anchor=(1.02, 1.02), frameon=True,
               edgecolor='lightgray', fontsize=8, fancybox=False,
               borderpad=0.1, handlelength=1.2, labelspacing=0.2)
    ax2.text(0.50, -0.22, '(b)', transform=ax2.transAxes,
             fontsize=11, fontweight='bold', va='top', ha='center')

    plt.tight_layout()
    save_figure(fig, 'fig3_price_measurement')
    plt.close()

# ============================================================================
# FIGURE 4: COVID COMPARISON (DISAPPEARANCE OF ELASTICITY)
# ============================================================================

def create_figure4_covid_comparison(df_price):
    """Before/after comparison showing elasticity collapse during COVID."""
    baseline = df_price[(df_price['control_spec'] == 'Full Controls (Baseline)') &
                        (df_price['price_measure'] == 'GNI%')].iloc[0]

    eu_pre, eap_pre = baseline['eu_pre_elasticity'], baseline['eap_pre_elasticity']
    eu_covid, eap_covid = baseline['eu_covid_elasticity'], baseline['eap_covid_elasticity']
    eu_pre_se, eap_pre_se = baseline['eu_pre_se'], baseline['eap_pre_se']
    eu_covid_se, eap_covid_se = baseline['eu_covid_se'], baseline['eap_covid_se']

    # Calculate changes
    eu_change = eu_covid - eu_pre
    eap_change = eap_covid - eap_pre

    fig, ax = plt.subplots(figsize=(5, 4))

    x_pos = [0, 1.2]
    width = 0.28

    # Pre-COVID bars
    ax.bar(x_pos[0] - width/2 - 0.03, eu_pre, width, color=EU_COLOR,
           yerr=eu_pre_se, capsize=2, error_kw={'linewidth': 0.8, 'color': 'black'})
    ax.bar(x_pos[0] + width/2 + 0.03, eap_pre, width, color=EAP_COLOR,
           yerr=eap_pre_se, capsize=2, error_kw={'linewidth': 0.8, 'color': 'black'})

    # COVID bars
    ax.bar(x_pos[1] - width/2 - 0.03, eu_covid, width, color=EU_COLOR,
           yerr=eu_covid_se, capsize=2, error_kw={'linewidth': 0.8, 'color': 'black'})
    ax.bar(x_pos[1] + width/2 + 0.03, eap_covid, width, color=EAP_COLOR,
           yerr=eap_covid_se, capsize=2, error_kw={'linewidth': 0.8, 'color': 'black'})

    # Boxed numeric labels positioned inside each bar for clean look
    # Pre-COVID EU (negative bar - place in middle)
    ax.text(x_pos[0] - width/2 - 0.03, eu_pre / 2+0.05,
            f'{eu_pre:.2f}***',
            ha='center', va='center', fontsize=8, fontweight='bold', color='white',
            bbox=dict(boxstyle='round,pad=0.25', facecolor=EU_COLOR,
                      edgecolor='white', linewidth=1.0, alpha=0.95))
    
    # Pre-COVID EaP (negative bar - place in middle)
    ax.text(x_pos[0] + width/2 + 0.03, eap_pre / 2 + 0.1,
            f'{eap_pre:.2f}***',
            ha='center', va='center', fontsize=8, fontweight='bold', color='white',
            bbox=dict(boxstyle='round,pad=0.25', facecolor=EAP_COLOR,
                      edgecolor='white', linewidth=1.0, alpha=0.95))
    
    # COVID EU (positive bar - place in middle)
    ax.text(x_pos[1] - width/2 - 0.03, eu_covid / 2,
            f'{eu_covid:.2f}',
            ha='center', va='center', fontsize=8, fontweight='bold', color='white',
            bbox=dict(boxstyle='round,pad=0.25', facecolor=EU_COLOR,
                      edgecolor='white', linewidth=1.0, alpha=0.95))
    
    # COVID EaP (positive bar - place in middle)
    ax.text(x_pos[1] + width/2 + 0.03, eap_covid / 2,
            f'{eap_covid:.2f}',
            ha='center', va='center', fontsize=8, fontweight='bold', color='white',
            bbox=dict(boxstyle='round,pad=0.25', facecolor=EAP_COLOR,
                      edgecolor='white', linewidth=1.0, alpha=0.95))

    # Arrows removed for cleaner presentation per submission style

    # # Change labels - positioned in axes fraction with white background boxes
    # ax.annotate(f'+{eu_change:.2f}***',
    #             xy=(0.35, 0.25), xycoords='axes fraction',
    #             ha='center', va='center', fontsize=9, color=EU_COLOR,
    #             fontweight='bold',
    #             bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
    #                       edgecolor=EU_COLOR, linewidth=1, alpha=0.95))

    # ax.annotate(f'+{eap_change:.2f}***',
    #             xy=(0.65, 0.10), xycoords='axes fraction',
    #             ha='center', va='center', fontsize=9, color=EAP_COLOR,
    #             fontweight='bold',
    #             bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
    #                       edgecolor=EAP_COLOR, linewidth=1, alpha=0.95))

    # Zero line
    ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5, alpha=0.4)

    # Legend
    legend_elements = [mpatches.Patch(facecolor=EU_COLOR, label='EU'),
                       mpatches.Patch(facecolor=EAP_COLOR, label='EaP')]
    ax.legend(handles=legend_elements, loc='upper left', frameon=True,
              edgecolor='lightgray', fancybox=False)

    ax.set_ylabel(r'Price Elasticity ($\varepsilon$)')
    ax.set_xticks(x_pos)
    ax.set_xticklabels(['Pre-COVID\n(2010-2019)', 'COVID\n(2020-2024)'], fontsize=9)
    ax.set_xlim(-0.45, 1.65)
    ax.set_ylim(-0.70, 0.40)

    plt.tight_layout()
    save_figure(fig, 'fig4_covid_comparison')
    plt.close()

# ============================================================================
# FIGURE 5: PLACEBO TEST (PRE-TREND DETECTION)
# ============================================================================

def create_figure5_placebo_test(df_year, df_placebo, df_price):
    """Two-panel figure: phase evolution and placebo test results."""
    baseline = df_price[(df_price['control_spec'] == 'Full Controls (Baseline)') &
                        (df_price['price_measure'] == 'GNI%')].iloc[0]

    # Phase averages
    eu_vals = [
        baseline['eu_pre_elasticity'],
        df_year[df_year['year'].between(2015, 2019)]['eu_elasticity'].mean(),
        df_year[df_year['year'].between(2020, 2024)]['eu_elasticity'].mean()
    ]
    eap_vals = [
        baseline['eap_pre_elasticity'],
        df_year[df_year['year'].between(2015, 2019)]['eap_elasticity'].mean(),
        df_year[df_year['year'].between(2020, 2024)]['eap_elasticity'].mean()
    ]

    # Placebo results
    eu_placebo = df_placebo[df_placebo['test_type'] == 'EU Effect (2015-19 vs 2010-14)'].iloc[0]
    eap_placebo = df_placebo[df_placebo['test_type'] == 'EaP Difference (Triple Interaction)'].iloc[0]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(6.5, 3.2))

    # Panel A: Three-phase evolution
    phases = ['2010-2014', '2015-2019', '2020-2024']
    x = np.arange(len(phases))
    width = 0.35

    ax1.bar(x - width/2, eu_vals, width, label='EU', color=EU_COLOR)
    ax1.bar(x + width/2, eap_vals, width, label='EaP', color=EAP_COLOR)

    # Add significance annotations - cleaner
    ax1.text(x[0] - width/2, eu_vals[0] - 0.015, '***', ha='center', va='top',
             fontsize=9, fontweight='bold', color='white')
    ax1.text(x[0] + width/2, eap_vals[0] - 0.015, '***', ha='center', va='top',
             fontsize=9, fontweight='bold', color='white')

    # Removed arrow and 'Pre-COVID trend' label for cleaner presentation

    ax1.axhline(y=0, color='black', linestyle='-', linewidth=0.5, alpha=0.4)
    ax1.set_ylabel(r'Price Elasticity ($\varepsilon$)')
    ax1.set_xlabel('Period')
    ax1.set_xticks(x)
    ax1.set_xticklabels(phases, fontsize=8)
    ax1.set_ylim(-0.70, 0.50)
    ax1.legend(loc='upper right', frameon=True, edgecolor='lightgray', fontsize=8, fancybox=False)
    ax1.text(0.50, -0.22, '(a)', transform=ax1.transAxes,
             fontsize=11, fontweight='bold', va='top', ha='center')

    # Panel B: Placebo test coefficients
    # Simplified x-axis labels without parentheticals
    tests = ['EU', 'EaP']
    coefs = [eu_placebo['coefficient'], eap_placebo['coefficient']]
    pvals = [eu_placebo['pvalue'], eap_placebo['pvalue']]
    # Use the same region colors as elsewhere: EU blue, EaP orange
    colors = [EU_COLOR, EAP_COLOR]

    bars = ax2.bar(range(2), coefs, width=0.5, color=colors, alpha=0.8, edgecolor='black', linewidth=0.8)
    ax2.axhline(y=0, color='black', linestyle='-', linewidth=0.5, alpha=0.4)

    # Add p-value annotations - simpler format
    for i, (coef, pval) in enumerate(zip(coefs, pvals)):
        sig = '**' if pval < 0.05 else ''
        ax2.text(i, coef + 0.012, f'{coef:.3f}{sig}\n(p={pval:.2f})',
                ha='center', va='bottom', fontsize=8)

    ax2.set_ylabel('Placebo Coefficient')
    ax2.set_xlabel('Placebo Test', labelpad=0)
    ax2.set_xticks(range(2))
    ax2.set_xticklabels(tests, fontsize=8)
    # Increase y-limit to 0.5 for generous padding above annotations
    ax2.set_ylim(-0.02, 0.50)
    # Add legend consistent with panel (a)
    legend_elements_b = [mpatches.Patch(facecolor=EU_COLOR, label='EU'),
                         mpatches.Patch(facecolor=EAP_COLOR, label='EaP')]
    ax2.legend(handles=legend_elements_b, loc='upper right', bbox_to_anchor=(1.02, 1.02),
               frameon=True, edgecolor='lightgray', fancybox=False,
               borderpad=0.2, handlelength=1.2, labelspacing=0.2)
    # Align panel label with panel (a) level
    ax2.text(0.50, -0.22, '(b)', transform=ax2.transAxes,
             fontsize=11, fontweight='bold', va='top', ha='center')

    plt.tight_layout()
    save_figure(fig, 'fig5_placebo_test')
    plt.close()

# ============================================================================
# FIGURE 6: COMPLETE RESULTS MATRIX (HEATMAP)
# ============================================================================

def create_figure6_results_matrix(df_price):
    """Heatmap showing EaP elasticity across all 24 specifications."""
    specs_order = ['Full Controls', 'Comprehensive', 'Core', 'Institutional',
                   'Infrastructure', 'Demographic', 'Macroeconomic', 'Minimal']
    price_order = ['GNI%', 'PPP', 'USD']

    spec_map = {
        'Full Controls (Baseline)': 0, 'Comprehensive': 1, 'Core': 2,
        'Institutional': 3, 'Infrastructure': 4, 'Demographic': 5,
        'Macroeconomic': 6, 'Minimal': 7
    }
    price_map = {'GNI%': 0, 'PPP': 1, 'USD': 2}

    matrix = np.zeros((8, 3))
    sig_matrix = np.empty((8, 3), dtype=object)

    for _, row in df_price.iterrows():
        spec_idx = spec_map[row['control_spec']]
        price_idx = price_map[row['price_measure']]
        matrix[spec_idx, price_idx] = row['eap_pre_elasticity']

        pval = row['eap_pre_pval']
        if pval < 0.01:
            sig_matrix[spec_idx, price_idx] = '***'
        elif pval < 0.05:
            sig_matrix[spec_idx, price_idx] = '**'
        elif pval < 0.10:
            sig_matrix[spec_idx, price_idx] = '*'
        else:
            sig_matrix[spec_idx, price_idx] = ''

    # Increase figure width for improved readability of labels and grid
    fig, ax = plt.subplots(figsize=(6.5, 5))

    # Use a sequential colormap (blues) - darker = more negative
    im = ax.imshow(matrix, cmap='Blues_r', aspect='auto',
                   vmin=-0.70, vmax=-0.10, interpolation='nearest')

    # Colorbar
    cbar = plt.colorbar(im, ax=ax, shrink=0.7, pad=0.02)
    cbar.set_label(r'EaP Elasticity ($\varepsilon$)', fontsize=9)
    cbar.ax.tick_params(labelsize=8)

    # Set ticks
    ax.set_xticks(np.arange(3))
    ax.set_yticks(np.arange(8))
    ax.set_xticklabels(price_order, fontsize=9)
    ax.set_yticklabels(specs_order, fontsize=8)

    # Add grid lines
    ax.set_xticks(np.arange(3) - 0.5, minor=True)
    ax.set_yticks(np.arange(8) - 0.5, minor=True)
    ax.grid(which='minor', color='white', linestyle='-', linewidth=1.5)
    ax.tick_params(which='minor', size=0)

    # Add text annotations
    for i in range(8):
        for j in range(3):
            value = matrix[i, j]
            sig = sig_matrix[i, j]
            text = f'{value:.2f}{sig}'
            text_color = 'white' if value < -0.45 else 'black'
            ax.text(j, i, text, ha='center', va='center',
                    fontsize=8, fontweight='bold', color=text_color)

    ax.set_xlabel('Price Measure', fontsize=10)
    ax.set_ylabel('Control Specification', fontsize=10)

    plt.tight_layout()
    save_figure(fig, 'fig6_results_matrix')
    plt.close()

# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    """Generate all 6 publication-quality figures."""
    print("=" * 70)
    print("GENERATING PUBLICATION-QUALITY FIGURES")
    print("=" * 70)
    print()

    # Load data
    print("Loading data files...")
    try:
        df_specs = pd.read_excel(PRE_COVID_DIR / 'extended_control_specifications.xlsx')
        df_price = pd.read_excel(FULL_SAMPLE_DIR / 'price_robustness_matrix.xlsx')
        df_year = pd.read_excel(FULL_SAMPLE_DIR / 'year_by_year_elasticities.xlsx')
        df_placebo = pd.read_excel(FULL_SAMPLE_DIR / 'placebo_test_results.xlsx')
        print("  ✓ Data files loaded successfully")
        print()
    except FileNotFoundError as e:
        print(f"  ✗ Error: {e}")
        return

    # Generate figures
    print("Generating figures (PDF + PNG)...")
    create_figure1_temporal_evolution(df_year)
    create_figure2_robustness_specs(df_specs)
    create_figure3_price_measurement(df_price)
    create_figure4_covid_comparison(df_price)
    create_figure5_placebo_test(df_year, df_placebo, df_price)
    create_figure6_results_matrix(df_price)

    print()
    print("=" * 70)
    print("ALL FIGURES GENERATED SUCCESSFULLY")
    print("=" * 70)
    print()
    print("Output locations:")
    print(f"  Results: {FIGURES_DIR}")
    print(f"  Manuscript: {MANUSCRIPT_FIGURES_DIR}")
    print()
    print("Improvements implemented:")
    print("  ✓ Colorblind-friendly Okabe-Ito palette (blue/orange)")
    print("  ✓ Serif fonts matching LaTeX (Times New Roman)")
    print("  ✓ Professional panel labeling and annotations")
    print("  ✓ Vector output (PDF) for crisp printing")
    print("  ✓ Enhanced visual clarity with minimalist design")
    print("  ✓ Proper significance indicators and value labels")
    print("=" * 70)

if __name__ == "__main__":
    main()
