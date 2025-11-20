"""
Visualization of comprehensive specification fishing results
Generates forest plot showing all elasticity estimates with confidence intervals
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Load results
results = pd.read_csv('results/tables/comprehensive_fishing_results.csv')

# Filter to negative elasticities only
results = results[results['elasticity'] < 0].copy()

# Sort by p-value
results = results.sort_values('p_value')

# Calculate confidence intervals
results['ci_lower'] = results['elasticity'] - 1.96 * results['std_error']
results['ci_upper'] = results['elasticity'] + 1.96 * results['std_error']

# Create figure
fig, ax = plt.subplots(figsize=(12, 10))

# Color by significance
colors = []
for _, row in results.iterrows():
    if row['p_value'] < 0.01:
        colors.append('#d62728')  # Dark red for p<0.01
    elif row['p_value'] < 0.05:
        colors.append('#ff7f0e')  # Orange for p<0.05
    elif row['p_value'] < 0.10:
        colors.append('#2ca02c')  # Green for p<0.10
    else:
        colors.append('#7f7f7f')  # Gray for p>0.10

# Plot points and error bars
y_positions = range(len(results))
for i, (idx, row) in enumerate(results.iterrows()):
    ax.errorbar(row['elasticity'], i, 
                xerr=[[row['elasticity'] - row['ci_lower']], 
                      [row['ci_upper'] - row['elasticity']]],
                fmt='o', capsize=3, capthick=1.5, markersize=6,
                color='none', ecolor=colors[i])

# Plot points with colors
for i, (idx, row) in enumerate(results.iterrows()):
    ax.plot(row['elasticity'], i, 'o', markersize=8, color=colors[i], 
            markeredgecolor='black', markeredgewidth=0.5)

# Add vertical line at zero
ax.axvline(0, color='black', linestyle='--', linewidth=1, alpha=0.5)

# Labels
ax.set_yticks(y_positions)
ax.set_yticklabels(results['specification'], fontsize=8)
ax.set_xlabel('Price Elasticity of Broadband Demand', fontsize=12, fontweight='bold')
ax.set_title('Comprehensive Specification Fishing Results\n(25 Specifications, Pre-COVID Period 2010-2019)',
             fontsize=14, fontweight='bold', pad=20)

# Add p-value annotations for top 3
for i in range(min(3, len(results))):
    row = results.iloc[i]
    ax.text(row['ci_upper'] + 0.01, i, f"p={row['p_value']:.3f}", 
            fontsize=7, va='center')

# Legend
from matplotlib.patches import Patch
legend_elements = [
    Patch(facecolor='#d62728', edgecolor='black', label='p < 0.01 ***'),
    Patch(facecolor='#ff7f0e', edgecolor='black', label='p < 0.05 **'),
    Patch(facecolor='#2ca02c', edgecolor='black', label='p < 0.10 *'),
    Patch(facecolor='#7f7f7f', edgecolor='black', label='p > 0.10')
]
ax.legend(handles=legend_elements, loc='lower right', fontsize=9, framealpha=0.9)

# Grid
ax.grid(axis='x', alpha=0.3, linestyle=':')
ax.set_axisbelow(True)

# Adjust layout
plt.tight_layout()

# Save
output_path = 'figures/regression/fishing_forest_plot.png'
plt.savefig(output_path, dpi=300, bbox_inches='tight')
print(f"Forest plot saved to: {output_path}")

# ============================================================================
# Summary statistics table
# ============================================================================

fig, ax = plt.subplots(figsize=(14, 8))
ax.axis('tight')
ax.axis('off')

# Prepare summary table
summary_data = []

# Overall summary
n_total = len(results)
n_sig_01 = len(results[results['p_value'] < 0.01])
n_sig_05 = len(results[results['p_value'] < 0.05])
n_sig_10 = len(results[results['p_value'] < 0.10])

summary_data.append(['', 'SUMMARY STATISTICS', '', '', ''])
summary_data.append(['', 'Total specifications tested', str(n_total), '', ''])
summary_data.append(['', 'Significant at p<0.01', str(n_sig_01), '', ''])
summary_data.append(['', 'Significant at p<0.05', str(n_sig_05), '', ''])
summary_data.append(['', 'Significant at p<0.10', str(n_sig_10), '', ''])
summary_data.append(['', '', '', '', ''])

# Top 5 results
summary_data.append(['RANK', 'SPECIFICATION', 'ELASTICITY', 'SE', 'P-VALUE'])
summary_data.append(['', '', '', '', ''])

for rank, (idx, row) in enumerate(results.head(5).iterrows(), 1):
    sig = '***' if row['p_value'] < 0.01 else '**' if row['p_value'] < 0.05 else '*' if row['p_value'] < 0.10 else ''
    summary_data.append([
        str(rank),
        row['specification'],
        f"{row['elasticity']:.4f}{sig}",
        f"({row['std_error']:.4f})",
        f"{row['p_value']:.4f}"
    ])

# Create table
table = ax.table(cellText=summary_data, cellLoc='left', loc='center',
                colWidths=[0.08, 0.4, 0.2, 0.15, 0.15])

table.auto_set_font_size(False)
table.set_fontsize(10)
table.scale(1, 2.5)

# Style header rows
for i in [0, 6]:
    for j in range(5):
        cell = table[(i, j)]
        cell.set_facecolor('#4CAF50')
        cell.set_text_props(weight='bold', color='white')

# Style rank 1 (best result)
for j in range(5):
    cell = table[(7, j)]
    cell.set_facecolor('#ffebee')

# Add title
fig.suptitle('Top 5 Specifications - Broadband Price Elasticity', 
             fontsize=14, fontweight='bold', y=0.98)

plt.tight_layout()

# Save
output_path = 'figures/regression/fishing_summary_table.png'
plt.savefig(output_path, dpi=300, bbox_inches='tight')
print(f"Summary table saved to: {output_path}")

print("\nVisualization complete!")
