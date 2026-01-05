#!/usr/bin/env python3
"""
Create extended depth analysis visualization for paper.
Shows performance vs depth (2, 4, 8, 16, 32, 64, 128 layers) for baseline and mHC-GNN.
"""

import re
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from collections import defaultdict

# Set publication-quality style
plt.rcParams.update({
    'font.size': 16,
    'font.family': 'serif',
    'font.serif': ['Times New Roman'],
    'text.usetex': False,  # Set to True if you have LaTeX installed
    'figure.figsize': (12, 4),
    'axes.labelsize': 18,
    'axes.titlesize': 20,
    'legend.fontsize': 14,
    'xtick.labelsize': 16,
    'ytick.labelsize': 16,
    'lines.linewidth': 2,
    'lines.markersize': 8,
})

results_base = Path("results/deep_depth_analysis")
pattern = r"Test Acc at Best Val: ([\d.]+)"

# Collect results
data = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))

for depth_dir in sorted(results_base.glob("*layers")):
    depth = int(depth_dir.name.replace("layers", ""))

    for log_file in depth_dir.glob("*.log"):
        exp_name = log_file.stem
        parts = exp_name.split('_')
        if len(parts) < 4:
            continue

        dataset = parts[0]

        if 'baseline' in exp_name:
            config = 'baseline'
        elif 'mhc' in exp_name and 'n2' in exp_name:
            config = 'mhc_n2'
        elif 'mhc' in exp_name and 'n4' in exp_name:
            config = 'mhc_n4'
        else:
            continue

        try:
            with open(log_file) as f:
                log_content = f.read()
                match = re.search(pattern, log_content)
                if match:
                    test_acc = float(match.group(1)) * 100
                    data[dataset][depth][config].append(test_acc)
        except Exception as e:
            print(f"Error reading {log_file}: {e}")

# Prepare data for plotting
datasets = ['Cora', 'CiteSeer', 'PubMed']
depths = [2, 4, 8, 16, 32, 64, 128]
colors = {
    'baseline': '#E74C3C',  # Red
    'mhc_n2': '#3498DB',    # Blue
    'mhc_n4': '#2ECC71',    # Green
}
labels = {
    'baseline': 'Baseline GCN',
    'mhc_n2': 'mHC-GNN (n=2)',
    'mhc_n4': 'mHC-GNN (n=4)',
}

# Create figure with subplots
fig, axes = plt.subplots(1, 3, figsize=(16, 5))

for idx, dataset in enumerate(datasets):
    ax = axes[idx]

    for config in ['baseline', 'mhc_n2', 'mhc_n4']:
        means = []
        stds = []
        valid_depths = []

        for depth in depths:
            accs = data[dataset][depth][config]
            if accs:
                means.append(np.mean(accs))
                stds.append(np.std(accs))
                valid_depths.append(depth)

        if not means:
            continue

        means = np.array(means)
        stds = np.array(stds)
        valid_depths = np.array(valid_depths)

        # Plot line with error band
        ax.plot(valid_depths, means,
                color=colors[config],
                label=labels[config],
                marker='o',
                linewidth=2.5,
                markersize=8)

        # Add shaded error region
        ax.fill_between(valid_depths,
                        means - stds,
                        means + stds,
                        color=colors[config],
                        alpha=0.2)

    # Styling
    ax.set_xlabel('Number of Layers', fontsize=18)
    ax.set_ylabel('Test Accuracy (%)', fontsize=18)
    ax.set_title(dataset, fontsize=20)
    ax.set_xscale('log', base=2)  # Log scale for depth
    ax.set_xticks(depths)
    ax.set_xticklabels(['2', '4', '8', '16', '32', '64', '128'])
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.legend(loc='best', framealpha=0.95, fontsize=14)

    # Set y-axis limits for better visualization
    ax.set_ylim([10, 85])

plt.tight_layout()

# Save figure
output_file = "figures/deep_depth_analysis.pdf"
Path("figures").mkdir(exist_ok=True)
plt.savefig(output_file, dpi=300, bbox_inches='tight')
print(f"✅ Saved figure to: {output_file}")

# Also save PNG version
output_png = "figures/deep_depth_analysis.png"
plt.savefig(output_png, dpi=300, bbox_inches='tight')
print(f"✅ Saved PNG to: {output_png}")

plt.show()

# Create second figure: Improvement vs Depth
fig, ax = plt.subplots(1, 1, figsize=(10, 6))

for dataset in datasets:
    improvements_n2 = []
    improvements_n4 = []
    valid_depths_n2 = []
    valid_depths_n4 = []

    for depth in depths:
        baseline_accs = data[dataset][depth]['baseline']
        mhc_n2_accs = data[dataset][depth]['mhc_n2']
        mhc_n4_accs = data[dataset][depth]['mhc_n4']

        if baseline_accs and mhc_n2_accs:
            improvement = np.mean(mhc_n2_accs) - np.mean(baseline_accs)
            improvements_n2.append(improvement)
            valid_depths_n2.append(depth)

        if baseline_accs and mhc_n4_accs:
            improvement = np.mean(mhc_n4_accs) - np.mean(baseline_accs)
            improvements_n4.append(improvement)
            valid_depths_n4.append(depth)

    # Plot improvements
    if improvements_n2:
        ax.plot(valid_depths_n2, improvements_n2,
                marker='o',
                label=f'{dataset} (n=2)',
                linewidth=2.5,
                markersize=8)

# Add horizontal line at y=0
ax.axhline(y=0, color='black', linestyle='--', linewidth=1, alpha=0.5)

ax.set_xlabel('Number of Layers', fontsize=18)
ax.set_ylabel('mHC-GNN Improvement over Baseline (%)', fontsize=18)
ax.set_title('Performance Gain vs Network Depth', fontsize=20)
ax.set_xscale('log', base=2)
ax.set_xticks(depths)
ax.set_xticklabels(['2', '4', '8', '16', '32', '64', '128'])
ax.grid(True, alpha=0.3, linestyle='--')
ax.legend(loc='best', framealpha=0.95, fontsize=14)

plt.tight_layout()

# Save improvement figure
improvement_file = "figures/deep_depth_improvement.pdf"
plt.savefig(improvement_file, dpi=300, bbox_inches='tight')
print(f"✅ Saved improvement figure to: {improvement_file}")

improvement_png = "figures/deep_depth_improvement.png"
plt.savefig(improvement_png, dpi=300, bbox_inches='tight')
print(f"✅ Saved improvement PNG to: {improvement_png}")

plt.show()

# Create third figure: Semi-log plot showing over-smoothing
fig, axes = plt.subplots(1, 3, figsize=(16, 5))

for idx, dataset in enumerate(datasets):
    ax = axes[idx]

    for config in ['baseline', 'mhc_n2', 'mhc_n4']:
        means = []
        valid_depths = []

        for depth in depths:
            accs = data[dataset][depth][config]
            if accs:
                means.append(np.mean(accs))
                valid_depths.append(depth)

        if not means:
            continue

        means = np.array(means)
        valid_depths = np.array(valid_depths)

        # Plot line
        ax.plot(valid_depths, means,
                color=colors[config],
                label=labels[config],
                marker='o',
                linewidth=2.5,
                markersize=8)

    # Styling
    ax.set_xlabel('Number of Layers (log scale)', fontsize=18)
    ax.set_ylabel('Test Accuracy (%)', fontsize=18)
    ax.set_title(f'{dataset}: Over-smoothing Effect', fontsize=20)
    ax.set_xscale('log', base=2)
    ax.set_xticks(depths)
    ax.set_xticklabels(['2', '4', '8', '16', '32', '64', '128'])
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.legend(loc='best', framealpha=0.95, fontsize=14)

    # Highlight region where baseline starts to fail
    ax.axvspan(16, 128, alpha=0.1, color='red', label='Over-smoothing region')

plt.tight_layout()

# Save over-smoothing figure
oversmoothing_file = "figures/oversmoothing_analysis.pdf"
plt.savefig(oversmoothing_file, dpi=300, bbox_inches='tight')
print(f"✅ Saved over-smoothing figure to: {oversmoothing_file}")

oversmoothing_png = "figures/oversmoothing_analysis.png"
plt.savefig(oversmoothing_png, dpi=300, bbox_inches='tight')
print(f"✅ Saved over-smoothing PNG to: {oversmoothing_png}")

plt.show()

print("\n✅ All figures generated successfully!")
print("\nGenerated figures:")
print("  1. deep_depth_analysis.pdf - Main accuracy vs depth plot")
print("  2. deep_depth_improvement.pdf - mHC improvement over baseline")
print("  3. oversmoothing_analysis.pdf - Over-smoothing effect visualization")
print("\nNext steps:")
print("  1. Review figures in figures/ directory")
print("  2. Add to paper LaTeX with:")
print("     \\includegraphics[width=\\textwidth]{figures/deep_depth_analysis.pdf}")
print("  3. Update paper narrative with extended depth analysis results")
print("  4. Highlight severe over-smoothing in baseline at 32+, 64+, 128+ layers")
