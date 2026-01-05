"""
Analyze results from heterophilic dataset experiments.
Generates tables and plots comparing performance.
"""

import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path


def load_results(results_dir):
    """
    Load all results from experiment directories.

    Returns:
        DataFrame with columns: dataset, model, n_streams, seed, test_acc, val_acc
    """
    results = []

    for dataset_dir in Path(results_dir).iterdir():
        if not dataset_dir.is_dir():
            continue

        # Parse directory name
        parts = dataset_dir.name.split('_')
        dataset = parts[0]

        if 'baseline' in dataset_dir.name:
            model_type = 'baseline'
            n_streams = 1
        elif 'mhc' in dataset_dir.name:
            model_type = 'mhc'
            n_streams = int(parts[-1][1:])  # Extract from "nX"
        else:
            continue

        # Load results from each seed
        for results_file in dataset_dir.glob('seed_*/results.json'):
            try:
                with open(results_file, 'r') as f:
                    data = json.load(f)

                seed = int(results_file.parent.name.split('_')[1])

                results.append({
                    'dataset': dataset,
                    'model': model_type,
                    'n_streams': n_streams,
                    'seed': seed,
                    'test_acc': data.get('test_acc', 0.0),
                    'val_acc': data.get('val_acc', 0.0),
                    'best_epoch': data.get('best_epoch', 0),
                })
            except Exception as e:
                print(f"Warning: Could not load {results_file}: {e}")

    df = pd.DataFrame(results)
    return df


def compute_statistics(df):
    """
    Compute mean and std for each configuration.

    Returns:
        DataFrame with aggregated statistics
    """
    stats = df.groupby(['dataset', 'model', 'n_streams']).agg({
        'test_acc': ['mean', 'std', 'count'],
        'val_acc': ['mean', 'std'],
    }).reset_index()

    # Flatten column names
    stats.columns = ['_'.join(col).strip('_') for col in stats.columns.values]

    return stats


def generate_table(stats_df, output_file='heterophilic_results.txt'):
    """
    Generate publication-ready table.
    """
    datasets = sorted(stats_df['dataset'].unique())

    with open(output_file, 'w') as f:
        # Header
        f.write("=" * 100 + "\n")
        f.write("Heterophilic Dataset Results: Test Accuracy (Mean ± Std over 5 seeds)\n")
        f.write("=" * 100 + "\n\n")

        # Table header
        f.write(f"{'Dataset':<12} {'Baseline':<20} {'mHC (n=2)':<20} {'mHC (n=4)':<20} {'mHC (n=8)':<20}\n")
        f.write("-" * 100 + "\n")

        for dataset in datasets:
            dataset_stats = stats_df[stats_df['dataset'] == dataset]

            row = [dataset]

            # Baseline
            baseline = dataset_stats[dataset_stats['model'] == 'baseline']
            if len(baseline) > 0:
                mean = baseline.iloc[0]['test_acc_mean'] * 100
                std = baseline.iloc[0]['test_acc_std'] * 100
                row.append(f"{mean:.2f} ± {std:.2f}")
            else:
                row.append("N/A")

            # mHC with n=2, 4, 8
            for n in [2, 4, 8]:
                mhc = dataset_stats[(dataset_stats['model'] == 'mhc') & (dataset_stats['n_streams'] == n)]
                if len(mhc) > 0:
                    mean = mhc.iloc[0]['test_acc_mean'] * 100
                    std = mhc.iloc[0]['test_acc_std'] * 100
                    row.append(f"{mean:.2f} ± {std:.2f}")
                else:
                    row.append("N/A")

            f.write(f"{row[0]:<12} {row[1]:<20} {row[2]:<20} {row[3]:<20} {row[4]:<20}\n")

        f.write("=" * 100 + "\n")

    print(f"Table saved to {output_file}")


def plot_results(df, stats_df, output_dir='plots'):
    """
    Generate plots comparing performance.
    """
    os.makedirs(output_dir, exist_ok=True)

    # Set style
    sns.set_style("whitegrid")
    sns.set_palette("Set2")

    # Plot 1: Bar chart of test accuracy per dataset
    fig, ax = plt.subplots(figsize=(12, 6))

    datasets = sorted(stats_df['dataset'].unique())
    x = np.arange(len(datasets))
    width = 0.2

    baseline_means = []
    mhc2_means = []
    mhc4_means = []
    mhc8_means = []

    baseline_stds = []
    mhc2_stds = []
    mhc4_stds = []
    mhc8_stds = []

    for dataset in datasets:
        dataset_stats = stats_df[stats_df['dataset'] == dataset]

        # Baseline
        baseline = dataset_stats[dataset_stats['model'] == 'baseline']
        if len(baseline) > 0:
            baseline_means.append(baseline.iloc[0]['test_acc_mean'] * 100)
            baseline_stds.append(baseline.iloc[0]['test_acc_std'] * 100)
        else:
            baseline_means.append(0)
            baseline_stds.append(0)

        # mHC variants
        for n, means_list, stds_list in [(2, mhc2_means, mhc2_stds),
                                          (4, mhc4_means, mhc4_stds),
                                          (8, mhc8_means, mhc8_stds)]:
            mhc = dataset_stats[(dataset_stats['model'] == 'mhc') & (dataset_stats['n_streams'] == n)]
            if len(mhc) > 0:
                means_list.append(mhc.iloc[0]['test_acc_mean'] * 100)
                stds_list.append(mhc.iloc[0]['test_acc_std'] * 100)
            else:
                means_list.append(0)
                stds_list.append(0)

    ax.bar(x - 1.5*width, baseline_means, width, label='Baseline', yerr=baseline_stds, capsize=3)
    ax.bar(x - 0.5*width, mhc2_means, width, label='mHC (n=2)', yerr=mhc2_stds, capsize=3)
    ax.bar(x + 0.5*width, mhc4_means, width, label='mHC (n=4)', yerr=mhc4_stds, capsize=3)
    ax.bar(x + 1.5*width, mhc8_means, width, label='mHC (n=8)', yerr=mhc8_stds, capsize=3)

    ax.set_xlabel('Dataset', fontsize=12)
    ax.set_ylabel('Test Accuracy (%)', fontsize=12)
    ax.set_title('Performance on Heterophilic Datasets', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(datasets, rotation=45, ha='right')
    ax.legend(fontsize=10)
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'heterophilic_comparison.png'), dpi=300)
    plt.savefig(os.path.join(output_dir, 'heterophilic_comparison.pdf'))
    print(f"Plot saved to {output_dir}/heterophilic_comparison.png")
    plt.close()

    # Plot 2: Improvement over baseline
    fig, ax = plt.subplots(figsize=(10, 6))

    improvements = []
    for dataset in datasets:
        dataset_stats = stats_df[stats_df['dataset'] == dataset]
        baseline = dataset_stats[dataset_stats['model'] == 'baseline']

        if len(baseline) > 0:
            baseline_acc = baseline.iloc[0]['test_acc_mean']

            for n in [2, 4, 8]:
                mhc = dataset_stats[(dataset_stats['model'] == 'mhc') & (dataset_stats['n_streams'] == n)]
                if len(mhc) > 0:
                    mhc_acc = mhc.iloc[0]['test_acc_mean']
                    improvement = (mhc_acc - baseline_acc) * 100

                    improvements.append({
                        'dataset': dataset,
                        'n_streams': n,
                        'improvement': improvement,
                    })

    imp_df = pd.DataFrame(improvements)

    sns.barplot(data=imp_df, x='dataset', y='improvement', hue='n_streams', ax=ax)
    ax.axhline(y=0, color='black', linestyle='--', linewidth=1)
    ax.set_xlabel('Dataset', fontsize=12)
    ax.set_ylabel('Improvement over Baseline (%)', fontsize=12)
    ax.set_title('mHC-GNN Improvement on Heterophilic Datasets', fontsize=14, fontweight='bold')
    ax.legend(title='Stream Count', fontsize=10)
    ax.grid(axis='y', alpha=0.3)

    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'heterophilic_improvements.png'), dpi=300)
    plt.savefig(os.path.join(output_dir, 'heterophilic_improvements.pdf'))
    print(f"Plot saved to {output_dir}/heterophilic_improvements.png")
    plt.close()


def main():
    """Main analysis function."""
    results_dir = '../../results/heterophilic'

    if not os.path.exists(results_dir):
        print(f"Error: Results directory not found: {results_dir}")
        print("Please run experiments first with: bash run_heterophilic_experiments.sh")
        return

    print("Loading results...")
    df = load_results(results_dir)

    if len(df) == 0:
        print("No results found!")
        return

    print(f"Loaded {len(df)} result entries")
    print(f"Datasets: {sorted(df['dataset'].unique())}")
    print(f"Models: {sorted(df['model'].unique())}")

    print("\nComputing statistics...")
    stats_df = compute_statistics(df)

    print("\nGenerating table...")
    generate_table(stats_df, output_file='../../results/heterophilic_results.txt')

    print("\nGenerating plots...")
    plot_results(df, stats_df, output_dir='../../results/plots')

    print("\n✅ Analysis complete!")
    print("\nResults:")
    print("  - Table: results/heterophilic_results.txt")
    print("  - Plots: results/plots/heterophilic_*.png")


if __name__ == '__main__':
    main()
