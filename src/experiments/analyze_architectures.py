#!/usr/bin/env python3
"""
Analyze multi-architecture experiment results
"""

import os
import re
from pathlib import Path
from collections import defaultdict
import numpy as np

def parse_train_log(log_path):
    """Extract best validation accuracy and test accuracy"""
    try:
        with open(log_path, 'r') as f:
            content = f.read()

        best_val_match = re.search(r'Best Val Acc:\s+([\d.]+)', content)
        test_acc_match = re.search(r'Test Acc at Best Val:\s+([\d.]+)', content)

        if best_val_match and test_acc_match:
            return {
                'val_acc': float(best_val_match.group(1)),
                'test_acc': float(test_acc_match.group(1))
            }
    except Exception as e:
        print(f"Error parsing {log_path}: {e}")
    return None

def collect_architecture_results(results_dir):
    """Collect multi-architecture experiment results"""
    results = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))

    results_path = Path(results_dir)

    for category in ['heterophilic', 'homophilic']:
        category_path = results_path / category
        if not category_path.exists():
            continue

        for exp_dir in category_path.iterdir():
            if not exp_dir.is_dir():
                continue

            exp_name = exp_dir.name

            # Parse: dataset_gnntype_config_sSEED
            # Examples:
            #   chameleon_gcn_baseline_s42
            #   chameleon_sage_mhc_n2_s42
            #   Cora_gat_mhc_n4_s42

            if '_s' not in exp_name:
                continue

            # Find last _s occurrence (for seed)
            last_s_idx = exp_name.rfind('_s')
            base_name = exp_name[:last_s_idx]
            seed = int(exp_name[last_s_idx+2:])

            # Extract dataset, gnn_type, and config
            parts = base_name.split('_')

            # Dataset is first part
            dataset = parts[0]

            # GNN type is second part
            gnn_type = parts[1]

            # Config is remaining parts
            config = '_'.join(parts[2:])

            # Parse train.log
            log_path = exp_dir / 'train.log'
            if log_path.exists():
                result = parse_train_log(log_path)
                if result:
                    result['seed'] = seed
                    result['category'] = category
                    results[dataset][gnn_type][config].append(result)

    return results

def compute_statistics(result_list):
    """Compute mean and std from list of results"""
    if not result_list:
        return None

    test_accs = [r['test_acc'] for r in result_list]

    return {
        'test_mean': np.mean(test_accs),
        'test_std': np.std(test_accs),
        'num_seeds': len(result_list)
    }

def print_architecture_comparison(results, datasets):
    """Print architecture comparison tables"""
    print(f"\n{'='*80}")
    print("MULTI-ARCHITECTURE RESULTS")
    print(f"{'='*80}\n")

    gnn_types = ['gcn', 'sage', 'gat', 'gin']
    configs = ['baseline', 'mhc_n2', 'mhc_n4', 'mhc_n8']

    for dataset in datasets:
        if dataset not in results:
            continue

        print(f"\n## {dataset.upper()}")
        print("-" * 100)
        print(f"{'GNN Type':<12} {'Baseline':<20} {'mHC n=2':<20} {'mHC n=4':<20} {'mHC n=8':<20}")
        print("-" * 100)

        for gnn_type in gnn_types:
            if gnn_type not in results[dataset]:
                continue

            row = [gnn_type.upper()]

            for config in configs:
                if config in results[dataset][gnn_type]:
                    stats = compute_statistics(results[dataset][gnn_type][config])
                    if stats:
                        mean = stats['test_mean'] * 100
                        std = stats['test_std'] * 100
                        row.append(f"{mean:>6.2f} ± {std:<4.2f}")
                    else:
                        row.append("-")
                else:
                    row.append("-")

            print(f"{row[0]:<12} {row[1]:<20} {row[2]:<20} {row[3]:<20} {row[4]:<20}")

def print_improvement_summary(results, datasets):
    """Print improvement summary: mHC best vs baseline"""
    print(f"\n{'='*80}")
    print("IMPROVEMENT SUMMARY (Best mHC vs Baseline)")
    print(f"{'='*80}\n")

    gnn_types = ['gcn', 'sage', 'gat', 'gin']

    for dataset in datasets:
        if dataset not in results:
            continue

        print(f"\n## {dataset.upper()}")
        print("-" * 60)
        print(f"{'GNN Type':<12} {'Baseline':<15} {'Best mHC':<15} {'Δ (%)':<10}")
        print("-" * 60)

        for gnn_type in gnn_types:
            if gnn_type not in results[dataset]:
                continue

            # Get baseline
            baseline_stats = compute_statistics(results[dataset][gnn_type].get('baseline', []))
            if not baseline_stats:
                continue

            baseline_mean = baseline_stats['test_mean'] * 100

            # Find best mHC
            best_mhc_mean = 0
            best_config = None

            for config in ['mhc_n2', 'mhc_n4', 'mhc_n8']:
                if config in results[dataset][gnn_type]:
                    stats = compute_statistics(results[dataset][gnn_type][config])
                    if stats and stats['test_mean'] * 100 > best_mhc_mean:
                        best_mhc_mean = stats['test_mean'] * 100
                        best_config = config

            if best_config:
                improvement = best_mhc_mean - baseline_mean
                marker = "✓" if improvement > 0.5 else " "
                print(f"{gnn_type.upper():<12} {baseline_mean:>6.2f}%        {best_mhc_mean:>6.2f}% ({best_config})  {improvement:+6.2f}% {marker}")

def print_latex_table(results, datasets):
    """Print LaTeX table for paper"""
    print(f"\n{'='*80}")
    print("LATEX TABLE - ARCHITECTURE COMPARISON")
    print(f"{'='*80}\n")

    print(r"\begin{table}[t]")
    print(r"\centering")
    print(r"\caption{Multi-architecture results comparing baseline vs mHC-GNN across GCN, GraphSAGE, GAT, and GIN.}")
    print(r"\label{tab:multi_arch}")

    # Determine number of columns
    num_datasets = len(datasets)
    col_spec = "l" + "c" * num_datasets
    print(f"\\begin{{tabular}}{{{col_spec}}}")
    print(r"\toprule")

    # Header
    header = "Architecture & " + " & ".join(datasets) + r" \\"
    print(header)
    print(r"\midrule")

    gnn_types = ['gcn', 'sage', 'gat', 'gin']
    gnn_names = ['GCN', 'GraphSAGE', 'GAT', 'GIN']

    for gnn_type, gnn_name in zip(gnn_types, gnn_names):
        # Baseline row
        baseline_row = [f"{gnn_name} (baseline)"]
        for dataset in datasets:
            if dataset in results and gnn_type in results[dataset]:
                stats = compute_statistics(results[dataset][gnn_type].get('baseline', []))
                if stats:
                    mean = stats['test_mean'] * 100
                    std = stats['test_std'] * 100
                    baseline_row.append(f"{mean:.2f} $\\pm$ {std:.2f}")
                else:
                    baseline_row.append("-")
            else:
                baseline_row.append("-")
        print(" & ".join(baseline_row) + r" \\")

        # mHC-GNN row (best of n=2,4,8)
        mhc_row = [f"{gnn_name} + mHC"]
        for dataset in datasets:
            if dataset in results and gnn_type in results[dataset]:
                best_mean = 0
                best_std = 0
                for config in ['mhc_n2', 'mhc_n4', 'mhc_n8']:
                    stats = compute_statistics(results[dataset][gnn_type].get(config, []))
                    if stats and stats['test_mean'] * 100 > best_mean:
                        best_mean = stats['test_mean'] * 100
                        best_std = stats['test_std'] * 100
                if best_mean > 0:
                    mhc_row.append(f"{best_mean:.2f} $\\pm$ {best_std:.2f}")
                else:
                    mhc_row.append("-")
            else:
                mhc_row.append("-")
        print(" & ".join(mhc_row) + r" \\")

        if gnn_type != 'gin':  # Don't add midrule after last architecture
            print(r"\midrule")

    print(r"\bottomrule")
    print(r"\end{tabular}")
    print(r"\end{table}")

def print_average_improvements(results):
    """Print average improvements across all architectures"""
    print(f"\n{'='*80}")
    print("AVERAGE IMPROVEMENTS ACROSS ARCHITECTURES")
    print(f"{'='*80}\n")

    gnn_types = ['gcn', 'sage', 'gat', 'gin']

    heterophilic_improvements = []
    homophilic_improvements = []

    for dataset in results:
        for gnn_type in gnn_types:
            if gnn_type not in results[dataset]:
                continue

            baseline_stats = compute_statistics(results[dataset][gnn_type].get('baseline', []))
            if not baseline_stats:
                continue

            baseline_mean = baseline_stats['test_mean'] * 100

            # Find best mHC
            best_mhc_mean = 0
            for config in ['mhc_n2', 'mhc_n4', 'mhc_n8']:
                stats = compute_statistics(results[dataset][gnn_type].get(config, []))
                if stats and stats['test_mean'] * 100 > best_mhc_mean:
                    best_mhc_mean = stats['test_mean'] * 100

            if best_mhc_mean > 0:
                improvement = best_mhc_mean - baseline_mean

                # Categorize by dataset type
                if dataset.lower() in ['chameleon', 'squirrel', 'actor']:
                    heterophilic_improvements.append(improvement)
                else:  # Cora, CiteSeer, PubMed
                    homophilic_improvements.append(improvement)

    if heterophilic_improvements:
        print(f"Heterophilic datasets:")
        print(f"  Average improvement: {np.mean(heterophilic_improvements):+.2f}%")
        print(f"  Std: {np.std(heterophilic_improvements):.2f}%")
        print(f"  Max: {np.max(heterophilic_improvements):+.2f}%")
        print(f"  Min: {np.min(heterophilic_improvements):+.2f}%")

    if homophilic_improvements:
        print(f"\nHomophilic datasets:")
        print(f"  Average improvement: {np.mean(homophilic_improvements):+.2f}%")
        print(f"  Std: {np.std(homophilic_improvements):.2f}%")
        print(f"  Max: {np.max(homophilic_improvements):+.2f}%")
        print(f"  Min: {np.min(homophilic_improvements):+.2f}%")

    all_improvements = heterophilic_improvements + homophilic_improvements
    if all_improvements:
        print(f"\nOverall (all architectures, all datasets):")
        print(f"  Average improvement: {np.mean(all_improvements):+.2f}%")
        print(f"  Std: {np.std(all_improvements):.2f}%")

def main():
    results_dir = '../../results_architectures'

    print("Collecting multi-architecture results...")
    results = collect_architecture_results(results_dir)

    # Dataset order
    heterophilic = ['chameleon', 'squirrel', 'actor']
    homophilic = ['Cora', 'CiteSeer', 'PubMed']

    print(f"\n{'='*80}")
    print("HETEROPHILIC DATASETS")
    print(f"{'='*80}")
    print_architecture_comparison(results, heterophilic)
    print_improvement_summary(results, heterophilic)

    print(f"\n{'='*80}")
    print("HOMOPHILIC DATASETS")
    print(f"{'='*80}")
    print_architecture_comparison(results, homophilic)
    print_improvement_summary(results, homophilic)

    # Overall statistics
    print_average_improvements(results)

    # LaTeX table
    all_datasets = heterophilic + homophilic
    print_latex_table(results, all_datasets)

    print(f"\n{'='*80}")
    print("Analysis complete!")
    print(f"{'='*80}\n")

if __name__ == '__main__':
    main()
