#!/usr/bin/env python3
"""
Analyze ablation study results
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

def collect_ablation_results(results_dir):
    """Collect ablation experiment results"""
    results = defaultdict(lambda: defaultdict(list))

    results_path = Path(results_dir)

    for category in ['heterophilic', 'homophilic']:
        category_path = results_path / category
        if not category_path.exists():
            continue

        for exp_dir in category_path.iterdir():
            if not exp_dir.is_dir():
                continue

            exp_name = exp_dir.name

            # Parse: dataset_ablation_sSEED
            # Examples:
            #   chameleon_dynamic_only_s42
            #   chameleon_full_s42
            #   chameleon_no_manifold_s42
            #   chameleon_sinkhorn10_s42

            if '_s' not in exp_name:
                continue

            # Find last _s occurrence (for seed)
            last_s_idx = exp_name.rfind('_s')
            base_name = exp_name[:last_s_idx]
            seed = int(exp_name[last_s_idx+2:])

            # Extract dataset and ablation type
            if 'chameleon' in base_name:
                dataset = 'chameleon'
                ablation = base_name.replace('chameleon_', '')
            elif 'texas' in base_name:
                dataset = 'texas'
                ablation = base_name.replace('texas_', '')
            elif 'Cora' in base_name:
                dataset = 'Cora'
                ablation = base_name.replace('Cora_', '')
            else:
                continue

            # Parse train.log
            log_path = exp_dir / 'train.log'
            if log_path.exists():
                result = parse_train_log(log_path)
                if result:
                    result['seed'] = seed
                    result['category'] = category
                    results[dataset][ablation].append(result)

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

def print_ablation_table(results, dataset_order):
    """Print ablation results in table format"""
    print(f"\n{'='*80}")
    print("ABLATION STUDY RESULTS")
    print(f"{'='*80}\n")

    # Ablation order for display
    ablation_order = [
        ('full', 'Full mHC-GNN'),
        ('dynamic_only', 'w/o static connections'),
        ('static_only', 'w/o dynamic routing'),
        ('no_manifold', 'w/o manifold constraints'),
        ('sinkhorn5', 'Sinkhorn 5 iters'),
        ('sinkhorn10', 'Sinkhorn 10 iters'),
        ('sinkhorn50', 'Sinkhorn 50 iters'),
        ('tau0_01', 'τ = 0.01'),
        ('tau0_05', 'τ = 0.05'),
        ('tau0_5', 'τ = 0.5'),
    ]

    for dataset in dataset_order:
        if dataset not in results:
            continue

        print(f"\n## {dataset.upper()}")
        print("-" * 60)
        print(f"{'Configuration':<35} {'Test Acc (%)':<15} {'# Seeds'}")
        print("-" * 60)

        dataset_results = results[dataset]

        # Get baseline (full) result for comparison
        baseline_stats = compute_statistics(dataset_results.get('full', []))
        baseline_mean = baseline_stats['test_mean'] * 100 if baseline_stats else 0

        for ablation_key, ablation_name in ablation_order:
            if ablation_key in dataset_results:
                stats = compute_statistics(dataset_results[ablation_key])
                if stats:
                    mean = stats['test_mean'] * 100
                    std = stats['test_std'] * 100

                    # Mark with * if worse than baseline
                    marker = '*' if mean < baseline_mean - 0.5 else ' '

                    print(f"{ablation_name:<35} {mean:>6.2f} ± {std:<4.2f}  {marker}  {stats['num_seeds']}")

def print_latex_table(results, datasets):
    """Print ablation results in LaTeX format"""
    print(f"\n{'='*80}")
    print("LATEX TABLE")
    print(f"{'='*80}\n")

    print(r"\begin{table}[t]")
    print(r"\centering")
    print(r"\caption{Ablation study results. * indicates significant degradation from full model.}")
    print(r"\label{tab:ablation}")

    # Determine number of columns
    num_datasets = len(datasets)
    col_spec = "l" + "c" * num_datasets
    print(f"\\begin{{tabular}}{{{col_spec}}}")
    print(r"\toprule")

    # Header
    header = "Configuration & " + " & ".join(datasets) + r" \\"
    print(header)
    print(r"\midrule")

    ablation_configs = [
        ('full', 'Full mHC-GNN'),
        ('dynamic_only', 'w/o static'),
        ('static_only', 'w/o dynamic'),
        ('no_manifold', 'w/o Sinkhorn'),
        ('sinkhorn10', 'Sinkhorn 10 iters'),
    ]

    for ablation_key, ablation_name in ablation_configs:
        row = [ablation_name]

        for dataset in datasets:
            if dataset in results and ablation_key in results[dataset]:
                stats = compute_statistics(results[dataset][ablation_key])
                if stats:
                    mean = stats['test_mean'] * 100
                    std = stats['test_std'] * 100
                    row.append(f"{mean:.2f} $\\pm$ {std:.2f}")
                else:
                    row.append("-")
            else:
                row.append("-")

        print(" & ".join(row) + r" \\")

    print(r"\bottomrule")
    print(r"\end{tabular}")
    print(r"\end{table}")

def print_summary_statistics(results):
    """Print summary of component contributions"""
    print(f"\n{'='*80}")
    print("COMPONENT CONTRIBUTION ANALYSIS")
    print(f"{'='*80}\n")

    for dataset in results:
        print(f"\n{dataset}:")
        print("-" * 40)

        full_stats = compute_statistics(results[dataset].get('full', []))
        if not full_stats:
            continue

        baseline = full_stats['test_mean'] * 100
        print(f"Full mHC-GNN: {baseline:.2f}%")

        # Component contributions
        components = [
            ('dynamic_only', 'Static connections'),
            ('static_only', 'Dynamic routing'),
            ('no_manifold', 'Manifold constraints'),
        ]

        for ablation_key, component_name in components:
            if ablation_key in results[dataset]:
                stats = compute_statistics(results[dataset][ablation_key])
                if stats:
                    without = stats['test_mean'] * 100
                    contribution = baseline - without
                    print(f"  {component_name:25s}: {contribution:+.2f}% "
                          f"(without: {without:.2f}%)")

def main():
    results_dir = '../../results_ablation'

    print("Collecting ablation results...")
    results = collect_ablation_results(results_dir)

    # Dataset order
    datasets = ['chameleon', 'texas', 'Cora']

    # Print tables
    print_ablation_table(results, datasets)
    print_latex_table(results, datasets)
    print_summary_statistics(results)

    print(f"\n{'='*80}")
    print("Analysis complete!")
    print(f"{'='*80}\n")

if __name__ == '__main__':
    main()
