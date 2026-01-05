#!/usr/bin/env python3
"""
Analyze all experimental results from heterophilic and homophilic datasets.
Generates tables and statistics for the paper.
"""

import os
import re
import json
from pathlib import Path
from collections import defaultdict
import numpy as np

def parse_train_log(log_path):
    """Extract best validation accuracy and test accuracy from train.log"""
    try:
        with open(log_path, 'r') as f:
            content = f.read()

        # Extract final results
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

def collect_results(results_dir):
    """Collect all experimental results"""
    results = defaultdict(lambda: defaultdict(list))

    results_path = Path(results_dir)

    # Process both heterophilic and homophilic
    for category in ['heterophilic', 'homophilic']:
        category_path = results_path / category
        if not category_path.exists():
            continue

        for exp_dir in category_path.iterdir():
            if not exp_dir.is_dir():
                continue

            # Parse experiment name
            # Format: {dataset}_{model}_{config}_seed{seed}
            # Examples:
            #   chameleon_baseline_seed42
            #   chameleon_mhc_n4_seed42
            #   Cora_baseline_seed42
            #   Cora_mhc_n4_seed42

            exp_name = exp_dir.name
            parts = exp_name.split('_')

            # Extract components
            # Skip if no seed (incomplete experiment)
            if 'seed' not in exp_name:
                continue

            if 'baseline' in exp_name:
                dataset = parts[0]
                model = 'baseline'
                n_streams = 1
                seed = int(parts[-1].replace('seed', ''))
            elif 'mhc' in exp_name:
                dataset = parts[0]
                model = 'mhc'
                # Extract n from 'n2', 'n4', 'n8'
                n_match = re.search(r'n(\d+)', exp_name)
                n_streams = int(n_match.group(1)) if n_match else None
                seed = int(parts[-1].replace('seed', ''))
            else:
                continue

            # Parse train.log
            log_path = exp_dir / 'train.log'
            if log_path.exists():
                result = parse_train_log(log_path)
                if result:
                    result['seed'] = seed
                    result['n_streams'] = n_streams
                    result['category'] = category

                    # Create key: dataset_model_n
                    if model == 'baseline':
                        key = f"{dataset}_baseline"
                    else:
                        key = f"{dataset}_mhc_n{n_streams}"

                    results[category][key].append(result)

    return results

def compute_statistics(result_list):
    """Compute mean and std from list of results"""
    if not result_list:
        return None

    val_accs = [r['val_acc'] for r in result_list]
    test_accs = [r['test_acc'] for r in result_list]

    return {
        'val_mean': np.mean(val_accs),
        'val_std': np.std(val_accs),
        'test_mean': np.mean(test_accs),
        'test_std': np.std(test_accs),
        'num_seeds': len(result_list)
    }

def print_latex_table(results, category, dataset_order):
    """Print results in LaTeX table format"""
    print(f"\n{'='*80}")
    print(f"{category.upper()} DATASETS - LaTeX Table")
    print(f"{'='*80}\n")

    # Header
    print(r"\begin{table}[t]")
    print(r"\centering")
    print(r"\caption{Results on " + category + r" datasets}")
    print(r"\label{tab:" + category + r"}")
    print(r"\begin{tabular}{lcccc}")
    print(r"\toprule")
    print(r"Dataset & Baseline & mHC-GNN ($n=2$) & mHC-GNN ($n=4$) & mHC-GNN ($n=8$) \\")
    print(r"\midrule")

    category_results = results[category]

    for dataset in dataset_order:
        if dataset not in [k.split('_')[0] for k in category_results.keys()]:
            continue

        row = [dataset]

        # Baseline
        baseline_key = f"{dataset}_baseline"
        if baseline_key in category_results:
            stats = compute_statistics(category_results[baseline_key])
            if stats:
                row.append(f"{stats['test_mean']*100:.2f} $\\pm$ {stats['test_std']*100:.2f}")
            else:
                row.append("-")
        else:
            row.append("-")

        # mHC-GNN variants
        for n in [2, 4, 8]:
            mhc_key = f"{dataset}_mhc_n{n}"
            if mhc_key in category_results:
                stats = compute_statistics(category_results[mhc_key])
                if stats:
                    # Bold if better than baseline
                    baseline_stats = compute_statistics(category_results.get(baseline_key, []))
                    if baseline_stats and stats['test_mean'] > baseline_stats['test_mean']:
                        row.append(f"\\textbf{{{stats['test_mean']*100:.2f}}} $\\pm$ {stats['test_std']*100:.2f}")
                    else:
                        row.append(f"{stats['test_mean']*100:.2f} $\\pm$ {stats['test_std']*100:.2f}")
                else:
                    row.append("-")
            else:
                row.append("-")

        print(" & ".join(row) + r" \\")

    print(r"\bottomrule")
    print(r"\end{tabular}")
    print(r"\end{table}")

def print_markdown_table(results, category, dataset_order):
    """Print results in Markdown table format"""
    print(f"\n{'='*80}")
    print(f"{category.upper()} DATASETS - Markdown Table")
    print(f"{'='*80}\n")

    # Header
    print("| Dataset | Baseline | mHC-GNN (n=2) | mHC-GNN (n=4) | mHC-GNN (n=8) |")
    print("|---------|----------|---------------|---------------|---------------|")

    category_results = results[category]

    for dataset in dataset_order:
        if dataset not in [k.split('_')[0] for k in category_results.keys()]:
            continue

        row = [dataset]

        # Baseline
        baseline_key = f"{dataset}_baseline"
        if baseline_key in category_results:
            stats = compute_statistics(category_results[baseline_key])
            if stats:
                row.append(f"{stats['test_mean']*100:.2f} ± {stats['test_std']*100:.2f}")
            else:
                row.append("-")
        else:
            row.append("-")

        # mHC-GNN variants
        for n in [2, 4, 8]:
            mhc_key = f"{dataset}_mhc_n{n}"
            if mhc_key in category_results:
                stats = compute_statistics(category_results[mhc_key])
                if stats:
                    # Add ** if better than baseline
                    baseline_stats = compute_statistics(category_results.get(baseline_key, []))
                    if baseline_stats and stats['test_mean'] > baseline_stats['test_mean']:
                        row.append(f"**{stats['test_mean']*100:.2f} ± {stats['test_std']*100:.2f}**")
                    else:
                        row.append(f"{stats['test_mean']*100:.2f} ± {stats['test_std']*100:.2f}")
                else:
                    row.append("-")
            else:
                row.append("-")

        print("| " + " | ".join(row) + " |")

def print_summary_statistics(results):
    """Print overall summary statistics"""
    print(f"\n{'='*80}")
    print("SUMMARY STATISTICS")
    print(f"{'='*80}\n")

    for category in ['heterophilic', 'homophilic']:
        if category not in results:
            continue

        print(f"\n{category.upper()}:")
        print("-" * 40)

        category_results = results[category]

        # Count experiments
        total_experiments = sum(len(v) for v in category_results.values())
        print(f"Total experiments: {total_experiments}")

        # Average improvement
        improvements = []
        for dataset_key in category_results:
            if 'baseline' in dataset_key:
                dataset = dataset_key.replace('_baseline', '')
                baseline_stats = compute_statistics(category_results[dataset_key])

                if baseline_stats:
                    for n in [2, 4, 8]:
                        mhc_key = f"{dataset}_mhc_n{n}"
                        if mhc_key in category_results:
                            mhc_stats = compute_statistics(category_results[mhc_key])
                            if mhc_stats:
                                improvement = (mhc_stats['test_mean'] - baseline_stats['test_mean']) * 100
                                improvements.append({
                                    'dataset': dataset,
                                    'n': n,
                                    'improvement': improvement,
                                    'baseline': baseline_stats['test_mean'] * 100,
                                    'mhc': mhc_stats['test_mean'] * 100
                                })

        if improvements:
            avg_improvement = np.mean([i['improvement'] for i in improvements])
            positive_improvements = [i for i in improvements if i['improvement'] > 0]

            print(f"Average improvement: {avg_improvement:.2f}%")
            print(f"Positive improvements: {len(positive_improvements)}/{len(improvements)}")

            # Best improvement
            best = max(improvements, key=lambda x: x['improvement'])
            print(f"Best improvement: {best['dataset']} (n={best['n']}): "
                  f"{best['baseline']:.2f}% → {best['mhc']:.2f}% "
                  f"(+{best['improvement']:.2f}%)")

def main():
    results_dir = '../../results'

    print("Collecting results...")
    results = collect_results(results_dir)

    # Dataset order
    heterophilic_datasets = ['chameleon', 'squirrel', 'actor', 'texas', 'wisconsin', 'cornell']
    homophilic_datasets = ['Cora', 'CiteSeer', 'PubMed']

    # Print tables
    if 'heterophilic' in results:
        print_markdown_table(results, 'heterophilic', heterophilic_datasets)
        print_latex_table(results, 'heterophilic', heterophilic_datasets)

    if 'homophilic' in results:
        print_markdown_table(results, 'homophilic', homophilic_datasets)
        print_latex_table(results, 'homophilic', homophilic_datasets)

    # Summary statistics
    print_summary_statistics(results)

    # Save results to JSON
    output_path = Path(results_dir) / 'analysis_results.json'

    # Convert to serializable format
    serializable_results = {}
    for category, category_data in results.items():
        serializable_results[category] = {}
        for key, result_list in category_data.items():
            stats = compute_statistics(result_list)
            serializable_results[category][key] = {
                'statistics': stats,
                'raw_results': result_list
            }

    with open(output_path, 'w') as f:
        json.dump(serializable_results, f, indent=2)

    print(f"\n{'='*80}")
    print(f"Results saved to: {output_path}")
    print(f"{'='*80}\n")

if __name__ == '__main__':
    main()
