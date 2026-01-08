#!/bin/bash

# Extended Depth Analysis: Performance across 2, 4, 8, 16, 32, 64, 128 layers
# Shows mHC-GNN benefits increase with network depth (over-smoothing mitigation)

set -e

DATASETS=("Cora" "CiteSeer" "PubMed")
DEPTHS=(2 4 8 16 32 64 128)
SEEDS=(42 123 456 789 2024)
RESULTS_BASE="results/deep_depth_analysis"

# Create results directory
mkdir -p $RESULTS_BASE

echo "=========================================="
echo "EXTENDED DEPTH ANALYSIS EXPERIMENTS"
echo "=========================================="
echo "Datasets: ${DATASETS[@]}"
echo "Depths: ${DEPTHS[@]} layers"
echo "Seeds: ${SEEDS[@]}"
echo ""
echo "Configuration strategy:"
echo "  2 layers:   64 hidden (baseline), 128 (mHC n=2), 256 (mHC n=4)"
echo "  4 layers:   128 hidden (baseline), 256 (mHC n=2), 512 (mHC n=4)"
echo "  8 layers:   128 hidden (baseline), 256 (mHC n=2), 512 (mHC n=4)"
echo "  16 layers:  128 hidden (baseline), 256 (mHC n=2), 512 (mHC n=4)"
echo "  32 layers:  128 hidden (baseline), 256 (mHC n=2), 512 (mHC n=4)"
echo "  64 layers:  128 hidden (baseline), 256 (mHC n=2), 512 (mHC n=4)"
echo "  128 layers: 128 hidden (baseline), 256 (mHC n=2), 512 (mHC n=4)"
echo ""
echo "Rationale: Maintain 64+ hidden dims per stream"
echo "=========================================="

# Function to get hidden dimensions based on depth and config
get_hidden_dims() {
    local depth=$1
    local config=$2

    if [ "$depth" -eq 2 ]; then
        if [ "$config" = "baseline" ]; then
            echo 64
        elif [ "$config" = "mhc_n2" ]; then
            echo 128
        elif [ "$config" = "mhc_n4" ]; then
            echo 256
        fi
    else
        # For 4, 8, 16, 32, 64, 128 layers
        if [ "$config" = "baseline" ]; then
            echo 128
        elif [ "$config" = "mhc_n2" ]; then
            echo 256
        elif [ "$config" = "mhc_n4" ]; then
            echo 512
        fi
    fi
}

# Function to run single experiment
run_experiment() {
    local dataset=$1
    local depth=$2
    local config=$3
    local seed=$4

    local hidden_dims=$(get_hidden_dims $depth $config)
    local exp_name="${dataset}_${depth}L_${config}_seed${seed}"
    local results_dir="${RESULTS_BASE}/${depth}layers"
    local result_marker="${results_dir}/${exp_name}.done"

    # Skip if already exists
    if [ -f "$result_marker" ]; then
        echo "SKIP: $exp_name already completed"
        return
    fi

    mkdir -p "$results_dir"

    echo "Running: $dataset | ${depth}L | $config | hidden=${hidden_dims} | seed=${seed}"

    # Build command
    if [ "$config" = "baseline" ]; then
        cmd="conda run -n mhc-gnn python src/experiments/run_node_classification.py \
            --dataset $dataset \
            --gnn_type gcn \
            --model standard_gnn \
            --num_layers $depth \
            --hidden_channels $hidden_dims \
            --seed $seed \
            --exp_name $exp_name \
            --save_dir $results_dir \
            --epochs 500 \
            --patience 100"
    else
        # Extract n from mhc_n2 or mhc_n4
        n_streams=${config:5}  # Extract "2" or "4" from "mhc_n2"
        cmd="conda run -n mhc-gnn python src/experiments/run_node_classification.py \
            --dataset $dataset \
            --gnn_type gcn \
            --model mhc_gnn \
            --num_layers $depth \
            --hidden_channels $hidden_dims \
            --n_streams $n_streams \
            --seed $seed \
            --exp_name $exp_name \
            --save_dir $results_dir \
            --epochs 500 \
            --patience 100"
    fi

    # Run and capture output
    eval $cmd 2>&1 | tee "${results_dir}/${exp_name}.log"

    if [ ${PIPESTATUS[0]} -eq 0 ]; then
        touch "$result_marker"
        echo "✅ SUCCESS: $exp_name"
    else
        echo "❌ FAILED: $exp_name"
    fi
}

# Export function for parallel execution
export -f run_experiment
export -f get_hidden_dims
export RESULTS_BASE

# Generate all experiment combinations
experiments=()
for dataset in "${DATASETS[@]}"; do
    for depth in "${DEPTHS[@]}"; do
        for seed in "${SEEDS[@]}"; do
            experiments+=("$dataset $depth baseline $seed")
            experiments+=("$dataset $depth mhc_n2 $seed")
            experiments+=("$dataset $depth mhc_n4 $seed")
        done
    done
done

echo "Total experiments: ${#experiments[@]}"
echo ""
echo "Starting parallel execution with 16 parallel jobs across 4 GPUs..."
echo "Estimated time: ~4-5 hours (longer for deep networks)"
echo ""

# Run experiments in parallel across 4 GPUs
printf '%s\n' "${experiments[@]}" | parallel -j 16 --colsep ' ' \
    'CUDA_VISIBLE_DEVICES=$(( ({%} - 1) % 4 )) run_experiment {1} {2} {3} {4}'

echo ""
echo "=========================================="
echo "EXTENDED DEPTH ANALYSIS EXPERIMENTS COMPLETE!"
echo "=========================================="

# Count completed experiments per depth
for depth in "${DEPTHS[@]}"; do
    completed=$(ls ${RESULTS_BASE}/${depth}layers/*.done 2>/dev/null | wc -l)
    total=$((3 * 5))  # 3 configs × 5 seeds × 3 datasets = 45 per depth
    echo "${depth} layers: $completed / 45 experiments completed"
done

echo ""
echo "Generating summary..."

# Generate summary
conda run -n mhc-gnn python - <<'PYTHON_SCRIPT'
import re
import numpy as np
from pathlib import Path
from collections import defaultdict

results_base = Path("results/deep_depth_analysis")

# Pattern to extract test accuracy from logs
pattern = r"Test Acc at Best Val: ([\d.]+)"

# Collect results
data = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))

for depth_dir in sorted(results_base.glob("*layers")):
    depth = int(depth_dir.name.replace("layers", ""))

    for log_file in depth_dir.glob("*.log"):
        exp_name = log_file.stem

        # Parse: dataset_depth_config_seed
        parts = exp_name.split('_')
        if len(parts) < 4:
            continue

        dataset = parts[0]

        # Determine config
        if 'baseline' in exp_name:
            config = 'baseline'
        elif 'mhc' in exp_name and 'n2' in exp_name:
            config = 'mhc_n2'
        elif 'mhc' in exp_name and 'n4' in exp_name:
            config = 'mhc_n4'
        else:
            continue

        # Extract test accuracy
        try:
            with open(log_file) as f:
                log_content = f.read()
                match = re.search(pattern, log_content)
                if match:
                    test_acc = float(match.group(1))
                    data[dataset][depth][config].append(test_acc)
        except Exception as e:
            print(f"Error reading {log_file}: {e}")

# Print summary
print("\n" + "="*120)
print("EXTENDED DEPTH ANALYSIS RESULTS SUMMARY")
print("="*120)

datasets = ['Cora', 'CiteSeer', 'PubMed']
depths = [2, 4, 8, 16, 32, 64, 128]

for dataset in datasets:
    print(f"\n{'='*120}")
    print(f"{dataset.upper()}")
    print(f"{'='*120}")
    print(f"{'Depth':<10} | {'Baseline':<20} | {'mHC n=2':<20} | {'mHC n=4':<20} | {'Best Δ':<10}")
    print("-"*120)

    baseline_trend = []
    mhc_n2_trend = []
    mhc_n4_trend = []
    depth_labels = []

    for depth in depths:
        baseline = data[dataset][depth]['baseline']
        mhc_n2 = data[dataset][depth]['mhc_n2']
        mhc_n4 = data[dataset][depth]['mhc_n4']

        if baseline:
            baseline_mean = np.mean(baseline) * 100
            baseline_std = np.std(baseline) * 100
            baseline_str = f"{baseline_mean:6.2f} ± {baseline_std:5.2f}"
            baseline_trend.append(baseline_mean)
            depth_labels.append(depth)
        else:
            baseline_str = "N/A"
            baseline_mean = 0

        if mhc_n2:
            n2_mean = np.mean(mhc_n2) * 100
            n2_std = np.std(mhc_n2) * 100
            n2_str = f"{n2_mean:6.2f} ± {n2_std:5.2f}"
            mhc_n2_trend.append(n2_mean)
        else:
            n2_str = "N/A"
            n2_mean = 0

        if mhc_n4:
            n4_mean = np.mean(mhc_n4) * 100
            n4_std = np.std(mhc_n4) * 100
            n4_str = f"{n4_mean:6.2f} ± {n4_std:5.2f}"
            mhc_n4_trend.append(n4_mean)
        else:
            n4_str = "N/A"
            n4_mean = 0

        # Calculate best improvement
        best_mhc = max(n2_mean, n4_mean)
        if baseline_mean > 0 and best_mhc > 0:
            delta = best_mhc - baseline_mean
            delta_str = f"{delta:+6.2f}%"
        else:
            delta_str = "N/A"

        print(f"{depth}L {' ':<6} | {baseline_str:<20} | {n2_str:<20} | {n4_str:<20} | {delta_str:<10}")

    # Show trend
    if len(baseline_trend) >= 2:
        baseline_degradation = baseline_trend[0] - baseline_trend[-1]
        print(f"\n  Baseline degradation (2L → {depth_labels[-1]}L): {baseline_degradation:+.2f}%")

        if len(mhc_n2_trend) >= 2:
            mhc_degradation = mhc_n2_trend[0] - mhc_n2_trend[-1]
            print(f"  mHC n=2 degradation (2L → {depth_labels[-1]}L):  {mhc_degradation:+.2f}%")
            print(f"  Over-smoothing mitigation:      {baseline_degradation - mhc_degradation:+.2f}%")

print("\n" + "="*120)
print("SUMMARY: mHC Improvement vs Depth")
print("="*120)
print(f"{'Dataset':<12} | {'2L':<10} | {'4L':<10} | {'8L':<10} | {'16L':<10} | {'32L':<10} | {'64L':<10} | {'128L':<10}")
print("-"*120)

for dataset in datasets:
    improvements = []
    for depth in depths:
        baseline = data[dataset][depth]['baseline']
        mhc_n2 = data[dataset][depth]['mhc_n2']
        mhc_n4 = data[dataset][depth]['mhc_n4']

        if baseline and (mhc_n2 or mhc_n4):
            baseline_mean = np.mean(baseline) * 100
            best_mhc = 0
            if mhc_n2:
                best_mhc = max(best_mhc, np.mean(mhc_n2) * 100)
            if mhc_n4:
                best_mhc = max(best_mhc, np.mean(mhc_n4) * 100)
            improvement = best_mhc - baseline_mean
            improvements.append(f"{improvement:+6.2f}%")
        else:
            improvements.append("N/A")

    print(f"{dataset:<12} | {improvements[0]:<10} | {improvements[1]:<10} | {improvements[2]:<10} | {improvements[3]:<10} | {improvements[4]:<10} | {improvements[5]:<10} | {improvements[6]:<10}")

print("="*120)
print("\nKey Finding: mHC improvement should increase with depth (mitigating over-smoothing)")
print("Expected: Baseline accuracy drops significantly at 32+, 64+, 128+ layers")
print("Expected: mHC-GNN maintains accuracy even at extreme depths")

# Save summary
summary_file = results_base / "summary.txt"
with open(summary_file, 'w') as f:
    f.write("See console output for extended depth analysis summary\n")

print(f"\nResults saved to: {results_base}")
PYTHON_SCRIPT

echo ""
echo "✅ COMPLETE! Check results/deep_depth_analysis/ for all results"
echo ""
echo "Next steps:"
echo "  1. Review summary above"
echo "  2. Create visualization: python create_deep_depth_plots.py"
echo "  3. Add extended depth analysis to paper"
echo ""
