#!/bin/bash

#############################################
# mHC-GNN: Complete Experiment Suite
#############################################
#
# This script runs all experiments reported in the paper:
# 1. Main benchmarks (8 layers, multiple datasets)
# 2. Multi-architecture experiments (GCN, GraphSAGE, GAT, GIN)
# 3. Extended depth analysis (2-128 layers)
# 4. Ablation studies
#
# Usage:
#   ./run_all_experiments.sh [experiment_type]
#
# Options:
#   main       - Run main 8-layer benchmarks (default)
#   multi-arch - Run multi-architecture experiments
#   depth      - Run extended depth analysis (requires 4 GPUs)
#   ablation   - Run ablation studies
#   all        - Run all experiments
#

set -e  # Exit on error

EXPERIMENT_TYPE="${1:-main}"
RESULTS_DIR="results"
mkdir -p "$RESULTS_DIR"

# Color output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

log_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

log_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

#############################################
# Main Benchmarks (8 Layers)
#############################################
run_main_benchmarks() {
    log_info "Running main 8-layer benchmarks..."

    DATASETS=("Cora" "CiteSeer" "PubMed" "Texas" "Chameleon" "ogbn-arxiv")
    SEEDS=(42 123 456 789 2024)

    for dataset in "${DATASETS[@]}"; do
        log_info "Dataset: $dataset"

        for seed in "${SEEDS[@]}"; do
            # Baseline GCN
            python src/experiments/run_node_classification.py \
                --dataset "$dataset" \
                --model standard_gnn \
                --gnn_type gcn \
                --num_layers 8 \
                --hidden_channels 128 \
                --epochs 500 \
                --seed "$seed" \
                --output_dir "$RESULTS_DIR/main/${dataset}_baseline_s${seed}"

            # mHC-GNN n=2
            python src/experiments/run_node_classification.py \
                --dataset "$dataset" \
                --model mhc_gnn \
                --gnn_type gcn \
                --num_layers 8 \
                --hidden_channels 256 \
                --n_streams 2 \
                --epochs 500 \
                --seed "$seed" \
                --output_dir "$RESULTS_DIR/main/${dataset}_mhc_n2_s${seed}"

            # mHC-GNN n=4
            python src/experiments/run_node_classification.py \
                --dataset "$dataset" \
                --model mhc_gnn \
                --gnn_type gcn \
                --num_layers 8 \
                --hidden_channels 512 \
                --n_streams 4 \
                --epochs 500 \
                --seed "$seed" \
                --output_dir "$RESULTS_DIR/main/${dataset}_mhc_n4_s${seed}"
        done
    done

    log_info "Main benchmarks completed!"
}

#############################################
# Multi-Architecture Experiments
#############################################
run_multi_arch() {
    log_info "Running multi-architecture experiments..."

    ARCHITECTURES=("gcn" "sage" "gat" "gin")
    DATASETS=("Cora" "CiteSeer" "PubMed" "Chameleon" "Texas" "Actor")
    SEEDS=(42 123 456 789 2024)

    for arch in "${ARCHITECTURES[@]}"; do
        log_info "Architecture: $arch"

        for dataset in "${DATASETS[@]}"; do
            for seed in "${SEEDS[@]}"; do
                # Baseline
                python src/experiments/run_node_classification.py \
                    --dataset "$dataset" \
                    --model standard_gnn \
                    --gnn_type "$arch" \
                    --num_layers 8 \
                    --hidden_channels 128 \
                    --epochs 500 \
                    --seed "$seed" \
                    --output_dir "$RESULTS_DIR/multi_arch/${arch}/${dataset}_baseline_s${seed}"

                # mHC-GNN n=4
                python src/experiments/run_node_classification.py \
                    --dataset "$dataset" \
                    --model mhc_gnn \
                    --gnn_type "$arch" \
                    --num_layers 8 \
                    --hidden_channels 512 \
                    --n_streams 4 \
                    --epochs 500 \
                    --seed "$seed" \
                    --output_dir "$RESULTS_DIR/multi_arch/${arch}/${dataset}_mhc_n4_s${seed}"
            done
        done
    done

    log_info "Multi-architecture experiments completed!"
}

#############################################
# Extended Depth Analysis (requires 4 GPUs)
#############################################
run_depth_analysis() {
    log_info "Running extended depth analysis..."
    log_warn "This requires 4 GPUs and will take 4-5 hours"

    # Check if run_deep_depth_analysis.sh exists
    if [ -f "run_deep_depth_analysis.sh" ]; then
        ./run_deep_depth_analysis.sh
    else
        log_error "run_deep_depth_analysis.sh not found!"
        log_info "Running depth experiments manually..."

        DEPTHS=(2 4 8 16 32 64 128)
        DATASETS=("Cora" "CiteSeer" "PubMed")
        SEEDS=(42 123 456 789 2024)

        for depth in "${DEPTHS[@]}"; do
            # Adjust hidden channels based on depth
            if [ "$depth" -eq 2 ]; then
                baseline_hidden=64
                n2_hidden=128
                n4_hidden=256
            else
                baseline_hidden=128
                n2_hidden=256
                n4_hidden=512
            fi

            log_info "Depth: ${depth} layers"

            for dataset in "${DATASETS[@]}"; do
                for seed in "${SEEDS[@]}"; do
                    # Baseline
                    python src/experiments/run_node_classification.py \
                        --dataset "$dataset" \
                        --model standard_gnn \
                        --gnn_type gcn \
                        --num_layers "$depth" \
                        --hidden_channels "$baseline_hidden" \
                        --epochs 500 \
                        --seed "$seed" \
                        --output_dir "$RESULTS_DIR/depth/${depth}L/${dataset}_baseline_s${seed}"

                    # mHC-GNN n=2
                    python src/experiments/run_node_classification.py \
                        --dataset "$dataset" \
                        --model mhc_gnn \
                        --gnn_type gcn \
                        --num_layers "$depth" \
                        --hidden_channels "$n2_hidden" \
                        --n_streams 2 \
                        --epochs 500 \
                        --seed "$seed" \
                        --output_dir "$RESULTS_DIR/depth/${depth}L/${dataset}_mhc_n2_s${seed}"

                    # mHC-GNN n=4
                    python src/experiments/run_node_classification.py \
                        --dataset "$dataset" \
                        --model mhc_gnn \
                        --gnn_type gcn \
                        --num_layers "$depth" \
                        --hidden_channels "$n4_hidden" \
                        --n_streams 4 \
                        --epochs 500 \
                        --seed "$seed" \
                        --output_dir "$RESULTS_DIR/depth/${depth}L/${dataset}_mhc_n4_s${seed}"
                done
            done
        done
    fi

    log_info "Depth analysis completed!"
}

#############################################
# Ablation Studies
#############################################
run_ablation() {
    log_info "Running ablation studies..."

    DATASETS=("Cora" "Chameleon" "Texas")
    SEEDS=(42 123 456 789 2024)

    for dataset in "${DATASETS[@]}"; do
        # Adjust hidden dimensions per dataset
        if [ "$dataset" == "Cora" ]; then
            hidden=8
        else
            hidden=16
        fi

        log_info "Dataset: $dataset (hidden=${hidden})"

        for seed in "${SEEDS[@]}"; do
            # Full mHC-GNN
            python src/experiments/run_node_classification.py \
                --dataset "$dataset" \
                --model mhc_gnn \
                --gnn_type gcn \
                --num_layers 4 \
                --hidden_channels "$hidden" \
                --n_streams 4 \
                --use_dynamic \
                --use_static \
                --epochs 500 \
                --seed "$seed" \
                --output_dir "$RESULTS_DIR/ablation/${dataset}_full_s${seed}"

            # Dynamic-only
            python src/experiments/run_node_classification.py \
                --dataset "$dataset" \
                --model mhc_gnn \
                --gnn_type gcn \
                --num_layers 4 \
                --hidden_channels "$hidden" \
                --n_streams 4 \
                --use_dynamic \
                --epochs 500 \
                --seed "$seed" \
                --output_dir "$RESULTS_DIR/ablation/${dataset}_dynamic_only_s${seed}"

            # Static-only
            python src/experiments/run_node_classification.py \
                --dataset "$dataset" \
                --model mhc_gnn \
                --gnn_type gcn \
                --num_layers 4 \
                --hidden_channels "$hidden" \
                --n_streams 4 \
                --use_static \
                --epochs 500 \
                --seed "$seed" \
                --output_dir "$RESULTS_DIR/ablation/${dataset}_static_only_s${seed}"

            # No-Sinkhorn (unconstrained)
            python src/experiments/run_node_classification.py \
                --dataset "$dataset" \
                --model mhc_gnn \
                --gnn_type gcn \
                --num_layers 4 \
                --hidden_channels "$hidden" \
                --n_streams 4 \
                --use_dynamic \
                --use_static \
                --no_sinkhorn \
                --epochs 500 \
                --seed "$seed" \
                --output_dir "$RESULTS_DIR/ablation/${dataset}_no_sinkhorn_s${seed}"
        done
    done

    log_info "Ablation studies completed!"
}

#############################################
# Main Script Logic
#############################################

echo "========================================"
echo "  mHC-GNN Experiment Suite"
echo "========================================"
echo ""

case "$EXPERIMENT_TYPE" in
    main)
        run_main_benchmarks
        ;;
    multi-arch)
        run_multi_arch
        ;;
    depth)
        run_depth_analysis
        ;;
    ablation)
        run_ablation
        ;;
    all)
        log_info "Running ALL experiments..."
        run_main_benchmarks
        run_multi_arch
        run_depth_analysis
        run_ablation
        ;;
    *)
        log_error "Unknown experiment type: $EXPERIMENT_TYPE"
        echo ""
        echo "Usage: $0 [main|multi-arch|depth|ablation|all]"
        echo ""
        echo "Options:"
        echo "  main       - Run main 8-layer benchmarks (default)"
        echo "  multi-arch - Run multi-architecture experiments"
        echo "  depth      - Run extended depth analysis (requires 4 GPUs)"
        echo "  ablation   - Run ablation studies"
        echo "  all        - Run all experiments"
        exit 1
        ;;
esac

echo ""
echo "========================================"
log_info "Experiments completed successfully!"
echo "Results saved to: $RESULTS_DIR/"
echo "========================================"
