#!/bin/bash

# Monitor deep depth analysis experiments

echo "========================================"
echo "Deep Depth Analysis - Progress Monitor"
echo "========================================"
echo ""

# Count total completed
total_completed=$(find results/deep_depth_analysis -name "*.done" 2>/dev/null | wc -l)
echo "Total completed: $total_completed / 315"
echo ""

# Breakdown by depth
echo "Progress by depth:"
echo "-------------------------------------------"
for depth in 2 4 8 16 32 64 128; do
    completed=$(ls results/deep_depth_analysis/${depth}layers/*.done 2>/dev/null | wc -l)
    printf "%3dL: %2d / 45 experiments (" $depth $completed
    percent=$((completed * 100 / 45))
    printf "%3d%%)\n" $percent
done

echo ""
echo "-------------------------------------------"

# Show GPU usage
echo ""
echo "Current GPU usage:"
nvidia-smi --query-gpu=index,name,utilization.gpu,memory.used,memory.total --format=csv,noheader,nounits | \
    awk -F, '{printf "GPU %d: %3d%% util, %5dMB / %5dMB mem\n", $1, $3, $4, $5}'

echo ""
echo "Running processes: $(ps aux | grep 'run_node_classification.py' | grep -v grep | wc -l)"

echo ""
echo "Recent completions:"
find results/deep_depth_analysis -name "*.done" -mmin -5 2>/dev/null | tail -10 | \
    xargs -I {} basename {} .done | sed 's/^/  - /'

echo ""
echo "========================================"
echo "Estimated completion: $(echo "scale=1; (315 - $total_completed) / 16 * 3" | bc) minutes"
echo "========================================"
