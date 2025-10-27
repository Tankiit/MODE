#!/bin/bash
# Monitor experiment progress

echo "==================================="
echo "CIFAR-10 Experiment Monitor"
echo "==================================="
echo

# Check if experiment is running
if ps aux | grep -q "[p]ython run_cifar10_quick.py"; then
    echo "✓ Experiment is RUNNING"
    echo

    # Show recent log output
    echo "Recent logs:"
    echo "-----------------------------------"
    tail -n 30 nohup.out 2>/dev/null || echo "No log file found yet"
    echo

    # Check for results directory
    if [ -d "experiment_results" ]; then
        echo "Results directory exists:"
        ls -lh experiment_results/
        echo
    fi

    # Check for checkpoints
    if [ -d "checkpoints" ]; then
        echo "Checkpoints directory exists:"
        ls -lh checkpoints/
        echo
    fi

else
    echo "✗ Experiment is NOT running"
    echo

    # Check if it completed
    if [ -d "experiment_results" ]; then
        echo "✓ Results found - experiment may have completed"
        find experiment_results -name "*.csv" -o -name "*.json" | head -10
        echo
    fi
fi

echo "==================================="
