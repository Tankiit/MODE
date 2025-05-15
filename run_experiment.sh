#!/bin/bash

# Script to easily run SPARROW experiments with timm models

# Default parameters
DATA_DIR="./data"
OUTPUT_DIR="./timm_experiment_results"
EPOCHS=30
DEVICE="cuda"
MODEL="resnet18"
LR=0.001

# Help function
show_help() {
    echo "Usage: $0 [options]"
    echo ""
    echo "Options:"
    echo "  -d, --dataset DATASET    Dataset to use (cifar10, cifar100, svhn, all)"
    echo "  -m, --model MODEL        timm model name (default: resnet18)"
    echo "  -e, --epochs EPOCHS      Number of epochs (default: 30)"
    echo "  -l, --lr LR              Learning rate (default: 0.001)"
    echo "  -o, --output DIR         Output directory (default: ./timm_experiment_results)"
    echo "  -g, --gpu DEVICE         GPU device (default: cuda)"
    echo "  -f, --full               Run full comparison experiments"
    echo "  -a, --ablation           Run ablation studies"
    echo "  -b, --both               Run both full and ablation"
    echo "  -h, --help               Show this help message"
    echo ""
    echo "Popular timm models:"
    echo "  - resnet18, resnet34, resnet50"
    echo "  - efficientnet_b0, efficientnet_b1, efficientnet_b2"
    echo "  - vit_tiny_patch16_224, vit_small_patch16_224, vit_base_patch16_224"
    echo "  - swin_tiny_patch4_window7_224, swin_small_patch4_window7_224"
    echo "  - mobilenetv3_small_100, mobilenetv3_large_100"
    echo ""
    echo "Examples:"
    echo "  $0 -d cifar10 -m resnet18 -f                    # Run full experiments on CIFAR-10 with ResNet-18"
    echo "  $0 -d all -m efficientnet_b0 -b                 # Run both full and ablation on all datasets with EfficientNet-B0"
    echo "  $0 -d cifar100 -m vit_small_patch16_224 -a      # Run ablation on CIFAR-100 with ViT-Small"
}

# Parse command line arguments
DATASET=""
MODE=""

while [ $# -gt 0 ]; do
    case $1 in
        -d|--dataset)
            DATASET="$2"
            shift 2
            ;;
        -m|--model)
            MODEL="$2"
            shift 2
            ;;
        -e|--epochs)
            EPOCHS="$2"
            shift 2
            ;;
        -l|--lr)
            LR="$2"
            shift 2
            ;;
        -o|--output)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        -g|--gpu)
            DEVICE="$2"
            shift 2
            ;;
        -f|--full)
            MODE="full"
            shift
            ;;
        -a|--ablation)
            MODE="ablation"
            shift
            ;;
        -b|--both)
            MODE="both"
            shift
            ;;
        -h|--help)
            show_help
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            show_help
            exit 1
            ;;
    esac
done

# Check if dataset is specified
if [ -z "$DATASET" ]; then
    echo "Error: Dataset must be specified"
    show_help
    exit 1
fi

# Check if mode is specified
if [ -z "$MODE" ]; then
    echo "Error: Experiment mode must be specified (--full, --ablation, or --both)"
    show_help
    exit 1
fi

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Function to run single dataset experiment
run_single_dataset() {
    local dataset=$1
    local mode=$2
    
    echo "----------------------------------------"
    echo "Running $mode experiments on $dataset with $MODEL"
    echo "----------------------------------------"
    
    python sparrow_experiment_with_timm.py \
        --dataset "$dataset" \
        --model "$MODEL" \
        --data-dir "$DATA_DIR" \
        --output-dir "$OUTPUT_DIR" \
        --epochs "$EPOCHS" \
        --lr "$LR" \
        --device "$DEVICE" \
        --mode "$mode"
    
    if [ $? -eq 0 ]; then
        echo "✓ Completed $mode experiments on $dataset with $MODEL"
    else
        echo "✗ Failed $mode experiments on $dataset with $MODEL"
    fi
}

# Check if timm model is available
echo "Checking if $MODEL is available in timm..."
python -c "import timm; print(timm.create_model('$MODEL', pretrained=False))" > /dev/null 2>&1
if [ $? -ne 0 ]; then
    echo "Error: Model '$MODEL' is not available in timm."
    echo "Available models can be listed with: python -m timm.list_models"
    exit 1
fi

# Run experiments
if [ "$DATASET" = "all" ]; then
    echo "Running experiments on all datasets with $MODEL..."
    
    # Create comprehensive experiment runner for timm
    cat > run_all_timm_experiments.py << 'EOF'
#!/usr/bin/env python3
"""
Comprehensive experiment runner for MODE with timm models
"""

import os
import sys
import argparse
import subprocess
import json
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
import numpy as np
from pathlib import Path

def run_command(cmd):
    """Run a command and capture output"""
    print(f"Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"Error running command: {result.stderr}")
        return False
    return True

def main():
    parser = argparse.ArgumentParser(description='Run comprehensive SPARROW experiments with timm models')
    
    # Base parameters
    parser.add_argument('--data-dir', type=str, default='./data',
                        help='Path to dataset')
    parser.add_argument('--output-base-dir', type=str, default='./comprehensive_timm_results',
                        help='Base output directory for results')
    parser.add_argument('--epochs', type=int, default=30,
                        help='Number of epochs for training')
    parser.add_argument('--device', type=str, default=None,
                        help='Device to use (cuda, mps, cpu)')
    parser.add_argument('--model', type=str, default='resnet18',
                        help='timm model name')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='Learning rate')
    
    # Experiment control
    parser.add_argument('--datasets', type=str, nargs='+', 
                        default=['cifar10', 'cifar100', 'svhn'],
                        help='Datasets to run experiments on')
    parser.add_argument('--run-full', action='store_true',
                        help='Run full comparison experiments')
    parser.add_argument('--run-ablation', action='store_true',
                        help='Run ablation studies')
    parser.add_argument('--all', action='store_true',
                        help='Run both full and ablation experiments')
    
    args = parser.parse_args()
    
    # Create output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_output_dir = os.path.join(args.output_base_dir, f"comprehensive_exp_{args.model}_{timestamp}")
    os.makedirs(base_output_dir, exist_ok=True)
    
    # Save arguments
    with open(os.path.join(base_output_dir, "experiment_args.json"), "w") as f:
        json.dump(vars(args), f, indent=4)
    
    # Track all results
    all_results = {
        'full_experiments': {},
        'ablation_studies': {}
    }
    
    # Run experiments for each dataset
    for dataset in args.datasets:
        print(f"\n{'='*60}")
        print(f"Running experiments for {dataset.upper()} with {args.model}")
        print(f"{'='*60}\n")
        
        dataset_output_dir = os.path.join(base_output_dir, dataset)
        os.makedirs(dataset_output_dir, exist_ok=True)
        
        # Base command
        base_cmd = [
            'python', 'sparrow_experiment_with_timm.py',
            '--dataset', dataset,
            '--model', args.model,
            '--data-dir', args.data_dir,
            '--output-dir', dataset_output_dir,
            '--epochs', str(args.epochs),
            '--lr', str(args.lr)
        ]
        
        if args.device:
            base_cmd.extend(['--device', args.device])
        
        # Run full experiments
        if args.run_full or args.all:
            print(f"\nRunning full experiments for {dataset}...")
            full_cmd = base_cmd + ['--mode', 'full']
            if run_command(full_cmd):
                print(f"Full experiments completed for {dataset}")
                all_results['full_experiments'][dataset] = 'completed'
            else:
                print(f"Full experiments failed for {dataset}")
                all_results['full_experiments'][dataset] = 'failed'
        
        # Run ablation studies
        if args.run_ablation or args.all:
            print(f"\nRunning ablation studies for {dataset}...")
            ablation_cmd = base_cmd + ['--mode', 'ablation']
            if run_command(ablation_cmd):
                print(f"Ablation studies completed for {dataset}")
                all_results['ablation_studies'][dataset] = 'completed'
            else:
                print(f"Ablation studies failed for {dataset}")
                all_results['ablation_studies'][dataset] = 'failed'
    
    # Generate comprehensive analysis
    print(f"\n{'='*60}")
    print("Generating comprehensive analysis...")
    print(f"{'='*60}\n")
    
    # Generate analysis (similar to previous script but for timm)
    analyze_results(base_output_dir, args.datasets, args.model)
    
    # Save completion status
    with open(os.path.join(base_output_dir, "completion_status.json"), "w") as f:
        json.dump(all_results, f, indent=4)
    
    print(f"\nAll experiments completed! Results saved to {base_output_dir}")

def analyze_results(base_output_dir, datasets, model_name):
    """Generate comprehensive analysis across all datasets for timm model"""
    analysis_dir = os.path.join(base_output_dir, "comprehensive_analysis")
    os.makedirs(analysis_dir, exist_ok=True)
    
    # Your analysis code here...
    # Similar to the previous analyze_results function but adapted for timm

if __name__ == "__main__":
    main()
EOF

    # Make the script executable
    chmod +x run_all_timm_experiments.py
    
    # Run comprehensive experiments
    if [ "$MODE" = "both" ]; then
        python run_all_timm_experiments.py \
            --data-dir "$DATA_DIR" \
            --output-base-dir "$OUTPUT_DIR" \
            --epochs "$EPOCHS" \
            --model "$MODEL" \
            --lr "$LR" \
            --device "$DEVICE" \
            --all
    elif [ "$MODE" = "full" ]; then
        python run_all_timm_experiments.py \
            --data-dir "$DATA_DIR" \
            --output-base-dir "$OUTPUT_DIR" \
            --epochs "$EPOCHS" \
            --model "$MODEL" \
            --lr "$LR" \
            --device "$DEVICE" \
            --run-full
    elif [ "$MODE" = "ablation" ]; then
        python run_all_timm_experiments.py \
            --data-dir "$DATA_DIR" \
            --output-base-dir "$OUTPUT_DIR" \
            --epochs "$EPOCHS" \
            --model "$MODEL" \
            --lr "$LR" \
            --device "$DEVICE" \
            --run-ablation
    fi
else
    # Run single dataset
    run_single_dataset "$DATASET" "$MODE"
fi

echo ""
echo "All experiments completed!"
echo "Results saved to: $OUTPUT_DIR"

# Show summary of results
if [ "$DATASET" != "all" ]; then
    echo ""
    echo "----------------------------------------"
    echo "Experiment Summary"
    echo "----------------------------------------"
    echo "Dataset: $DATASET"
    echo "Model: $MODEL"
    echo "Mode: $MODE"
    echo "Results directory: $OUTPUT_DIR"
    
    # Find the latest experiment directory
    LATEST_EXP=$(find "$OUTPUT_DIR" -maxdepth 2 -name "sparrow_timm_${MODEL}_${DATASET}_*" -type d | sort -r | head -n 1)
    
    if [ -n "$LATEST_EXP" ]; then
        RESULTS_CSV="$LATEST_EXP/results_summary.csv"
        if [ -f "$RESULTS_CSV" ]; then
            echo ""
            echo "Final Results:"
            echo "----------------------------------------"
            python -c "
import pandas as pd
df = pd.read_csv('$RESULTS_CSV')
print(df.to_string(index=False))
"
        fi
    fi
fi