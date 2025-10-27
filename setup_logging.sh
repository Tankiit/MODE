#!/bin/bash
# Setup script for TensorBoard & WandB logging

echo "=========================================="
echo "Setting up Logging Infrastructure"
echo "=========================================="
echo

# Check Python version
python_version=$(python --version 2>&1)
echo "✓ Python version: $python_version"
echo

# Install dependencies
echo "Installing dependencies..."
pip install -r requirements_logging.txt

echo
echo "=========================================="
echo "Installation Complete!"
echo "=========================================="
echo

# Check installations
echo "Checking installations:"
echo

if python -c "import torch" 2>/dev/null; then
    echo "✓ PyTorch installed"
else
    echo "✗ PyTorch NOT installed"
fi

if python -c "import tensorboard" 2>/dev/null; then
    echo "✓ TensorBoard installed"
else
    echo "✗ TensorBoard NOT installed"
fi

if python -c "import wandb" 2>/dev/null; then
    echo "✓ WandB installed"
    echo "  Run 'wandb login' to authenticate"
else
    echo "⚠  WandB NOT installed (optional)"
fi

if python -c "import timm" 2>/dev/null; then
    echo "✓ timm installed"
else
    echo "✗ timm NOT installed"
fi

echo
echo "=========================================="
echo "Next Steps:"
echo "=========================================="
echo
echo "1. (Optional) Setup WandB:"
echo "   wandb login"
echo
echo "2. Run quick test:"
echo "   python run_with_logging.py"
echo
echo "3. View TensorBoard (in another terminal):"
echo "   tensorboard --logdir=./runs"
echo
echo "4. Open browser:"
echo "   http://localhost:6006"
echo
echo "=========================================="
