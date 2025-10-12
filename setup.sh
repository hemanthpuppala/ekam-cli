#!/bin/bash
# VLM CLI Setup Script
# Automated setup for VLM testing tool

set -e

echo "╔══════════════════════════════════════════════════════════╗"
echo "║          VLM CLI Setup - Quick Installation              ║"
echo "╚══════════════════════════════════════════════════════════╝"
echo ""

# Check Python version
echo "📋 Checking Python version..."
PYTHON_VERSION=$(python3 --version 2>&1 | awk '{print $2}')
echo "   Found: Python $PYTHON_VERSION"

# Check if we're in the right directory
if [ ! -f "vlm_cli.py" ]; then
    echo "❌ Error: Please run this script from the vlm-tester directory"
    exit 1
fi

# Install dependencies
echo ""
echo "📦 Installing dependencies..."
pip3 install -r requirements.txt

# Create results directory
echo ""
echo "📁 Creating results directory..."
mkdir -p results
echo "   ✓ Created: ./results"

# Check Ollama (optional)
echo ""
echo "🔍 Checking for Ollama..."
if command -v ollama &> /dev/null; then
    echo "   ✓ Ollama found: $(ollama --version)"
    echo ""
    echo "   Available Ollama models:"
    ollama list | head -n 5
else
    echo "   ⚠️  Ollama not found (optional)"
    echo "   To install: https://ollama.ai"
fi

# Check CUDA (optional)
echo ""
echo "🎮 Checking for GPU support..."
python3 -c "import torch; print('   ✓ PyTorch found:', torch.__version__); print('   CUDA available:', torch.cuda.is_available())" 2>/dev/null || echo "   ⚠️  PyTorch not fully configured"

# Summary
echo ""
echo "╔══════════════════════════════════════════════════════════╗"
echo "║                    Setup Complete! 🎉                    ║"
echo "╚══════════════════════════════════════════════════════════╝"
echo ""
echo "Quick Start:"
echo "  1. Run the CLI:       python3 vlm_cli.py"
echo "  2. Run the example:   python3 example.py"
echo "  3. Read the docs:     cat README.md"
echo ""
echo "Default model: qwen2.5vl:3b (Ollama - already downloaded!)"
echo ""
