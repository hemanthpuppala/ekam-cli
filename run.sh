#!/usr/bin/env bash

# VLM/LLM CLI Launcher Script
# Handles prerequisites, environment setup, and application launch

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}=== VLM/LLM CLI Launcher ===${NC}"
echo

# Check Python version (require 3.8+)
echo -e "${BLUE}[1/5]${NC} Checking Python version..."
if ! command -v python3 &> /dev/null; then
    echo -e "${RED}Error: python3 not found${NC}"
    echo "Please install Python 3.8 or higher"
    exit 1
fi

PYTHON_VERSION=$(python3 -c 'import sys; print(".".join(map(str, sys.version_info[:2])))')
REQUIRED_VERSION="3.8"

if ! python3 -c "import sys; exit(0 if sys.version_info >= (3,8) else 1)"; then
    echo -e "${RED}Error: Python ${PYTHON_VERSION} is too old${NC}"
    echo "Required: Python ${REQUIRED_VERSION}+"
    exit 1
fi

echo -e "${GREEN}✓${NC} Python ${PYTHON_VERSION} found"

# Create virtual environment if it doesn't exist
echo -e "${BLUE}[2/5]${NC} Checking virtual environment..."
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
    echo -e "${GREEN}✓${NC} Virtual environment created"
else
    echo -e "${GREEN}✓${NC} Virtual environment exists"
fi

# Activate virtual environment
echo -e "${BLUE}[3/5]${NC} Activating virtual environment..."
source venv/bin/activate
echo -e "${GREEN}✓${NC} Virtual environment activated"

# Install/update dependencies
echo -e "${BLUE}[4/5]${NC} Checking dependencies..."
if [ ! -f "venv/.dependencies_installed" ] || [ "requirements.txt" -nt "venv/.dependencies_installed" ]; then
    echo "Installing dependencies (this may take a few minutes)..."
    pip install --upgrade pip > /dev/null 2>&1
    pip install -r requirements.txt > /dev/null 2>&1
    touch venv/.dependencies_installed
    echo -e "${GREEN}✓${NC} Dependencies installed"
else
    echo -e "${GREEN}✓${NC} Dependencies up to date"
fi

# Create config.yaml if it doesn't exist
echo -e "${BLUE}[5/5]${NC} Preparing configuration..."
if [ ! -f "config.yaml" ]; then
    echo -e "${YELLOW}⚠${NC}  config.yaml not found"
    if [ -f "config.yaml.example" ]; then
        echo "Creating config.yaml from template..."
        cp config.yaml.example config.yaml
        echo -e "${GREEN}✓${NC} Created config.yaml"
    else
        echo -e "${YELLOW}⚠${NC}  Will use default configuration"
    fi
else
    echo -e "${GREEN}✓${NC} Configuration file found"
fi

echo

# Provider detection and status
echo -e "${BLUE}=== Provider Status ===${NC}"
echo

# Check Ollama
echo -n "Ollama: "
if command -v ollama &> /dev/null; then
    if curl -s http://localhost:11434/api/tags > /dev/null 2>&1; then
        echo -e "${GREEN}✓ Running${NC}"
        OLLAMA_STATUS="running"
    else
        echo -e "${YELLOW}⚠ Installed but not running${NC}"
        echo -e "  ${BLUE}→${NC} Start with: ${YELLOW}ollama serve${NC}"
        OLLAMA_STATUS="installed"
    fi
else
    echo -e "${RED}✗ Not installed${NC}"
    echo -e "  ${BLUE}→${NC} Install from: ${YELLOW}https://ollama.ai${NC}"
    OLLAMA_STATUS="missing"
fi

# Check LM Studio
echo -n "LM Studio: "
if curl -s http://localhost:1234/v1/models > /dev/null 2>&1; then
    echo -e "${GREEN}✓ Running${NC}"
    LMSTUDIO_STATUS="running"
else
    echo -e "${YELLOW}⚠ Not running${NC}"
    echo -e "  ${BLUE}→${NC} Optional provider (not required)"
    LMSTUDIO_STATUS="not_running"
fi

# Check Python packages
echo -n "HuggingFace (transformers): "
if python3 -c "import transformers" 2>/dev/null; then
    echo -e "${GREEN}✓ Installed${NC}"
    HF_STATUS="installed"
else
    echo -e "${YELLOW}⚠ Not installed${NC}"
    echo -e "  ${BLUE}→${NC} Optional provider (not required)"
    HF_STATUS="missing"
fi

echo -n "GGUF (llama-cpp-python): "
if python3 -c "import llama_cpp" 2>/dev/null; then
    echo -e "${GREEN}✓ Installed${NC}"
    GGUF_STATUS="installed"
else
    echo -e "${YELLOW}⚠ Not installed${NC}"
    echo -e "  ${BLUE}→${NC} Optional provider (not required)"
    GGUF_STATUS="missing"
fi

echo

# Warn if no providers available
if [ "$OLLAMA_STATUS" = "missing" ] && [ "$LMSTUDIO_STATUS" != "running" ] && [ "$HF_STATUS" = "missing" ] && [ "$GGUF_STATUS" = "missing" ]; then
    echo -e "${RED}⚠ WARNING: No providers available!${NC}"
    echo -e "  Install Ollama to get started: ${YELLOW}https://ollama.ai${NC}"
    echo
    read -p "Continue anyway? [y/N]: " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        echo "Setup cancelled"
        deactivate
        exit 1
    fi
fi

# Launch application
echo
echo -e "${BLUE}=== Launching VLM/LLM CLI ===${NC}"
echo -e "${BLUE}→${NC} The application will automatically start Ollama if installed"
echo
python3 -m src.core.app

# Deactivate virtual environment on exit
deactivate
