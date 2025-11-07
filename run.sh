#!/usr/bin/env bash

# VLM/LLM CLI Launcher Script (Denali Version)
# Handles prerequisites, environment setup, and application launch
# Cross-platform: Linux, macOS, Windows (Git Bash)

set -e  # Exit on error

# Detect OS
if [[ "$OSTYPE" == "msys" ]] || [[ "$OSTYPE" == "win32" ]] || [[ "$OSTYPE" == "cygwin" ]]; then
    OS="windows"
    VENV_ACTIVATE="ekam-venv/Scripts/activate"
elif [[ "$OSTYPE" == "darwin"* ]]; then
    OS="mac"
    VENV_ACTIVATE="ekam-venv/bin/activate"
else
    OS="linux"
    VENV_ACTIVATE="ekam-venv/bin/activate"
fi

# Detect Python command (prefer python3.12, then python3, then python)
if command -v python3.12 &> /dev/null; then
    PYTHON_CMD="python3.12"
elif command -v python3 &> /dev/null; then
    PYTHON_CMD="python3"
elif command -v python &> /dev/null; then
    PYTHON_CMD="python"
else
    PYTHON_CMD=""
fi

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}=== VLM/LLM CLI Launcher (Denali) ===${NC}"
echo

# Check Python version (require 3.8+)
echo -e "${BLUE}[1/6]${NC} Checking Python version..."
if [ -z "$PYTHON_CMD" ]; then
    echo -e "${RED}Error: Python not found${NC}"
    echo "Please install Python 3.8 or higher"
    if [ "$OS" = "windows" ]; then
        echo "Download from: https://www.python.org/downloads/"
        echo "Make sure to check 'Add Python to PATH' during installation"
    fi
    exit 1
fi

PYTHON_VERSION=$($PYTHON_CMD -c 'import sys; print(".".join(map(str, sys.version_info[:2])))')
REQUIRED_VERSION="3.8"

if ! $PYTHON_CMD -c "import sys; exit(0 if sys.version_info >= (3,8) else 1)"; then
    echo -e "${RED}Error: Python ${PYTHON_VERSION} is too old${NC}"
    echo "Required: Python ${REQUIRED_VERSION}+"
    exit 1
fi

echo -e "${GREEN}✓${NC} Python ${PYTHON_VERSION} found"

# Create virtual environment if it doesn't exist
echo -e "${BLUE}[2/6]${NC} Checking virtual environment..."
if [ ! -d "ekam-venv" ]; then
    echo "Creating virtual environment..."
    $PYTHON_CMD -m venv ekam-venv
    # Remove any AppleDouble files that might cause issues
    if [ "$OS" = "mac" ]; then
        find ekam-venv -name "._*" -delete 2>/dev/null || true
    fi
    echo -e "${GREEN}✓${NC} Virtual environment created"
else
    echo -e "${GREEN}✓${NC} Virtual environment exists"
fi

# Activate virtual environment
echo -e "${BLUE}[3/6]${NC} Activating virtual environment..."
if [ -f "$VENV_ACTIVATE" ]; then
    source "$VENV_ACTIVATE"
    echo -e "${GREEN}✓${NC} Virtual environment activated"
else
    echo -e "${RED}Error: Virtual environment activation script not found${NC}"
    echo "Expected: $VENV_ACTIVATE"
    exit 1
fi

# Install/update dependencies
echo -e "${BLUE}[4/6]${NC} Checking dependencies..."
if [ ! -f "ekam-venv/.dependencies_installed" ] || [ "requirements.txt" -nt "ekam-venv/.dependencies_installed" ]; then
    echo "Installing dependencies (this may take a few minutes)..."
    pip install --upgrade pip > /dev/null 2>&1
    pip install -r requirements.txt > /dev/null 2>&1
    touch ekam-venv/.dependencies_installed
    echo -e "${GREEN}✓${NC} Dependencies installed"
else
    echo -e "${GREEN}✓${NC} Dependencies up to date"
fi

# Create config.yaml if it doesn't exist
echo -e "${BLUE}[5/6]${NC} Preparing configuration..."
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

# System specs check
echo -e "${BLUE}[6/6]${NC} Detecting system specifications..."
SYS_SPECS=$($PYTHON_CMD -c "
try:
    import psutil
    import platform
    mem = psutil.virtual_memory()
    print(f'{platform.system()}|{platform.machine()}|{psutil.cpu_count(logical=False)}|{psutil.cpu_count(logical=True)}|{mem.total/(1024**3):.1f}')
except:
    print('Unknown|Unknown|0|0|0')
" 2>/dev/null)

IFS='|' read -r SYS_PLATFORM SYS_ARCH SYS_CORES_PHYS SYS_CORES_LOG SYS_RAM <<< "$SYS_SPECS"

echo -e "${GREEN}✓${NC} Platform: $SYS_PLATFORM ($SYS_ARCH)"
echo -e "${GREEN}✓${NC} CPU: $SYS_CORES_PHYS cores ($SYS_CORES_LOG logical)"
echo -e "${GREEN}✓${NC} RAM: ${SYS_RAM} GB"

# Check temperature if available
TEMP_CHECK=$($PYTHON_CMD -c "
try:
    import psutil
    temps = psutil.sensors_temperatures()
    if temps:
        for name, entries in temps.items():
            for entry in entries:
                if hasattr(entry, 'current'):
                    print(f'{entry.current:.1f}')
                    break
            break
    else:
        print('N/A')
except:
    print('N/A')
" 2>/dev/null)

if [ "$TEMP_CHECK" != "N/A" ]; then
    if (( $(echo "$TEMP_CHECK >= 85" | bc -l 2>/dev/null || echo 0) )); then
        echo -e "${RED}⚠${NC}  CPU Temp: ${TEMP_CHECK}°C (High!)"
    elif (( $(echo "$TEMP_CHECK >= 70" | bc -l 2>/dev/null || echo 0) )); then
        echo -e "${YELLOW}⚠${NC}  CPU Temp: ${TEMP_CHECK}°C"
    else
        echo -e "${GREEN}✓${NC} CPU Temp: ${TEMP_CHECK}°C"
    fi
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
if $PYTHON_CMD -c "import transformers" 2>/dev/null; then
    echo -e "${GREEN}✓ Installed${NC}"
    HF_STATUS="installed"
else
    echo -e "${YELLOW}⚠ Not installed${NC}"
    echo -e "  ${BLUE}→${NC} Optional provider (not required)"
    HF_STATUS="missing"
fi

echo -n "GGUF (llama-cpp-python): "
if $PYTHON_CMD -c "import llama_cpp" 2>/dev/null; then
    echo -e "${GREEN}✓ Installed${NC}"
    GGUF_STATUS="installed"
else
    echo -e "${YELLOW}⚠ Not installed${NC}"
    echo -e "  ${BLUE}→${NC} Run: ${YELLOW}pip install -r requirements.txt${NC}"
    GGUF_STATUS="missing"
fi

echo -n "GGUF Quantization Tools: "
if command -v llama-quantize &> /dev/null; then
    echo -e "${GREEN}✓ Available${NC}"
    echo -e "  ${BLUE}→${NC} llama.cpp tools installed"
    GGUF_QUANT_STATUS="available"
elif $PYTHON_CMD -c "import llama_cpp; exit(0 if hasattr(llama_cpp, 'llama_model_quantize') else 1)" 2>/dev/null; then
    echo -e "${YELLOW}⚠ Partial support${NC}"
    echo -e "  ${BLUE}→${NC} llama-cpp-python installed but llama.cpp CLI tools missing"
    if [ "$OS" = "mac" ]; then
        echo -e "  ${BLUE}→${NC} HF→GGUF conversion requires: ${YELLOW}brew install llama.cpp${NC}"
    elif [ "$OS" = "linux" ]; then
        echo -e "  ${BLUE}→${NC} HF→GGUF conversion requires: ${YELLOW}Build llama.cpp from source${NC}"
    else
        echo -e "  ${BLUE}→${NC} HF→GGUF conversion requires: ${YELLOW}Build llama.cpp from source${NC}"
    fi
    echo -e "  ${BLUE}→${NC} Use Generic quantization for HF models instead"
    GGUF_QUANT_STATUS="partial"
else
    echo -e "${YELLOW}⚠ Not available${NC}"
    echo -e "  ${BLUE}→${NC} GGUF quantization will not be available"
    if [ "$OS" = "mac" ]; then
        echo -e "  ${BLUE}→${NC} Install: ${YELLOW}brew install llama.cpp${NC}"
    elif [ "$OS" = "linux" ]; then
        echo -e "  ${BLUE}→${NC} Install: ${YELLOW}Build llama.cpp from source${NC}"
    else
        echo -e "  ${BLUE}→${NC} Install: ${YELLOW}Build llama.cpp from source${NC}"
    fi
    echo -e "  ${BLUE}→${NC} Or use Generic quantization instead"
    GGUF_QUANT_STATUS="missing"
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
$PYTHON_CMD -m src.core.app

# Deactivate virtual environment on exit
deactivate
