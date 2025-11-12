#!/usr/bin/env bash

# EKAM CLI - Setup Script
# One-time setup for development environment
# Creates venv, installs dependencies, and optionally installs global CLI command

# Note: We do NOT use set -e here because we want to continue even if non-critical steps fail
# User can always Ctrl+C to exit at any time

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# Get script directory (project root)
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

# Configuration
VENV_DIR="$SCRIPT_DIR/ekam-venv"
MIN_PYTHON_VERSION="3.10"
ENTRY_POINT="src.core.app"

# Helper functions
print_header() {
    echo -e "\n${BLUE}═══════════════════════════════════════════════════════════${NC}"
    echo -e "${BLUE}$1${NC}"
    echo -e "${BLUE}═══════════════════════════════════════════════════════════${NC}\n"
}

print_section() {
    echo -e "\n${CYAN}▶ $1${NC}"
}

print_success() {
    echo -e "${GREEN}✓${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}⚠${NC} $1"
}

print_error() {
    echo -e "${RED}✗${NC} $1"
}

# Detect OS
detect_os() {
    # Check if running in WSL (Windows Subsystem for Linux)
    # Method 1: Check /proc/version
    if [ -f /proc/version ] && grep -qEi "(Microsoft|WSL)" /proc/version 2>/dev/null; then
        echo "wsl"
        return 0
    fi
    # Method 2: Check for WSL environment variable
    if [ -n "$WSL_DISTRO_NAME" ] || [ -n "$WSLENV" ]; then
        echo "wsl"
        return 0
    fi
    # Method 3: Check for /mnt/c directory (typical WSL mount)
    if [ -d /mnt/c ] && [ -d /mnt/c/Windows ]; then
        echo "wsl"
        return 0
    fi
    # Not WSL - check other OS types
    if [[ "$OSTYPE" == "msys" ]] || [[ "$OSTYPE" == "win32" ]] || [[ "$OSTYPE" == "cygwin" ]]; then
        echo "windows"
    elif [[ "$OSTYPE" == "darwin"* ]]; then
        echo "macos"
    else
        echo "linux"
    fi
}

# Find Python command
find_python() {
    if command -v python3.12 &> /dev/null; then
        echo "python3.12"
    elif command -v python3.11 &> /dev/null; then
        echo "python3.11"
    elif command -v python3.10 &> /dev/null; then
        echo "python3.10"
    elif command -v python3 &> /dev/null; then
        echo "python3"
    elif command -v python &> /dev/null; then
        echo "python"
    else
        echo ""
    fi
}

# Check Python version
check_python_version() {
    local python_cmd=$1
    local version_str=$($python_cmd -c 'import sys; print(".".join(map(str, sys.version_info[:2])))')

    if ! $python_cmd -c "import sys; exit(0 if sys.version_info >= (3,10) else 1)" 2>/dev/null; then
        print_error "Python ${version_str} is too old (require ${MIN_PYTHON_VERSION}+)"
        return 1
    fi

    echo "$version_str"
}

# Main setup
print_header "EKAM CLI - Development Setup"

# Check if already in a venv and warn user
if [ -n "$VIRTUAL_ENV" ]; then
    print_warning "You are already in a virtual environment: $VIRTUAL_ENV"
    echo "It's recommended to deactivate it first:"
    echo -e "  ${YELLOW}deactivate${NC}"
    echo
    read -p "Continue anyway? [y/N]: " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        echo "Exiting setup. Please deactivate current venv and run again."
        exit 0
    fi
fi

OS=$(detect_os)
print_section "Detecting system..."
print_success "OS: $OS"

# Find Python
print_section "Checking Python installation..."
PYTHON_CMD=$(find_python)

if [ -z "$PYTHON_CMD" ]; then
    print_error "Python not found!"
    echo -e "  Please install Python ${MIN_PYTHON_VERSION}+ from https://python.org"
    exit 1
fi

PYTHON_VERSION=$(check_python_version "$PYTHON_CMD")
if [ $? -ne 0 ]; then
    exit 1
fi

print_success "Python $PYTHON_VERSION found"

# Create venv
print_section "Setting up virtual environment..."
if [ -d "$VENV_DIR" ]; then
    # Check if existing venv is compatible with current OS
    VENV_COMPATIBLE=true
    if [ "$OS" = "wsl" ] || [ "$OS" = "linux" ] || [ "$OS" = "macos" ]; then
        # Unix-like systems need bin/activate
        if [ ! -f "$VENV_DIR/bin/activate" ] && [ -f "$VENV_DIR/Scripts/activate" ]; then
            print_warning "Existing venv is Windows-style, but you're on $OS"
            print_warning "The venv needs to be recreated for $OS"
            VENV_COMPATIBLE=false
        fi
    elif [ "$OS" = "windows" ]; then
        # Windows needs Scripts/activate
        if [ ! -f "$VENV_DIR/Scripts/activate" ] && [ -f "$VENV_DIR/bin/activate" ]; then
            print_warning "Existing venv is Unix-style, but you're on Windows"
            print_warning "The venv needs to be recreated for Windows"
            VENV_COMPATIBLE=false
        fi
    fi

    if [ "$VENV_COMPATIBLE" = false ]; then
        print_warning "Virtual environment is incompatible with current OS"
        read -p "Recreate for $OS? [Y/n]: " -n 1 -r
        echo
        if [[ ! $REPLY =~ ^[Nn]$ ]]; then
            rm -rf "$VENV_DIR"
            $PYTHON_CMD -m venv "$VENV_DIR"
            print_success "Virtual environment recreated for $OS"
        else
            print_error "Cannot continue with incompatible venv"
            exit 1
        fi
    else
        print_warning "Virtual environment already exists at $VENV_DIR"
        read -p "Recreate? [y/N]: " -n 1 -r
        echo
        if [[ $REPLY =~ ^[Yy]$ ]]; then
            rm -rf "$VENV_DIR"
            $PYTHON_CMD -m venv "$VENV_DIR"
            print_success "Virtual environment recreated"
        else
            print_success "Using existing virtual environment"
        fi
    fi
else
    if $PYTHON_CMD -m venv "$VENV_DIR"; then
        print_success "Virtual environment created at $VENV_DIR"
    else
        print_error "Failed to create virtual environment"
        exit 1
    fi
fi

# Activate venv
print_section "Activating virtual environment..."
if [ "$OS" = "windows" ]; then
    ACTIVATE_SCRIPT="$VENV_DIR/Scripts/activate"
else
    # Linux, macOS, and WSL all use bin/activate
    ACTIVATE_SCRIPT="$VENV_DIR/bin/activate"
fi

if [ ! -f "$ACTIVATE_SCRIPT" ]; then
    print_error "Activation script not found at $ACTIVATE_SCRIPT"
    exit 1
fi

source "$ACTIVATE_SCRIPT"
print_success "Virtual environment activated"

# Upgrade pip
print_section "Upgrading pip..."
if pip install --upgrade pip setuptools wheel; then
    print_success "pip upgraded"
else
    print_warning "pip upgrade had issues, but continuing..."
fi

# Choose dependency installation method
print_section "Selecting dependency installation method..."
echo
echo "Which dependency file should we use?"
echo "  1) pyproject.toml (modern, recommended)"
echo "  2) requirements.txt (simple, compatible)"
echo

read -p "Enter choice [1-2]: " dep_choice

case $dep_choice in
    1)
        if [ -f "$SCRIPT_DIR/pyproject.toml" ]; then
            print_section "Installing from pyproject.toml..."
            if pip install -e .; then
                print_success "Dependencies installed from pyproject.toml"
            else
                print_error "Failed to install from pyproject.toml"
                echo "  Falling back to requirements.txt..."
                if [ -f "$SCRIPT_DIR/requirements.txt" ]; then
                    if pip install -r requirements.txt; then
                        print_success "Dependencies installed from requirements.txt"
                    else
                        print_error "Failed to install from requirements.txt"
                        print_warning "You may need to install dependencies manually"
                    fi
                else
                    print_error "Neither pyproject.toml nor requirements.txt found!"
                fi
            fi
        else
            print_error "pyproject.toml not found!"
            echo "  Falling back to requirements.txt..."
            if [ -f "$SCRIPT_DIR/requirements.txt" ]; then
                if pip install -r requirements.txt; then
                    print_success "Dependencies installed from requirements.txt"
                else
                    print_error "Failed to install from requirements.txt"
                    print_warning "You may need to install dependencies manually"
                fi
            else
                print_error "Neither pyproject.toml nor requirements.txt found!"
            fi
        fi
        ;;
    2)
        print_section "Installing from requirements.txt..."
        if [ -f "$SCRIPT_DIR/requirements.txt" ]; then
            if pip install -r requirements.txt; then
                print_success "Dependencies installed from requirements.txt"
            else
                print_error "Failed to install from requirements.txt"
                echo "  Falling back to pyproject.toml..."
                if [ -f "$SCRIPT_DIR/pyproject.toml" ]; then
                    if pip install -e .; then
                        print_success "Dependencies installed from pyproject.toml"
                    else
                        print_error "Failed to install from pyproject.toml"
                        print_warning "You may need to install dependencies manually"
                    fi
                else
                    print_error "Neither pyproject.toml nor requirements.txt found!"
                fi
            fi
        else
            print_error "requirements.txt not found!"
            echo "  Falling back to pyproject.toml..."
            if [ -f "$SCRIPT_DIR/pyproject.toml" ]; then
                if pip install -e .; then
                    print_success "Dependencies installed from pyproject.toml"
                else
                    print_error "Failed to install from pyproject.toml"
                    print_warning "You may need to install dependencies manually"
                fi
            else
                print_error "Neither pyproject.toml nor requirements.txt found!"
            fi
        fi
        ;;
    *)
        print_error "Invalid choice. Using default (requirements.txt)..."
        if [ -f "$SCRIPT_DIR/requirements.txt" ]; then
            if pip install -r requirements.txt; then
                print_success "Dependencies installed from requirements.txt"
            else
                print_error "Failed to install from requirements.txt"
                echo "  Falling back to pyproject.toml..."
                if [ -f "$SCRIPT_DIR/pyproject.toml" ]; then
                    if pip install -e .; then
                        print_success "Dependencies installed from pyproject.toml"
                    else
                        print_error "Failed to install from pyproject.toml"
                        print_warning "You may need to install dependencies manually"
                    fi
                else
                    print_error "Neither pyproject.toml nor requirements.txt found!"
                fi
            fi
        else
            print_error "requirements.txt not found!"
            if [ -f "$SCRIPT_DIR/pyproject.toml" ]; then
                if pip install -e .; then
                    print_success "Dependencies installed from pyproject.toml"
                else
                    print_error "Failed to install from pyproject.toml"
                    print_warning "You may need to install dependencies manually"
                fi
            else
                print_error "Neither pyproject.toml nor requirements.txt found!"
            fi
        fi
        ;;
esac

# Install optional quantization packages (GPTQ, AWQ)
print_section "Installing optional quantization support..."

# Try GPTQ
echo "Attempting to install gptqmodel (requires torch to be installed first)..."
if pip install gptqmodel 2>&1 | grep -v "WARNING"; then
    print_success "gptqmodel installed successfully"
else
    print_warning "Failed to install gptqmodel (this is optional)"
    print_warning "You can try installing it manually later with:"
    echo -e "  ${YELLOW}pip install gptqmodel${NC}"
fi

echo

# Try AWQ
echo "Attempting to install autoawq (requires torch to be installed first)..."
if pip install autoawq 2>&1 | grep -v "WARNING"; then
    print_success "autoawq installed successfully"
else
    print_warning "Failed to install autoawq (this is optional)"
    print_warning "You can try installing it manually later with:"
    echo -e "  ${YELLOW}pip install autoawq${NC}"
fi

# Create .gitignore entry
print_section "Updating .gitignore..."
if [ ! -f "$SCRIPT_DIR/.gitignore" ]; then
    echo "ekam-venv/" > "$SCRIPT_DIR/.gitignore"
    print_success ".gitignore created"
else
    if ! grep -q "^ekam-venv/$" "$SCRIPT_DIR/.gitignore"; then
        echo "ekam-venv/" >> "$SCRIPT_DIR/.gitignore"
        print_success "Added ekam-venv/ to .gitignore"
    else
        print_success ".gitignore already has ekam-venv/ entry"
    fi
fi

# Ollama installation and model setup (MANDATORY)
print_section "Setting up Ollama (LLM provider - REQUIRED)..."
echo

# Check if Ollama is installed
if command -v ollama &> /dev/null; then
    print_success "Ollama is already installed"
    OLLAMA_INSTALLED=true
else
    print_warning "Ollama not found. Installing..."
    OLLAMA_INSTALLED=false

    case $OS in
        macos)
            echo "Installing Ollama for macOS..."
            if command -v brew &> /dev/null; then
                if brew install ollama; then
                    print_success "Ollama installed via Homebrew"
                    OLLAMA_INSTALLED=true
                else
                    print_error "Failed to install Ollama via Homebrew"
                    echo "Please install manually from: ${YELLOW}https://ollama.ai/download${NC}"
                    exit 1
                fi
            else
                print_error "Homebrew not found. Please install Ollama manually:"
                echo -e "  ${YELLOW}https://ollama.ai/download${NC}"
                echo "  Then run: bash $SCRIPT_DIR/setup.sh"
                exit 1
            fi
            ;;
        wsl)
            echo "Installing Ollama for WSL (Windows Subsystem for Linux)..."
            if command -v curl &> /dev/null; then
                if curl -fsSL https://ollama.ai/install.sh | sh; then
                    print_success "Ollama installed in WSL"
                    OLLAMA_INSTALLED=true
                else
                    print_error "Ollama installation failed. Please install manually:"
                    echo -e "  ${YELLOW}curl -fsSL https://ollama.ai/install.sh | sh${NC}"
                    echo "Or install Ollama for Windows and it will work in WSL:"
                    echo -e "  ${YELLOW}https://ollama.ai/download${NC}"
                    exit 1
                fi
            else
                print_error "curl not found. Please install curl first:"
                echo -e "  ${YELLOW}sudo apt-get install curl${NC}"
                exit 1
            fi
            ;;
        linux)
            echo "Installing Ollama for Linux..."
            if command -v curl &> /dev/null; then
                if curl -fsSL https://ollama.ai/install.sh | sh; then
                    print_success "Ollama installed"
                    OLLAMA_INSTALLED=true
                else
                    print_error "Ollama installation failed. Please install manually:"
                    echo -e "  ${YELLOW}curl -fsSL https://ollama.ai/install.sh | sh${NC}"
                    exit 1
                fi
            else
                print_error "curl not found. Please install curl first:"
                echo -e "  ${YELLOW}sudo apt-get install curl${NC}"
                exit 1
            fi
            ;;
        windows)
            print_error "Windows requires manual Ollama installation"
            echo "Please download and install Ollama from:"
            echo -e "  ${YELLOW}https://ollama.ai/download${NC}"
            echo "After installing, run this script again in WSL or Git Bash"
            exit 1
            ;;
    esac
fi

# If Ollama is installed, set up model
if command -v ollama &> /dev/null; then
    print_section "Configuring Ollama model..."
    echo

    # Check if Ollama service is running
    if curl -s http://localhost:11434/api/tags > /dev/null 2>&1; then
        print_success "Ollama service is running"
    else
        print_warning "Ollama service is not running. Starting..."

        if [ "$OS" = "macos" ]; then
            # macOS: Ollama typically runs as launchd agent
            launchctl start com.ollama.OllamaServer 2>/dev/null || ollama serve > /dev/null 2>&1 &
            print_warning "Starting Ollama in background..."
            sleep 2
        else
            # Linux: Start ollama service
            sudo systemctl start ollama 2>/dev/null || ollama serve > /dev/null 2>&1 &
            print_warning "Starting Ollama in background..."
            sleep 2
        fi
    fi

    # Wait for Ollama to be ready
    MAX_ATTEMPTS=10
    ATTEMPT=0
    while ! curl -s http://localhost:11434/api/tags > /dev/null 2>&1; do
        ATTEMPT=$((ATTEMPT + 1))
        if [ $ATTEMPT -ge $MAX_ATTEMPTS ]; then
            print_warning "Ollama service not responding. You may need to start it manually with: ollama serve"
            break
        fi
        sleep 1
    done

    # Prompt for model selection
    echo
    echo -e "${CYAN}Ollama Model Selection:${NC}"
    echo
    echo "Choose a model to download:"
    echo "  • qwen3:0.5b   - Ultra-lightweight (fastest, ~500MB)"
    echo "  • qwen3:0.6b   - Lightweight (fast, ~600MB) [DEFAULT]"
    echo "  • qwen3:1.8b   - Balanced (good quality, ~1.8GB)"
    echo "  • qwen3:7b     - High quality (slower, ~7GB)"
    echo
    echo "Or enter a custom model ID (e.g., 'mistral', 'llama2:13b')"
    echo "See available models at: https://ollama.ai/library"
    echo
    read -p "Model name [qwen3:0.6b]: " model_input

    # Set default model
    MODEL_NAME="${model_input:-qwen3:0.6b}"

    # Validate model name format (should be alphanumeric with optional : and numbers)
    if [[ ! "$MODEL_NAME" =~ ^[a-zA-Z0-9]+$ ]] && [[ ! "$MODEL_NAME" =~ ^[a-zA-Z0-9]+:[a-zA-Z0-9.]+$ ]]; then
        print_warning "Invalid model ID format: $MODEL_NAME"
        print_warning "Using default: qwen3:0.6b"
        MODEL_NAME="qwen3:0.6b"
    fi

    # Pull the model
    print_section "Downloading model: $MODEL_NAME..."
    echo "(This may take several minutes depending on model size and internet speed)"
    echo

    if ollama pull "$MODEL_NAME" 2>&1; then
        print_success "Model '$MODEL_NAME' downloaded successfully"
    else
        print_warning "Failed to download '$MODEL_NAME'. Attempting default: qwen3:0.6b"
        if ollama pull qwen3:0.6b 2>&1; then
            print_success "Default model 'qwen3:0.6b' downloaded"
            MODEL_NAME="qwen3:0.6b"
        else
            print_warning "Could not download any model. You can try manually with:"
            echo -e "  ${YELLOW}ollama pull qwen3:0.6b${NC}"
        fi
    fi

    echo
else
    print_error "Ollama installation is required but not available."
    print_error "Please install Ollama manually and run setup again."
    echo -e "  ${YELLOW}https://ollama.ai/download${NC}"
    exit 1
fi

# HuggingFace token setup
print_section "HuggingFace Token (optional)..."
echo "Some models require HuggingFace authentication."
echo "You can add your token now or skip it."
echo
read -p "Enter HuggingFace token (or press Enter to skip): " hf_token

if [ -n "$hf_token" ]; then
    # Create or update .env file
    if [ ! -f "$SCRIPT_DIR/.env" ]; then
        touch "$SCRIPT_DIR/.env"
    fi

    # Check if HF_TOKEN already exists in .env
    if grep -q "^HF_TOKEN=" "$SCRIPT_DIR/.env"; then
        # Update existing HF_TOKEN
        if [[ "$OS" == "macos" ]]; then
            sed -i '' "s/^HF_TOKEN=.*/HF_TOKEN=$hf_token/" "$SCRIPT_DIR/.env"
        else
            sed -i "s/^HF_TOKEN=.*/HF_TOKEN=$hf_token/" "$SCRIPT_DIR/.env"
        fi
    else
        # Add new HF_TOKEN
        echo "HF_TOKEN=$hf_token" >> "$SCRIPT_DIR/.env"
    fi

    print_success "HuggingFace token saved to .env"
    print_warning "Make sure .env is in .gitignore (it should be by default)"
else
    print_warning "Skipped HuggingFace token setup"
fi

echo

# Setup complete
print_header "Setup Complete!"

echo -e "${GREEN}Your development environment is ready!${NC}\n"

# Show usage instructions
echo -e "${CYAN}Quick Start:${NC}"
echo "  1. Activate ekam-venv in current shell:"
echo -e "     ${YELLOW}source $VENV_DIR/bin/activate${NC} (macOS/Linux)"
echo -e "     ${YELLOW}source $VENV_DIR/Scripts/activate${NC} (Windows)"
echo
echo "  2. Run the application:"
echo -e "     ${YELLOW}python -m $ENTRY_POINT${NC}"
echo

# Ask about global CLI installation
echo -e "${CYAN}Global CLI Installation:${NC}"
echo "Install 'ekam-cli' as a global command for easy access from anywhere?"
echo
read -p "Install ekam-cli globally? [y/N]: " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    if [ -f "$SCRIPT_DIR/install-ekam-cli.sh" ]; then
        print_section "Installing ekam-cli globally..."
        bash "$SCRIPT_DIR/install-ekam-cli.sh" "$SCRIPT_DIR"
        if [ $? -eq 0 ]; then
            print_success "ekam-cli installed globally!"
            echo -e "\nYou can now run ${YELLOW}ekam-cli${NC} from anywhere"
        else
            print_warning "Global installation failed. You can try later with:"
            echo -e "  ${YELLOW}bash $SCRIPT_DIR/install-ekam-cli.sh $SCRIPT_DIR${NC}"
        fi
    else
        print_warning "install-ekam-cli.sh not found in project root"
        echo "Please create it first or run: bash install-ekam-cli.sh <project-directory>"
    fi
else
    echo -e "\nYou can install the global command later with:"
    echo -e "  ${YELLOW}bash $SCRIPT_DIR/install-ekam-cli.sh $SCRIPT_DIR${NC}"
fi

# Helper function: Install system dependencies based on OS
install_system_dependencies() {
    local deps_needed=()
    local cmake_installed=false
    local git_installed=false
    local build_tools_installed=false

    # Check what's missing
    if ! command -v cmake &> /dev/null; then
        deps_needed+=("cmake")
    else
        cmake_installed=true
    fi

    if ! command -v git &> /dev/null; then
        deps_needed+=("git")
    else
        git_installed=true
    fi

    if [ "$OS" != "macos" ] && ! command -v make &> /dev/null; then
        deps_needed+=("build-essential")
    else
        build_tools_installed=true
    fi

    # If nothing is missing, return success
    if [ ${#deps_needed[@]} -eq 0 ]; then
        print_success "All system dependencies are already installed"
        return 0
    fi

    # Install missing dependencies
    print_section "Installing missing system dependencies: ${deps_needed[*]}"
    echo

    case $OS in
        macos)
            if ! command -v brew &> /dev/null; then
                print_error "Homebrew not found. Please install Homebrew first:"
                echo -e "  ${YELLOW}/bin/bash -c \"\$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)\"${NC}"
                return 1
            fi

            echo "Installing via Homebrew..."
            for dep in "${deps_needed[@]}"; do
                if brew install "$dep" 2>&1 | grep -v "Warning"; then
                    print_success "$dep installed"
                else
                    print_warning "Failed to install $dep - you may need to install manually"
                fi
            done
            ;;

        linux)
            # Detect package manager
            if command -v apt-get &> /dev/null; then
                echo "Installing via apt-get..."
                if sudo apt-get update && sudo apt-get install -y "${deps_needed[@]}"; then
                    print_success "Dependencies installed successfully"
                else
                    print_error "Failed to install dependencies via apt-get"
                    return 1
                fi
            elif command -v yum &> /dev/null; then
                echo "Installing via yum..."
                if sudo yum install -y "${deps_needed[@]}"; then
                    print_success "Dependencies installed successfully"
                else
                    print_error "Failed to install dependencies via yum"
                    return 1
                fi
            elif command -v pacman &> /dev/null; then
                echo "Installing via pacman..."
                if sudo pacman -S --noconfirm "${deps_needed[@]}"; then
                    print_success "Dependencies installed successfully"
                else
                    print_error "Failed to install dependencies via pacman"
                    return 1
                fi
            else
                print_error "Could not detect package manager"
                print_warning "Please install manually: ${deps_needed[*]}"
                return 1
            fi
            ;;

        wsl)
            echo "Installing via apt-get (WSL)..."
            if sudo apt-get update && sudo apt-get install -y "${deps_needed[@]}"; then
                print_success "Dependencies installed successfully"
            else
                print_error "Failed to install dependencies"
                return 1
            fi
            ;;

        windows)
            print_error "Windows native installation not supported via bash"
            print_warning "Please install dependencies manually:"
            for dep in "${deps_needed[@]}"; do
                case $dep in
                    cmake) echo -e "  ${YELLOW}https://cmake.org/download/${NC}" ;;
                    git) echo -e "  ${YELLOW}https://git-scm.com/download/win${NC}" ;;
                esac
            done
            return 1
            ;;
    esac

    return 0
}

# Helper function: Detect GPU type and return build flags
detect_gpu_and_build_flags() {
    local gpu_type=""
    local gpu_flags=""
    local gpu_description=""

    case $OS in
        macos)
            # Check for Apple Silicon vs Intel
            if sysctl -a | grep -q "arm64"; then
                gpu_type="metal_silicon"
                gpu_flags="-DGGML_METAL=ON"
                gpu_description="Apple Silicon (Metal GPU)"
            else
                gpu_type="metal_intel"
                gpu_flags="-DGGML_METAL=ON"
                gpu_description="Intel Mac (Metal GPU)"
            fi
            ;;

        linux|wsl)
            # Check for NVIDIA GPU
            if command -v nvidia-smi &> /dev/null; then
                if lspci 2>/dev/null | grep -i nvidia &> /dev/null || nvidia-smi &> /dev/null; then
                    gpu_type="nvidia"
                    gpu_flags="-DGGML_CUDA=ON"
                    gpu_description="NVIDIA GPU (CUDA)"
                fi
            fi

            # Check for AMD GPU if NVIDIA not found
            if [ -z "$gpu_type" ]; then
                if command -v rocm-smi &> /dev/null || [ -d "/opt/rocm" ]; then
                    gpu_type="amd"
                    gpu_flags="-DGGML_HIPBLAS=ON"
                    gpu_description="AMD GPU (ROCm)"
                fi
            fi

            # Check for Intel Arc GPU
            if [ -z "$gpu_type" ]; then
                if lspci 2>/dev/null | grep -i "intel.*arc" &> /dev/null; then
                    gpu_type="intel_arc"
                    gpu_flags="-DGGML_ONEAPI=ON"
                    gpu_description="Intel Arc GPU (oneAPI)"
                fi
            fi

            # Default to CPU-only
            if [ -z "$gpu_type" ]; then
                gpu_type="cpu"
                gpu_flags=""
                gpu_description="CPU-only mode (no GPU detected)"
            fi
            ;;

        *)
            gpu_type="cpu"
            gpu_flags=""
            gpu_description="CPU-only mode"
            ;;
    esac

    echo "$gpu_type|$gpu_flags|$gpu_description"
}

# Main llama.cpp installation
print_section "Setting up llama.cpp (GGUF model support - REQUIRED)..."
echo

LLAMACPP_DIR="$SCRIPT_DIR/llama.cpp"
LLAMACPP_BINARY="$LLAMACPP_DIR/build/bin/llama-server"
LLAMACPP_INSTALL_FAILED=false
REBUILD_LLAMA=false

# Step 1: Check if llama.cpp directory exists
if [ -d "$LLAMACPP_DIR" ]; then
    print_success "llama.cpp directory found"

    # Update llama.cpp to latest version
    print_section "Updating llama.cpp to latest version..."
    echo "Checking for updates..."
    cd "$LLAMACPP_DIR"

    # Fetch latest changes
    if git fetch origin master 2>&1 | grep -v "Already up to date"; then
        # Check if there are updates
        LOCAL=$(git rev-parse HEAD)
        REMOTE=$(git rev-parse origin/master)

        if [ "$LOCAL" != "$REMOTE" ]; then
            print_warning "Updates available. Pulling latest changes..."
            if git pull origin master 2>&1; then
                print_success "llama.cpp updated to latest version"
                print_warning "Rebuild required due to updates"
                rm -rf "$LLAMACPP_DIR/build"
                # Remove patch marker so patches are reapplied after update
                rm -f "$LLAMACPP_DIR/.ekam_patches_applied"
                rm -f "$LLAMACPP_DIR/tools/mtmd/mtmd.cpp.ekam-backup"
                REBUILD_LLAMA=true
            else
                print_error "Failed to update llama.cpp. Continuing with existing version..."
            fi
        else
            print_success "llama.cpp is already up to date"
        fi
    fi

    cd "$SCRIPT_DIR"

    # Check if binary is built and working
    if [ -f "$LLAMACPP_BINARY" ] && [ "$REBUILD_LLAMA" != true ]; then
        print_success "llama-server binary already exists"
        echo "Testing binary..."
        if "$LLAMACPP_BINARY" --version &> /dev/null; then
            print_success "llama-server is working correctly"
        else
            print_warning "Binary exists but doesn't work (possibly incompatible architecture)"
            echo "Marking for rebuild..."
            rm -rf "$LLAMACPP_DIR/build"
            REBUILD_LLAMA=true
        fi
    else
        if [ "$REBUILD_LLAMA" != true ]; then
            print_warning "llama-server binary not found"
            REBUILD_LLAMA=true
        fi
    fi
else
    print_warning "llama.cpp not found. Need to clone from GitHub..."
    REBUILD_LLAMA=true
fi

# Step 2: If rebuild needed, install dependencies first
if [ "$REBUILD_LLAMA" = true ] && [ "$LLAMACPP_INSTALL_FAILED" = false ]; then
    print_section "Installing system dependencies for llama.cpp..."
    if ! install_system_dependencies; then
        print_error "Failed to install system dependencies"
        LLAMACPP_INSTALL_FAILED=true
    else
        echo
    fi
fi

# Step 3: Clone llama.cpp if needed
if [ "$REBUILD_LLAMA" = true ] && [ ! -d "$LLAMACPP_DIR" ] && [ "$LLAMACPP_INSTALL_FAILED" = false ]; then
    print_section "Cloning llama.cpp repository..."
    echo "This may take a minute..."
    echo

    if git clone --depth 1 https://github.com/ggerganov/llama.cpp "$LLAMACPP_DIR" 2>&1; then
        print_success "llama.cpp cloned successfully"
        echo
    else
        print_error "Failed to clone llama.cpp repository"
        LLAMACPP_INSTALL_FAILED=true
    fi
fi

# Step 3.5: Apply production patches to llama.cpp
if [ -d "$LLAMACPP_DIR" ] && [ "$LLAMACPP_INSTALL_FAILED" = false ]; then
    print_section "Applying production patches to llama.cpp..."
    echo

    if [ -f "$SCRIPT_DIR/patches/apply_patches.sh" ]; then
        if bash "$SCRIPT_DIR/patches/apply_patches.sh"; then
            print_success "Production patches applied successfully"
            echo
            echo -e "${CYAN}Applied fixes:${NC}"
            echo "  ✓ Vision model state corruption fix (mtmd_encode buffer clearing)"
            echo "  ✓ Prevents HTTP 500 errors in continuous batching mode"
            echo "  ✓ Ensures stable multi-image processing"
            echo
        else
            print_warning "Some patches failed to apply"
            print_warning "Vision model support may be less stable"
            echo
            echo "This is usually fine - the patches may already be in upstream llama.cpp"
            echo
        fi
    else
        print_warning "Patch script not found at: $SCRIPT_DIR/patches/apply_patches.sh"
        print_warning "Continuing without patches - vision models may be less stable"
        echo
    fi
fi

# Step 4: Build llama.cpp if needed
if [ -d "$LLAMACPP_DIR" ] && [ ! -f "$LLAMACPP_BINARY" ] && [ "$LLAMACPP_INSTALL_FAILED" = false ]; then
    print_section "Building llama.cpp..."
    echo

    # Detect GPU and get build flags
    GPU_INFO=$(detect_gpu_and_build_flags)
    GPU_TYPE=$(echo "$GPU_INFO" | cut -d'|' -f1)
    GPU_FLAGS=$(echo "$GPU_INFO" | cut -d'|' -f2)
    GPU_DESCRIPTION=$(echo "$GPU_INFO" | cut -d'|' -f3)

    if [ -n "$GPU_FLAGS" ]; then
        print_success "Detected: $GPU_DESCRIPTION"
    else
        print_warning "No GPU detected: $GPU_DESCRIPTION"
    fi

    echo "Build configuration:"
    echo "  GPU Support: $GPU_DESCRIPTION"
    echo "  Build Flags: ${GPU_FLAGS:-'(none - CPU only)'}"
    echo

    # Create build directory
    cd "$LLAMACPP_DIR"
    if [ -d build ]; then
        print_warning "Cleaning previous build..."
        rm -rf build
    fi

    mkdir -p build
    cd build

    # Run CMake configuration
    echo "Configuring CMake..."
    if ! cmake .. $GPU_FLAGS 2>&1 | tail -20; then
        print_error "CMake configuration failed"
        LLAMACPP_INSTALL_FAILED=true
    else
        print_success "CMake configuration successful"
        echo

        # Detect number of CPU cores for parallel build
        if command -v nproc &> /dev/null; then
            CORES=$(nproc)
        elif command -v sysctl &> /dev/null; then
            CORES=$(sysctl -n hw.ncpu 2>/dev/null || echo 4)
        else
            CORES=4
        fi

        # Build with appropriate parallelism
        echo "Building with $CORES parallel jobs..."
        echo "This may take 5-15 minutes depending on your system..."
        echo

        if cmake --build . --config Release -j"$CORES" 2>&1 | tail -50; then
            print_success "llama.cpp built successfully!"
            echo

            # Verify binary was created
            if [ -f "$LLAMACPP_BINARY" ]; then
                print_success "llama-server binary created successfully"
                print_success "Location: $LLAMACPP_BINARY"

                # Test the binary
                echo
                echo "Testing binary..."
                if "$LLAMACPP_BINARY" --version &> /dev/null; then
                    print_success "Binary test passed - llama.cpp is ready!"
                else
                    print_warning "Binary was created but test failed"
                    LLAMACPP_INSTALL_FAILED=true
                fi
            else
                print_error "Build completed but binary not found at: $LLAMACPP_BINARY"
                LLAMACPP_INSTALL_FAILED=true
            fi
        else
            print_error "llama.cpp build failed"
            print_warning "Common solutions:"
            echo "  1. Ensure you have enough disk space (~2GB)"
            echo "  2. Try rebuilding: rm -rf '$LLAMACPP_DIR/build' && bash $SCRIPT_DIR/setup.sh"
            echo "  3. For GPU issues, verify drivers are installed"
            LLAMACPP_INSTALL_FAILED=true
        fi
    fi

    # Return to script directory
    cd "$SCRIPT_DIR"
fi

# Final llama.cpp status report
echo
echo -e "${BLUE}════════════════════════════════════════════════════════════${NC}"
if [ -f "$LLAMACPP_BINARY" ]; then
    print_success "llama.cpp is ready for GGUF model support"
    echo
    print_success "Binary location: $LLAMACPP_BINARY"
    echo
    echo "You can now use GGUF models with the application!"
elif [ "$LLAMACPP_INSTALL_FAILED" = true ]; then
    print_warning "llama.cpp installation encountered issues"
    print_warning "GGUF model support will not be available"
    echo
    echo "To troubleshoot and install manually:"
    echo "  1. Verify dependencies:"
    case $OS in
        macos)
            echo -e "     ${YELLOW}brew install cmake git${NC}"
            ;;
        linux|wsl)
            echo -e "     ${YELLOW}sudo apt-get install cmake git build-essential${NC}"
            ;;
    esac
    echo
    echo "  2. Clone and build:"
    echo -e "     ${YELLOW}cd \"$SCRIPT_DIR\"${NC}"
    echo -e "     ${YELLOW}git clone https://github.com/ggerganov/llama.cpp${NC}"
    echo -e "     ${YELLOW}cd llama.cpp && mkdir build && cd build${NC}"
    echo
    echo "  3. Configure for your system:"
    case $OS in
        macos)
            echo -e "     ${YELLOW}cmake .. -DGGML_METAL=ON${NC}"
            ;;
        linux)
            echo -e "     For NVIDIA GPU: ${YELLOW}cmake .. -DGGML_CUDA=ON${NC}"
            echo -e "     For AMD GPU: ${YELLOW}cmake .. -DGGML_HIPBLAS=ON${NC}"
            echo -e "     For CPU only: ${YELLOW}cmake ..${NC}"
            ;;
    esac
    echo
    echo "  4. Build:"
    echo -e "     ${YELLOW}cmake --build . --config Release -j\$(nproc)${NC}"
    echo
else
    print_warning "llama.cpp setup status unknown"
fi
echo -e "${BLUE}════════════════════════════════════════════════════════════${NC}"
echo
print_success "Setup completed successfully!"
echo
