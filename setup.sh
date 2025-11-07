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

# Ollama installation and model setup
print_section "Setting up Ollama (LLM provider)..."
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
                brew install ollama > /dev/null 2>&1
                print_success "Ollama installed via Homebrew"
            else
                print_warning "Homebrew not found. Please install Ollama manually:"
                echo -e "  ${YELLOW}https://ollama.ai/download${NC}"
                echo "  Then run: bash $SCRIPT_DIR/setup.sh"
                OLLAMA_INSTALLED=false
            fi
            ;;
        linux)
            echo "Installing Ollama for Linux..."
            if command -v curl &> /dev/null; then
                curl -fsSL https://ollama.ai/install.sh 2>/dev/null | sh > /dev/null 2>&1
                if [ $? -eq 0 ]; then
                    print_success "Ollama installed"
                    OLLAMA_INSTALLED=true
                else
                    print_warning "Ollama installation had issues. Please install manually:"
                    echo -e "  ${YELLOW}https://ollama.ai/download${NC}"
                fi
            else
                print_warning "curl not found. Please install Ollama manually:"
                echo -e "  ${YELLOW}https://ollama.ai/download${NC}"
            fi
            ;;
        windows)
            print_warning "Windows requires manual Ollama installation"
            echo "Please download and install Ollama from:"
            echo -e "  ${YELLOW}https://ollama.ai/download${NC}"
            echo "Then run this script again"
            OLLAMA_INSTALLED=false
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
    print_warning "Ollama is not installed. Skipping model setup."
    print_warning "You can install Ollama later and pull a model with: ollama pull qwen3:0.6b"
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

echo
print_success "Setup completed successfully!"
echo
