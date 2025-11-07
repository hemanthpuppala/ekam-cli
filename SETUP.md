# EKAM CLI - Setup Guide

One-time setup to create a development environment and install the global `ekam-cli` command.

## Quick Start (< 1 minute)

```bash
bash setup.sh
```

This will:
1. ✓ Check Python version (requires 3.10+)
2. ✓ Create a virtual environment (`ekam-venv/`)
3. ✓ Let you choose: `pyproject.toml` or `requirements.txt`
4. ✓ Install all dependencies
5. ✓ Ask if you want to install `ekam-cli` globally

## After Setup

### Running the Application

**Option 1: Global Command (recommended)**
```bash
ekam-cli
```

**Option 2: From Project Directory**
```bash
source ekam-venv/bin/activate  # macOS/Linux
source ekam-venv/Scripts/activate  # Windows
python -m src.core.app
```

**Option 3: Direct (without activation)**
```bash
./ekam-venv/bin/python -m src.core.app
```

## Installation Details

### What the Setup Script Does

| Step | Action | Details |
|------|--------|---------|
| 1 | Python Check | Verifies Python 3.10+ is installed |
| 2 | venv Creation | Creates isolated `ekam-venv/` directory |
| 3 | Dependency Install | Choose between `pyproject.toml` (modern) or `requirements.txt` (simple) |
| 4 | Global CLI | Optionally installs `ekam-cli` command system-wide |

### Virtual Environment Location

- **Directory**: `./ekam-venv/` (in project root)
- **Size**: ~500MB (includes all dependencies)
- **.gitignore**: Already configured to exclude `ekam-venv/`

### Global Command Installation

The `ekam-cli` command is installed in one of:
- `/usr/local/bin/ekam-cli` (system-wide, Linux/macOS)
- `~/.local/bin/ekam-cli` (user-local, Linux/macOS)
- `C:\...\Scripts\` (Windows)

**It uses your project's `ekam-venv/` - not system Python!**

This means:
- ✓ No global Python pollution
- ✓ Dependencies isolated to project
- ✓ Safe to delete `ekam-venv/` and recreate anytime
- ✓ Can have multiple projects with different dependencies

## Manual Installation

If you need to install `ekam-cli` globally after setup:

```bash
bash install-ekam-cli.sh /path/to/project
```

Or if you're in the project directory:

```bash
bash install-ekam-cli.sh
```

## Troubleshooting

### Python Not Found
```bash
# Install Python 3.10+
# macOS:
brew install python@3.10

# Ubuntu/Debian:
sudo apt-get install python3.10 python3.10-venv

# Windows:
# Download from https://python.org
```

### ekam-venv Directory Exists
Setup will ask if you want to recreate it. Choose:
- **y** - Delete old ekam-venv and create new one (clean slate)
- **n** - Reuse existing ekam-venv (faster)

### ekam-cli Not in PATH
After global installation, you may need to:

```bash
# Add to shell profile (~/.bashrc, ~/.zshrc, etc.)
export PATH="$HOME/.local/bin:$PATH"

# Then reload:
exec $SHELL
```

### Permission Denied
If you get permission errors during global installation:

```bash
# Option 1: Install to user-local directory (recommended)
bash install-ekam-cli.sh

# Option 2: Use sudo (if prompted)
sudo bash install-ekam-cli.sh /path/to/project
```

## Uninstalling

### Remove Global Command
```bash
# macOS/Linux:
sudo rm /usr/local/bin/ekam-cli

# User-local:
rm ~/.local/bin/ekam-cli
```

### Clean Virtual Environment
```bash
# Delete the ekam-venv directory (safe, can recreate anytime)
rm -rf ekam-venv/

# Recreate by running:
bash setup.sh
```

## Environment Variables

Create a `.env` file in the project root for configuration:

```bash
# .env (excluded from git)
PYTHONPATH=/path/to/project
DEBUG=1
```

See `.env.example` (if provided) for all available options.

## For Production Deployment

This setup is for **development**. For production:

1. Use `pyproject.toml` with version pins
2. Generate `requirements.lock` for reproducible builds
3. Use containerization (Docker) for deployment
4. See [Ollama](https://ollama.ai) for production VLM setup

## Support

- Check Python version: `python3 --version`
- Verify venv: `ls -la ekam-venv/`
- Test activation: `source ekam-venv/bin/activate`
- Run diagnostics: `python -m pip list`
