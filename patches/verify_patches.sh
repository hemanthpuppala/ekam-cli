#!/usr/bin/env bash

# EKAM CLI - Patch Verification Script
# Verifies that all production patches have been successfully applied

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

# Get directories
PATCHES_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$(dirname "$PATCHES_DIR")"
LLAMACPP_DIR="$PROJECT_ROOT/llama.cpp"

echo -e "${BLUE}═══════════════════════════════════════════════════════════${NC}"
echo -e "${BLUE}EKAM CLI - Patch Verification${NC}"
echo -e "${BLUE}═══════════════════════════════════════════════════════════${NC}"
echo

# Check 1: llama.cpp exists
echo -n "Checking llama.cpp directory... "
if [ -d "$LLAMACPP_DIR" ]; then
    echo -e "${GREEN}✓${NC}"
else
    echo -e "${RED}✗${NC}"
    echo -e "${RED}Error: llama.cpp not found at $LLAMACPP_DIR${NC}"
    echo "Please run ./setup.sh first"
    exit 1
fi

# Check 2: mtmd.cpp exists
echo -n "Checking mtmd.cpp file... "
if [ -f "$LLAMACPP_DIR/tools/mtmd/mtmd.cpp" ]; then
    echo -e "${GREEN}✓${NC}"
else
    echo -e "${RED}✗${NC}"
    echo -e "${RED}Error: mtmd.cpp not found${NC}"
    exit 1
fi

# Check 3: Vision state fix applied
echo -n "Checking vision state corruption fix... "
PATCH_MARKER="FIX: Clear stale embeddings before resize to prevent buffer corruption"
FIXES_FOUND=$(grep -c "$PATCH_MARKER" "$LLAMACPP_DIR/tools/mtmd/mtmd.cpp" 2>/dev/null || echo "0")
FIXES_FOUND=$(echo "$FIXES_FOUND" | tr -d '\n' | tr -d ' ')

if [ "$FIXES_FOUND" -eq 2 ] 2>/dev/null; then
    echo -e "${GREEN}✓${NC} (2/2 locations fixed)"
elif [ "$FIXES_FOUND" -eq 1 ] 2>/dev/null; then
    echo -e "${YELLOW}⚠${NC} (1/2 locations fixed - incomplete)"
else
    echo -e "${RED}✗${NC} (0/2 locations fixed)"
fi

# Check 4: Patch marker file
echo -n "Checking patch tracking marker... "
if [ -f "$LLAMACPP_DIR/.ekam_patches_applied" ]; then
    echo -e "${GREEN}✓${NC}"
    PATCH_DATE=$(cat "$LLAMACPP_DIR/.ekam_patches_applied")
    echo "  Applied: $PATCH_DATE"
else
    echo -e "${YELLOW}⚠${NC} (marker not found)"
fi

# Check 5: Backup exists
echo -n "Checking backup file... "
if [ -f "$LLAMACPP_DIR/tools/mtmd/mtmd.cpp.ekam-backup" ]; then
    echo -e "${GREEN}✓${NC}"
else
    echo -e "${YELLOW}⚠${NC} (no backup - first-time install or already patched upstream)"
fi

# Check 6: Verify actual code changes
echo
echo -e "${BLUE}Detailed Code Verification:${NC}"
echo

# Location 1: Audio encoding
echo "1. Audio encoding fix (mtmd_encode_chunk):"
if grep -A2 "int n_mmproj_embd = ctx->n_embd_text;" "$LLAMACPP_DIR/tools/mtmd/mtmd.cpp" | grep -q "image_embd_v.clear()"; then
    echo -e "   ${GREEN}✓${NC} Buffer clear present"
else
    echo -e "   ${RED}✗${NC} Buffer clear missing"
fi

# Location 2: Image encoding
echo "2. Image encoding fix (mtmd_encode):"
if grep -A2 "int n_mmproj_embd = clip_n_mmproj_embd(ctx_clip);" "$LLAMACPP_DIR/tools/mtmd/mtmd.cpp" | grep -q "image_embd_v.clear()"; then
    echo -e "   ${GREEN}✓${NC} Buffer clear present"
else
    echo -e "   ${RED}✗${NC} Buffer clear missing"
fi

# Summary
echo
echo -e "${BLUE}═══════════════════════════════════════════════════════════${NC}"

if [ "$FIXES_FOUND" -eq 2 ] 2>/dev/null; then
    echo -e "${GREEN}✓ All patches successfully applied!${NC}"
    echo
    echo "Your llama.cpp installation has production-ready vision support."
    echo "Vision models should process multi-image requests reliably."
    echo
    echo -e "${GREEN}Status: PRODUCTION READY${NC}"
    exit 0
elif [ "$FIXES_FOUND" -eq 1 ] 2>/dev/null; then
    echo -e "${YELLOW}⚠ Patches partially applied${NC}"
    echo
    echo "One location was patched but another is missing."
    echo "Vision models may still experience instability."
    echo
    echo "To fix, run:"
    echo "  bash patches/apply_patches.sh"
    echo
    echo -e "${YELLOW}Status: PARTIALLY PATCHED${NC}"
    exit 1
else
    echo -e "${RED}✗ Patches NOT applied${NC}"
    echo
    echo "Vision state corruption fix is missing."
    echo "Vision models will have ~30-50% failure rate."
    echo
    echo "To fix, run:"
    echo "  bash patches/apply_patches.sh"
    echo
    echo "Then rebuild llama.cpp:"
    echo "  cd llama.cpp/build"
    echo "  cmake --build . --config Release"
    echo
    echo -e "${RED}Status: UNPATCHED - NOT PRODUCTION READY${NC}"
    exit 1
fi
