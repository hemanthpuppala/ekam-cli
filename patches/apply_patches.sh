#!/usr/bin/env bash

# EKAM CLI - llama.cpp Patch Applicator
# Applies critical fixes to llama.cpp for production stability

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# Get script directory
PATCHES_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$(dirname "$PATCHES_DIR")"
LLAMACPP_DIR="$PROJECT_ROOT/llama.cpp"

# Helper functions
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

# Manual patch application fallback
apply_patch_manually() {
    print_section "Applying patches with sed/awk..."

    local mtmd_file="tools/mtmd/mtmd.cpp"
    local temp_file=$(mktemp)

    # Check if file exists
    if [ ! -f "$mtmd_file" ]; then
        print_error "File not found: $mtmd_file"
        rm -f "$temp_file"
        return 1
    fi

    # Apply patch using awk
    awk '
    BEGIN {
        fixed_audio = 0
        fixed_image = 0
        in_encode_chunk = 0
        in_encode = 0
    }

    # Track function context
    /^int32_t mtmd_encode_chunk/ {
        in_encode_chunk = 1
        in_encode = 0
    }
    /^int32_t mtmd_encode\(/ {
        in_encode = 1
        in_encode_chunk = 0
    }
    /^[a-zA-Z]/ && !/^int32_t mtmd_encode/ {
        in_encode_chunk = 0
        in_encode = 0
    }

    # Fix 1: Audio encoding in mtmd_encode_chunk (around line 787)
    /^[ \t]*int n_mmproj_embd = ctx->n_embd_text;/ && in_encode_chunk && fixed_audio == 0 {
        print $0
        print "        // FIX: Clear stale embeddings before resize to prevent buffer corruption"
        print "        ctx->image_embd_v.clear();"
        fixed_audio = 1
        next
    }

    # Fix 2: Image encoding in mtmd_encode (around line 807)
    /^[ \t]*int n_mmproj_embd = clip_n_mmproj_embd\(ctx_clip\);/ && in_encode && fixed_image == 0 {
        print $0
        print "    // FIX: Clear stale embeddings before resize to prevent buffer corruption"
        print "    ctx->image_embd_v.clear();"
        fixed_image = 1
        next
    }

    { print }

    END {
        if (fixed_audio && fixed_image) {
            exit 0
        } else {
            exit 1
        }
    }
    ' "$mtmd_file" > "$temp_file"

    if [ $? -eq 0 ]; then
        mv "$temp_file" "$mtmd_file"
        print_success "Patches applied successfully with sed/awk"
        return 0
    else
        rm -f "$temp_file"
        print_error "Failed to apply patches - couldn't find all target lines"
        return 1
    fi
}

# Check if llama.cpp directory exists
if [ ! -d "$LLAMACPP_DIR" ]; then
    print_error "llama.cpp directory not found at: $LLAMACPP_DIR"
    print_error "Please run setup.sh first to clone llama.cpp"
    exit 1
fi

print_section "Applying llama.cpp production patches..."

# Change to llama.cpp directory
cd "$LLAMACPP_DIR" || exit 1

# Check if patch is already applied
PATCH_MARKER="FIX: Clear stale embeddings before resize to prevent buffer corruption"
FIXES_APPLIED=$(grep -c "$PATCH_MARKER" tools/mtmd/mtmd.cpp 2>/dev/null || echo "0")

if [ "$FIXES_APPLIED" -ge 2 ]; then
    print_success "Vision state fix already applied - skipping"
    print_success "Found $FIXES_APPLIED fix markers in code"
    exit 0
elif [ "$FIXES_APPLIED" -gt 0 ]; then
    print_warning "Partial fix detected ($FIXES_APPLIED/2 locations)"
    print_warning "Will attempt to complete the patch..."
fi

# Backup original file
print_section "Creating backup of original file..."
if [ ! -f "tools/mtmd/mtmd.cpp.ekam-backup" ]; then
    cp tools/mtmd/mtmd.cpp tools/mtmd/mtmd.cpp.ekam-backup
    print_success "Backup created: tools/mtmd/mtmd.cpp.ekam-backup"
else
    print_warning "Backup already exists - skipping"
fi

# Try patch command first
print_section "Applying vision state corruption fix..."
if patch -p1 --dry-run < "$PATCHES_DIR/llamacpp_vision_state_fix.patch" > /dev/null 2>&1; then
    print_success "Patch format is compatible"
    if patch -p1 < "$PATCHES_DIR/llamacpp_vision_state_fix.patch" 2>&1 | tail -10; then
        print_success "Patch applied successfully with patch command!"
        PATCH_SUCCESS=true
    else
        print_warning "patch command failed, trying manual application..."
        PATCH_SUCCESS=false
    fi
else
    print_warning "Patch format not compatible with this llama.cpp version"
    print_warning "Using manual sed/awk application..."
    PATCH_SUCCESS=false
fi

# Fallback to manual application if patch command failed
if [ "$PATCH_SUCCESS" != "true" ]; then
    if apply_patch_manually; then
        PATCH_SUCCESS=true
    else
        print_error "All patch methods failed"
        print_warning "Vision model support may be unstable"
        echo
        echo "To restore original file:"
        echo "  cp tools/mtmd/mtmd.cpp.ekam-backup tools/mtmd/mtmd.cpp"
        echo
        exit 1
    fi
fi

if [ "$PATCH_SUCCESS" = "true" ]; then
    echo
    echo -e "${GREEN}What was fixed:${NC}"
    echo "  • Prevents state corruption in vision model embeddings buffer"
    echo "  • Adds buffer clearing before resize operations"
    echo "  • Fixes intermittent HTTP 500 errors in continuous batching mode"
    echo "  • Affects: tools/mtmd/mtmd.cpp (2 locations)"
    echo
    echo -e "${YELLOW}Note:${NC} llama.cpp will need to be rebuilt for changes to take effect"
    echo

    # Create marker file to track patch status
    echo "Vision state fix applied on $(date)" > "$LLAMACPP_DIR/.ekam_patches_applied"
    print_success "Patch tracking marker created"

    echo
    print_success "All patches applied successfully!"
    echo

    exit 0
else
    exit 1
fi
