# Quantization Module Refactoring TODO

## Current Status
✅ Phase 1 implementation complete (GGUF quantization)
✅ All core functionality working
⏳ File organization needs refactoring for clarity

## Refactoring Steps

### 1. Reorganize Files

#### Move to core/
```bash
mv src/quantization/background.py src/quantization/core/
mv src/quantization/recommendations.py src/quantization/core/
```

#### Move to ui/
```bash
mv src/quantization/ui.py src/quantization/ui/display.py
mv src/quantization/workflow.py src/quantization/ui/workflow.py
```

#### Already in techniques/
- ✅ techniques/base.py (created)
- ✅ techniques/gguf.py (created)

#### Delete old file
```bash
rm src/quantization/gguf_quantizer.py  # Replaced by techniques/gguf.py
```

### 2. Create __init__.py Files

#### techniques/__init__.py
```python
"""Quantization techniques."""

from .base import BaseQuantizer
from .gguf import GGUFQuantizer

__all__ = ["BaseQuantizer", "GGUFQuantizer"]
```

#### core/__init__.py
```python
"""Core quantization functionality."""

from .background import BackgroundJobManager
from .recommendations import (
    check_can_quantize_multiple,
    get_quantization_recommendations,
)

__all__ = [
    "BackgroundJobManager",
    "get_quantization_recommendations",
    "check_can_quantize_multiple",
]
```

#### ui/__init__.py
```python
"""UI components for quantization."""

from .display import (
    ask_background_mode,
    ask_gpu_preference,
    confirm_quantization,
    select_model_to_quantize,
    select_quantization_type,
    show_live_progress,
    show_quantization_intro,
)
from .workflow import run_quantization_workflow, show_background_jobs_monitor

__all__ = [
    "show_quantization_intro",
    "select_model_to_quantize",
    "ask_gpu_preference",
    "select_quantization_type",
    "confirm_quantization",
    "ask_background_mode",
    "show_live_progress",
    "run_quantization_workflow",
    "show_background_jobs_monitor",
]
```

#### methods/__init__.py
```python
"""Quantization methods and utilities."""

# Phase 2: Add calibration and validation
__all__ = []
```

### 3. Update Imports

#### In manager.py
```python
# OLD:
from .background import BackgroundJobManager
from .gguf_quantizer import GGUFQuantizer
from .recommendations import get_quantization_recommendations

# NEW:
from .core.background import BackgroundJobManager
from .core.recommendations import get_quantization_recommendations
from .techniques.gguf import GGUFQuantizer
```

#### In ui/workflow.py
```python
# OLD:
from ..manager import QuantizationManager
from .ui import (...)

# NEW:
from ..manager import QuantizationManager
from .display import (...)
```

#### In core/background.py
```python
# OLD:
from .gguf_quantizer import GGUFQuantizer

# NEW:
from ..techniques.gguf import GGUFQuantizer
```

### 4. Create Placeholder Files for Phase 2

#### techniques/gptq.py
```python
"""GPTQ quantization technique (Phase 2)."""

from .base import BaseQuantizer

class GPTQQuantizer(BaseQuantizer):
    """GPTQ quantization for GPU inference."""
    # TODO: Implement in Phase 2
    pass
```

#### techniques/awq.py
```python
"""AWQ quantization technique (Phase 2)."""

from .base import BaseQuantizer

class AWQQuantizer(BaseQuantizer):
    """AWQ quantization for GPU inference."""
    # TODO: Implement in Phase 2
    pass
```

#### techniques/bnb.py
```python
"""BitsAndBytes quantization technique (Phase 2)."""

from .base import BaseQuantizer

class BnBQuantizer(BaseQuantizer):
    """BitsAndBytes quantization."""
    # TODO: Implement in Phase 2
    pass
```

#### methods/calibration.py
```python
"""Calibration methods for quantization (Phase 2)."""

# TODO: Implement calibration dataset generation
# TODO: Implement calibration methods
```

#### methods/validation.py
```python
"""Validation methods for quantized models (Phase 2)."""

# TODO: Implement quality metrics
# TODO: Implement perplexity testing
# TODO: Implement speed benchmarks
```

### 5. Update Main __init__.py

```python
"""Model quantization module."""

from .models import (
    QuantizationModule,
    QuantizationRecommendation,
    QuantizationTask,
    QuantizationType,
    TaskStatus,
)
from .manager import QuantizationManager
from .core.background import BackgroundJobManager
from .techniques.base import BaseQuantizer
from .techniques.gguf import GGUFQuantizer

__all__ = [
    # Data models
    "QuantizationType",
    "QuantizationModule",
    "QuantizationTask",
    "QuantizationRecommendation",
    "TaskStatus",
    # Core
    "QuantizationManager",
    "BackgroundJobManager",
    # Techniques
    "BaseQuantizer",
    "GGUFQuantizer",
]
```

### 6. Test After Refactoring

```bash
# Verify imports
python3 -c "from src.quantization import QuantizationManager; print('✓ Imports work')"

# Verify structure
ls -R src/quantization/

# Expected structure:
# src/quantization/
# ├── __init__.py
# ├── models.py
# ├── manager.py
# ├── ARCHITECTURE.md
# ├── REFACTORING_TODO.md
# ├── core/
# │   ├── __init__.py
# │   ├── background.py
# │   └── recommendations.py
# ├── techniques/
# │   ├── __init__.py
# │   ├── base.py
# │   ├── gguf.py
# │   ├── gptq.py (placeholder)
# │   ├── awq.py (placeholder)
# │   └── bnb.py (placeholder)
# ├── methods/
# │   ├── __init__.py
# │   ├── calibration.py (placeholder)
# │   └── validation.py (placeholder)
# └── ui/
#     ├── __init__.py
#     ├── display.py
#     └── workflow.py
```

## Benefits of This Structure

1. **Clear Separation** - Each folder has a single responsibility
2. **Scalability** - Easy to add new techniques in Phase 2
3. **Testability** - Each module can be tested independently
4. **Discoverability** - New developers can easily understand the architecture
5. **Maintainability** - Changes are localized to specific modules

## After Refactoring

1. Update integration in `src/core/app_quantization.py`
2. Add to main app menu
3. Add status bar integration
4. Test end-to-end workflow
5. Update documentation
6. Push to GitHub

## Notes

- All functionality is already implemented and working
- This is purely organizational refactoring
- No logic changes required
- Just moving files and updating imports
- Takes ~15 minutes to complete
