<!--
SYNC IMPACT REPORT
==================

Version Change: TEMPLATE → 1.0.0 (Initial ratification)

Rationale: MINOR version - Initial constitution creation for VLM CLI project.
This is the first concrete constitution, transitioning from template placeholders
to project-specific governance.

Modified Principles:
- [PRINCIPLE_1_NAME] → I. Provider Abstraction
- [PRINCIPLE_2_NAME] → II. Configuration-Driven Architecture
- [PRINCIPLE_3_NAME] → III. Production-Ready Reliability
- [PRINCIPLE_4_NAME] → IV. Interactive User Experience
- [PRINCIPLE_5_NAME] → V. Extensibility First

Added Sections:
- Development Standards (code organization, error handling, testing strategy)
- Performance & Resource Management (constraints and performance standards)

Removed Sections:
- Generic placeholder sections [SECTION_2_NAME], [SECTION_3_NAME] replaced with
  concrete "Development Standards" and "Performance & Resource Management"

Template Updates:
✅ .specify/templates/plan-template.md - No updates needed (generic Constitution Check section)
✅ .specify/templates/spec-template.md - No updates needed (requirements-focused, agnostic)
✅ .specify/templates/tasks-template.md - No updates needed (task structure is generic)
✅ .specify/templates/agent-file-template.md - No updates needed (auto-generated placeholder)
✅ .claude/commands/*.md - Verified: No agent-specific references like "CLAUDE" found

Follow-up TODOs: None - All placeholders resolved

Last Validated: 2025-10-11
-->

# VLM CLI Constitution

## Core Principles

### I. Provider Abstraction

All VLM functionality MUST be accessible through a unified interface that abstracts
provider-specific implementations. Each provider (Ollama, HuggingFace, etc.) MUST
implement the same core endpoints (QA, Caption, Detect, Point). New providers MUST
integrate without modifying existing provider code.

**Rationale**: Ensures extensibility and prevents vendor lock-in while maintaining
consistent user experience across different VLM backends.

### II. Configuration-Driven Architecture

Model capabilities, inference parameters, and provider settings MUST be defined in
external configuration (YAML). Code MUST NOT hardcode model names, endpoints, or
provider-specific URLs. All runtime behavior MUST be configurable without code changes.

**Rationale**: Enables rapid model switching and testing without deployments; supports
diverse testing scenarios through configuration alone.

### III. Production-Ready Reliability

All endpoints MUST include comprehensive error handling for network failures, model
errors, and invalid inputs. Results MUST be persistable with full context (timestamp,
model, parameters). Inference timing MUST be tracked and reported. The CLI MUST handle
graceful degradation when providers are unavailable.

**Rationale**: Ensures the tool is reliable for production testing scenarios where
failures must be diagnosed and results must be reproducible.

### IV. Interactive User Experience

The CLI MUST provide rich terminal UI with clear menus, progress indicators, and
formatted results. Users MUST be able to switch models without restarting. Statistics
and model information MUST be easily accessible. Error messages MUST be actionable and
user-friendly.

**Rationale**: Developer productivity depends on fast iteration cycles and clear
feedback during VLM testing.

### V. Extensibility First

New endpoints MUST be addable by implementing a single method per provider. New models
MUST be addable via configuration alone. The architecture MUST support model-specific
optimizations (e.g., Moondream native methods) without breaking the unified interface.

**Rationale**: VLM landscape evolves rapidly; the tool must accommodate new models and
capabilities without architectural rewrites.

## Development Standards

### Code Organization

- **Provider implementations**: Isolated in provider-specific modules
- **Shared interfaces**: Defined in base classes with clear contracts
- **Configuration**: Single YAML file with validation on load
- **CLI logic**: Separated from VLM client logic

### Error Handling

- **Network errors**: Retry with exponential backoff, clear timeout messages
- **Model errors**: Capture full error context, suggest configuration fixes
- **Invalid input**: Validate early, provide examples of valid input
- **Provider unavailable**: Graceful fallback with helpful setup instructions

### Testing Strategy (when specified)

- **Contract tests**: Verify each provider implements all required endpoints
- **Integration tests**: Test end-to-end workflows with real models (when available) or mocks
- **Configuration tests**: Validate YAML schema and default configurations

## Performance & Resource Management

### Resource Constraints

- Model loading MUST be lazy (load on first use or explicit switch)
- Memory usage MUST be monitored and reported in statistics
- Large images MUST be processed efficiently (streaming or chunked when possible)
- Multiple models MUST NOT be loaded simultaneously unless explicitly requested

### Performance Standards

- **CLI startup**: <1 second (excluding model loading)
- **Model switching**: Display progress, allow cancellation
- **Results display**: Immediate streaming for long responses
- **Statistics calculation**: <100ms regardless of history size

## Governance

### Amendment Process

Constitution changes require:

1. Documentation of rationale in commit message
2. Review of impact on existing features
3. Update of all affected templates and commands
4. Version increment per semantic versioning

### Compliance

- All new features MUST align with Provider Abstraction and Configuration-Driven principles
- PRs MUST NOT introduce hardcoded model names or provider URLs
- New endpoints MUST be added to all providers or marked as optional with clear documentation
- Breaking changes to configuration format MUST include migration guide

### Versioning Policy

- **MAJOR**: Breaking changes to configuration schema or provider interface
- **MINOR**: New endpoints, providers, or models
- **PATCH**: Bug fixes, performance improvements, documentation

**Version**: 1.0.0 | **Ratified**: 2025-10-11 | **Last Amended**: 2025-10-11
