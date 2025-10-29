# Testing Structure

This directory contains automated tests for the agentic_shac project.

## Directory Structure

```
tests/
├── unit/              # Unit tests (pytest)
│   └── test_*.py      # Test individual functions
└── integration/       # Integration tests (AI validation)
    └── phaseN_test_*.py  # End-to-end tests per phase
```

## Integration Tests (`integration/`)

**Purpose**: These are **complete, runnable scripts** that the AI runs to validate each phase works correctly.

### Naming Convention
- `phase1_test_brat_loader.py` - Test Phase 1: BRAT loading
- `phase2_test_prompts.py` - Test Phase 2: Prompt generation
- `phase2_test_model_loading.py` - Test Phase 2: Model loading
- etc.

### Requirements
- Must run end-to-end without user interaction
- Must test on REAL data (not mocked)
- Must exit with code 0 on success, non-zero on failure
- Must print clear ✅ success or ❌ failure messages

### Running Integration Tests

```bash
# Run a specific phase test
conda run -n temp python tests/integration/phase1_test_brat_loader.py

# Run all integration tests
for test in tests/integration/phase*.py; do
    echo "Running $test..."
    conda run -n temp python "$test" || echo "FAILED: $test"
done
```

## Unit Tests (`unit/`)

**Purpose**: Test individual functions in isolation using pytest.

### Running Unit Tests

```bash
# Run all unit tests
conda run -n temp pytest tests/unit/

# Run specific test file
conda run -n temp pytest tests/unit/test_brat_loader.py

# Run with verbose output
conda run -n temp pytest tests/unit/ -v
```

## Testing Workflow

For each phase:

1. **AI creates integration test** → `tests/integration/phaseN_test_*.py`
2. **AI runs test and verifies** → Must pass before proceeding
3. **AI creates notebook** → `notebooks/NN_test_*.ipynb`
4. **User runs notebook** → Validates outputs visually
5. **✅ Proceed to next phase**

## Key Principle

**Integration tests must pass BEFORE claiming something works.**

The AI should:
- ✅ Run integration tests and show actual results
- ✅ Report failures honestly
- ❌ Never claim code works without running tests
- ❌ Never assume code will work without verification

