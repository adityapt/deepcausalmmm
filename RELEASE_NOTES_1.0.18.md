# Release Notes: DeepCausalMMM v1.0.18

**Release Date:** October 23, 2025

## Major Improvements: Professional Code Standards & Documentation

This release focuses on improving code quality, professionalism, and documentation standards in preparation for JOSS (Journal of Open Source Software) publication.

---

## Key Changes

### Logging System Overhaul

**Replaced Print Statements with Professional Logging:**
- Migrated all core library print statements to Python's standard logging module
- Added centralized logging configuration in `deepcausalmmm/__init__.py`
- Provides better control over verbosity and output management
- Users can now control logging levels programmatically

**Package-wide Logger:**
```python
import logging

# Configure logging (optional)
logging.getLogger('deepcausalmmm').setLevel(logging.DEBUG)  # More verbose
logging.getLogger('deepcausalmmm').setLevel(logging.WARNING)  # Less verbose
```

### Code Quality Improvements

**Emoji Removal:**
- Removed all emojis from core library files
- Removed all emojis from postprocessing modules
- Removed all emojis from CHANGELOG.md
- Removed all emojis from CODE_OF_CONDUCT.md
- Ensures better Unicode compatibility and professional presentation

**Files Updated:**
- `core/train_model.py`: 70 print statements → logging
- `core/data.py`: 46 print statements → logging
- `core/unified_model.py`: 17 print statements → logging
- `core/seasonality.py`: 7 print statements → logging
- `core/trainer.py`: 3 print statements → logging
- `core/visualization.py`: 1 print statement → logging
- `postprocess/analysis.py`: 4 print statements → logging
- `postprocess/response_curves.py`: 4 print statements → logging
- `postprocess/comprehensive_analysis.py`: 65 print statements → logging, 8 emojis removed from plot titles

### Documentation Updates

**JOSS Paper Refinements (`JOSS/paper.md`):**
- Updated test suite description (removed specific count, now references `tests/` directory)
- Updated Zenodo DOI to concept DOI (10.5281/zenodo.16934842) that always resolves to latest version
- Clarified Python 3.9+ and PyTorch 2.0+ requirements
- Removed visualization count claims for verifiability

**arXiv Submission Updates (`JOSS/arxiv_submission/paper_arxiv.tex`):**
- Synced all changes from JOSS paper
- Fixed Robyn description (evolutionary → evolutionary hyperparameter search)
- Ensured consistency across both submission formats

---

## Technical Details

### Logging Configuration

**Default Setup:**
- **Logger Name**: `'deepcausalmmm'`
- **Default Level**: `logging.INFO`
- **Format**: `'%(levelname)s - %(message)s'`
- **Output**: `sys.stdout`

**Logging Levels Used:**
- `logger.debug()`: Detailed diagnostic information
- `logger.info()`: General informational messages
- `logger.warning()`: Warning messages for potential issues
- `logger.error()`: Error messages for failures

**User Control:**
Users can customize logging behavior:
```python
import logging

# Get the package logger
logger = logging.getLogger('deepcausalmmm')

# Change level
logger.setLevel(logging.WARNING)  # Only warnings and errors

# Add custom handler
handler = logging.FileHandler('mmm_training.log')
handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
logger.addHandler(handler)
```

---

## Backward Compatibility

### Maintained Compatibility

**No Breaking Changes:**
- All existing APIs remain unchanged
- Function signatures are identical
- Return values are the same
- Only internal output mechanism changed

**CLI and Examples:**
- Command-line interface (`cli.py`) retains print statements
- Example scripts retain print statements
- User-facing terminal output unchanged

**Migration:**
- No code changes required for existing users
- Upgrade with: `pip install --upgrade deepcausalmmm`

---

## Benefits

### For Library Users

**Better Control:**
- Can suppress verbose output in production
- Can redirect logs to files
- Can integrate with existing logging infrastructure

**Professional Output:**
- Cleaner, more consistent messages
- Proper severity levels (info/warning/error)
- No emoji-related Unicode issues

### For Contributors

**Code Quality:**
- Follows Python best practices for library logging
- More professional codebase
- Easier to debug and maintain

**Testing:**
- Can capture and assert on log messages in tests
- Better visibility into internal operations

---

## Documentation Standards

### Paper Refinements

**Verifiable Claims:**
- Removed exact test count (28 → "comprehensive test suite")
- References actual code location (`tests/` directory)
- Removed unverifiable visualization counts

**Accurate Citations:**
- Updated Zenodo DOI to concept DOI
- Ensures citation always points to latest version
- Matches JOSS requirements

**Requirement Clarity:**
- Confirmed Python 3.9+ requirement
- Confirmed PyTorch 2.0+ requirement
- Matches `pyproject.toml` specifications

---

## Performance

### Model Performance (Unchanged)

- **Training R²**: 0.947
- **Holdout R²**: 0.918
- **Performance Gap**: 3.0%
- **Training RMSE**: 314,692 KPI units (42.8% relative error)
- **Holdout RMSE**: 351,602 KPI units (41.9% relative error)

### Runtime Performance

- No performance impact from logging changes
- Logging overhead is negligible
- Can be disabled entirely if needed

---

## Migration Guide

### Upgrading from v1.0.17

**Installation:**
```bash
pip install --upgrade deepcausalmmm
```

**For Most Users:**
- No changes needed
- Everything works as before
- Upgrade seamlessly

**For Advanced Users (Optional):**

**Control Logging Verbosity:**
```python
import logging

# Quiet mode (warnings and errors only)
logging.getLogger('deepcausalmmm').setLevel(logging.WARNING)

# Verbose mode (include debug messages)
logging.getLogger('deepcausalmmm').setLevel(logging.DEBUG)
```

**Redirect to File:**
```python
import logging

logger = logging.getLogger('deepcausalmmm')
handler = logging.FileHandler('training.log')
handler.setFormatter(logging.Formatter('%(asctime)s - %(message)s'))
logger.addHandler(handler)
```

**Disable Logging:**
```python
import logging
logging.getLogger('deepcausalmmm').disabled = True
```

---

## Quality Assurance

### Testing

- All existing tests pass (100% success rate)
- Logging changes verified across all modules
- No functional regressions detected

### Code Review

- Reviewed all print → logging conversions
- Verified appropriate logging levels
- Confirmed emoji removal completeness

### Documentation

- README.md remains current
- API documentation unchanged
- JOSS paper refined and validated

---

## Future Enhancements

This release sets the foundation for:
- JOSS publication acceptance
- arXiv preprint submission
- Broader academic and industry adoption
- Enhanced testing and CI/CD integration

---

## Acknowledgments

Special thanks to the open-source community and JOSS reviewers whose feedback drove these improvements.

---

## Support

- **Documentation**: [deepcausalmmm.readthedocs.io](https://deepcausalmmm.readthedocs.io/)
- **Issues**: [GitHub Issues](https://github.com/adityapt/deepcausalmmm/issues)
- **JOSS Paper**: Submitted for review
- **arXiv Preprint**: Coming soon

---

**DeepCausalMMM v1.0.18** - Professional, production-ready, and publication-quality code.

