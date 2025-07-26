# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- Initial package structure and core functionality
- GRU-based causal MMM model with Bayesian Network integration
- Adstock and Hill saturation transformations
- Multi-region support
- Comprehensive CLI interface
- Training and inference utilities
- Visualization and analytics tools
- Test suite and documentation

## [0.1.0] - 2024-01-XX

### Added
- Initial release of DeepCausalMMM
- Core model architecture with CausalEncoder and GRUCausalMMM
- Data preprocessing and validation utilities
- Training pipeline with early stopping and validation
- Inference and forecasting capabilities
- Feature importance and contribution analysis
- Causal effects analysis and ROAS calculations
- Command-line interface for easy usage
- Comprehensive configuration system
- Visualization tools for results and analytics
- Test suite with unit tests
- Documentation and examples

### Features
- **Model Architecture**: GRU + Bayesian Network causal structure
- **Data Processing**: Automatic data loading, validation, and preprocessing
- **Training**: Configurable training with validation and early stopping
- **Inference**: Prediction, forecasting, and uncertainty quantification
- **Analytics**: Feature importance, contributions, and causal effects
- **CLI**: Command-line interface for all major operations
- **Visualization**: Comprehensive plotting and reporting tools

### Technical Details
- PyTorch-based implementation
- Support for multiple regions and time series
- Configurable hyperparameters and model architecture
- Production-ready with proper error handling
- Extensive documentation and examples 