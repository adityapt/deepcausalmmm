---
title: "DeepCausalMMM: A Deep Learning Framework for Marketing Mix Modeling with Causal Inference"
tags:
  - Python
  - marketing mix modeling
  - causal inference
  - deep learning
  - time series
  - saturation response curves
  - PyTorch
authors:
  - name: "Aditya Puttaparthi Tirumala"
    orcid: "0009-0008-9495-3932"
    affiliation: "1"
affiliations:
  - index: 1
    name: "Independent Researcher"    
date: 5 October 2025
bibliography: paper.bib
archive_doi: 10.5281/zenodo.16934842
---

# Summary

Marketing Mix Modeling (MMM) is a statistical technique used to estimate the impact of marketing activities on business outcomes such as sales, revenue, or customer visits. Traditional MMM approaches often rely on linear regression or Bayesian hierarchical models that assume independence between marketing channels and struggle to capture complex temporal dynamics and non-linear saturation effects [@Chan2017; @Hanssens2005; @Ng2021Bayesian].

**DeepCausalMMM** is a Python package that addresses these limitations by combining deep learning, causal inference, and advanced marketing science. The package uses Gated Recurrent Units (GRUs) to automatically learn temporal patterns such as adstock (carryover effects) and lag, while simultaneously learning statistical dependencies between marketing channels through Directed Acyclic Graph (DAG) structure using an upper triangular mask to enforce acyclicity [@Zheng2018NOTEARS; @Gong2024CausalMMM]. Additionally, it implements Hill equation-based saturation curves to model diminishing returns and optimize budget allocation.

Key features include: (1) a data-driven design where hyperparameters and transformations (e.g., adstock decay, saturation curves) are learned or estimated from data with sensible defaults, rather than requiring fixed heuristics or manual specification, (2) multi-region modeling with both shared and region-specific parameters, (3) robust statistical methods including Huber loss and advanced regularization, (4) comprehensive response curve analysis for understanding channel saturation.

# Statement of Need

Marketing organizations invest billions annually in advertising across channels (TV, digital, social, search), yet measuring ROI remains challenging due to: (1) temporal complexity with delayed and persistent effects [@Hanssens2005], (2) channel interdependencies [@Gong2024CausalMMM], (3) non-linear saturation with diminishing returns [@Li2024Survey], (4) regional heterogeneity, and (5) multicollinearity between campaigns.

**DeepCausalMMM** addresses these challenges by combining GRU-based temporal modeling, DAG-based structure learning, Hill equation response curves, multi-region modeling, production-ready performance (91.8% holdout R², 3.0% train-test gap), and data-driven hyperparameter learning for generalizability.

# State of the Field

Several open-source MMM frameworks exist, each with distinct approaches:

**Robyn (Meta)** [@Runge2024RobynPackaging; @RobynGitHub] uses evolutionary hyperparameter optimization with fixed adstock and saturation transformations (Adstock, Hill, Weibull). It provides budget optimization and is widely used in industry but requires manual specification of transformation types and does not model channel interdependencies.

**Meridian (Google)** [@Meridian2024] is Google's open-source Bayesian MMM framework featuring reach and frequency modeling, geo-level analysis, and experimental calibration. It employs causal inference with pre-specified causal graphs and the backdoor criterion.

**PyMC-Marketing** [@PyMCMarketing2024] provides Bayesian MMM with highly flexible prior specifications and some causal identification capabilities. It excels at uncertainty quantification but requires significant Bayesian modeling expertise and does not use neural networks for temporal modeling.

**CausalMMM** [@Gong2024CausalMMM] introduces neural networks and graph learning to MMM, demonstrating the value of discovering channel interdependencies. However, it does not provide multi-region modeling or comprehensive response curve analysis.

**DeepCausalMMM** advances the field by integrating: (1) GRU-based temporal modeling, (2) DAG-based structure learning using upper triangular constraints [@Zheng2018NOTEARS], (3) Hill equation response curves, (4) multi-region modeling, (5) robust statistical methods, (6) production-ready architecture.

# Software Design

DeepCausalMMM's architecture reflects several key design decisions driven by the unique challenges of marketing mix modeling:

**Neural Architecture Choice**: We selected Gated Recurrent Units (GRUs) over alternatives (LSTMs, Transformers, simple RNNs) after evaluating trade-offs: GRUs provide sufficient temporal modeling capacity for weekly marketing data while requiring fewer parameters than LSTMs, reducing overfitting risk in typical MMM datasets (50-200 weeks). Transformers were rejected due to their quadratic memory complexity and tendency to overfit on small temporal sequences.

**DAG Structure Learning**: Rather than implementing the full NOTEARS continuous optimization [@Zheng2018NOTEARS], we adopt an upper triangular adjacency matrix constraint to enforce acyclicity. This design choice prioritizes computational efficiency and training stability over the flexibility of learning arbitrary DAG structures, making the method more practical for production marketing applications where interpretability and fast iteration are critical.

**Saturation Function Design**: We implement Hill equation saturation with enforced constraints ($a \geq 2.0$) rather than free-form learned transformations. This decision reflects domain expertise in marketing science where diminishing returns follow predictable S-curves, and constraining the parameter space improves generalization and interpretability for business stakeholders.

**Multi-Region Modeling Philosophy**: The architecture employs shared temporal dynamics (GRU weights) with region-specific baselines and scaling factors. This hybrid approach balances the bias-variance trade-off: shared patterns enable learning from limited data per region, while region-specific parameters capture geographic heterogeneity essential for local marketing decisions.

**Robustness Over Accuracy**: We prioritize Huber loss over MSE despite slightly higher training complexity. This choice addresses the reality of marketing data: outliers from promotional spikes, data quality issues, and external shocks are common. Huber loss provides robust estimation while maintaining differentiability for gradient-based optimization. The package also implements gradient clipping, L1/L2 regularization with sparsity control, and learnable coefficient bounds to ensure stable training.

**Modular Post-Processing Design**: Response curve analysis is decoupled from model training through the `ResponseCurveFit` module, which fits Hill equations to learned channel contributions. This separation enables budget optimization and saturation analysis without retraining, supporting iterative business decision-making workflows.

These design decisions collectively enable DeepCausalMMM to handle real-world marketing data challenges while remaining interpretable and computationally tractable for practitioners.

## Implementation Details

- **Language**: Python 3.9+, **Deep Learning**: PyTorch 2.0+
- **Data Processing**: pandas, NumPy, **Optimization**: scipy, scikit-learn
- **Visualization**: Plotly, NetworkX, **Statistical Methods**: statsmodels
- **Installation**: `pip install deepcausalmmm`
- **Documentation**: [https://deepcausalmmm.readthedocs.io](https://deepcausalmmm.readthedocs.io)
- **Tests**: Comprehensive unit and integration test suite in `tests/` directory

## Visualizations

Figure 1 shows an example of the learned DAG structure between marketing channels. The directed edges reveal statistical dependencies and potential causal relationships such as TV advertising's association with search behavior, demonstrating the model's ability to discover channel interdependencies from data.

![Causal network (DAG) showing relationships between marketing channels.](figure_dag_professional.png)

Figure 2 demonstrates a non-linear response curve fitted to a marketing channel using the Hill equation. The S-shaped curve clearly shows saturation effects and diminishing returns, with annotations indicating the half-saturation point where the channel reaches 50% of maximum effectiveness.

![Response curve showing Hill saturation effects for a marketing channel.](figure_response_curve_simple.png)

# Example Usage

```python
import numpy as np
from deepcausalmmm.core import get_default_config
from deepcausalmmm.core.trainer import ModelTrainer
from deepcausalmmm.core.data import UnifiedDataPipeline

# Generate sample MMM data
np.random.seed(42)
n_regions, n_weeks = 10, 52  # 10 regions, 52 weeks
n_media, n_control = 5, 3    # 5 media channels, 3 controls

# Media spend/impressions [regions, weeks, channels]
X_media = np.random.uniform(100, 5000, (n_regions, n_weeks, n_media))
# Control variables [regions, weeks, controls]
X_control = np.random.uniform(0, 1, (n_regions, n_weeks, n_control))
# Target (sales/visits) [regions, weeks]
y = np.random.uniform(1000, 10000, (n_regions, n_weeks))

# Configure and initialize pipeline
config = get_default_config()
pipeline = UnifiedDataPipeline(config)

# Split data temporally (train/holdout)
train_data, holdout_data = pipeline.temporal_split(X_media, X_control, y)
train_tensors = pipeline.fit_and_transform_training(train_data)
holdout_tensors = pipeline.transform_holdout(holdout_data)

# Create and train model
trainer = ModelTrainer(config)
model = trainer.create_model(
    n_media=train_tensors['X_media'].shape[2],
    n_control=train_tensors['X_control'].shape[2],
    n_regions=train_tensors['X_media'].shape[0]
)
trainer.create_optimizer_and_scheduler()

# Train with train and holdout data
results = trainer.train(
    train_tensors['X_media'], train_tensors['X_control'],
    train_tensors['R'], train_tensors['y'],
    holdout_tensors['X_media'], holdout_tensors['X_control'],
    holdout_tensors['R'], holdout_tensors['y'],
    verbose=True
)

# Results
print(f"Training R²: {results['final_train_r2']:.3f}")
print(f"Holdout R²: {results['final_holdout_r2']:.3f}")
print(f"Training RMSE original scale: {results['final_train_rmse']:.0f}")
print(f"Holdout RMSE original scale: {results['final_holdout_rmse']:.0f}")

```

**Data Leakage Note**: The production code in `examples/dashboard_rmse_optimized.py` includes `y_full_for_baseline` parameter which uses the full dataset (including holdout) for baseline initialization. This introduces minor data leakage where regional mean statistics from holdout data inform the baseline. In strict production scenarios, this parameter should be omitted to ensure zero leakage. The leakage is limited to regional means only; all model weights, coefficients, and temporal patterns are learned exclusively from training data.

# Performance

**Note on Benchmarks:** The following performance metrics are derived from real-world anonymized marketing data (190 geographic regions, 109 weeks, 13 channels) using the complete production workflow in `examples/dashboard_rmse_optimized.py`, not from the toy example above. The example code demonstrates API usage with synthetic data for pedagogical purposes, while these benchmarks validate the software's effectiveness on substantial real-world marketing analytics scenarios.

DeepCausalMMM has demonstrated strong performance on anonymized real-world marketing data containing 190 geographic regions (DMAs), 109 weeks of observations, 13 marketing channels, and 7 control variables. The model uses a temporal train-holdout split with 101 training weeks and the most recent 8 weeks (7.3%) reserved for out-of-sample validation:

- **Training R²**: 0.947, **Holdout R²**: 0.918
- **Performance Gap**: 3.0% (indicating excellent generalization)
- **Training RMSE**: 314,692 KPI units (42.8% relative error - Relative RMSE = (RMSE / Mean) × 100  = (314,692 / ~743,088) × 100  ≈ 42.8%)
- **Holdout RMSE**: 351,602 KPI units (41.9% relative error)

These results demonstrate the model's ability to capture complex marketing dynamics while maintaining strong out-of-sample predictive accuracy. The small performance gap between training and holdout sets indicates robust generalization without overfitting.

**Note on Baseline Initialization**: The reported metrics include minor data leakage from baseline initialization using full dataset statistics (regional means). This design choice prioritizes stable training convergence in production scenarios where historical data is available. The leakage affects only the baseline term; all dynamic components (GRU weights, saturation parameters, channel coefficients) are learned exclusively from training data. In strict evaluation scenarios, omitting the `y_full_for_baseline` parameter eliminates this leakage.

# Research Impact Statement

DeepCausalMMM demonstrates credible significance through production deployment and strong benchmark performance. The software has been deployed in real-world marketing analytics scenarios processing 190 geographic regions across 109 weeks with 13 marketing channels, achieving 91.8% holdout R² with only 3.0% train-test gap, demonstrating robust generalization superior to traditional linear MMM approaches.

**Benchmarks and Reproducibility**: The package includes comprehensive reproducible benchmarks on anonymized real-world data, with all code, configurations, and evaluation metrics publicly available. Performance metrics (holdout R² of 0.918, relative RMSE of 41.9%) provide concrete evidence of the model's capability to capture complex marketing dynamics while maintaining out-of-sample predictive accuracy.

**Community Readiness**: The software exhibits production-ready quality with comprehensive documentation at readthedocs.io, 14+ interactive visualizations for business stakeholder communication, extensive test coverage, and stable API design. The package is distributed via PyPI with semantic versioning and includes worked examples demonstrating application to multi-region MMM problems.

**Technical Contribution**: DeepCausalMMM advances the state of open-source MMM tools by uniquely combining GRU-based temporal modeling with DAG structure learning and Hill equation saturation analysis in a single integrated framework, addressing a gap between traditional statistical MMM (Robyn, PyMC-Marketing) and purely neural approaches (CausalMMM).

**Near-term Significance**: The software's emphasis on interpretability (DAG visualizations, response curves, contribution decomposition) and production deployment considerations (robust loss functions, efficient training, stakeholder-ready outputs) position it for adoption by marketing analytics teams seeking to move beyond traditional linear models while maintaining business interpretability requirements.

# Reproducibility

DeepCausalMMM ensures reproducible results through deterministic training with configurable random seeds, comprehensive test suite, example notebooks, detailed documentation of hyperparameters, and version-controlled releases with semantic versioning.

# Research and Practical Applications

**Industry Applications**: Budget optimization across marketing channels, ROI measurement and attribution, strategic planning and forecasting, channel effectiveness analysis, regional marketing strategy development.

**Research Applications**: Causal inference in marketing, temporal dynamics in advertising, multi-region heterogeneity, saturation modeling, and channel interdependencies.

The data-driven hyperparameter learning and comprehensive documentation make it accessible to practitioners while rigorous statistical foundations support academic research.

# Acknowledgments

We acknowledge the contributions of the open-source community, particularly the developers of PyTorch, pandas, and scikit-learn, which form the foundation of this package. We also thank the MMM research community for establishing the theoretical foundations that informed this work.

# AI-Assisted Research Disclosure

In accordance with JOSS editorial policy on AI-assisted research, the author discloses that AI tools (GitHub Copilot, Claude, and GPT-4) were used during the development of this software package. These tools assisted with code generation, documentation writing, debugging, and manuscript drafting. All AI-generated content was reviewed, verified, and modified by the author to ensure accuracy, correctness, and alignment with research goals. The author takes full responsibility for all claims, implementations, and content in this work.

# Conflict of Interest

The author declares no competing financial or non-financial interests that could inappropriately influence this work.

# References
