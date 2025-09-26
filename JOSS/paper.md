---
title: "DeepCausalMMM: Deep Learning & Causal Inference for Marketing Mix Modeling"
tags:
  - Python
  - marketing mix modeling
  - causal inference
  - deep learning
  - time series
authors:
  - name: "Aditya Puttaparthi Tirumala"
    orcid: "0009-0008-9495-3932"
    affiliation: 1
affiliations:
  - name: Zillow Group
    index: 1
date: 2025-09-25
bibliography: paper.bib
---

# Summary

**DeepCausalMMM** is a Python package for marketing mix modeling (MMM) that integrates
deep learning with causal structure discovery to estimate the incremental
impact of marketing channels on key business metrics. It uses Gated Recurrent Units (GRUs) to capture temporal dynamics such as lag, carryover, and decay, and learns a Directed Acyclic Graph (DAG) among channels to reveal interdependencies [Zheng2018NOTEARS, Gong2024CausalMMM].

The package is designed for practical use in both industry and research, offering
multi-region support, robust loss functions (e.g., Huber), configurable regularization, and visualization tools. Its "zero hardcoding" philosophy ensures coefficients and modeling decisions can be learned from data or user-configured.

# Statement of Need

Marketing mix modeling is critical for organizations to attribute outcomes to
marketing channels and optimize budgets. Traditional MMM methods rely on regression
or Bayesian hierarchical models, often assuming channels act independently
[Ng2021Bayesian, Runge2024RobynPackaging]. These approaches struggle with
multicollinearity, causal interdependencies, and non-linear or delayed effects.

**DeepCausalMMM** addresses these gaps by combining:

- **Temporal modeling** with GRUs to capture lag and carryover automatically.
- **Causal discovery** via DAG learning among channels [Zheng2018NOTEARS, Gong2024CausalMMM].
- **Multi-region modeling** for geographic or segment-level analysis.
- **Robust estimation** with configurable loss functions and regularization.

This tool serves data scientists, marketing analysts, and researchers seeking
causal and interpretable MMM models beyond predictive accuracy [Li2024Survey, Hanssens2005].

# State of the Field

Existing open-source MMM frameworks include:

- **Robyn (Meta)**: Bayesian hyperparameter search with fixed adstock/saturation [Runge2024RobynPackaging, RobynGitHub].
- **LightweightMMM (Google)**: Bayesian MMM with JAX/Numpyro; supports budget optimization and adstock effects [LightweightMMM2022].
- **PyMC-Marketing**: Bayesian MMM with flexible priors and causal identification in some settings [PyMCMarketing2024].
- **CausalMMM**: Neural approach with causal graph learning and saturation functions [Gong2024CausalMMM].

DeepCausalMMM is distinct in combining GRU-based temporal dynamics, DAG causal
discovery, multi-region modeling, robust regularization, and zero-hardcoding, offering
flexibility and interpretability.

# Functionality

Key features:

- **GRU-based temporal modeling**: automatically learns lag and carryover effects.
- **Causal graph learning**: estimates directed channel interdependencies.
- **Multi-region analysis**: shared and region-specific parameters.
- **Configurable robustness**: Huber loss, L1/L2 penalties, learnable coefficient bounds.
- **Visualization tools**: attribution curves, DAG plots, region-level insights.

Implementation:

- Python package, installable via `pip install deepcausalmmm`.
- Built on PyTorch, pandas, scikit-learn.
- Configurable via default or user-provided config files.
- Documentation and examples: [https://deepcausalmmm.readthedocs.io](https://deepcausalmmm.readthedocs.io).

# Example Usage

```python
from deepcausalmmm import DeepCausalMMM

# X: marketing channel spends (time × channels)
# y: business KPI (e.g., sales, revenue)
# region_labels: optional, for multi-region analysis

model = DeepCausalMMM()

model.fit(
    X=X,
    y=y,
    regions=region_labels,
    config_overrides={
        "loss": "huber",
        "regularization": {"l1": 1e-4, "l2": 1e-3}
    }
)

contributions = model.get_channel_contributions()
model.plot_contributions()
model.plot_dag()
