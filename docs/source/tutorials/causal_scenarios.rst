Causal scenario workflow
========================

DeepCausalMMM v2 keeps predictive MMM and causal scenario analysis separate.
The GRU MMM remains responsible for predictive modeling. The causal scenario
workflow is a post-fit structural analysis for a panel of non-negative media,
measured controls, and a KPI.

Run the example
---------------

From a cloned repository, run:

.. code-block:: bash

   python examples/run_causal_attribution.py

The SVG renderer requires the Graphviz ``dot`` executable in addition to the
Python ``graphviz`` package. Install it with your system package manager (for
example, ``brew install graphviz`` on macOS).

For a bounded stability screen, add blocked time refits:

.. code-block:: bash

   python examples/run_causal_attribution.py \
     --bootstrap-repetitions 3 \
     --bootstrap-block-size 10 \
     --bootstrap-seed 42

The default is zero refits, which is appropriate for an exploratory run. A
larger number of refits is a release-validation task, not an interactive
default.

Method
------

For media ``M``, measured pre-treatment controls ``C``, DMA indicator ``D``,
and Fourier time terms ``T``, the workflow first fits conditional media
innovations:

.. math::

   Z = \operatorname{standardize}(\log(1 + M) - E[\log(1 + M) \mid C, D, T]).

Conditional NOTEARS learns a sparse acyclic candidate graph among ``Z``. For
interventions, the thresholded graph is projected to an exact DAG and negative
media-to-media edges are excluded by the reporting prior. The outcome equation
is a monotone nonlinear model of ``C, D, T, Z``. A total effect changes a
channel and its learned descendants; a direct effect changes only that
channel's innovation. Their difference is the reported mediated component.

Outputs
-------

``causal_attribution_report.json`` contains aggregate effects, fit metrics,
assumptions, diagnostics, raw diagnostic edges, and the exact intervention
edges. ``causal_attribution_dma_week_channel.csv`` contains one row per DMA,
week, and channel with these fields:

* ``predicted_visits``;
* direct and total counterfactual predictions;
* ``time_varying_direct_effect_visits``;
* ``time_varying_mediated_effect_visits``;
* ``time_varying_total_effect_visits``.

These are time-varying scenario effects because the model conditions on each
row's controls, DMA, seasonality, and media state. They are not independent
weekly coefficients.

Assumptions and validation
--------------------------

The workflow assumes the supplied controls, DMA effects, and time terms block
the relevant measured backdoor paths; that the contemporaneous graph is an
adequate structural approximation; and that the nonlinear outcome model is
well specified. These are not all testable from observational data.

The validation suite can surface problems through zero-spend support checks,
future-media lead-placebo associations, and blocked-time edge-selection
stability. A passed diagnostic is not causal identification. Geo experiments,
lift studies, or other external calibration remain the strongest evidence for
incrementality.

Interpretation
--------------

Single-channel ``do(0)`` total effects overlap in nonlinear models and shared
mediator paths. Their percentages must not be treated as an additive budget
allocation. Use an additive cooperative allocation such as causal Shapley
before presenting channel shares as a partition of total impact.
