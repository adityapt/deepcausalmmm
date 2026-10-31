Migrating to v2.0
=================

What remains unchanged
----------------------

The ``DeepCausalMMM`` GRU model, data pipeline, predictive training workflow,
response curves, and existing predictive outputs remain the baseline MMM
interface. v2 does not place the new observational DAG learner inside the GRU
loss.

What changed
------------

Causal scenario analysis is a mandatory post-fit stage of
``ModelTrainer.train()``. It emits a report, an intervention DAG DOT file, and
a DMA-week-channel CSV. This separation protects the predictive GRU fit from
observational structural assumptions while ensuring every training run refreshes
the learned graph.

Remove configuration that belonged to the abandoned experimental
causal-training path, including ``causal_graph_mode``,
``causal_treatment_model``, and media-share regularization settings claimed as
causal priors. Use the scenario workflow and external evidence for priors or
calibration instead.

Review before adoption
----------------------

1. Confirm all adjustment controls are measured before media and KPI outcomes.
2. Inspect zero-spend support and future-media placebo diagnostics.
3. Inspect the positive, exact intervention DAG rather than raw diagnostic
   NOTEARS edges.
4. Validate important channel effects against holdouts and, where available,
   experiments or geo lift.
5. Do not use overlapping ``do(0)`` totals as additive channel contributions.

Performance
-----------

The NOTEARS solver uses the media-innovation covariance in its least-squares
objective. This avoids recomputing panel-sized products at each optimizer
step. A production DAG fit is still a multi-minute CPU operation; repeated
bootstrap refits should be treated as a bounded screening or offline release
validation task.
