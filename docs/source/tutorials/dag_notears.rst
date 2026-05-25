DAG and NOTEARS structure learning
====================================

DeepCausalMMM learns how media channels influence each other through a
**Directed Acyclic Graph (DAG)** embedded in the forward pass. Each channel's
effective input can blend its own signal with a weighted sum of causal parents.

Two modes are available (``config['dag_mode']``):

* **``triangular`` (default)** — acyclicity enforced by an upper-triangular
  adjacency mask. Stable, fast, and backward compatible with earlier releases.
* **``notears`` (opt-in)** — continuous structure learning via the NOTEARS
  smooth penalty `h(W) = tr(exp(W ⊙ W)) − d` (Zheng et al., 2018), optimised
  under an augmented Lagrangian with periodic dual updates.

Triangular mode (default)
-------------------------

No configuration change is required:

.. code-block:: python

    from deepcausalmmm.core import get_default_config

    config = get_default_config()
    assert config['dag_mode'] == 'triangular'

The model learns sparse adjacency weights subject to the triangular mask.
Inspect edges after training with ``model.threshold_dag()`` or the DAG network
plot in ``examples/dashboard_rmse_optimized.py``.

NOTEARS mode (opt-in)
---------------------

Enable data-driven topology discovery:

.. code-block:: python

    from deepcausalmmm.core import get_default_config
    from deepcausalmmm.core.trainer import ModelTrainer

    config = get_default_config()
    config['dag_mode'] = 'notears'

    # Recommended starting points (see config.py for defaults):
    config['notears_warmup_epochs'] = 500   # Huber-only, then enable penalty
    config['notears_lambda1'] = 0.005       # L1 sparsity on adjacency
    config['dag_temperature'] = 0.5           # Sharper {0,1} edge weights
    config['notears_group_l1'] = 0.01         # Focused parents per channel
    config['notears_dual_factor'] = 3.0       # Gentler rho growth when h stalls
    config['notears_dual_update_every'] = 100 # Outer-loop cadence (epochs)

    trainer = ModelTrainer(config)
    # ... create model, prepare data, trainer.train(...) as in quickstart ...

Key config keys
---------------

.. list-table::
   :header-rows: 1
   :widths: 28 72

   * - Key
     - Role
   * - ``dag_mode``
     - ``'triangular'`` or ``'notears'``
   * - ``notears_warmup_epochs``
     - Epochs of Huber-only training before the NOTEARS penalty activates
   * - ``notears_lambda1``
     - L1 sparsity on the learned adjacency
   * - ``notears_rho_init`` / ``notears_alpha_init``
     - Initial augmented-Lagrangian penalty and dual variable
   * - ``notears_dual_update_every``
     - How often ``notears_update_duals()`` runs during training
   * - ``notears_dual_factor``
     - Multiplier applied to ``rho`` when acyclicity progress stalls
   * - ``dag_temperature``
     - Sigmoid temperature for edge weights (``< 1`` sharpens toward {0, 1})
   * - ``notears_group_l1``
     - Column-group L1 encouraging focused parent sets per channel
   * - ``notears_threshold``
     - Pruning cutoff for ``threshold_dag(eps)`` after training
   * - ``visualization.correlation_threshold``
     - Minimum edge weight shown in dashboard DAG plot (often ``0.05`` for NOTEARS)
   * - ``visualization.dag_top_n_edges``
     - Global cap on strongest edges in the dashboard network chart

Training behaviour
------------------

When ``dag_mode='notears'`` and ``notears_warmup_epochs > 0``:

1. **Warmup** — prediction (Huber) loss only; ``notears_active`` is False.
2. **Activation** — at the warmup epoch, the NOTEARS penalty and dual updates
   turn on. Verbose logs print ``[NOTEARS] warmup complete ...``.
3. **Outer loop** — every ``notears_dual_update_every`` epochs,
   ``model.notears_update_duals(factor=notears_dual_factor)`` adjusts
   ``rho`` and ``alpha`` based on the current ``h(W)``.

Huber prediction loss is unchanged; NOTEARS terms are added in
``get_dag_loss()`` only.

Inspecting the learned graph
----------------------------

After training:

.. code-block:: python

    W = model.threshold_dag(eps=0.3)   # pruned adjacency tensor
    print(W)

With ``examples/dashboard_rmse_optimized.py``, the DAG network plot uses global
top-N edges and writes ``dag_adjacency.csv`` beside the HTML output.

API reference
-------------

NOTEARS logic lives on :class:`~deepcausalmmm.core.unified_model.DeepCausalMMM`:

* :meth:`~deepcausalmmm.core.unified_model.DeepCausalMMM.h_acyclicity`
* :meth:`~deepcausalmmm.core.unified_model.DeepCausalMMM.get_dag_loss`
* :meth:`~deepcausalmmm.core.unified_model.DeepCausalMMM.notears_update_duals`
* :meth:`~deepcausalmmm.core.unified_model.DeepCausalMMM.threshold_dag`

Defaults and tunables are in :func:`~deepcausalmmm.core.config.get_default_config`.

See also :doc:`../quickstart` (NOTEARS subsection) and the v1.0.21 entry in
``CHANGELOG.md``.
