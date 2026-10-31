Core Model Components
=====================

DeepCausalMMM Model
-------------------

``DeepCausalMMM`` supports two DAG modes via ``dag_mode`` (set in
:func:`~deepcausalmmm.core.config.get_default_config` or passed to the
constructor): **triangular** (default) and **notears**. See
:doc:`../tutorials/dag_notears` for configuration and inspection.

.. automodule:: deepcausalmmm.core.unified_model
   :members:
   :undoc-members:
   :show-inheritance:

Configuration
-------------

.. automodule:: deepcausalmmm.core.config
   :members:
   :undoc-members:
   :show-inheritance:

DAG Model
---------

.. automodule:: deepcausalmmm.core.dag_model
   :members:
   :undoc-members:
   :show-inheritance:

Scaling
-------

.. automodule:: deepcausalmmm.core.scaling
   :members:
   :undoc-members:
   :show-inheritance:

Causal scenario layer
---------------------

The causal scenario modules are post-fit tools. They do not alter the GRU
prediction loss; see :doc:`../tutorials/causal_scenarios` for assumptions and
interpretation.

.. automodule:: deepcausalmmm.core.conditional_notears
   :members:
   :undoc-members:

.. automodule:: deepcausalmmm.core.causal_attribution
   :members:
   :undoc-members:

.. automodule:: deepcausalmmm.core.causal_validation
   :members:
   :undoc-members:

.. automodule:: deepcausalmmm.core.causal_visualization
   :members:
   :undoc-members:
