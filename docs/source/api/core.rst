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
