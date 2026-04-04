Installation
============

Requirements
------------

These match ``pyproject.toml`` (``requires-python`` and runtime dependencies):

* Python **3.9+**
* PyTorch **2.0+**
* pandas **1.5+**
* numpy **1.21+** (currently pinned as **< 2.0** in the package metadata)
* scipy **1.7+**
* plotly **5.0+**
* NetworkX **2.6+**
* scikit-learn **1.0+**
* statsmodels **0.13+**
* tqdm **4.60+**

Install from PyPI
-----------------

.. code-block:: bash

    pip install deepcausalmmm

Install from GitHub
-------------------

For the latest commit on the default branch:

.. code-block:: bash

    pip install git+https://github.com/adityapt/deepcausalmmm.git

Optional dependencies
---------------------

The package declares a **test** extra (pytest tooling) for contributors:

.. code-block:: bash

    pip install deepcausalmmm[test]
    # or, from a clone:
    pip install -e .[test]

Documentation builds use ``docs/requirements.txt`` (Sphinx, theme, and doc import dependencies).

Development Installation
------------------------

For contributors and developers:

.. code-block:: bash

    # Clone repository
    git clone https://github.com/adityapt/deepcausalmmm.git
    cd deepcausalmmm
    
    # Install in development mode
    pip install -e .

    # With test dependencies (pytest, coverage)
    pip install -e .[test]

Verify Installation
-------------------

Test that the package is installed correctly:

.. code-block:: python

    from deepcausalmmm import DeepCausalMMM, get_device
    from deepcausalmmm.core import get_default_config

    print("DeepCausalMMM package imported successfully!")
    print(f"Device: {get_device()}")

GPU Support
-----------

DeepCausalMMM automatically detects and uses CUDA if available. For GPU acceleration:

1. Install PyTorch with CUDA support:

.. code-block:: bash

    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

2. Verify GPU detection:

.. code-block:: python

    from deepcausalmmm.utils.device import get_device
    device = get_device()
    print(f"Using device: {device}")

Docker Installation
-------------------

A Docker image will be available soon for easy deployment:

.. code-block:: bash

    # Coming soon
    docker pull deepcausalmmm/deepcausalmmm:latest

Troubleshooting
---------------

**Import Errors**
    Make sure all dependencies are installed. Try reinstalling with ``pip install --upgrade``.

**CUDA Issues**
    Ensure your PyTorch installation matches your CUDA version. Check with ``torch.cuda.is_available()``.

**Memory Issues**
    For large datasets, consider reducing batch size or using gradient checkpointing.

**Performance Issues**
    Enable mixed precision training and use appropriate hardware (GPU recommended for large models).