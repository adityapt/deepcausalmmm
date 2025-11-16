Installation
============

Requirements
------------

* Python 3.8+
* PyTorch 2.0+
* pandas 1.5+
* numpy 1.21+
* plotly 5.11+
* statsmodels 0.13+
* scikit-learn 1.1+

Install from GitHub
-------------------

The recommended way to install DeepCausalMMM is directly from GitHub:

.. code-block:: bash

    pip install git+https://github.com/adityapt/deepcausalmmm.git

Install with Optional Dependencies
----------------------------------

For development and visualization features:

.. code-block:: bash

    # With visualization support
    pip install git+https://github.com/adityapt/deepcausalmmm.git[visualization]

    # With development tools
    pip install git+https://github.com/adityapt/deepcausalmmm.git[dev]

    # Install everything
    pip install git+https://github.com/adityapt/deepcausalmmm.git[dev,visualization,docs]

Development Installation
------------------------

For contributors and developers:

.. code-block:: bash

    # Clone repository
    git clone https://github.com/adityapt/deepcausalmmm.git
    cd deepcausalmmm
    
    # Install in development mode
    pip install -e .
    
    # Or with development dependencies
    pip install -e .[dev]

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