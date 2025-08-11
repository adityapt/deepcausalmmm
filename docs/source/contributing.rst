Contributing
============

We welcome contributions to DeepCausalMMM! This guide will help you get started.

Development Setup
-----------------

1. **Fork and clone the repository**:

.. code-block:: bash

    git clone https://github.com/adityapt/deepcausalmmm.git
    cd deepcausalmmm

2. **Create a virtual environment**:

.. code-block:: bash

    python -m venv venv
    source venv/bin/activate  # On Windows: venv\Scripts\activate

3. **Install in development mode**:

.. code-block:: bash

    pip install -e .[dev]

4. **Run tests to verify setup**:

.. code-block:: bash

    python -m pytest tests/ -v

Contributing Guidelines
-----------------------

Zero Hardcoding Philosophy
~~~~~~~~~~~~~~~~~~~~~~~~~~~

* **All parameters must be configurable** via ``config.py``
* **No magic numbers** in model code
* **Dataset agnostic** - works on any MMM dataset
* **Learnable parameters** preferred over fixed constants

Code Quality Standards
~~~~~~~~~~~~~~~~~~~~~~

* **Type hints** for all function parameters and returns
* **Comprehensive docstrings** with examples (Google/NumPy style)
* **Error handling** with informative messages
* **Modular design** with clear separation of concerns

Performance Standards
~~~~~~~~~~~~~~~~~~~~~

* **Benchmark against baseline** before and after changes
* **Maintain generalization** with proper train/holdout validation
* **Document performance impact** in pull requests
* **Training stability** - no coefficient explosion or divergence
* **Business logic** - contributions should be realistic and interpretable

Submitting Changes
------------------

1. **Create a feature branch**:

.. code-block:: bash

    git checkout -b feature/your-feature-name

2. **Make your changes** following the guidelines above

3. **Add tests** for new functionality

4. **Run the test suite**:

.. code-block:: bash

    python -m pytest tests/ -v --cov=deepcausalmmm

5. **Update documentation** if needed

6. **Submit a pull request** with:
   - Clear description of changes
   - Performance impact analysis
   - Test results
   - Documentation updates

Documentation
-------------

We use Sphinx with Read the Docs for documentation. To build docs locally:

.. code-block:: bash

    cd docs/
    make html
    open build/html/index.html

All public APIs should have comprehensive docstrings in Google/NumPy style:

.. code-block:: python

    def example_function(param1: int, param2: str = "default") -> Dict[str, Any]:
        """
        Brief description of the function.
        
        Longer description explaining the purpose and behavior.
        
        Parameters
        ----------
        param1 : int
            Description of param1
        param2 : str, default="default"
            Description of param2
            
        Returns
        -------
        Dict[str, Any]
            Description of return value
            
        Examples
        --------
        >>> result = example_function(42, "test")
        >>> print(result)
        {'status': 'success'}
        """

Testing
-------

We use pytest for testing. Tests should cover:

* **Unit tests** for individual functions/classes
* **Integration tests** for complete workflows
* **Performance tests** for critical paths
* **Edge cases** and error conditions

Run specific test categories:

.. code-block:: bash

    # Unit tests only
    python -m pytest tests/unit/ -v

    # Integration tests
    python -m pytest tests/integration/ -v

    # With coverage report
    python -m pytest tests/ --cov=deepcausalmmm --cov-report=html

Code Style
----------

We follow PEP 8 with some modifications:

* **100-character line limit**
* **Type hints** for all function signatures
* **Descriptive variable names** (no abbreviations)
* **Consistent formatting** using ``black`` formatter

Format your code before submitting:

.. code-block:: bash

    black deepcausalmmm/
    isort deepcausalmmm/

Naming Conventions
~~~~~~~~~~~~~~~~~~

* **Functions/variables**: ``snake_case``
* **Classes**: ``PascalCase``
* **Constants**: ``UPPER_SNAKE_CASE``
* **Private methods**: ``_leading_underscore``

Import Organization
~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    # Standard library imports
    import os
    import sys
    from typing import Dict, List, Optional, Tuple

    # Third-party imports
    import torch
    import torch.nn as nn
    import pandas as pd
    import numpy as np

    # Local imports
    from deepcausalmmm.core.config import get_default_config
    from deepcausalmmm.utils.device import get_device

Recognition
-----------

Contributors will be recognized in:

* **CHANGELOG.md** for each release
* **README.md** contributors section  
* **GitHub releases** with contributor highlights

Community Guidelines
--------------------

* **Be respectful** and constructive in discussions
* **Help others** learn and contribute
* **Share knowledge** through documentation and examples
* **Celebrate successes** and learn from failures

Getting Help
------------

* **GitHub Issues**: For bug reports and feature requests
* **GitHub Discussions**: For questions and general discussion
* **Documentation**: Check the docs first for common questions

Thank you for contributing to DeepCausalMMM! 🚀