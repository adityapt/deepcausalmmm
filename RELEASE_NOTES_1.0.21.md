# DeepCausalMMM 1.0.21 Release Notes

Released: 2026-05-23

This minor release adds **opt-in NOTEARS DAG structure learning** while keeping
the default **upper-triangular** acyclicity mask unchanged for existing workflows.

## Highlights

### NOTEARS causal structure learning (opt-in)
- Set `config['dag_mode'] = 'notears'` to learn channel ordering from data via the
  smooth acyclicity penalty `h(W) = tr(exp(W ⊙ W)) − d` under an augmented Lagrangian.
- Huber-first **warmup** (`notears_warmup_epochs`), **per-channel parent blending**
  in `dag_interaction()`, **temperature-scaled** edges (`dag_temperature`), and
  **column-group L1** (`notears_group_l1`) for focused parent sets.
- New APIs: `h_acyclicity()`, `notears_update_duals(factor=...)`, `threshold_dag(eps=...)`.
- Trainer logs `[NOTEARS]` warmup and dual-update lines when `verbose=True`.

### Bug fixes
- **Dashboard DAG plots** now use `threshold_dag(eps=notears_threshold)` for the
  network chart and CSV export, and masked temperature-scaled adjacency for the
  heatmap—matching the model forward pass instead of raw `sigmoid(adj_logits)`.
- **Viz defaults** aligned: `get_viz_params()` and heatmap use `correlation_threshold=0.05`.
- **Trainer fallbacks** for `notears_lambda1` and `notears_dual_factor` match `config.py`.

### Tests
- `tests/unit/test_notears.py`: forward/backward smoke tests for triangular and NOTEARS,
  acyclicity scalar, dual updates, threshold pruning, warmup gate.
- `tests/unit/test_config.py`: asserts NOTEARS config keys and defaults.

### Documentation
- Sphinx tutorial: `docs/source/tutorials/dag_notears.rst`.
- README NOTEARS section, quickstart cross-links, JOSS Software Design + Summary updates.
- `RELEASE_NOTES_1.0.21.md` and `[1.0.21]` in `CHANGELOG.md`.

## Compatibility

- **Python**: 3.9 – 3.13 (tested in CI).
- **API**: No breaking changes. Default `dag_mode='triangular'` is unchanged.
- **Saved models**: Checkpoints from 1.0.20 load without modification.

## Upgrade

```bash
pip install --upgrade deepcausalmmm==1.0.21
# or latest from the feature branch until PyPI publish:
pip install --upgrade git+https://github.com/adityapt/deepcausalmmm.git@feature/notears-v1.0.21
```

## Enable NOTEARS

```python
from deepcausalmmm.core import get_default_config

config = get_default_config()
config['dag_mode'] = 'notears'
```

See `CHANGELOG.md` and the Sphinx guide
[DAG and NOTEARS structure learning](https://deepcausalmmm.readthedocs.io/en/latest/tutorials/dag_notears.html).
