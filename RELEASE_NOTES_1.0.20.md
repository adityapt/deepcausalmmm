# DeepCausalMMM 1.0.20 Release Notes

Released: 2026-04-18

This is a patch release that finalizes the JOSS review cycle for the paper
*"DeepCausalMMM: A Deep Learning Framework for Marketing Mix Modeling with Causal Structure Learning"*.
It ships bug fixes and documentation improvements collected since 1.0.19 and is the version
archived on Zenodo for the JOSS submission.

## Highlights

### Bug fixes
- **Inference forward-pass contract aligned with callers** (commit `3f4fc56`, PR #11).
  `InferenceManager.predict()`, `UnifiedDataPipeline.predict_and_postprocess`, and the
  `train_model` paths now unpack `(predictions, media_coeffs, media_contributions, outputs)`
  and source control contributions from `outputs['control_contributions']`.
  This fixes: wrong media/control tensors when `n_media != n_control`, a broken 3-value
  tuple unpack, and `return_contributions=False` incorrectly returning a tuple.
- **Benchmark loader on pandas 3** (commit `bd0a434`, PR #12). Region-wise missing-value
  fill in `examples/dashboard_rmse_optimized.py` replaces the deprecated grouped
  `fillna(method=...)` with `groupby(...).transform(lambda s: s.ffill().bfill())`.
- **Log-scaling residue cleanup** (commit `e5157be`). Removed the vestigial
  `log_transform` parameter from `core/scaling.py` and renamed ~30 `_log` variables
  to `_scaled` across `core/scaling.py`, `core/train_model.py`, and `core/trainer.py`.
  Behavior matches 1.0.19; this only removes dead parameters and improves clarity.

### JOSS paper (no code impact)
- `JOSS/paper.bib`: fixed malformed citations flagged by reviewer
  (`Meridian2024`, `PyMCMarketing2024`, `RobynGitHub`) by double-bracing corporate
  author names and protecting product names in titles.
- `JOSS/paper.md`: aligned `archive_doi` with the Zenodo concept/version record,
  updated Table 1 and Reproducibility section to match `examples/mmm_three_way_benchmark.ipynb`.

### Tests
- New `tests/integration/test_dashboard_rmse_optimized.py` covering the
  `examples/data/MMM Data.csv` data path used in the paper.
- New `tests/unit/test_inference.py` covering `InferenceManager.predict()` with
  `return_contributions` true/false; shape/consistency assertions on coefficients
  vs. contributions vs. `outputs`.

### Documentation
- `docs/source/installation.rst`, `README.md`, `CONTRIBUTING.md` aligned with the actual
  `pyproject.toml` dependency list (Python 3.9+, `scipy`/`networkx`/`tqdm` listed,
  numpy `<2.0` note, `[test]` extra documented, `[dev]`/`[visualization]`/`[docs]` extras
  removed).
- Sphinx: added missing `docs/source/api/cli.rst`, `visualization.rst`, and example pages
  (`retail_mmm.rst`, `multi_region.rst`) so API and Examples toctrees resolve.
- `CITATION.cff` title aligned with the JOSS paper title, version bumped to `1.0.20`.
- `RELEASE_NOTES_1.0.20.md` (this file) and corresponding `[1.0.20]` section added to
  `CHANGELOG.md`.

## Compatibility

- **Python**: 3.9 – 3.13 (tested in CI).
- **API**: No breaking changes relative to 1.0.19. The only public-surface changes
  are the inference unpack contract (already broken in 1.0.19; now correct) and
  the removal of the previously no-op `log_transform` parameter from internal
  scaling utilities.
- **Saved models**: Checkpoints from 1.0.19 load and run without modification.

## Upgrade

```bash
pip install --upgrade deepcausalmmm==1.0.20
```

## JOSS archive

This release is the version archived on Zenodo for the JOSS submission; the DOI is
posted on the JOSS review thread.
