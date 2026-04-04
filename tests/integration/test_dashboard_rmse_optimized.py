from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path

import numpy as np


def _load_dashboard_module():
    repo_root = Path(__file__).resolve().parents[2]
    module_path = repo_root / "examples" / "dashboard_rmse_optimized.py"
    spec = spec_from_file_location("dashboard_rmse_optimized", module_path)
    module = module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_load_real_mmm_data_handles_current_pandas():
    module = _load_dashboard_module()
    data_path = Path(__file__).resolve().parents[2] / "examples" / "data" / "MMM Data.csv"

    X_media, X_control, y, media_names, control_names = module.load_real_mmm_data(str(data_path))

    assert X_media.ndim == 3
    assert X_control.ndim == 3
    assert y.ndim == 2
    assert X_media.shape[:2] == X_control.shape[:2]
    assert X_media.shape[:2] == y.shape
    assert len(media_names) == X_media.shape[2]
    assert len(control_names) == X_control.shape[2]
    assert np.isfinite(X_media).all()
    assert np.isfinite(X_control).all()
    assert np.isfinite(y).all()
