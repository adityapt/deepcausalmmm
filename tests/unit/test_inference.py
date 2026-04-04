import pytest
import numpy as np
import torch

from deepcausalmmm import DeepCausalMMM
from deepcausalmmm.core.inference import InferenceManager


def _build_model_and_inputs():
    torch.manual_seed(0)
    model = DeepCausalMMM(n_media=3, ctrl_dim=2, n_regions=2)
    model.eval()

    X_media = torch.randn(2, 5, 3)
    X_control = torch.randn(2, 5, 2)
    regions = torch.zeros(2, dtype=torch.long)

    return model, X_media, X_control, regions


def test_inference_manager_predict_returns_expected_contributions():
    model, X_media, X_control, regions = _build_model_and_inputs()
    with pytest.warns(
        UserWarning,
        match="Neither pipeline nor scaler provided",
    ):
        manager = InferenceManager(model)

    with torch.no_grad():
        predictions, media_coeffs, media_contributions, outputs = model(X_media, X_control, regions)

    results = manager.predict(
        X_media.numpy(),
        X_control.numpy(),
        return_contributions=True,
        remove_padding=False,
    )

    assert np.allclose(results['predictions'], predictions.cpu().numpy())
    assert np.allclose(results['media_contributions'], media_contributions.cpu().numpy())
    assert np.allclose(results['control_contributions'], outputs['control_contributions'].cpu().numpy())
    assert media_coeffs.shape == X_media.shape
    assert outputs['control_contributions'].shape == X_control.shape


def test_inference_manager_predict_without_contributions_returns_predictions_only():
    model, X_media, X_control, regions = _build_model_and_inputs()
    with pytest.warns(
        UserWarning,
        match="Neither pipeline nor scaler provided",
    ):
        manager = InferenceManager(model)

    with torch.no_grad():
        predictions, _, _, _ = model(X_media, X_control, regions)

    results = manager.predict(
        X_media.numpy(),
        X_control.numpy(),
        return_contributions=False,
        remove_padding=False,
    )

    assert list(results.keys()) == ['predictions']
    assert np.allclose(results['predictions'], predictions.cpu().numpy())


def test_inference_manager_predict_can_return_media_coefficients():
    model, X_media, X_control, regions = _build_model_and_inputs()
    with pytest.warns(
        UserWarning,
        match="Neither pipeline nor scaler provided",
    ):
        manager = InferenceManager(model)

    with torch.no_grad():
        _, media_coeffs, _, _ = model(X_media, X_control, regions)

    results = manager.predict(
        X_media.numpy(),
        X_control.numpy(),
        return_contributions=False,
        remove_padding=False,
        return_media_coefficients=True,
    )

    assert set(results.keys()) == {'predictions', 'media_coefficients'}
    assert np.allclose(results['media_coefficients'], media_coeffs.cpu().numpy())
