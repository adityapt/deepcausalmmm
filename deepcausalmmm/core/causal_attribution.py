"""Post-fit accounting for direct and DAG-mediated media effects.

This module deliberately does not alter the predictive MMM.  A learned DAG
only identifies candidate mediator paths; it is *not* a contribution vector.
Callers must supply counterfactual total effects from a separately validated
causal outcome model before mediated credit is reported.
"""

from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import r2_score

from deepcausalmmm.core.conditional_notears import ConditionalNotears, ConditionalNotearsResult


@dataclass(frozen=True)
class CausalAttributionResult:
    """Direct, mediated, and total allocation in outcome units."""

    channel_names: tuple[str, ...]
    direct: np.ndarray
    mediated: np.ndarray
    total: np.ndarray

    @property
    def total_share(self) -> np.ndarray:
        return self.total / self.total.sum().clip(min=1e-12)

    @property
    def positive_total_share(self) -> np.ndarray:
        """Share of positive incremental effects only.

        A signed effect is retained in ``total``: clipping it before reporting
        would conceal evidence that an assumed causal relation is implausible.
        This companion view is only for a conventional non-negative budget
        allocation display.
        """
        positive = np.maximum(self.total, 0.0)
        return positive / positive.sum().clip(min=1e-12)


def combine_direct_and_total_effects(
    channel_names: Sequence[str],
    direct_effect: Sequence[float],
    total_effect: Sequence[float],
) -> CausalAttributionResult:
    """Combine separately estimated direct and intervention total effects.

    ``total_effect`` must come from valid ``do(channel)`` counterfactuals or
    experiments.  Passing NOTEARS adjacency weights here is intentionally
    invalid: graph edges have no outcome-unit interpretation by themselves.
    """
    names = tuple(channel_names)
    direct = np.asarray(direct_effect, dtype=float)
    total = np.asarray(total_effect, dtype=float)
    if direct.ndim != 1 or total.ndim != 1 or len(names) != direct.size or direct.shape != total.shape:
        raise ValueError("names, direct_effect, and total_effect must be aligned one-dimensional vectors")
    if not np.isfinite(direct).all() or not np.isfinite(total).all():
        raise ValueError("effects must be finite")
    return CausalAttributionResult(names, direct, total - direct, total)


@dataclass(frozen=True)
class StructuralCausalEstimate:
    """Model-based causal effects and diagnostics for a fitted media DAG.

    Effects are the observed-vs-``do(media_channel=0)`` change in outcome
    units, summed over DMA-week observations.  They are causal estimates only
    under the stated backdoor and structural assumptions; they are not a
    replacement for a randomized incrementality study.
    """

    attribution: CausalAttributionResult
    outcome_coefficients: np.ndarray
    ridge_alpha: float
    residual_outcome_r2: float
    residual_outcome_cv_r2: float
    dag_acyclicity: float


class StructuralCausalAttributor:
    """Estimate direct and DAG-mediated outcome effects after NOTEARS.

    This is a separate, post-fit partial-linear causal model.  Let ``C`` be
    measured pre-treatment controls, ``M`` log-media innovations after removing
    ``C``, DMA and seasonal terms, and ``W`` the NOTEARS adjacency.  It fits

    ``Y - E[Y|C] = M beta + error``

    and evaluates ``do(M_j=0)`` by holding each fitted structural shock fixed
    while propagating the intervention through ``W``.  The GRU MMM is never
    modified by this class.
    """

    def __init__(self, ridge_alphas: Sequence[float] = (0.0, 0.01, 0.1, 1.0, 10.0, 100.0)):
        alphas = np.asarray(ridge_alphas, dtype=float)
        if alphas.ndim != 1 or not len(alphas) or np.any(alphas < 0):
            raise ValueError("ridge_alphas must be a non-empty sequence of non-negative values")
        self.ridge_alphas = alphas

    @staticmethod
    def _innovations(
        media: np.ndarray, controls: np.ndarray, dag: ConditionalNotearsResult
    ) -> np.ndarray:
        design = ConditionalNotears._design(controls)
        values = np.log1p(media.reshape(-1, media.shape[-1]))
        return (values - design @ dag.adjustment_coefficients - dag.innovation_mean) / dag.innovation_scale

    @staticmethod
    def _ridge_fit(x: np.ndarray, y: np.ndarray, alpha: float) -> np.ndarray:
        return np.linalg.solve(x.T @ x + alpha * np.eye(x.shape[1]), x.T @ y)

    def _select_alpha(self, x: np.ndarray, y: np.ndarray, n_dma: int, weeks: int) -> float:
        """Choose shrinkage using chronological held-out weeks, not row shuffling."""
        if weeks < 12:
            return float(self.ridge_alphas[0])
        # Three expanding-window evaluations preserve shared DMA-week timing.
        split_points = np.unique(np.linspace(max(4, weeks // 2), weeks - 1, 3, dtype=int))
        scores = []
        for alpha in self.ridge_alphas:
            fold_mse = []
            for split in split_points:
                train_rows = np.concatenate([np.arange(dma * weeks, dma * weeks + split) for dma in range(n_dma)])
                test_rows = np.concatenate([np.arange(dma * weeks + split, (dma + 1) * weeks) for dma in range(n_dma)])
                if not len(test_rows):
                    continue
                beta = self._ridge_fit(x[train_rows], y[train_rows], float(alpha))
                fold_mse.append(float(np.mean((y[test_rows] - x[test_rows] @ beta) ** 2)))
            scores.append(np.mean(fold_mse) if fold_mse else np.inf)
        return float(self.ridge_alphas[int(np.argmin(scores))])

    def fit(
        self,
        media: np.ndarray,
        controls: np.ndarray,
        outcome: np.ndarray,
        dag: ConditionalNotearsResult,
        threshold: float = 0.1,
    ) -> StructuralCausalEstimate:
        media = np.asarray(media, dtype=float)
        controls = np.asarray(controls, dtype=float)
        outcome = np.asarray(outcome, dtype=float)
        if media.ndim != 3 or controls.ndim != 3 or outcome.shape != media.shape[:2]:
            raise ValueError("media, controls, and outcome must align as [DMA, week, feature], [DMA, week, control], [DMA, week]")
        if media.shape[-1] != len(dag.node_names):
            raise ValueError("DAG node count must match media channel count")
        if not 0 <= threshold:
            raise ValueError("threshold must be non-negative")

        n_dma, weeks, channels = media.shape
        design = ConditionalNotears._design(controls)
        z = self._innovations(media, controls, dag)
        y = outcome.reshape(-1)
        # Partial out only variables assumed to be pre-treatment backdoors.
        baseline = np.linalg.lstsq(design, y, rcond=None)[0]
        y_residual = y - design @ baseline
        alpha = self._select_alpha(z, y_residual, n_dma, weeks)
        beta = self._ridge_fit(z, y_residual, alpha)
        fitted = z @ beta
        residual_r2 = 1.0 - np.sum((y_residual - fitted) ** 2) / (np.sum(y_residual ** 2) + 1e-12)

        # Report an honest temporal out-of-fold diagnostic for the causal
        # outcome equation.  It need not equal the predictive GRU's R2.
        split = max(4, int(weeks * 0.8))
        train_rows = np.concatenate([np.arange(dma * weeks, dma * weeks + split) for dma in range(n_dma)])
        test_rows = np.concatenate([np.arange(dma * weeks + split, (dma + 1) * weeks) for dma in range(n_dma)])
        if len(test_rows):
            cv_beta = self._ridge_fit(z[train_rows], y_residual[train_rows], alpha)
            cv_r2 = 1.0 - np.sum((y_residual[test_rows] - z[test_rows] @ cv_beta) ** 2) / (np.sum(y_residual[test_rows] ** 2) + 1e-12)
        else:
            cv_r2 = float("nan")

        direct = np.zeros(channels)
        total = np.zeros(channels)
        for channel in range(channels):
            counterfactual_media = dag.intervene(media, controls, channel, target=0.0, threshold=threshold)
            z_cf = self._innovations(counterfactual_media, controls, dag)
            delta = z - z_cf
            direct[channel] = float(np.sum(delta[:, channel] * beta[channel]))
            total[channel] = float(np.sum(delta @ beta))

        return StructuralCausalEstimate(
            attribution=combine_direct_and_total_effects(dag.node_names, direct, total),
            outcome_coefficients=beta,
            ridge_alpha=alpha,
            residual_outcome_r2=float(residual_r2),
            residual_outcome_cv_r2=float(cv_r2),
            dag_acyclicity=float(dag.acyclicity),
        )


@dataclass(frozen=True)
class NonlinearStructuralCausalEstimate:
    """Counterfactual effects from a nonlinear structural outcome equation.

    The three effect tensors have shape ``[DMA, week, channel]``.  They are
    deliberately retained before aggregation so users can inspect the final
    scenario estimate at the same DMA-week resolution as the input panel.
    They are *time-varying effects*, not per-week coefficients: the nonlinear
    response is conditioned on that row's controls, DMA fixed effect, Fourier
    time features, and media state.
    """

    attribution: CausalAttributionResult
    outcome_r2_in_sample: float
    outcome_r2_temporal_holdout: float
    dag_acyclicity: float
    observed_prediction: np.ndarray
    direct_counterfactual_prediction: np.ndarray
    total_counterfactual_prediction: np.ndarray
    direct_effect: np.ndarray
    mediated_effect: np.ndarray
    total_effect: np.ndarray

    def dma_week_channel_frame(
        self,
        media: np.ndarray,
        dma_ids: Optional[Sequence[object]] = None,
        week_ids: Optional[Sequence[object]] = None,
    ) -> pd.DataFrame:
        """Return one audit-ready, time-varying scenario row per DMA-week-channel.

        ``total_effect_visits`` is the model prediction difference between the
        observed row and a ``do(channel=0)`` row with effects propagated along
        the intervention DAG.  ``direct_effect_visits`` holds descendants
        fixed; their difference is the mediated component.  This is the final
        granular output to aggregate by DMA, week, or channel.
        """
        media = np.asarray(media, dtype=float)
        n_dma, weeks, channels = media.shape
        expected = (n_dma, weeks, channels)
        if self.total_effect.shape != expected:
            raise ValueError("media shape does not match this causal estimate")
        dma_ids = np.asarray(dma_ids if dma_ids is not None else np.arange(n_dma))
        week_ids = np.asarray(week_ids if week_ids is not None else np.arange(weeks))
        if dma_ids.shape != (n_dma,) or week_ids.shape != (weeks,):
            raise ValueError("dma_ids and week_ids must align with media dimensions")
        dma_grid = np.broadcast_to(dma_ids[:, None, None], expected).reshape(-1)
        week_grid = np.broadcast_to(week_ids[None, :, None], expected).reshape(-1)
        channel_grid = np.broadcast_to(np.asarray(self.attribution.channel_names)[None, None, :], expected).reshape(-1)
        observed_prediction = np.broadcast_to(self.observed_prediction[:, :, None], expected).reshape(-1)
        return pd.DataFrame(
            {
                "dma": dma_grid,
                "week": week_grid,
                "channel": channel_grid,
                "observed_media": media.reshape(-1),
                "predicted_visits": observed_prediction,
                "direct_counterfactual_predicted_visits": self.direct_counterfactual_prediction.reshape(-1),
                "total_counterfactual_predicted_visits": self.total_counterfactual_prediction.reshape(-1),
                "time_varying_direct_effect_visits": self.direct_effect.reshape(-1),
                "time_varying_mediated_effect_visits": self.mediated_effect.reshape(-1),
                "time_varying_total_effect_visits": self.total_effect.reshape(-1),
            }
        )


class NonlinearStructuralCausalAttributor(StructuralCausalAttributor):
    """Nonlinear structural outcome model with DAG-consistent interventions.

    The fitted equation is ``log1p(Y) = f(C, Z)`` where ``C`` contains only
    backdoor adjustment variables and ``Z`` is the conditional media
    innovation used by NOTEARS.  A direct counterfactual changes one element
    of ``Z`` only; a total counterfactual changes that element and every
    learned descendant through the exact-DAG intervention.  Differences are
    calculated after ``expm1``, so reported effects are visits, not log units.
    """

    def __init__(self, max_iter: int = 300, l2_regularization: float = 10.0):
        self.max_iter = max_iter
        self.l2_regularization = l2_regularization

    def _model(self, n_adjustment_features: int) -> HistGradientBoostingRegressor:
        return HistGradientBoostingRegressor(
            learning_rate=0.05,
            max_iter=self.max_iter,
            max_leaf_nodes=15,
            min_samples_leaf=40,
            l2_regularization=self.l2_regularization,
            # Controls may move outcome either way.  Media innovations are
            # constrained non-decreasing, matching the MMM's positive-media
            # business rule and preventing negative direct counterfactuals.
            monotonic_cst=[0] * n_adjustment_features + [1] * self._channels,
            early_stopping=False,
            random_state=42,
        )

    def fit(
        self,
        media: np.ndarray,
        controls: np.ndarray,
        outcome: np.ndarray,
        dag: ConditionalNotearsResult,
        threshold: float = 0.1,
    ) -> NonlinearStructuralCausalEstimate:
        media = np.asarray(media, dtype=float)
        controls = np.asarray(controls, dtype=float)
        outcome = np.asarray(outcome, dtype=float)
        if media.ndim != 3 or controls.ndim != 3 or outcome.shape != media.shape[:2]:
            raise ValueError("media, controls, and outcome must have aligned DMA-week dimensions")
        if media.shape[-1] != len(dag.node_names):
            raise ValueError("DAG node count must match media channel count")

        n_dma, weeks, channels = media.shape
        self._channels = channels
        design = ConditionalNotears._design(controls)
        z = self._innovations(media, controls, dag)
        features = np.concatenate([design, z], axis=1)
        target = np.log1p(outcome.reshape(-1))
        split = max(4, int(weeks * 0.8))
        train_rows = np.concatenate([np.arange(dma * weeks, dma * weeks + split) for dma in range(n_dma)])
        test_rows = np.concatenate([np.arange(dma * weeks + split, (dma + 1) * weeks) for dma in range(n_dma)])
        validation = self._model(design.shape[1]).fit(features[train_rows], target[train_rows])
        cv_r2 = r2_score(target[test_rows], validation.predict(features[test_rows])) if len(test_rows) else float("nan")

        model = self._model(design.shape[1]).fit(features, target)
        in_sample_r2 = r2_score(target, model.predict(features))
        direct_cf = np.empty((n_dma * weeks, channels))
        total_cf = np.empty((n_dma * weeks, channels))
        observed_visits = np.expm1(model.predict(features))
        for channel in range(channels):
            counterfactual_media = dag.intervene(
                media, controls, channel, target=0.0, threshold=threshold, positive_only=True
            )
            z_total = self._innovations(counterfactual_media, controls, dag)
            z_direct = z.copy()
            z_direct[:, channel] = z_total[:, channel]
            direct_features = np.concatenate([design, z_direct], axis=1)
            total_features = np.concatenate([design, z_total], axis=1)
            direct_cf[:, channel] = np.expm1(model.predict(direct_features))
            total_cf[:, channel] = np.expm1(model.predict(total_features))

        direct_effect = observed_visits[:, None] - direct_cf
        total_effect = observed_visits[:, None] - total_cf
        mediated_effect = total_effect - direct_effect
        direct = direct_effect.sum(axis=0)
        total = total_effect.sum(axis=0)

        return NonlinearStructuralCausalEstimate(
            attribution=combine_direct_and_total_effects(dag.node_names, direct, total),
            outcome_r2_in_sample=float(in_sample_r2),
            outcome_r2_temporal_holdout=float(cv_r2),
            dag_acyclicity=float(dag.acyclicity),
            observed_prediction=observed_visits.reshape(n_dma, weeks),
            direct_counterfactual_prediction=direct_cf.reshape(n_dma, weeks, channels),
            total_counterfactual_prediction=total_cf.reshape(n_dma, weeks, channels),
            direct_effect=direct_effect.reshape(n_dma, weeks, channels),
            mediated_effect=mediated_effect.reshape(n_dma, weeks, channels),
            total_effect=total_effect.reshape(n_dma, weeks, channels),
        )
