"""Conditional linear NOTEARS for a media-channel structural graph.

The discovery stage is intentionally separate from outcome-model fitting.  It
first removes measured backdoor variables (controls, DMA and seasonal time)
from every channel, then estimates a sparse acyclic graph among the remaining
media innovations.  This avoids allowing an unstable prediction loss to decide
the causal graph.
"""

from dataclasses import dataclass
from typing import List, Optional

import numpy as np
from scipy.linalg import expm
from scipy.optimize import minimize


@dataclass
class ConditionalNotearsResult:
    """A learned media adjacency where rows are parents and columns children."""

    node_names: List[str]
    adjacency: np.ndarray
    acyclicity: float
    objective_history: List[float]
    adjustment_coefficients: np.ndarray
    innovation_mean: np.ndarray
    innovation_scale: np.ndarray

    def edges(self, threshold: float = 0.1):
        return [
            (self.node_names[i], self.node_names[j], float(self.adjacency[i, j]))
            for i in range(len(self.node_names))
            for j in range(len(self.node_names))
            if abs(self.adjacency[i, j]) >= threshold
        ]

    def _thresholded_order(self, threshold: float, positive_only: bool = False):
        """Return an exact-DAG projection and its topological order.

        NOTEARS controls a smooth acyclicity penalty; finite optimization can
        leave tiny cycles.  Thresholding alone may therefore produce a
        discrete cycle even when ``h(W)`` is close to zero.  For interventions
        we greedily retain the largest edges that do not close a directed
        cycle, rather than pretending that a cyclic graph has an SCM meaning.
        """
        candidate = self.adjacency.copy()
        candidate[np.abs(candidate) < threshold] = 0.0
        if positive_only:
            candidate[candidate < 0.0] = 0.0
        count = candidate.shape[0]
        weights = np.zeros_like(candidate)
        children = [set() for _ in range(count)]

        def creates_cycle(parent: int, child: int) -> bool:
            """Adding parent -> child cycles iff child already reaches parent."""
            pending, seen = [child], set()
            while pending:
                node = pending.pop()
                if node == parent:
                    return True
                if node not in seen:
                    seen.add(node)
                    pending.extend(children[node] - seen)
            return False

        ordered_edges = sorted(
            ((abs(candidate[parent, child]), parent, child)
             for parent in range(count) for child in range(count)
             if candidate[parent, child] != 0.0),
            reverse=True,
        )
        for _, parent, child in ordered_edges:
            if not creates_cycle(parent, child):
                weights[parent, child] = candidate[parent, child]
                children[parent].add(child)

        parents = [set(np.flatnonzero(weights[:, child])) for child in range(count)]
        ready = [node for node in range(count) if not parents[node]]
        order = []
        while ready:
            node = ready.pop()
            order.append(node)
            for child in children[node]:
                parents[child].discard(node)
                if not parents[child]:
                    ready.append(child)
        if len(order) != count:  # defensive: the greedy projection must be a DAG
            raise RuntimeError("acyclic projection unexpectedly contains a cycle")
        return weights, order

    def intervene(
        self,
        media: np.ndarray,
        controls: np.ndarray,
        channel: int,
        target: float = 0.0,
        threshold: float = 0.1,
        positive_only: bool = False,
    ) -> np.ndarray:
        """Return media under ``do(channel=target)`` and DAG propagation.

        The graph operates on log-media innovations conditional on the measured
        backdoor variables. The intervention fixes one channel, preserves each
        channel's estimated structural shock, then recomputes all descendants.
        """
        media = np.asarray(media, dtype=float)
        controls = np.asarray(controls, dtype=float)
        if media.ndim != 3 or controls.shape[:2] != media.shape[:2]:
            raise ValueError("media and controls must have matching [DMA, week] dimensions")
        if not 0 <= channel < media.shape[-1] or target < 0:
            raise ValueError("channel must be valid and target must be non-negative")

        design = ConditionalNotears._design(controls)
        values = np.log1p(media.reshape(-1, media.shape[-1]))
        conditional_mean = design @ self.adjustment_coefficients
        innovation = (values - conditional_mean - self.innovation_mean) / self.innovation_scale
        weights, order = self._thresholded_order(threshold, positive_only=positive_only)
        shocks = innovation - innovation @ weights
        counterfactual = innovation.copy()
        target_innovation = (
            np.log1p(target) - conditional_mean[:, channel] - self.innovation_mean[channel]
        ) / self.innovation_scale[channel]

        for node in order:
            if node == channel:
                counterfactual[:, node] = target_innovation
            else:
                counterfactual[:, node] = shocks[:, node] + counterfactual @ weights[:, node]

        transformed = counterfactual * self.innovation_scale + self.innovation_mean + conditional_mean
        return np.maximum(np.expm1(transformed), 0.0).reshape(media.shape)


class ConditionalNotears:
    """Learn a conditional, contemporaneous channel DAG with linear NOTEARS.

    Assumptions: all relevant common causes are included in ``controls`` (plus
    DMA and time fixed effects), the within-period channel system is acyclic,
    and its conditional structural equations are approximately linear.  The
    output is a structure-learning candidate, not causal validation.
    """

    def __init__(
        self,
        lambda1: float = 0.01,
        max_outer: int = 12,
        max_iter: int = 300,
        h_tolerance: float = 1e-6,
    ):
        if lambda1 < 0:
            raise ValueError("lambda1 must be non-negative")
        self.lambda1 = lambda1
        self.max_outer = max_outer
        self.max_iter = max_iter
        self.h_tolerance = h_tolerance

    @staticmethod
    def _acyclicity(weights: np.ndarray) -> float:
        return float(np.trace(expm(weights * weights)) - weights.shape[0])

    @staticmethod
    def _design(controls: np.ndarray) -> np.ndarray:
        n_dma, weeks, _ = controls.shape
        rows = n_dma * weeks
        dma = np.repeat(np.arange(n_dma), weeks)
        week = np.tile(np.arange(weeks), n_dma)
        # DMA fixed effects, a constant, measured controls, plus two annual
        # Fourier harmonics. Drop the reference DMA to avoid collinearity.
        periods = np.array([max(weeks, 1), max(weeks / 2, 1)])
        time = np.concatenate(
            [
                np.sin(2 * np.pi * week[:, None] / periods),
                np.cos(2 * np.pi * week[:, None] / periods),
            ],
            axis=1,
        )
        return np.concatenate(
            [
                np.ones((rows, 1)),
                controls.reshape(rows, -1),
                np.eye(n_dma)[dma, 1:],
                time,
            ],
            axis=1,
        )

    def fit(
        self,
        media: np.ndarray,
        controls: np.ndarray,
        media_names: Optional[List[str]] = None,
    ) -> ConditionalNotearsResult:
        media = np.asarray(media, dtype=float)
        controls = np.asarray(controls, dtype=float)
        if media.ndim != 3 or controls.ndim != 3 or media.shape[:2] != controls.shape[:2]:
            raise ValueError("media and controls must be [DMA, week, feature] arrays")
        if np.any(media < 0):
            raise ValueError("media must be non-negative before log1p transformation")

        _, _, d = media.shape
        names = media_names or [f"channel_{i + 1:02d}" for i in range(d)]
        if len(names) != d:
            raise ValueError("media_names must have one entry per media channel")

        design = self._design(controls)
        values = np.log1p(media.reshape(-1, d))
        adjustment_coefficients = np.linalg.lstsq(design, values, rcond=None)[0]
        residuals = values - design @ adjustment_coefficients
        innovation_mean = residuals.mean(axis=0)
        innovation_scale = residuals.std(axis=0) + 1e-8
        innovations = (residuals - innovation_mean) / innovation_scale
        mask = 1.0 - np.eye(d)
        epsilon = 1e-6
        weights = np.zeros((d, d))
        rho, alpha, previous_h = 1.0, 0.0, np.inf
        objective_history = []

        for _ in range(self.max_outer):
            def objective(flat_weights):
                w = flat_weights.reshape(d, d) * mask
                residual = innovations - innovations @ w
                fit_loss = 0.5 * np.mean(residual * residual)
                h = self._acyclicity(w)
                l1 = np.sum(np.sqrt(w * w + epsilon))
                objective_value = fit_loss + self.lambda1 * l1 + alpha * h + 0.5 * rho * h * h

                grad_h = 2.0 * (expm(w * w).T * w)
                gradient = (
                    -innovations.T @ residual / innovations.shape[0]
                    + self.lambda1 * w / np.sqrt(w * w + epsilon)
                    + (alpha + rho * h) * grad_h
                ) * mask
                return objective_value, gradient.ravel()

            result = minimize(
                objective,
                weights.ravel(),
                jac=True,
                method="L-BFGS-B",
                options={"maxiter": self.max_iter, "ftol": 1e-10, "gtol": 1e-8},
            )
            weights = result.x.reshape(d, d) * mask
            h = self._acyclicity(weights)
            objective_history.append(float(result.fun))
            if h <= self.h_tolerance:
                break
            if h > 0.25 * previous_h:
                rho = min(rho * 10.0, 1e8)
            alpha += rho * h
            previous_h = h

        return ConditionalNotearsResult(
            names, weights, self._acyclicity(weights), objective_history,
            adjustment_coefficients, innovation_mean, innovation_scale,
        )
