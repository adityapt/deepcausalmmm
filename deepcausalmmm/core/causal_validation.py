"""Diagnostics for the assumptions behind the causal scenario layer.

These routines can *falsify* or qualify parts of a causal story; they cannot
turn observational data into experimental proof.  In particular, passing a
lead placebo, having overlap at zero spend, or finding a stable NOTEARS edge
does not establish that all unmeasured confounding has been removed.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from scipy.stats import pearsonr

from deepcausalmmm.core.conditional_notears import ConditionalNotears, ConditionalNotearsResult


@dataclass(frozen=True)
class MediaSupportDiagnostic:
    """Observed support for the zero-spend intervention for one channel."""

    channel: str
    zero_fraction: float
    minimum: float
    percentile_01: float
    percentile_05: float
    percentile_50: float
    zero_intervention_observed: bool

    def to_dict(self) -> Dict[str, object]:
        return {
            "channel": self.channel,
            "zero_fraction": self.zero_fraction,
            "minimum": self.minimum,
            "percentile_01": self.percentile_01,
            "percentile_05": self.percentile_05,
            "percentile_50": self.percentile_50,
            "zero_intervention_observed": self.zero_intervention_observed,
        }


@dataclass(frozen=True)
class LeadPlaceboDiagnostic:
    """Association of future media innovation with the current adjusted outcome.

    A material lead association is a warning sign for residual confounding,
    reverse timing, or common time dynamics.  It is not a formal causal test:
    persistent media and outcomes can also create a lead association.
    """

    lead_periods: int
    channel: str
    correlation: float
    p_value: float

    def to_dict(self) -> Dict[str, object]:
        return {
            "lead_periods": self.lead_periods,
            "channel": self.channel,
            "correlation": self.correlation,
            "p_value": self.p_value,
        }


@dataclass(frozen=True)
class EdgeStabilityDiagnostic:
    """Selection frequency for an edge across blocked-time bootstrap refits."""

    parent: str
    child: str
    selection_frequency: float
    mean_weight_when_selected: float

    def to_dict(self) -> Dict[str, object]:
        return {
            "parent": self.parent,
            "child": self.child,
            "selection_frequency": self.selection_frequency,
            "mean_weight_when_selected": self.mean_weight_when_selected,
        }


@dataclass(frozen=True)
class CausalValidationReport:
    """Non-identifying checks for a fitted observational causal scenario model."""

    support: Tuple[MediaSupportDiagnostic, ...]
    lead_placebo: Tuple[LeadPlaceboDiagnostic, ...]
    edge_stability: Tuple[EdgeStabilityDiagnostic, ...]
    bootstrap_repetitions: int
    edge_stability_method: str
    warnings: Tuple[str, ...]

    def to_dict(self) -> Dict[str, object]:
        return {
            "scope": (
                "Falsification and robustness diagnostics only; these checks do not "
                "identify causal effects or replace randomized calibration."
            ),
            "support": [item.to_dict() for item in self.support],
            "lead_placebo": [item.to_dict() for item in self.lead_placebo],
            "edge_stability": [item.to_dict() for item in self.edge_stability],
            "bootstrap_repetitions": self.bootstrap_repetitions,
            "edge_stability_method": self.edge_stability_method,
            "warnings": list(self.warnings),
        }


class CausalValidationSuite:
    """Run pre-specified diagnostics without changing the predictive MMM.

    The blocked bootstrap resamples contiguous time blocks jointly for every
    DMA.  This preserves cross-DMA timing and local temporal dependence better
    than row resampling, while making edge selection frequency interpretable as
    a robustness statistic rather than a confidence interval.
    """

    def __init__(
        self,
        lead_periods: int = 4,
        placebo_warning_correlation: float = 0.10,
        support_warning_zero_fraction: float = 0.01,
        discovery_lambda1: float = 0.01,
        discovery_max_outer: int = 12,
        discovery_max_iter: int = 300,
        bootstrap_max_outer: int = 3,
        bootstrap_max_iter: int = 75,
    ) -> None:
        if lead_periods < 1:
            raise ValueError("lead_periods must be at least one")
        if not 0 <= placebo_warning_correlation <= 1:
            raise ValueError("placebo_warning_correlation must be in [0, 1]")
        if not 0 <= support_warning_zero_fraction <= 1:
            raise ValueError("support_warning_zero_fraction must be in [0, 1]")
        if (
            discovery_lambda1 < 0
            or discovery_max_outer < 1
            or discovery_max_iter < 1
            or bootstrap_max_outer < 1
            or bootstrap_max_iter < 1
        ):
            raise ValueError("discovery hyperparameters must be positive (lambda1 non-negative)")
        self.lead_periods = lead_periods
        self.placebo_warning_correlation = placebo_warning_correlation
        self.support_warning_zero_fraction = support_warning_zero_fraction
        self.discovery_lambda1 = discovery_lambda1
        self.discovery_max_outer = discovery_max_outer
        self.discovery_max_iter = discovery_max_iter
        self.bootstrap_max_outer = bootstrap_max_outer
        self.bootstrap_max_iter = bootstrap_max_iter

    @staticmethod
    def _innovations(media: np.ndarray, controls: np.ndarray, dag: ConditionalNotearsResult) -> np.ndarray:
        design = ConditionalNotears._design(controls)
        values = np.log1p(media.reshape(-1, media.shape[-1]))
        return (values - design @ dag.adjustment_coefficients - dag.innovation_mean) / dag.innovation_scale

    @staticmethod
    def _outcome_residual(outcome: np.ndarray, controls: np.ndarray) -> np.ndarray:
        design = ConditionalNotears._design(controls)
        target = np.log1p(np.asarray(outcome, dtype=float).reshape(-1))
        coefficients = np.linalg.lstsq(design, target, rcond=None)[0]
        return (target - design @ coefficients).reshape(outcome.shape)

    def support(self, media: np.ndarray, channel_names: Sequence[str]) -> Tuple[MediaSupportDiagnostic, ...]:
        media = np.asarray(media, dtype=float)
        if media.ndim != 3 or media.shape[-1] != len(channel_names):
            raise ValueError("media and channel_names must align as [DMA, week, channel]")
        return tuple(
            MediaSupportDiagnostic(
                channel=name,
                zero_fraction=float(np.mean(media[..., index] == 0.0)),
                minimum=float(np.min(media[..., index])),
                percentile_01=float(np.quantile(media[..., index], 0.01)),
                percentile_05=float(np.quantile(media[..., index], 0.05)),
                percentile_50=float(np.quantile(media[..., index], 0.50)),
                zero_intervention_observed=bool(np.any(media[..., index] == 0.0)),
            )
            for index, name in enumerate(channel_names)
        )

    def lead_placebo(
        self,
        media: np.ndarray,
        controls: np.ndarray,
        outcome: np.ndarray,
        dag: ConditionalNotearsResult,
    ) -> Tuple[LeadPlaceboDiagnostic, ...]:
        if outcome.shape[1] <= self.lead_periods:
            raise ValueError("lead_periods must be smaller than the number of weeks")
        innovations = self._innovations(media, controls, dag).reshape(media.shape)
        target = self._outcome_residual(outcome, controls)
        current = target[:, :-self.lead_periods].reshape(-1)
        diagnostics = []
        for index, name in enumerate(dag.node_names):
            future = innovations[:, self.lead_periods:, index].reshape(-1)
            correlation, p_value = pearsonr(future, current)
            diagnostics.append(LeadPlaceboDiagnostic(self.lead_periods, name, float(correlation), float(p_value)))
        return tuple(diagnostics)

    @staticmethod
    def _blocked_time_indices(weeks: int, block_size: int, rng: np.random.Generator) -> np.ndarray:
        """Sample a week-length sequence of contiguous blocks with replacement."""
        if not 1 <= block_size <= weeks:
            raise ValueError("block_size must lie between 1 and number of weeks")
        blocks: List[np.ndarray] = []
        while sum(len(block) for block in blocks) < weeks:
            start = int(rng.integers(0, weeks - block_size + 1))
            blocks.append(np.arange(start, start + block_size))
        return np.concatenate(blocks)[:weeks]

    def blocked_bootstrap_edge_stability(
        self,
        media: np.ndarray,
        controls: np.ndarray,
        dag: ConditionalNotearsResult,
        threshold: float,
        repetitions: int,
        block_size: Optional[int] = None,
        seed: int = 42,
        positive_only: bool = True,
    ) -> Tuple[EdgeStabilityDiagnostic, ...]:
        """Refit conditional NOTEARS on blocked-time bootstraps.

        ``repetitions`` should be substantially larger than the small smoke
        tests used in CI (for example, 100+) before using this for a release
        decision.  It is intentionally caller-controlled because each refit is
        computationally expensive.
        """
        if repetitions < 0:
            raise ValueError("repetitions must be non-negative")
        if repetitions == 0:
            return ()
        weeks = media.shape[1]
        size = block_size or max(2, int(np.sqrt(weeks)))
        rng = np.random.default_rng(seed)
        selected: Dict[Tuple[int, int], List[float]] = {}
        for _ in range(repetitions):
            sampled_weeks = self._blocked_time_indices(weeks, size, rng)
            learner = ConditionalNotears(
                lambda1=self.discovery_lambda1,
                max_outer=self.bootstrap_max_outer,
                max_iter=self.bootstrap_max_iter,
            )
            refit = learner.fit(
                media[:, sampled_weeks, :],
                controls[:, sampled_weeks, :],
                list(dag.node_names),
                initial_adjacency=dag.adjacency,
            )
            weights, _ = refit._thresholded_order(threshold, positive_only=positive_only)
            for parent, child in zip(*np.nonzero(weights)):
                selected.setdefault((int(parent), int(child)), []).append(float(weights[parent, child]))
        return tuple(
            EdgeStabilityDiagnostic(
                parent=dag.node_names[parent],
                child=dag.node_names[child],
                selection_frequency=len(weights) / repetitions,
                mean_weight_when_selected=float(np.mean(weights)),
            )
            for (parent, child), weights in sorted(selected.items())
        )

    def run(
        self,
        media: np.ndarray,
        controls: np.ndarray,
        outcome: np.ndarray,
        dag: ConditionalNotearsResult,
        threshold: float = 0.15,
        bootstrap_repetitions: int = 0,
        bootstrap_block_size: Optional[int] = None,
        bootstrap_seed: int = 42,
    ) -> CausalValidationReport:
        support = self.support(media, dag.node_names)
        placebo = self.lead_placebo(media, controls, outcome, dag)
        stability = self.blocked_bootstrap_edge_stability(
            media, controls, dag, threshold, bootstrap_repetitions, bootstrap_block_size, bootstrap_seed
        )
        warnings = [
            f"{item.channel}: do(media=0) is out of observed support (no zero-spend observations)."
            for item in support if not item.zero_intervention_observed
        ]
        warnings.extend(
            f"{item.channel}: future-media lead association is {item.correlation:.3f}; investigate timing or confounding."
            for item in placebo if abs(item.correlation) >= self.placebo_warning_correlation
        )
        if bootstrap_repetitions == 0:
            warnings.append("DAG edge stability was not run; use blocked bootstrap refits before relying on learned paths.")
        method = (
            "not run"
            if bootstrap_repetitions == 0
            else (
                "warm-started bounded NOTEARS stability screen "
                f"({self.bootstrap_max_outer} outer × {self.bootstrap_max_iter} inner iterations per refit)"
            )
        )
        return CausalValidationReport(support, placebo, stability, bootstrap_repetitions, method, tuple(warnings))
