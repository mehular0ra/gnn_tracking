from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Callable

import numpy as np
import optuna

from gnn_tracking.utils.earlystopping import no_early_stopping

metric_type = Callable[[np.ndarray, np.ndarray], float]


class AbstractClusterHyperParamScanner(ABC):
    @abstractmethod
    def scan(self, *args, **kwargs):
        pass


class ClusterHyperParamScanner(AbstractClusterHyperParamScanner):
    def __init__(
        self,
        algorithm: Callable[[np.ndarray, ...], np.ndarray],
        suggest: Callable[[optuna.trial.Trial], dict[str, Any]],
        graphs: list[np.ndarray],
        truth: list[np.ndarray],
        metric: metric_type,
        *,
        cheap_metric: metric_type | None = None,
        early_stopping=no_early_stopping,
    ):
        """Class to scan hyperparameters of a clustering algorithm.

        Args:
            algorithm: Takes graph and keyword arguments
            suggest: Function that suggest parameters to optuna
            graphs:
            truth: Truth labels for clustering
            metric: (Expensive) metric: Callable that takes truth and predicted labels
                and returns float
            cheap_metric: Cheap metric: Callable that takes truth and predicted labels
                and returns float and runs faster than $metric.
            early_stopping: Instance that can be called and has a reset method

        Example:
            # Note: This is also pre-implemented in dbscanner.py

            from sklearn import metrics
            from sklearn.cluster import DBSCAN

            def dbscan(graph, eps, min_samples):
                return DBSCAN(eps=eps, min_samples=min_samples).fit_predict(graph)

            def suggest(trial):
                eps = trial.suggest_float("eps", 1e-5, 1.0)
                min_samples = trial.suggest_int("min_samples", 1, 50)
                return dict(eps=eps, min_samples=min_samples)

            chps = ClusterHyperParamScanner(
                dbscan,
                suggest,
                graphs,
                truths,
                expensive_metric,
                cheap_metric=metrics.v_measure_score,
            )
            study = chps.scan(n_trials=100)
            print(study.best_params)
        """
        self.algorithm = algorithm
        self.suggest = suggest
        self.graphs: list[np.ndarray] = graphs
        self.truth: list[np.ndarray] = truth
        self._es = early_stopping
        self._study = None
        self._cheap_metric = cheap_metric
        self._expensive_metric = metric

    def _objective(self, trial: optuna.trial.Trial) -> float:
        params = self.suggest(trial)
        cheap_foms = []
        all_labels = []
        # Do a first run, looking only at the cheap metric, stopping early
        for i_graph, (graph, truth) in enumerate(zip(self.graphs, self.truth)):
            labels = self.algorithm(graph, **params)
            all_labels.append(labels)
            maybe_cheap_metric: metric_type = (
                self._cheap_metric or self._expensive_metric
            )
            cheap_foms.append(maybe_cheap_metric(truth, labels))
            if i_graph >= 2:
                trial.report(np.nanmean(cheap_foms).item(), i_graph)
            if trial.should_prune():
                raise optuna.TrialPruned()
        if self._cheap_metric is None:
            # What we just evaluated is actually already the expensive metric
            expensive_foms = cheap_foms
        else:
            expensive_foms = []
            # If we haven't stopped early, do a second run, looking at the expensive
            # metric
            for i_labels, (labels, truth) in enumerate(zip(all_labels, self.truth)):
                expensive_fom = self._expensive_metric(truth, labels)
                expensive_foms.append(expensive_fom)
                if i_labels >= 2:
                    trial.report(
                        np.nanmean(expensive_foms).item(), i_labels + len(self.graphs)
                    )
                if trial.should_prune():
                    raise optuna.TrialPruned()
        global_fom = np.nanmean(expensive_foms).item()
        if self._es(global_fom):
            print("Stopped early")
            trial.study.stop()
        return global_fom

    def scan(self, n_trials=100) -> optuna.study.Study:
        """Run scan

        Args:
            n_trials:

        Returns: Study object.
            Call study.best_param to get the best parameters
        """
        self._es.reset()
        if self._study is None:
            self._study = optuna.create_study(
                pruner=optuna.pruners.MedianPruner(),
                direction="maximize",
            )
        self._study.optimize(
            self._objective,
            n_trials=n_trials,
        )
        return self._study