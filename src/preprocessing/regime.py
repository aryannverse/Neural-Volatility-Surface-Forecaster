from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from statsmodels.tsa.regime_switching.markov_regression import MarkovRegression


@dataclass(frozen=True)
class RegimeConfig:
    n_regimes: int = 3
    random_state: int = 7


class RegimeTagger:
    def __init__(self, config: RegimeConfig | None = None):
        self.config = config or RegimeConfig()
        self._kmeans: KMeans | None = None

    @staticmethod
    def realized_vol(returns: pd.Series, window: int = 20) -> pd.Series:
        return returns.rolling(window).std() * np.sqrt(252.0)

    def fit_predict_from_surfaces(self, surfaces: np.ndarray) -> np.ndarray:
        """
        Regime clustering on compressed surface stats:
        [level, skew, smile curvature, term slope]
        """
        lvl = surfaces.mean(axis=(1, 2))
        skew = np.median(np.gradient(surfaces, axis=2), axis=(1, 2))
        curv = np.median(np.gradient(np.gradient(surfaces, axis=2), axis=2), axis=(1, 2))
        term = np.median(np.gradient(surfaces, axis=1), axis=(1, 2))
        X = np.column_stack([lvl, skew, curv, term])
        self._kmeans = KMeans(
            n_clusters=self.config.n_regimes,
            n_init=20,
            random_state=self.config.random_state,
        )
        return self._kmeans.fit_predict(X)

    def hidden_markov_regimes(self, series: np.ndarray) -> np.ndarray:
        """
        Markov-switching proxy for hidden regimes using statsmodels.
        The series is typically IV level or realized volatility.
        """
        x = pd.Series(series).dropna().astype(float)
        if len(x) < 40:
            raise ValueError("Need at least 40 observations for Markov regime estimation.")
        model = MarkovRegression(x, k_regimes=self.config.n_regimes, trend="c", switching_variance=True)
        res = model.fit(disp=False)
        probs = res.smoothed_marginal_probabilities.to_numpy()
        return probs.argmax(axis=1)
