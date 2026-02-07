"""Market regime detection."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler


class RegimeDetector:
    """Detect market regimes from return series."""

    def detect_kmeans(
        self, returns: pd.Series, n_regimes: int = 3
    ) -> pd.Series:
        """KMeans-based regime detection.

        Features used: rolling return, rolling volatility.

        Parameters
        ----------
        returns : pd.Series
            Period return series.
        n_regimes : int
            Number of regimes to detect (default 3: bear/neutral/bull).

        Returns
        -------
        pd.Series
            Integer regime labels (sorted so 0=lowest mean return, etc.).
        """
        if returns.empty or len(returns) < 20:
            return pd.Series(dtype=int, index=returns.index)

        window = min(20, len(returns) // 2)
        roll_ret = returns.rolling(window, min_periods=1).mean()
        roll_vol = returns.rolling(window, min_periods=1).std().fillna(0)

        features = pd.DataFrame(
            {"roll_ret": roll_ret, "roll_vol": roll_vol},
            index=returns.index,
        ).dropna()

        if len(features) < n_regimes:
            return pd.Series(0, index=returns.index, dtype=int)

        scaler = StandardScaler()
        X = scaler.fit_transform(features.values)

        km = KMeans(n_clusters=n_regimes, n_init=10, random_state=42)
        raw_labels = km.fit_predict(X)

        # Sort labels by cluster centroid mean return (ascending)
        centroids_ret = []
        for i in range(n_regimes):
            mask = raw_labels == i
            centroids_ret.append(features["roll_ret"].values[mask].mean())
        order = np.argsort(centroids_ret)
        label_map = {old: new for new, old in enumerate(order)}
        sorted_labels = np.array([label_map[l] for l in raw_labels])

        result = pd.Series(
            np.nan, index=returns.index, dtype=float, name="regime"
        )
        result.loc[features.index] = sorted_labels.astype(float)
        return result.astype("Int64")

    def detect_hmm(
        self, returns: pd.Series, n_regimes: int = 3
    ) -> pd.Series:
        """Hidden Markov Model regime detection.

        Requires the ``hmmlearn`` package. Raises ImportError with a helpful
        message if not installed.

        Parameters
        ----------
        returns : pd.Series
            Period return series.
        n_regimes : int
            Number of hidden states.

        Returns
        -------
        pd.Series
            Integer regime labels sorted by mean return.
        """
        try:
            from hmmlearn.hmm import GaussianHMM
        except ImportError:
            raise ImportError(
                "hmmlearn is required for HMM regime detection. "
                "Install it with: pip install hmmlearn"
            )

        if returns.empty or len(returns) < 20:
            return pd.Series(dtype=int, index=returns.index)

        clean = returns.dropna()
        X = clean.values.reshape(-1, 1)

        model = GaussianHMM(
            n_components=n_regimes,
            covariance_type="full",
            n_iter=100,
            random_state=42,
        )
        model.fit(X)
        raw_labels = model.predict(X)

        # Sort labels by state mean (ascending)
        means = model.means_.flatten()
        order = np.argsort(means)
        label_map = {old: new for new, old in enumerate(order)}
        sorted_labels = np.array([label_map[l] for l in raw_labels])

        result = pd.Series(
            np.nan, index=returns.index, dtype=float, name="regime"
        )
        result.loc[clean.index] = sorted_labels.astype(float)
        return result.astype("Int64")

    def regime_statistics(
        self, returns: pd.Series, regimes: pd.Series
    ) -> pd.DataFrame:
        """Per-regime statistics.

        Parameters
        ----------
        returns : pd.Series
            Period return series.
        regimes : pd.Series
            Regime label series (aligned with returns).

        Returns
        -------
        pd.DataFrame
            Columns: mean_return, volatility, sharpe, avg_duration.
        """
        aligned = pd.concat([returns, regimes], axis=1).dropna()
        if aligned.empty:
            return pd.DataFrame(
                columns=["mean_return", "volatility", "sharpe", "avg_duration"]
            )

        ret_col = aligned.iloc[:, 0]
        reg_col = aligned.iloc[:, 1]
        unique_regimes = sorted(reg_col.unique())

        rows = []
        for regime in unique_regimes:
            mask = reg_col == regime
            r = ret_col[mask]
            vol = r.std() * np.sqrt(252) if len(r) > 1 else 0.0
            mean_r = r.mean()
            sharpe = (mean_r * 252) / vol if vol > 0 else 0.0

            # Average consecutive duration
            changes = (reg_col != regime).astype(int)
            groups = changes.cumsum()
            regime_groups = groups[mask]
            if len(regime_groups) > 0:
                durations = regime_groups.groupby(regime_groups).count()
                avg_dur = durations.mean()
            else:
                avg_dur = 0.0

            rows.append(
                {
                    "regime": int(regime),
                    "mean_return": float(mean_r),
                    "volatility": float(vol),
                    "sharpe": float(sharpe),
                    "avg_duration": float(avg_dur),
                }
            )

        return pd.DataFrame(rows).set_index("regime")
