"""Win-probability model for commodity futures RFQ flow.

Implements a logistic regression model that estimates the probability
of winning an RFQ at a given quoted spread:

    P(win) = sigmoid(X @ beta)

where X is a feature matrix assembled from RFQ attributes (spread in
basis points, log notional value, volatility, session hour, one-hot
encoded client segment, and one-hot encoded product dummies) and beta
is a vector of calibrated coefficients.

The model supports end-to-end training on synthetic RFQ data with
temporal train-test split, decile calibration diagnostics, and
feature-distribution drift detection via Population Stability Index.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd


_PRODUCT_CODES = {"CL": 0, "HO": 1, "RB": 2, "NG": 3}
_SEGMENT_CODES = {"producer": 0, "refiner": 1, "airline": 2, "hedge_fund": 3, "utility": 4}
_CONTRACT_SIZES = {"CL": 1000, "HO": 42000, "RB": 42000, "NG": 10000}

_NUMERICAL_FEATURES = [
    "spread_bps",
    "log_notional",
    "volatility",
    "session_hour",
]

_FEATURE_NAMES = (
    ["intercept"]
    + _NUMERICAL_FEATURES
    + list(_SEGMENT_CODES.keys())
    + list(_PRODUCT_CODES.keys())
)


class WinProbabilityModel:
    """Logistic regression model estimating the probability of winning an RFQ.

    Assembles a 14-feature vector from RFQ attributes — spread in basis
    points, log notional value, implied volatility, session hour, one-hot
    encoded client segment (5 categories), and one-hot encoded product
    (4 categories) — and applies a sigmoid transformation to a linear
    combination with a calibrated beta vector.

    Training pipeline:
        1. generate_training_data() -- synthetic labeled RFQ generation
        2. temporal_train_test_split() -- time-ordered split
        3. calibrate() -- gradient descent on binary log-loss
        4. decile_calibration() -- predicted vs actual by probability bucket
        5. detect_drift() -- PSI-based feature distribution monitoring

    Attributes:
        beta: numpy.ndarray of shape (14,) containing the logistic
            regression coefficients in the order:
            [intercept, spread_bps, log_notional, volatility,
            session_hour, producer, refiner, airline, hedge_fund,
            utility, CL, HO, RB, NG].
    """

    N_FEATURES = 14  # 1 intercept + 4 numerical + 5 segments + 4 products

    def __init__(self, beta=None):
        """Initialise the win-probability model.

        Args:
            beta: Optional numpy.ndarray of regression coefficients
                with shape (14,). If provided, the model is ready for
                prediction immediately. If None, ``fit()`` must be
                called before ``predict_proba()``.
        """
        self.beta = beta

    @staticmethod
    def _sigmoid(x):
        """Apply the sigmoid function element-wise, clipped to avoid overflow.

        Args:
            x: Scalar or numpy array of logit values.

        Returns:
            Scalar or numpy array with values in (0.0, 1.0).
        """
        return 1.0 / (1.0 + np.exp(-np.clip(x, -500, 500)))

    def predict_proba(self, features):
        """Predict the win probability for each row in a features DataFrame.

        Args:
            features: pandas.DataFrame with RFQ rows. See
                ``prepare_features`` for expected columns.

        Returns:
            numpy.ndarray of shape (n_rfqs,) with probabilities in (0, 1).
        """
        if self.beta is None:
            raise RuntimeError(
                "Model has not been fitted. Call fit() before predict_proba()."
            )
        X = self.prepare_features(features)
        logit = X @ self.beta
        return self._sigmoid(logit)

    def prepare_features(self, df):
        """Build the numeric feature matrix from an RFQ DataFrame.

        Constructs a (n, 14) array with columns: intercept, spread_bps,
        log_notional, volatility, session_hour, 5 segment dummies,
        4 product dummies. Missing columns are filled with defaults.

        If ``log_notional`` is absent, it is computed from
        ``num_contracts`` and ``product``. If ``session_hour`` is
        absent, it is extracted from ``timestamp`` or defaulted to 11.5.

        Args:
            df: pandas.DataFrame with RFQ rows.

        Returns:
            numpy.ndarray of shape (n_rfqs, 14).
        """
        n = len(df)
        X = np.zeros((n, self.N_FEATURES))

        # Col 0: intercept
        X[:, 0] = 1.0

        # Col 1: spread_bps
        if "spread_bps" in df.columns:
            X[:, 1] = df["spread_bps"].values

        # Col 2: log_notional
        if "log_notional" in df.columns:
            X[:, 2] = df["log_notional"].values
        elif "num_contracts" in df.columns:
            cs = pd.Series([1000] * n, index=df.index)
            if "product" in df.columns:
                cs = df["product"].map(_CONTRACT_SIZES).fillna(1000)
            X[:, 2] = np.log(df["num_contracts"].values * cs.values)

        # Col 3: volatility
        if "volatility" in df.columns:
            X[:, 3] = df["volatility"].values
        else:
            X[:, 3] = 0.25

        # Col 4: session_hour
        if "session_hour" in df.columns:
            X[:, 4] = df["session_hour"].values
        elif "timestamp" in df.columns:
            try:
                X[:, 4] = pd.to_datetime(df["timestamp"]).dt.hour.values
            except Exception:
                X[:, 4] = 11.5
        else:
            X[:, 4] = 11.5

        # Cols 5-9: client_segment one-hot
        if "client_segment" in df.columns:
            for seg, code in _SEGMENT_CODES.items():
                mask = df["client_segment"] == seg
                X[mask, 5 + code] = 1.0

        # Cols 10-13: product one-hot
        if "product" in df.columns:
            for prod, code in _PRODUCT_CODES.items():
                mask = df["product"] == prod
                X[mask, 10 + code] = 1.0

        return X

    def calibrate(self, X, y, lr=0.01, n_iter=2000, standardize=True):
        """Calibrate coefficients via gradient descent on binary log-loss.

        Optionally standardises features (zero-mean, unit-variance) before
        training so that gradient steps are balanced across features with
        very different scales (e.g. log_notional ~11 vs spread_bps ~2.6).
        After convergence the beta vector is converted back to the
        original feature scale so that predict_proba works on raw inputs.

        Args:
            X: numpy.ndarray (n_samples, n_features) feature matrix.
            y: numpy.ndarray (n_samples,) binary labels (1=win, 0=loss).
            lr: Learning rate. Defaults to 0.01.
            n_iter: Gradient descent iterations. Defaults to 2000.
            standardize: If True, z-score normalise all non-intercept
                features before fitting and convert beta back afterward.
                Defaults to True.

        Returns:
            List of (iteration, loss) tuples sampled every 500 steps.
        """
        if standardize:
            feat_mean = np.zeros(X.shape[1])
            feat_std = np.ones(X.shape[1])
            for j in range(1, X.shape[1]):
                feat_mean[j] = X[:, j].mean()
                s = X[:, j].std()
                feat_std[j] = s if s > 1e-10 else 1.0
            X_fit = X.copy()
            X_fit[:, 1:] = (X[:, 1:] - feat_mean[1:]) / feat_std[1:]
        else:
            X_fit = X

        self.beta = np.zeros(X_fit.shape[1])
        losses = []
        for i in range(n_iter):
            p = self._sigmoid(X_fit @ self.beta)
            grad = X_fit.T @ (p - y) / len(y)
            self.beta -= lr * grad
            if i % 500 == 0 or i == n_iter - 1:
                loss = -np.mean(y * np.log(p + 1e-12) + (1 - y) * np.log(1 - p + 1e-12))
                losses.append((i, loss))

        if standardize:
            # Convert beta back to original (un-standardised) feature scale
            self.beta[1:] = self.beta[1:] / feat_std[1:]
            self.beta[0] -= np.dot(self.beta[1:], feat_mean[1:])

        return losses

    @staticmethod
    def generate_training_data(n_samples=5000, seed=42):
        """Generate synthetic labeled RFQ data for model training.

        Creates n_samples synthetic RFQs with realistic feature
        distributions spanning 60 trading days, then generates binary
        win/loss labels from a known logistic DGP so that calibrate()
        can recover the true coefficients.

        Features: spread_bps (uniform 0.2-5.0 bps), log_notional
        (log of contracts * contract_size), volatility (uniform
        0.15-0.50), session_hour (normal ~11.5), client_segment
        (weighted draw), product (weighted draw).

        Args:
            n_samples: Number of synthetic RFQs. Defaults to 5000.
            seed: Random seed for reproducibility. Defaults to 42.

        Returns:
            pandas.DataFrame with columns: timestamp, spread_bps,
            num_contracts, product, client_segment, volatility,
            session_hour, log_notional, won.
        """
        rng = np.random.RandomState(seed)

        products = list(_PRODUCT_CODES.keys())
        segments = list(_SEGMENT_CODES.keys())

        spread_bps = rng.uniform(0.2, 5.0, n_samples)
        num_contracts = rng.exponential(50, n_samples).astype(int).clip(1, 500)
        product_arr = rng.choice(products, n_samples, p=[0.45, 0.15, 0.20, 0.20])
        segment_arr = rng.choice(segments, n_samples, p=[0.30, 0.25, 0.10, 0.20, 0.15])
        volatility = rng.uniform(0.15, 0.50, n_samples)
        session_hour = rng.normal(11.5, 1.5, n_samples).clip(9, 15)

        log_notional = np.array([
            np.log(nc * _CONTRACT_SIZES.get(p, 1000))
            for nc, p in zip(num_contracts, product_arr)
        ])

        # Timestamps over 60 trading days for temporal ordering
        base = pd.Timestamp("2025-06-01 09:00:00")
        day_offsets = np.sort(rng.randint(0, 60, n_samples))
        timestamps = [
            base + pd.Timedelta(days=int(d), hours=int(h) - 9,
                                minutes=int(rng.randint(0, 60)))
            for d, h in zip(day_offsets, session_hour)
        ]

        df = pd.DataFrame({
            "timestamp": timestamps,
            "spread_bps": spread_bps,
            "num_contracts": num_contracts,
            "product": product_arr,
            "client_segment": segment_arr,
            "volatility": volatility,
            "session_hour": session_hour,
            "log_notional": log_notional,
        })

        # --- True DGP: known coefficients for label generation ---
        true_beta = np.array([
            1.8, -0.6, -0.08, -0.4, 0.02,
            0.1, 0.05, 0.15, -0.2, 0.0,
            0.1, -0.05, 0.0, -0.1,
        ])

        n = len(df)
        X = np.zeros((n, 14))
        X[:, 0] = 1.0
        X[:, 1] = spread_bps
        X[:, 2] = log_notional
        X[:, 3] = volatility
        X[:, 4] = session_hour
        for seg, code in _SEGMENT_CODES.items():
            X[segment_arr == seg, 5 + code] = 1.0
        for prod, code in _PRODUCT_CODES.items():
            X[product_arr == prod, 10 + code] = 1.0

        logit = X @ true_beta
        true_probs = 1.0 / (1.0 + np.exp(-np.clip(logit, -500, 500)))
        df["won"] = (rng.random(n_samples) < true_probs).astype(float)

        return df

    @staticmethod
    def temporal_train_test_split(df, train_frac=0.8):
        """Split data by timestamp for out-of-time validation.

        Sorts by timestamp and assigns the first ``train_frac``
        fraction to the training set and the remainder to the test set.

        Args:
            df: DataFrame with a ``timestamp`` column.
            train_frac: Fraction in the training set. Defaults to 0.8.

        Returns:
            Tuple of (train_df, test_df) DataFrames.
        """
        df_sorted = df.sort_values("timestamp").reset_index(drop=True)
        split_idx = int(len(df_sorted) * train_frac)
        return df_sorted.iloc[:split_idx].copy(), df_sorted.iloc[split_idx:].copy()

    @staticmethod
    def decile_calibration(y_true, y_pred, n_bins=10):
        """Check model calibration by binning predictions into deciles.

        For each bin, computes the mean predicted probability and the
        observed win rate. A well-calibrated model has predicted ~=
        actual in every bin.

        Args:
            y_true: numpy.ndarray of binary labels.
            y_pred: numpy.ndarray of predicted probabilities.
            n_bins: Number of equal-sized bins. Defaults to 10.

        Returns:
            List of dicts with keys: decile, pred_mean, actual_mean, count.
        """
        order = np.argsort(y_pred)
        y_true_s = y_true[order]
        y_pred_s = y_pred[order]

        bins = np.array_split(np.arange(len(y_true_s)), n_bins)
        results = []
        for i, idx in enumerate(bins):
            results.append({
                "decile": i + 1,
                "pred_mean": float(y_pred_s[idx].mean()),
                "actual_mean": float(y_true_s[idx].mean()),
                "count": len(idx),
            })
        return results

    @staticmethod
    def detect_drift(X_ref, X_test, feature_names=None, n_bins=10):
        """Detect feature distribution drift via Population Stability Index.

        PSI = sum_i (test_pct_i - ref_pct_i) * ln(test_pct_i / ref_pct_i)

        PSI < 0.10: no significant drift.
        0.10 <= PSI < 0.25: moderate drift.
        PSI >= 0.25: significant drift.

        Args:
            X_ref: numpy.ndarray reference (training) feature matrix.
            X_test: numpy.ndarray test feature matrix.
            feature_names: Optional list of feature name strings.
            n_bins: Number of histogram bins. Defaults to 10.

        Returns:
            List of dicts with keys: feature, psi, status.
        """
        if feature_names is None:
            feature_names = [f"feature_{j}" for j in range(X_ref.shape[1])]

        results = []
        for j in range(X_ref.shape[1]):
            ref_col = X_ref[:, j]
            test_col = X_test[:, j]

            if ref_col.std() < 1e-10 and test_col.std() < 1e-10:
                results.append({"feature": feature_names[j], "psi": 0.0, "status": "constant"})
                continue

            edges = np.percentile(ref_col, np.linspace(0, 100, n_bins + 1))
            edges[0], edges[-1] = -np.inf, np.inf
            edges = np.unique(edges)
            if len(edges) < 3:
                results.append({"feature": feature_names[j], "psi": 0.0, "status": "degenerate"})
                continue

            ref_pct = np.clip(np.histogram(ref_col, bins=edges)[0] / len(ref_col), 1e-4, None)
            test_pct = np.clip(np.histogram(test_col, bins=edges)[0] / len(test_col), 1e-4, None)

            psi = float(np.sum((test_pct - ref_pct) * np.log(test_pct / ref_pct)))

            if psi < 0.10:
                status = "no drift"
            elif psi < 0.25:
                status = "moderate"
            else:
                status = "SIGNIFICANT"

            results.append({"feature": feature_names[j], "psi": psi, "status": status})

        return results

    @staticmethod
    def feature_names():
        """Return the ordered list of feature names matching the beta vector."""
        return list(_FEATURE_NAMES)

    def __repr__(self):
        """Return a concise string representation of the model."""
        n = len(self.beta) if self.beta is not None else 0
        fitted = self.beta is not None
        return f"WinProbabilityModel(n_features={n}, fitted={fitted})"
