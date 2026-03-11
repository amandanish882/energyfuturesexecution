"""Win-probability model for commodity futures RFQ flow.

Implements a logistic regression model that estimates the probability
of winning an RFQ at a given quoted spread:

    P(win) = sigmoid(X @ beta)

where X is a feature matrix assembled from RFQ attributes (spread in
basis points, contract size, spread sensitivity, urgency, volatility,
and one-hot encoded product dummies) and beta is a vector of
pre-calibrated coefficients. The model can be re-calibrated from
labelled historical data using gradient-descent log-loss minimisation.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd


_PRODUCT_CODES = {"CL": 0, "HO": 1, "RB": 2, "NG": 3}

_NUMERICAL_FEATURES = [
    "spread_bps",
    "num_contracts",
    "spread_sensitivity",
    "urgency_score",
    "volatility",
]


class WinProbabilityModel:
    """Logistic regression model that estimates the probability of winning an RFQ.

    Assembles a feature matrix from RFQ attributes — spread in basis
    points, contract size, client spread sensitivity, urgency score,
    implied volatility, and one-hot encoded product dummies — and
    applies a sigmoid transformation to a linear combination of those
    features with a pre-calibrated beta vector.

    The default beta vector was calibrated on historical RFQ win/loss
    data and encodes intuitions such as: wider spreads reduce win
    probability, larger trades are marginally harder to win, and
    urgent clients are slightly easier to win.

    Attributes:
        beta: numpy.ndarray of shape (10,) containing the logistic
            regression coefficients in the order:
            [intercept, spread_bps, num_contracts, spread_sensitivity,
            urgency_score, volatility, CL, HO, RB, NG].
    """

    def __init__(self, beta=None):
        """Initialise the win-probability model.

        Args:
            beta: Optional numpy.ndarray of regression coefficients
                with shape matching the number of features (10 by
                default). If None, a hard-coded default vector is
                used that reflects typical energy market RFQ dynamics.
        """
        if beta is None:
            # [intercept, spread_bps, num_contracts, spread_sensitivity,
            #  urgency_score, volatility, CL, HO, RB, NG]
            self.beta = np.array([
                2.5,    # intercept (fixed from 1.2)
                -0.8,   # spread_bps
                -0.001, # num_contracts (fixed from -0.002)
                -0.8,   # spread_sensitivity
                0.4,    # urgency_score
                -0.5,   # volatility
                0.1, -0.1, 0.0, -0.2,  # CL, HO, RB, NG
            ])
        else:
            self.beta = beta

    @staticmethod
    def _sigmoid(x):
        """Apply the sigmoid (logistic) function element-wise.

        Clips the input to [-500, 500] before exponentiation to
        avoid numerical overflow in the exponential.

        Args:
            x: Scalar or numpy array of logit values.

        Returns:
            Scalar or numpy array of the same shape as ``x`` with
            values in (0.0, 1.0).
        """
        return 1.0 / (1.0 + np.exp(-np.clip(x, -500, 500)))

    def predict_proba(self, features):
        """Predict the win probability for each row in a features DataFrame.

        Calls ``prepare_features`` to build the numeric feature
        matrix, computes the logit as ``X @ beta``, and passes the
        result through the sigmoid function.

        Args:
            features: pandas.DataFrame where each row represents one
                RFQ. Expected columns include ``spread_bps``,
                ``num_contracts``, ``spread_sensitivity``,
                ``urgency`` (or ``urgency_score``), ``volatility``,
                and ``product``. Missing columns are filled with
                sensible defaults.

        Returns:
            numpy.ndarray of shape (n_rfqs,) containing float win
            probabilities in (0.0, 1.0) for each row.
        """
        X = self.prepare_features(features)
        logit = X @ self.beta
        return self._sigmoid(logit)

    def prepare_features(self, df):
        """Build the numeric feature matrix from an RFQ DataFrame.

        Constructs a 2-D numpy array with one row per RFQ and columns
        corresponding to the model's beta vector: intercept,
        numerical features (spread_bps, num_contracts,
        spread_sensitivity, urgency_score, volatility), and one-hot
        encoded product dummies (CL, HO, RB, NG).

        Missing numerical columns default to zero. The ``urgency``
        string column (``'urgent'``, ``'normal'``, ``'patient'``) is
        mapped to a numeric urgency_score (1.0, 0.5, 0.0) when
        ``urgency_score`` is absent. The ``volatility`` column
        defaults to 0.25 when absent.

        Args:
            df: pandas.DataFrame with RFQ rows. Recognised columns:
                ``spread_bps`` (float), ``num_contracts`` (int),
                ``spread_sensitivity`` (float),
                ``urgency_score`` (float) or ``urgency`` (str),
                ``volatility`` (float), ``product`` (str).

        Returns:
            numpy.ndarray of shape (n_rfqs, n_features) containing
            the assembled feature values ready for dot-product with
            ``beta``.
        """
        n = len(df)
        n_features = len(self.beta)
        X = np.zeros((n, n_features))

        X[:, 0] = 1.0  # intercept

        for i, feat in enumerate(_NUMERICAL_FEATURES):
            if feat in df.columns:
                X[:, i + 1] = df[feat].values
            elif feat == "urgency_score":
                urgency_map = {"urgent": 1.0, "normal": 0.5, "patient": 0.0}
                if "urgency" in df.columns:
                    X[:, i + 1] = df["urgency"].map(urgency_map).fillna(0.5).values
            elif feat == "volatility":
                X[:, i + 1] = 0.25  # default vol

        if "product" in df.columns:
            base_idx = len(_NUMERICAL_FEATURES) + 1
            for product, code in _PRODUCT_CODES.items():
                mask = df["product"] == product
                if base_idx + code < n_features:
                    X[mask, base_idx + code] = 1.0

        return X

    def calibrate(self, X, y, lr=0.01, n_iter=1000):
        """Calibrate the model coefficients via gradient descent on binary log-loss.

        Initialises ``beta`` to a zero vector of length
        ``X.shape[1]`` and iteratively updates it using the gradient
        of the average binary cross-entropy loss with respect to
        ``beta``:
            grad = X.T @ (sigmoid(X @ beta) - y) / n

        Args:
            X: numpy.ndarray of shape (n_samples, n_features)
                containing the pre-assembled feature matrix. Each row
                is one labelled RFQ observation.
            y: numpy.ndarray of shape (n_samples,) containing binary
                labels where 1 indicates a win and 0 indicates a loss.
            lr: Learning rate (step size) for the gradient descent
                update. Defaults to 0.01.
            n_iter: Number of gradient descent iterations to perform.
                Defaults to 1000.

        Returns:
            None. Updates ``self.beta`` in-place.
        """
        self.beta = np.zeros(X.shape[1])
        for i in range(n_iter):
            p = self._sigmoid(X @ self.beta)
            grad = X.T @ (p - y) / len(y)
            self.beta -= lr * grad

    def __repr__(self):
        """Return a concise string representation of the model.

        Returns:
            String of the form ``WinProbabilityModel(n_features=N)``
            where N is the length of the beta coefficient vector.
        """
        return f"WinProbabilityModel(n_features={len(self.beta)})"
