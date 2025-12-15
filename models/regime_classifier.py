# models/regime_classifier.py

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, classification_report


@dataclass
class RegimeClassifierConfig:
    """
    Configuration for the regime classifier.

    model_type : "logistic" | "rf"
        Type of classifier to use.
    C : float
        Regularisation strength for logistic regression.
    rf_n_estimators : int
        Number of trees for RandomForest.
    rf_max_depth : int or None
        Max depth for RandomForest.
    crisis_weight : float
        Class weight multiplier for the crisis class (label 1).
    """

    model_type: str = "logistic"
    C: float = 1.0
    rf_n_estimators: int = 200
    rf_max_depth: int | None = None
    crisis_weight: float = 2.0


class RegimeClassifier:
    """
    Thin wrapper around scikit-learn classifiers for
    HMM regime prediction (calm vs crisis).
    """

    def __init__(self, cfg: RegimeClassifierConfig):
        self.cfg = cfg
        self.model = self._build_model()

    def _build_model(self):
        if self.cfg.model_type == "logistic":
            return LogisticRegression(
                C=self.cfg.C,
                penalty="l2",
                solver="lbfgs",
                max_iter=1000,
                class_weight={0: 1.0, 1: self.cfg.crisis_weight},
            )
        elif self.cfg.model_type == "rf":
            return RandomForestClassifier(
                n_estimators=self.cfg.rf_n_estimators,
                max_depth=self.cfg.rf_max_depth,
                class_weight={0: 1.0, 1: self.cfg.crisis_weight},
                n_jobs=-1,
                random_state=42,
            )
        else:
            raise ValueError(f"Unknown model_type: {self.cfg.model_type}")

    # ---------- Data preparation ----------

    @staticmethod
    def make_xy(
        features: pd.DataFrame,
        regime_str: pd.Series,
    ) -> tuple[pd.DataFrame, pd.Series]:
        """
        Given features X_t and regime labels y_t (calm/crisis), build
        (X, y) for the supervised task X_t -> y_{t+1}.

        Parameters
        ----------
        features : DataFrame
            Indexed by date, containing engineered features.
        regime_str : Series[str]
            "calm" or "crisis" per date; same index as features.

        Returns
        -------
        X : DataFrame
            Features aligned and trimmed to valid samples.
        y : Series[int]
            Next-day crisis labels (1 = crisis, 0 = calm).
        """
        regime_str = regime_str.reindex(features.index)
        y_today = (regime_str == "crisis").astype(int)

        # Shift labels so that features at t predict regime at t+1
        y_next = y_today.shift(-1)

        df = features.join(y_next.rename("y")).dropna()
        X = df[features.columns]
        y = df["y"].astype(int)
        return X, y

    @staticmethod
    def time_split(
        X: pd.DataFrame,
        y: pd.Series,
        train_frac: float = 0.6,
        val_frac: float = 0.2,
    ):
        """
        Chronological train/validation/test split.

        Returns
        -------
        (X_train, y_train, X_val, y_val, X_test, y_test)
        """
        if not X.index.equals(y.index):
            raise ValueError("X and y must have the same index for time_split.")

        n = len(X)
        train_end = int(n * train_frac)
        val_end = int(n * (train_frac + val_frac))

        idx = X.index
        train_idx = idx[:train_end]
        val_idx = idx[train_end:val_end]
        test_idx = idx[val_end:]

        X_train = X.loc[train_idx]
        y_train = y.loc[train_idx]
        X_val = X.loc[val_idx]
        y_val = y.loc[val_idx]
        X_test = X.loc[test_idx]
        y_test = y.loc[test_idx]

        return X_train, y_train, X_val, y_val, X_test, y_test

    # ---------- Fit / predict / evaluate ----------

    def fit(self, X_train: pd.DataFrame, y_train: pd.Series):
        self.model.fit(X_train.values, y_train.values)

    def predict_proba(self, X: pd.DataFrame) -> pd.Series:
        """
        Return P(crisis=1) per sample.
        """
        proba = self.model.predict_proba(X.values)[:, 1]
        return pd.Series(proba, index=X.index, name="p_crisis")

    def evaluate(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        threshold: float = 0.5,
    ) -> tuple[float, dict]:
        """
        Compute ROC-AUC and classification_report dict for a given set.
        """
        p = self.predict_proba(X)
        y_hat = (p >= threshold).astype(int)

        auc = roc_auc_score(y, p)
        report = classification_report(y, y_hat, output_dict=True)
        return auc, report
