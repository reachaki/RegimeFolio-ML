# experiments/run_classifier.py

from __future__ import annotations

import pprint

import pandas as pd

from utils.data_loader import load_or_download_prices, compute_returns
from models.regime_labels import infer_hmm_regimes
from features.feature_engineering import build_regime_features
from models.regime_classifier import RegimeClassifier, RegimeClassifierConfig


def main():
    # 1. Load data (reuse your main config)
    tickers = ["XLK", "XLF", "XLV", "XLP", "XLE", "TLT", "GLD"]
    start = "2010-01-01"
    end = "2025-01-01"

    prices = load_or_download_prices(
        tickers,
        start,
        end,
        cache_path="data/raw/prices_classifier_experiment.csv",
    )
    returns = compute_returns(prices)

    # Optional: restrict to evaluation window
    eval_start = "2010-01-01"
    eval_end = "2023-12-31"
    returns_eval = returns.loc[eval_start:eval_end]

    # 2. Generate HMM regimes and features
    state_series, regime_str, _, _, _ = infer_hmm_regimes(returns_eval)

    bond_tickers = [t for t in tickers if "TLT" in t]
    equity_tickers = [t for t in tickers if t not in bond_tickers]

    feats = build_regime_features(
        returns_eval,
        equity_tickers=equity_tickers or tickers,
        bond_tickers=bond_tickers,
    )

    # 3. Build supervised dataset X_t -> y_{t+1}
    clf_cfg = RegimeClassifierConfig(
        model_type="logistic",
        C=0.5,
        crisis_weight=3.0,
    )
    clf = RegimeClassifier(clf_cfg)

    X, y = clf.make_xy(feats, regime_str)
    X_train, y_train, X_val, y_val, X_test, y_test = clf.time_split(
        X, y, train_frac=0.6, val_frac=0.2
    )

    # 4. Train and evaluate
    clf.fit(X_train, y_train)

    val_auc, val_rep = clf.evaluate(X_val, y_val, threshold=0.3)
    test_auc, test_rep = clf.evaluate(X_test, y_test, threshold=0.3)

    print("Validation ROC-AUC:", val_auc)
    print("Test ROC-AUC:", test_auc)
    print("\nTest classification report (dict):")
    pprint.pprint(test_rep)
    print("\nCrisis recall on test:", test_rep["1"]["recall"])


if __name__ == "__main__":
    main()
