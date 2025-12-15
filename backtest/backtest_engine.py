# backtest/backtest_engine.py

import numpy as np
import pandas as pd

from models.hmm_regime import RegimeHMM
from models.mvo import mean_variance_weights
from models.hrp import hrp_allocation

# NEW imports for predictive HRP
from models.regime_labels import infer_hmm_regimes
from models.regime_classifier import RegimeClassifier, RegimeClassifierConfig
from features.feature_engineering import build_regime_features


def rolling_backtest(
    returns_df,
    lookback=252,
    rebalance_freq=21,
    use_hrp=False,
    risk_aversion=1.0,
    l2_reg=1e-4,
):
    """
    Rolling-window backtest with HMM regime detection and either MVO or HRP.
    Uses regime-conditional means/covs: in each window, fit HMM, identify
    calm vs crisis regime by volatility, and compute cov from the current regime.
    """
    dates = returns_df.index
    assets = returns_df.columns
    weights_list = []
    weight_dates = []

    hmm = RegimeHMM(n_states=2)

    for i in range(lookback, len(dates), rebalance_freq):
        end = i
        start = i - lookback
        window_rets = returns_df.iloc[start:end]

        # 1. Fit HMM on window
        hmm.fit(window_rets)
        states = hmm.predict_states(window_rets)

        # 2. Identify crisis vs calm by state volatility
        state_vols = []
        for s in range(hmm.model.n_components):
            if np.any(states == s):
                state_vols.append(window_rets[states == s].std().mean())
            else:
                state_vols.append(0.0)
        crisis_state = int(np.argmax(state_vols))
        calm_state = 1 - crisis_state

        # 3. Decide current regime from last ~month of states
        if len(states) >= 21:
            recent_states = states[-21:]
        else:
            recent_states = states
        # crude mode via mean + rounding for 2 states
        current_state = int(np.round(recent_states.mean()))

        if current_state == crisis_state:
            mask = states == crisis_state
        else:
            mask = states == calm_state

        rets_regime = window_rets[mask]

        # Fallback: if too few rows, use full window
        if len(rets_regime) < 30:
            rets_regime = window_rets

        mu = rets_regime.mean()
        cov = rets_regime.cov()

        # 4. Allocate via MVO or HRP
        if use_hrp:
            w = hrp_allocation(cov)
        else:
            w_arr = mean_variance_weights(mu.values, cov.values, risk_aversion, l2_reg)
            w = pd.Series(w_arr, index=assets)

        weights_list.append(w)
        weight_dates.append(dates[end])

    weights_df = pd.DataFrame(weights_list, index=weight_dates)
    return weights_df


def rolling_backtest_predictive_hrp(
    returns_df: pd.DataFrame,
    lookback: int = 252,
    rebalance_freq: int = 21,
    clf_cfg: RegimeClassifierConfig | None = None,
    p_crisis_threshold: float = 0.6,
) -> pd.DataFrame:
    """
    Predictive HMM + HRP strategy.

    For each rolling window:
      - Fit an HMM and compute HRP weights for calm and crisis regimes.
      - Train a regime classifier on all history up to the window end.
      - Use P(crisis at t+1) to select calm vs crisis HRP portfolio
        at the rebalance date.

    Parameters
    ----------
    returns_df : DataFrame
        Asset returns, index = dates, columns = assets.
    lookback : int
        Window length in days.
    rebalance_freq : int
        Rebalance step in days.
    clf_cfg : RegimeClassifierConfig or None
        Configuration for the regime classifier.
    p_crisis_threshold : float
        Threshold on predicted crisis probability to pick the crisis HRP
        portfolio; below this threshold, choose the calm HRP portfolio.

    Returns
    -------
    DataFrame
        Weights at each rebalance date (index) x assets (columns).
    """
    dates = returns_df.index
    assets = list(returns_df.columns)

    weights_list: list[pd.Series] = []
    weight_dates: list[pd.Timestamp] = []

    if clf_cfg is None:
        clf_cfg = RegimeClassifierConfig()

    # Pre-compute global HMM regimes and feature matrix once
    state_series, regime_str, _, _, _ = infer_hmm_regimes(returns_df)

    # Heuristic: treat non-bond tickers as equity, "TLT" as bond if present
    bond_candidates = [c for c in assets if "TLT" in c]
    equity_candidates = [c for c in assets if c not in bond_candidates]

    feats_full = build_regime_features(
        returns_df,
        equity_tickers=equity_candidates or assets,
        bond_tickers=bond_candidates,
    )

    clf = RegimeClassifier(clf_cfg)
    X_all, y_all = clf.make_xy(feats_full, regime_str)

    for i in range(lookback, len(dates), rebalance_freq):
        end = i
        start = i - lookback

        window_dates = dates[start:end]
        window_rets = returns_df.iloc[start:end]

        # 1. Fit HMM on this window and identify calm/crisis
        hmm_local = RegimeHMM(n_states=2)
        hmm_local.fit(window_rets)
        states = hmm_local.predict_states(window_rets)

        state_vols = []
        for s in range(hmm_local.model.n_components):
            if np.any(states == s):
                state_vols.append(window_rets[states == s].std().mean())
            else:
                state_vols.append(0.0)

        crisis_state_local = int(np.argmax(state_vols))
        calm_state_local = 1 - crisis_state_local

        # 2. Build HRP portfolios for calm and crisis regimes in this window
        w_regime: dict[str, pd.Series] = {}
        for name, state_id in [
            ("calm", calm_state_local),
            ("crisis", crisis_state_local),
        ]:
            mask_state = states == state_id
            rets_regime = window_rets[mask_state]
            if len(rets_regime) < 30:
                rets_regime = window_rets
            cov = rets_regime.cov()
            w_regime[name] = hrp_allocation(cov).reindex(assets).fillna(0.0)

        # 3. Train classifier using history up to this window end (no leakage)
        cut_off_date = window_dates[-1]
        hist_mask = X_all.index <= cut_off_date
        X_hist, y_hist = X_all.loc[hist_mask], y_all.loc[hist_mask]

        if len(X_hist) < 200 or y_hist.nunique() < 2:
            # Not enough history or only one class -> reactive fallback
            last_state = states[-1]
            chosen_regime = "crisis" if last_state == crisis_state_local else "calm"
        else:
            n_hist = len(X_hist)
            train_end = int(n_hist * 0.8)
            X_train = X_hist.iloc[:train_end]
            y_train = y_hist.iloc[:train_end]

            clf.fit(X_train, y_train)

            # 4. Predict P(crisis at t+1) using features at window end
            if cut_off_date not in feats_full.index:
                # Safety; fallback to reactive
                last_state = states[-1]
                chosen_regime = "crisis" if last_state == crisis_state_local else "calm"
            else:
                x_now = feats_full.loc[[cut_off_date]]
                p_crisis = clf.predict_proba(x_now).iloc[0]
                chosen_regime = "crisis" if p_crisis >= p_crisis_threshold else "calm"

        w = w_regime[chosen_regime]
        weights_list.append(w)
        weight_dates.append(dates[end])

    weights_df = pd.DataFrame(weights_list, index=weight_dates)
    return weights_df
