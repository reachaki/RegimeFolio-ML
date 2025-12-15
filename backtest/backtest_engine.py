# backtest/backtest_engine.py
import numpy as np
import pandas as pd
from models.hmm_regime import RegimeHMM
from models.mvo import mean_variance_weights
from models.hrp import hrp_allocation


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
