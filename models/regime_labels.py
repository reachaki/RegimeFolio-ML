# models/regime_labels.py

from __future__ import annotations

import numpy as np
import pandas as pd

from .hmm_regime import RegimeHMM


def infer_hmm_regimes(
    returns_df: pd.DataFrame,
    n_states: int = 2,
) -> tuple[pd.Series, pd.Series, int, int, RegimeHMM]:
    """
    Fit an HMM on returns_df and infer calm/crisis regimes.

    Parameters
    ----------
    returns_df : DataFrame
        Asset returns, index = dates, columns = assets.
    n_states : int
        Number of HMM states (default 2).

    Returns
    -------
    state_series : Series[int]
        Raw HMM state indices per date.
    regime_str : Series[str]
        "calm" or "crisis" labels per date (based on state volatility).
    calm_state : int
        State id interpreted as calm / low-vol.
    crisis_state : int
        State id interpreted as crisis / high-vol.
    hmm : RegimeHMM
        Fitted HMM object.
    """
    hmm = RegimeHMM(n_states=n_states)
    hmm.fit(returns_df)

    states = hmm.predict_states(returns_df)
    state_series = pd.Series(states, index=returns_df.index, name="state")

    # Identify crisis vs calm by state-specific volatility
    state_vols: list[float] = []
    for s in range(n_states):
        mask_s = state_series == s
        if mask_s.any():
            state_vols.append(returns_df[mask_s].std().mean())
        else:
            state_vols.append(0.0)

    crisis_state = int(np.argmax(state_vols))
    calm_state = int(1 - crisis_state)

    regime_str = state_series.map({calm_state: "calm", crisis_state: "crisis"}).rename(
        "regime"
    )

    return state_series, regime_str, calm_state, crisis_state, hmm
