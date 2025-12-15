# features/feature_engineering.py

from __future__ import annotations

import pandas as pd
import numpy as np


def _rolling_return(series: pd.Series, window: int) -> pd.Series:
    """
    Rolling window simple return over `window` days:
    (1 + r_t-window+1) * ... * (1 + r_t) - 1
    """
    return series.rolling(window).apply(
        lambda x: (1.0 + x).prod() - 1.0,
        raw=False,
    )


def build_regime_features(
    returns_df: pd.DataFrame,
    equity_tickers: list[str] | None = None,
    bond_tickers: list[str] | None = None,
    vol_proxy: pd.Series | None = None,
    windows: tuple[int, ...] = (1, 5, 20),
) -> pd.DataFrame:
    """
    Build a panel of daily features for regime classification.

    Parameters
    ----------
    returns_df : DataFrame
        Asset returns, indexed by date, columns = tickers.
    equity_tickers : list[str] or None
        Tickers treated as "equity". If None, use all tickers.
    bond_tickers : list[str] or None
        Tickers treated as "bond" (for equity-minus-bond spreads).
    vol_proxy : Series or None
        Optional volatility proxy (e.g. VIX level or returns), aligned on dates.
    windows : tuple[int]
        Horizons (in days) for rolling returns.

    Returns
    -------
    DataFrame
        Features indexed by date, columns like:
        - eq_ret_1d, eq_ret_5d, eq_ret_20d
        - eq_vol_20d
        - eq_minus_bond_5d
        - vol_proxy_level, vol_proxy_chg_5d
    """
    if equity_tickers is None or len(equity_tickers) == 0:
        equity_tickers = list(returns_df.columns)

    eq_rets = returns_df[equity_tickers].mean(axis=1)

    feats = pd.DataFrame(index=returns_df.index)

    # Multi-horizon equity returns
    for w in windows:
        feats[f"eq_ret_{w}d"] = _rolling_return(eq_rets, w)

    # Realised volatility (20d)
    feats["eq_vol_20d"] = eq_rets.rolling(20).std()

    # Equity-minus-bond spread
    if bond_tickers is not None and len(bond_tickers) > 0:
        # Only keep bond tickers that are in the DataFrame
        bond_tickers_present = [b for b in bond_tickers if b in returns_df.columns]
        if len(bond_tickers_present) > 0:
            bond_rets = returns_df[bond_tickers_present].mean(axis=1)
            feats["eq_minus_bond_5d"] = _rolling_return(eq_rets, 5) - _rolling_return(
                bond_rets, 5
            )

    # Optional volatility proxy
    if vol_proxy is not None:
        vol_proxy = vol_proxy.reindex(returns_df.index)
        feats["vol_proxy_level"] = vol_proxy
        feats["vol_proxy_chg_5d"] = vol_proxy.pct_change(5)

    return feats
