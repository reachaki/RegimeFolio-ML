# utils/metrics.py
import numpy as np
import pandas as pd


def portfolio_returns(weights_df, returns_df):
    """
    weights_df: DataFrame (dates x assets) of portfolio weights
    returns_df: DataFrame (dates x assets) of asset returns
    """
    aligned_rets = returns_df.loc[weights_df.index]
    port_rets = (weights_df * aligned_rets).sum(axis=1)
    return port_rets


def sharpe_ratio(returns, rf=0.0, freq=252):
    """
    Annualised Sharpe.
    returns: Series of daily returns.
    """
    excess = returns - rf / freq
    mu = excess.mean() * freq
    sigma = excess.std() * (freq**0.5)
    return mu / sigma if sigma > 0 else np.nan


def max_drawdown(equity_curve):
    """
    equity_curve: Series of cumulative returns (e.g., (1+ret).cumprod()).
    """
    roll_max = equity_curve.cummax()
    drawdown = equity_curve / roll_max - 1.0
    return drawdown.min()


def turnover(weights_df):
    """
    Average daily turnover: mean of sum |w_t - w_{t-1}|.
    """
    diff = weights_df.diff().abs()
    return diff.sum(axis=1).mean()
