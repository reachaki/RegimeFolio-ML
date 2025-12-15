# main_run.py

import pandas as pd

from utils.data_loader import load_or_download_prices, compute_returns
from backtest.backtest_engine import (
    rolling_backtest,
    rolling_backtest_predictive_hrp,  # NEW
)
from utils.metrics import portfolio_returns, sharpe_ratio, max_drawdown, turnover
from models.regime_classifier import RegimeClassifierConfig  # NEW


# 1. Config
tickers = ["XLK", "XLF", "XLV", "XLP", "XLE", "TLT", "GLD"]  # sectors + bonds + gold
start = "2010-01-01"
end = "2025-01-01"


# 2. Data
prices = load_or_download_prices(tickers, start, end, cache_path="data/raw/prices.csv")
returns = compute_returns(prices)


# Restrict evaluation window to focus on recent period with COVID crisis
eval_start = "2018-01-01"
eval_end = "2023-12-31"
returns_eval = returns.loc[eval_start:eval_end]


# 3. Baseline: HMM + Mean-Variance (regime-conditional)
weights_mvo = rolling_backtest(
    returns_eval,
    lookback=252,  # one trading year
    rebalance_freq=21,  # roughly monthly
    use_hrp=False,
    risk_aversion=1.0,
    l2_reg=1e-4,
)
port_rets_mvo = portfolio_returns(weights_mvo, returns_eval)
eq_mvo = (1 + port_rets_mvo).cumprod()


# 4. Proposed: HMM + HRP (regime-conditional)
weights_hrp = rolling_backtest(
    returns_eval,
    lookback=252,
    rebalance_freq=21,
    use_hrp=True,
)
port_rets_hrp = portfolio_returns(weights_hrp, returns_eval)
eq_hrp = (1 + port_rets_hrp).cumprod()


# 4b. Predictive: HMM + HRP + supervised regime forecaster
clf_cfg = RegimeClassifierConfig(
    model_type="logistic",
    C=0.5,
    crisis_weight=3.0,
)

weights_hrp_pred = rolling_backtest_predictive_hrp(
    returns_df=returns_eval,
    lookback=252,
    rebalance_freq=21,
    clf_cfg=clf_cfg,
    p_crisis_threshold=0.3,  # match the threshold you evaluated in run_classifier.py
)
port_rets_hrp_pred = portfolio_returns(weights_hrp_pred, returns_eval)
eq_hrp_pred = (1 + port_rets_hrp_pred).cumprod()


# 5. Metrics
print("Evaluation window:", eval_start, "to", eval_end)

print("\nBaseline (HMM + MVO)")
print("Sharpe:", sharpe_ratio(port_rets_mvo))
print("Max Drawdown:", max_drawdown(eq_mvo))
print("Turnover:", turnover(weights_mvo))

print("\nProposed (HMM + HRP)")
print("Sharpe:", sharpe_ratio(port_rets_hrp))
print("Max Drawdown:", max_drawdown(eq_hrp))
print("Turnover:", turnover(weights_hrp))

print("\nPredictive (HMM + HRP + classifier)")
print("Sharpe:", sharpe_ratio(port_rets_hrp_pred))
print("Max Drawdown:", max_drawdown(eq_hrp_pred))
print("Turnover:", turnover(weights_hrp_pred))


# 6. Save results
eq_df = pd.DataFrame(
    {
        "MVO": eq_mvo,
        "HRP": eq_hrp,
        "HRP_Predictive": eq_hrp_pred,
    }
)
eq_df.to_csv("data/processed/equity_curves_2018_2023.csv")

weights_mvo.to_csv("data/processed/weights_mvo_2018_2023.csv")
weights_hrp.to_csv("data/processed/weights_hrp_2018_2023.csv")
weights_hrp_pred.to_csv("data/processed/weights_hrp_predictive_2018_2023.csv")
