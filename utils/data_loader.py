# utils/data_loader.py
import os
import pandas as pd
import numpy as np
import yfinance as yf


def download_price_data(tickers, start, end, out_path=None):
    """
    Download daily adjusted close prices from Yahoo for a list of tickers.
    Returns a DataFrame (Date index, columns = tickers).
    """
    data = yf.download(tickers, start=start, end=end, auto_adjust=True)

    # Handle multi-index vs single-index columns
    if isinstance(data, pd.DataFrame) and isinstance(data.columns, pd.MultiIndex):
        data = data["Close"]
    else:
        data = data["Close"]

    if isinstance(data, pd.Series):
        data = data.to_frame()

    data = data.dropna(how="all")
    if out_path:
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        data.to_csv(out_path)
    return data


def load_or_download_prices(tickers, start, end, cache_path="data/raw/prices.csv"):
    if os.path.exists(cache_path):
        prices = pd.read_csv(cache_path, index_col=0, parse_dates=True)
    else:
        prices = download_price_data(tickers, start, end, cache_path)
    return prices


def compute_returns(prices, method="log"):
    """
    Compute daily returns from prices.
    method = "log" for log returns, anything else for simple returns.
    """
    if method == "log":
        rets = np.log(prices / prices.shift(1))
    else:
        rets = prices.pct_change()
    return rets.dropna()
