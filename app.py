# app.py
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import altair as alt

from utils.data_loader import load_or_download_prices, compute_returns
from backtest.backtest_engine import rolling_backtest
from utils.metrics import portfolio_returns, sharpe_ratio, max_drawdown, turnover
from models.hmm_regime import RegimeHMM

st.set_page_config(page_title="Regime-Switching HRP vs MVO", layout="wide")


@st.cache_data
def get_prices_and_returns(tickers, start, end):
    # ticker‑specific cache file so columns always match the current universe
    cache_path = f"data/raw/prices_{'_'.join(tickers)}.csv"
    prices = load_or_download_prices(tickers, start, end, cache_path=cache_path)
    rets = compute_returns(prices)
    return prices, rets


@st.cache_data
def run_backtests(returns_eval, lookback, rebalance_freq, risk_aversion, l2_reg):
    # Baseline: HMM + MVO
    weights_mvo = rolling_backtest(
        returns_eval,
        lookback=lookback,
        rebalance_freq=rebalance_freq,
        use_hrp=False,
        risk_aversion=risk_aversion,
        l2_reg=l2_reg,
    )
    port_rets_mvo = portfolio_returns(weights_mvo, returns_eval)
    eq_mvo = (1 + port_rets_mvo).cumprod()

    # Proposed: HMM + HRP
    weights_hrp = rolling_backtest(
        returns_eval,
        lookback=lookback,
        rebalance_freq=rebalance_freq,
        use_hrp=True,
        risk_aversion=risk_aversion,
        l2_reg=l2_reg,
    )
    port_rets_hrp = portfolio_returns(weights_hrp, returns_eval)
    eq_hrp = (1 + port_rets_hrp).cumprod()

    return weights_mvo, weights_hrp, port_rets_mvo, port_rets_hrp, eq_mvo, eq_hrp


@st.cache_data
def fit_full_hmm(returns_eval, n_states=2):
    hmm = RegimeHMM(n_states=n_states)
    hmm.fit(returns_eval)
    states = hmm.predict_states(returns_eval)
    probs = hmm.state_probabilities(returns_eval)
    return states, probs


# Sidebar controls
st.sidebar.title("Configuration")

default_tickers = ["XLK", "XLF", "XLV", "XLP", "XLE", "TLT", "GLD"]
tickers = (
    st.sidebar.text_input("Tickers (comma-separated)", value=",".join(default_tickers))
    .replace(" ", "")
    .split(",")
)

start = st.sidebar.date_input("Start date", value=pd.to_datetime("2010-01-01"))
end = st.sidebar.date_input("End date", value=pd.to_datetime("2025-01-01"))

eval_start = st.sidebar.date_input(
    "Evaluation start", value=pd.to_datetime("2018-01-01")
)
eval_end = st.sidebar.date_input("Evaluation end", value=pd.to_datetime("2023-12-31"))

lookback = st.sidebar.slider(
    "Lookback window (days)", min_value=126, max_value=504, value=252, step=21
)
rebalance_freq = st.sidebar.slider(
    "Rebalance frequency (days)", min_value=5, max_value=63, value=21, step=5
)
risk_aversion = st.sidebar.slider(
    "Risk aversion (MVO)", min_value=0.1, max_value=5.0, value=1.0, step=0.1
)
l2_reg = st.sidebar.number_input("L2 regularisation (MVO)", value=1e-4, format="%.5f")

if st.sidebar.button("Run / Refresh"):
    st.rerun()

# Load data
prices, returns = get_prices_and_returns(
    tickers,
    start.strftime("%Y-%m-%d"),
    end.strftime("%Y-%m-%d"),
)

returns_eval = returns.loc[str(eval_start) : str(eval_end)]

st.title("Robust Asset Allocation Dashboard")
st.markdown(
    "HMM-based regime switching with Mean-Variance vs Hierarchical Risk Parity."
)

tab_regimes, tab_alloc, tab_backtest = st.tabs(["Regimes", "Allocations", "Backtest"])

# --- Regimes tab ---
with tab_regimes:
    st.subheader("Market regimes from HMM")

    # Fit HMM on evaluation window
    states, probs = fit_full_hmm(returns_eval)
    state_series = pd.Series(states, index=returns_eval.index, name="state")

    # Identify calm vs crisis by volatility
    state_vols = []
    for s in np.unique(states):
        mask_s = state_series == s
        state_vols.append(returns_eval[mask_s].std().mean())
    crisis_state = int(np.argmax(state_vols))
    calm_state = int(1 - crisis_state)

    regime_labels = {
        calm_state: "Calm / Low-vol",
        crisis_state: "Crisis / High-vol",
    }
    state_names = state_series.map(regime_labels)
    current_state = states[-1]

    axis_cfg = dict(
        grid=True,
        gridColor="#3a3a3a",
        gridOpacity=0.4,
        domain=False,
        tickColor="#888888",
        labelColor="#dddddd",
        titleColor="#ffffff",
    )

    # ------------------------------------------------------------------
    # Multi-asset price explorer
    # ------------------------------------------------------------------
    st.markdown("### Multi-asset price explorer")

    available_assets_all = [t for t in tickers if t in prices.columns]
    if len(available_assets_all) == 0:
        st.warning("No matching price columns for the selected tickers.")
    else:
        col_range, col_custom_from, col_custom_to = st.columns([2, 1, 1])
        with col_range:
            range_option = st.selectbox(
                "Range",
                ["1D", "5D", "1M", "6M", "1Y", "5Y", "Max", "Custom"],
                index=6,
            )
        with col_custom_from:
            custom_from = st.date_input("From", value=returns_eval.index.min())
        with col_custom_to:
            custom_to = st.date_input("To", value=returns_eval.index.max())

        end_date = returns_eval.index.max()
        if range_option == "1D":
            start_date = end_date - pd.Timedelta(days=1)
        elif range_option == "5D":
            start_date = end_date - pd.Timedelta(days=5)
        elif range_option == "1M":
            start_date = end_date - pd.DateOffset(months=1)
        elif range_option == "6M":
            start_date = end_date - pd.DateOffset(months=6)
        elif range_option == "1Y":
            start_date = end_date - pd.DateOffset(years=1)
        elif range_option == "5Y":
            start_date = end_date - pd.DateOffset(years=5)
        elif range_option == "Custom":
            start_date = pd.to_datetime(custom_from)
            end_date = pd.to_datetime(custom_to)
        else:
            start_date = returns_eval.index.min()

        prices_range = prices.loc[start_date:end_date, available_assets_all].dropna(
            how="all"
        )

        if prices_range.empty:
            st.warning("No price data in the selected date range.")
        else:
            # Normalise each series to 1 at start of the visible window
            norm_prices = prices_range / prices_range.iloc[0]

            prices_long = (
                norm_prices.reset_index()
                .melt(
                    id_vars=norm_prices.index.name or "index",
                    var_name="ticker",
                    value_name="price_norm",
                )
                .rename(columns={norm_prices.index.name or "index": "date"})
            )

            y_min = float(prices_long["price_norm"].min())
            y_max = float(prices_long["price_norm"].max())
            padding = (y_max - y_min) * 0.1 if y_max > y_min else 0.01
            y_domain = (y_min - padding, y_max + padding)

            legend_sel = alt.selection_multi(fields=["ticker"], bind="legend")

            chart_multi = (
                alt.Chart(prices_long)
                .mark_line(interpolate="monotone", strokeWidth=1.5)
                .encode(
                    x=alt.X(
                        "date:T",
                        title="Date",
                        axis=alt.Axis(
                            format="%Y-%m",
                            labelAngle=-40,
                            tickCount="year",
                        ),
                    ),
                    y=alt.Y(
                        "price_norm:Q",
                        title="Normalised price (start = 1)",
                        scale=alt.Scale(domain=y_domain),
                    ),
                    color=alt.Color(
                        "ticker:N", title="Ticker", scale=alt.Scale(scheme="tableau10")
                    ),
                    opacity=alt.condition(legend_sel, alt.value(1.0), alt.value(0.2)),
                    tooltip=[
                        alt.Tooltip("date:T", title="Date"),
                        alt.Tooltip("ticker:N", title="Ticker"),
                        alt.Tooltip("price_norm:Q", title="Norm price", format=".3f"),
                    ],
                )
                .add_params(legend_sel)
                .properties(height=340)  # taller chart
                .configure_view(strokeWidth=0)
                .configure_axis(**axis_cfg)
            )

            st.altair_chart(chart_multi, use_container_width=True)

    # ------------------------------------------------------------------
    # Price with regimes for a single asset
    # ------------------------------------------------------------------
    st.markdown("### Regime overlay on price")

    available_assets = [t for t in tickers if t in prices.columns]
    if len(available_assets) == 0:
        st.warning(
            "No matching price columns for the selected tickers. "
            "Check ticker symbols or date range."
        )
    else:
        asset_for_plot = st.selectbox(
            "Asset for regime overlay", options=available_assets
        )
        price_sub = prices.loc[returns_eval.index, asset_for_plot]

        price_df = pd.DataFrame(
            {
                "date": price_sub.index,
                "price": price_sub.values,
                "state": state_series.values,
            }
        )
        price_df["regime"] = price_df["state"].map(regime_labels)
        price_df["segment_id"] = (
            price_df["regime"] != price_df["regime"].shift()
        ).cumsum()

        y_min = float(price_df["price"].min())
        y_max = float(price_df["price"].max())
        padding = (y_max - y_min) * 0.1 if y_max > y_min else 0.02
        y_domain = (y_min - padding, y_max + padding)

        st.markdown(f"#### {asset_for_plot} price with regimes")

        # Grey price line with its own legend label
        base_line = (
            alt.Chart(price_df.assign(series="Price (no regime)"))
            .mark_line(interpolate="monotone", strokeWidth=1.2)
            .encode(
                x=alt.X(
                    "date:T",
                    title="Date",
                    axis=alt.Axis(
                        format="%Y-%m",
                        labelAngle=-40,
                        tickCount="year",
                    ),
                ),
                y=alt.Y(
                    "price:Q",
                    title=f"{asset_for_plot} price",
                    scale=alt.Scale(domain=y_domain),
                ),
                color=alt.Color(
                    "series:N",
                    title="Series / Regime",
                    scale=alt.Scale(range=["#9ca3af"]),
                ),
                tooltip=[
                    alt.Tooltip("date:T", title="Date"),
                    alt.Tooltip("price:Q", title="Price", format=".2f"),
                    alt.Tooltip("series:N", title="Type"),
                ],
            )
        )

        # Regime-coloured segments on top
        regime_line = (
            alt.Chart(price_df)
            .mark_line(interpolate="monotone", strokeWidth=1.8)
            .encode(
                x=alt.X(
                    "date:T",
                    title="Date",
                    axis=alt.Axis(
                        format="%Y-%m",
                        labelAngle=-40,
                        tickCount="year",
                    ),
                ),
                y=alt.Y(
                    "price:Q",
                    title=f"{asset_for_plot} price",
                    scale=alt.Scale(domain=y_domain),
                ),
                color=alt.Color(
                    "regime:N",
                    title="Series / Regime",
                    scale=alt.Scale(
                        domain=["Calm / Low-vol", "Crisis / High-vol"],
                        range=["#22c55e", "#ef4444"],
                    ),
                ),
                detail="segment_id:N",
                tooltip=[
                    alt.Tooltip("date:T", title="Date"),
                    alt.Tooltip("price:Q", title="Price", format=".2f"),
                    alt.Tooltip("regime:N", title="Regime"),
                ],
            )
        )

        chart_price = (
            (base_line + regime_line)
            .resolve_scale(color="independent")
            .configure_view(strokeWidth=0)
            .configure_axis(**axis_cfg)
        )

        st.altair_chart(chart_price, use_container_width=True)
        st.markdown(
            f"**Current inferred regime:** {regime_labels[current_state]} "
            f"(state {current_state})"
        )

    # ------------------------------------------------------------------
    # Regime probabilities (last 60 days)
    # ------------------------------------------------------------------
    st.markdown("### Regime probabilities (last 60 days)")
    probs_df = pd.DataFrame(probs, index=returns_eval.index)
    probs_last = probs_df.tail(60)
    probs_long = (
        probs_last.reset_index()
        .melt(
            id_vars=probs_last.index.name or "index",
            var_name="state",
            value_name="prob",
        )
        .rename(columns={probs_last.index.name or "index": "date"})
    )

    chart_probs = (
        alt.Chart(probs_long)
        .mark_line(interpolate="monotone", strokeWidth=1.5)
        .encode(
            x=alt.X(
                "date:T",
                title="Date",
                axis=alt.Axis(format="%b %Y", labelAngle=-40),
            ),
            y=alt.Y(
                "prob:Q", title="State probability", scale=alt.Scale(domain=(0, 1))
            ),
            color=alt.Color(
                "state:N", title="HMM state", scale=alt.Scale(scheme="dark2")
            ),
            tooltip=[
                alt.Tooltip("date:T", title="Date"),
                alt.Tooltip("state:N", title="State"),
                alt.Tooltip("prob:Q", title="Prob.", format=".2f"),
            ],
        )
        .properties(height=220)
        .configure_view(strokeWidth=0)
        .configure_axis(**axis_cfg)
    )
    st.altair_chart(chart_probs, use_container_width=True)

    # ------------------------------------------------------------------
    # State summary
    # ------------------------------------------------------------------
    st.markdown("### State summary")
    state_counts = state_names.value_counts().reset_index()
    state_counts.columns = ["regime", "count"]

    chart_counts = (
        alt.Chart(state_counts)
        .mark_bar()
        .encode(
            x=alt.X("regime:N", title="Regime"),
            y=alt.Y("count:Q", title="Days"),
            color=alt.Color(
                "regime:N",
                legend=None,
                scale=alt.Scale(range=["#22c55e", "#ef4444"]),
            ),
        )
        .configure_view(strokeWidth=0)
        .configure_axis(**axis_cfg)
    )
    st.altair_chart(chart_counts, use_container_width=True)


# --- Allocations tab ---
with tab_alloc:
    st.subheader("HRP vs Mean-Variance allocations")

    # Run backtests to get weights
    weights_mvo, weights_hrp, _, _, _, _ = run_backtests(
        returns_eval, lookback, rebalance_freq, risk_aversion, l2_reg
    )

    st.markdown("Select a rebalance date to inspect weights.")
    common_dates = weights_mvo.index.intersection(weights_hrp.index)
    if len(common_dates) == 0:
        st.warning(
            "No overlapping rebalance dates. Adjust evaluation window or parameters."
        )
    else:
        date_selected = st.selectbox(
            "Rebalance date",
            options=common_dates,
            index=len(common_dates) - 1,
        )
        w_mvo_sel = weights_mvo.loc[date_selected]
        w_hrp_sel = weights_hrp.loc[date_selected]

        # Long-format DataFrame: one row per (asset, model)
        df_long = pd.concat(
            [
                pd.DataFrame(
                    {
                        "asset": w_mvo_sel.index,
                        "weight": w_mvo_sel.values,
                        "model": "MVO",
                    }
                ),
                pd.DataFrame(
                    {
                        "asset": w_hrp_sel.index,
                        "weight": w_hrp_sel.values,
                        "model": "HRP",
                    }
                ),
            ],
            ignore_index=True,
        )

        st.markdown("**Mean-Variance vs HRP weights (grouped by asset)**")
        chart_alloc = (
            alt.Chart(df_long)
            .mark_bar()
            .encode(
                x=alt.X("asset:N", title="Asset"),
                y=alt.Y("weight:Q", title="Weight"),
                color=alt.Color(
                    "model:N",
                    title="Model",
                    scale=alt.Scale(range=["#1f77b4", "#ff7f0e"]),
                ),
                xOffset="model:N",
                tooltip=[
                    alt.Tooltip("model:N", title="Model"),
                    alt.Tooltip("asset:N", title="Asset"),
                    alt.Tooltip("weight:Q", title="Weight", format=".2%"),
                ],
            )
            .configure_view(strokeWidth=0)
            .configure_axis(
                grid=True,
                gridColor="#e6e6e6",
                gridOpacity=0.6,
                domain=False,
                tickColor="#888888",
                labelColor="#555555",
            )
        )
        st.altair_chart(chart_alloc, use_container_width=True)

        # Weights comparison table
        weights_compare = pd.DataFrame({"MVO": w_mvo_sel, "HRP": w_hrp_sel})
        st.markdown("**Weights comparison table**")
        st.dataframe(weights_compare.style.format("{:.2%}"))

        # ---------- Correlation heatmap (Altair) ----------
        st.markdown("### Correlation structure (last lookback window)")
        window_rets = returns_eval.iloc[-lookback:]
        corr = window_rets.corr()

        # Tidy format for Altair
        corr_long = (
            corr.reset_index()
            .melt(id_vars="index", var_name="col", value_name="corr")
            .rename(columns={"index": "row"})
        )

        heatmap = (
            alt.Chart(corr_long)
            .mark_rect()
            .encode(
                x=alt.X("col:N", title="", sort=corr.columns.tolist()),
                y=alt.Y("row:N", title="", sort=corr.index.tolist()),
                color=alt.Color(
                    "corr:Q",
                    title="Correlation",
                    scale=alt.Scale(scheme="redblue", domain=(-1, 1)),
                ),
                tooltip=[
                    alt.Tooltip("row:N", title="Row"),
                    alt.Tooltip("col:N", title="Col"),
                    alt.Tooltip("corr:Q", title="Corr", format=".2f"),
                ],
            )
        )

        text = (
            alt.Chart(corr_long)
            .mark_text(baseline="middle", fontSize=10)
            .encode(
                x="col:N",
                y="row:N",
                text=alt.Text("corr:Q", format=".2f"),
                color=alt.value("#000000"),
            )
        )

        chart_corr = (
            (heatmap + text)
            .properties(height=260)
            .configure_view(strokeWidth=0)
            .configure_axis(
                grid=False,
                domain=False,
                tickColor="#888888",
                labelColor="#555555",
            )
        )

        st.altair_chart(chart_corr, use_container_width=True)


# --- Backtest tab ---
with tab_backtest:
    st.subheader("Strategy backtest and performance")

    # Run backtests
    weights_mvo, weights_hrp, port_rets_mvo, port_rets_hrp, eq_mvo, eq_hrp = (
        run_backtests(returns_eval, lookback, rebalance_freq, risk_aversion, l2_reg)
    )

    # ---------- Equity curves (clean chart) ----------
    eq_df = pd.DataFrame({"MVO": eq_mvo, "HRP": eq_hrp})

    # Long-form for Altair
    eq_long = (
        eq_df.reset_index()
        .melt(
            id_vars=eq_df.index.name or "index",
            value_vars=["MVO", "HRP"],
            var_name="strategy",
            value_name="equity",
        )
        .rename(columns={eq_df.index.name or "index": "date"})
    )

    st.markdown("### Equity curves")

    # Tight y-range around data
    y_min = float(eq_long["equity"].min())
    y_max = float(eq_long["equity"].max())
    padding = (y_max - y_min) * 0.1 if y_max > y_min else 0.02
    y_domain = (max(0.8, y_min - padding), y_max + padding)

    base_eq = (
        alt.Chart(eq_long)
        .mark_line(interpolate="monotone", strokeWidth=2)
        .encode(
            x=alt.X("date:T", title="", axis=alt.Axis(format="%Y", grid=False)),
            y=alt.Y("equity:Q", title="", scale=alt.Scale(domain=y_domain)),
            color=alt.Color(
                "strategy:N",
                title="",
                scale=alt.Scale(range=["#1f77b4", "#ff7f0e"]),
            ),
        )
    )

    chart_eq = base_eq.configure_view(strokeWidth=0).configure_axis(
        grid=True,
        gridColor="#e6e6e6",
        gridOpacity=0.6,
        domain=False,
        tickColor="#888888",
        labelColor="#555555",
    )

    st.altair_chart(chart_eq, use_container_width=True)

    # ---------- Performance metrics ----------
    st.markdown("### Performance metrics")
    metrics_data = {
        "Sharpe": [
            sharpe_ratio(port_rets_mvo),
            sharpe_ratio(port_rets_hrp),
        ],
        "Max drawdown": [
            max_drawdown(eq_mvo),
            max_drawdown(eq_hrp),
        ],
        "Turnover": [
            turnover(weights_mvo),
            turnover(weights_hrp),
        ],
    }
    metrics_df = pd.DataFrame(metrics_data, index=["HMM + MVO", "HMM + HRP"])
    st.dataframe(
        metrics_df.style.format(
            {
                "Sharpe": "{:.2f}",
                "Max drawdown": "{:.1%}",
                "Turnover": "{:.2%}",
            }
        )
    )

    # ---------- Drawdown plot ----------
    st.markdown("### Drawdowns")
    dd_mvo = eq_mvo / eq_mvo.cummax() - 1.0
    dd_hrp = eq_hrp / eq_hrp.cummax() - 1.0
    dd_df = pd.DataFrame({"MVO": dd_mvo, "HRP": dd_hrp})
    st.area_chart(dd_df)
