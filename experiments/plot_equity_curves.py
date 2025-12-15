# experiments/plot_equity_curves.py

import pandas as pd
import matplotlib.pyplot as plt


def main():
    # 1. Load equity curves saved by main_run.py
    eq_df = pd.read_csv(
        "data/processed/equity_curves_2018_2023.csv",
        index_col=0,
        parse_dates=True,
    )

    # Expect columns: "MVO", "HRP", "HRP_Predictive"
    print(eq_df.head())

    # 2. Full-period plot
    plt.figure(figsize=(10, 5))
    plt.plot(eq_df.index, eq_df["MVO"], label="HMM + MVO")
    plt.plot(eq_df.index, eq_df["HRP"], label="HMM + HRP (Reactive)")
    plt.plot(eq_df.index, eq_df["HRP_Predictive"], label="HMM + HRP (Predictive)")
    plt.title("Equity curves 2018–2023")
    plt.xlabel("Date")
    plt.ylabel("Equity (growth of 1)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

    # 3. Zoomed crisis window: COVID crash (adjust dates as needed)
    crisis_start = "2020-02-01"
    crisis_end = "2020-05-31"
    eq_crisis = eq_df.loc[crisis_start:crisis_end]

    plt.figure(figsize=(10, 5))
    plt.plot(eq_crisis.index, eq_crisis["MVO"], label="HMM + MVO")
    plt.plot(eq_crisis.index, eq_crisis["HRP"], label="HMM + HRP (Reactive)")
    plt.plot(
        eq_crisis.index, eq_crisis["HRP_Predictive"], label="HMM + HRP (Predictive)"
    )
    plt.title(f"Equity curves in crisis window {crisis_start}–{crisis_end}")
    plt.xlabel("Date")
    plt.ylabel("Equity (growth of 1)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
