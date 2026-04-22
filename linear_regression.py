import numpy as np
import pandas as pd

def linear_regression(ticker_x, ticker_y, year):

    TICKER_X = ticker_x
    TICKER_Y = ticker_y
    YEAR = year - 1  

    df_x = pd.read_csv(f"/Lean/Data/custom/{TICKER_X}_1d_bars_{YEAR}.csv", parse_dates=["timestamp"])
    df_y = pd.read_csv(f"/Lean/Data/custom/{TICKER_Y}_1d_bars_{YEAR}.csv", parse_dates=["timestamp"])

    # Normalize dates (strip time component for merging)
    df_x["date"] = df_x["timestamp"].dt.date
    df_y["date"] = df_y["timestamp"].dt.date    

    # Merge on date so both stocks are aligned
    merged = pd.merge(
        df_x[["date", "close"]].rename(columns={"close": TICKER_X}),
        df_y[["date", "close"]].rename(columns={"close": TICKER_Y}),
        on="date",
        how="inner"
    )

    print(f"Date range: {merged['date'].iloc[0]} to {merged['date'].iloc[-1]}")
    print(f"Trading days: {len(merged)}\n")

    log_x = np.log(merged[TICKER_X].values)
    log_y = np.log(merged[TICKER_Y].values)

    x_bar = log_x.mean()
    y_bar = log_y.mean()

    x_dev = log_x - x_bar
    y_dev = log_y - y_bar

    beta = (x_dev * y_dev).sum() / (x_dev ** 2).sum()
    alpha = y_bar - beta * x_bar

    # Spread (residuals)
    spread = log_y - (alpha + beta * log_x)

    # R-squared
    ss_res = (spread ** 2).sum()
    ss_tot = (y_dev ** 2).sum()
    r_squared = 1 - ss_res / ss_tot

    R = ss_res / (len(merged) - 2) 
    n = len(merged)
    P_bb = R / (x_dev ** 2).sum()
    P_aa = R * (log_x ** 2).sum() / (n * (x_dev ** 2).sum())
    P_ab = -R * log_x.sum() / (n * (x_dev ** 2).sum())

    P = np.array([[P_aa, P_ab],
                [P_ab, P_bb]])

    window = 20
    rolling_betas = []
    rolling_alphas = []

    for i in range(window, len(log_x)):
        xw = log_x[i-window:i]
        yw = log_y[i-window:i]

        xw_bar = xw.mean()
        yw_bar = yw.mean()
        xw_dev = xw - xw_bar
        yw_dev = yw - yw_bar

        beta_w = (xw_dev * yw_dev).sum() / (xw_dev ** 2).sum()
        alpha_w = yw_bar - beta_w * xw_bar

        rolling_alphas.append(alpha_w)
        rolling_betas.append(beta_w)
    
    beta_diffs = np.diff(rolling_betas)
    Q_bb = np.var(beta_diffs)
    alpha_diffs = np.diff(rolling_alphas)
    Q_aa = np.var(alpha_diffs)
    Q = np.array([[Q_aa, 0],
                [0, Q_bb]])
    # ============================================================
    # RESULTS
    # ============================================================
    print("=" * 50)
    print(f"OLS REGRESSION: log({TICKER_X}) = α + β·log({TICKER_Y})")
    print("=" * 50)
    print(f"Beta (hedge ratio):  {beta:.4f}")
    print(f"Alpha (intercept):   {alpha:.4f}")
    print(f"R²:                  {r_squared:.4f}")
    print(f"Spread std:          {spread.std():.6f}")
    print()
    print(f"Interpretation:")
    print(f"  A 1% move in {TICKER_Y} corresponds to ~{beta:.2f}% move in {TICKER_X}")
    print(f"  R² of {r_squared:.4f} means {TICKER_Y} explains {r_squared*100:.1f}% of {TICKER_X}'s price variation")
    print()

    
    return alpha, beta, P, Q, R