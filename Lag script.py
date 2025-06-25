import pandas as pd
import numpy as np
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import os
import warnings

warnings.filterwarnings("ignore")

# -------- CONFIG --------
DATA_PATH = "GW data (missing filled).csv"
FORECAST_HORIZON = 60
LAGS_TO_TEST = [3, 6, 9, 12, 15, 18, 21, 24]
PLOT_DIR = "acf_pacf_lag_test_plots"

# -------- Create plot output directory --------
os.makedirs(PLOT_DIR, exist_ok=True)

# -------- Load and prepare data --------
df = pd.read_csv(DATA_PATH)
df.columns = df.columns.str.strip()
df["Date"] = pd.to_datetime(df["Date"])
df = df.set_index("Date").sort_index()

# -------- Data cleaning function --------
def clean_series(series):
    q1, q3 = series.quantile([0.25, 0.75])
    iqr = q3 - q1
    return series.where(series.between(q1 - 3 * iqr, q3 + 3 * iqr)).interpolate(limit_direction="both")

# -------- Run lag comparison across wells --------
summary_rows = []

for well in [col for col in df.columns if col.startswith("W")]:
    series = clean_series(df[well])
    if series.isnull().sum() > 0:
        print(f"Skipped {well} due to NaNs.")
        continue

    # ---- Plot ACF & PACF and save ----
    try:
        fig, axes = plt.subplots(2, 1, figsize=(12, 6))
        plot_acf(series.dropna(), lags=24, ax=axes[0])
        axes[0].set_title(f"ACF - {well}")
        plot_pacf(series.dropna(), lags=24, ax=axes[1])
        axes[1].set_title(f"PACF - {well}")
        plt.tight_layout()
        plot_path = os.path.join(PLOT_DIR, f"ACF_PACF_{well}.png")
        plt.savefig(plot_path)
        plt.close()
        print(f"Saved ACF/PACF plot: {plot_path}")
    except Exception as e:
        print(f"Error plotting ACF/PACF for {well}: {e}")

    # ---- Evaluate fixed lags ----
    best_rmse = np.inf
    best_lag = None
    best_aic = None

    for p in LAGS_TO_TEST:
        try:
            model = ARIMA(series, order=(p, 1, 0))
            fitted = model.fit()

            if len(series) > FORECAST_HORIZON:
                actual = series[-FORECAST_HORIZON:]
                pred = fitted.fittedvalues[-FORECAST_HORIZON:]
                rmse = np.sqrt(mean_squared_error(actual, pred))
            else:
                rmse = np.sqrt(fitted.mse)

            if rmse < best_rmse:
                best_rmse = rmse
                best_lag = p
                best_aic = fitted.aic

        except Exception as e:
            print(f"Error fitting ARIMA({p},1,0) for {well}: {e}")
            continue

    summary_rows.append({
        "Well": well,
        "Best_Lag(p)": best_lag,
        "RMSE": round(best_rmse, 4),
        "AIC": round(best_aic, 1) if best_aic is not None else None
    })

# -------- Save and print results --------
summary_df = pd.DataFrame(summary_rows)
print("\nBest Lag Evaluation Summary for ARIMA(p,1,0):\n")
print(summary_df)

output_file = "best_lag_arima_summary.csv"
summary_df.to_csv(output_file, index=False)
print(f"\nSaved results to: {output_file}")