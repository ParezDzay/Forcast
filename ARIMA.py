import pandas as pd
import numpy as np
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_squared_error
from pathlib import Path
import warnings

warnings.filterwarnings("ignore", category=UserWarning)

# -------- CONFIG --------
DATA_PATH = "GW data (missing filled).csv"
FORECAST_HORIZON = 60  # 5 years = 60 months

# -------- Load and prepare data --------
df = pd.read_csv(DATA_PATH)
df["Date"] = pd.to_datetime(df["Year"].astype(str) + "-" + df["Months"].astype(str).str.zfill(2) + "-01")
df = df.sort_values("Date").reset_index(drop=True)
df.set_index("Date", inplace=True)

# -------- Data cleaning --------
def clean_series(series):
    q1, q3 = series.quantile([0.25, 0.75])
    iqr = q3 - q1
    cleaned = series.where(series.between(q1 - 3 * iqr, q3 + 3 * iqr))
    return cleaned.interpolate(limit_direction="both")

# -------- SARIMA with R² and forecast --------
def sarima_forecast(series, H=FORECAST_HORIZON):
    model = SARIMAX(series, order=(0,1,1), seasonal_order=(0,1,1,12),
                    enforce_stationarity=False, enforce_invertibility=False)
    results = model.fit(disp=False, maxiter=25)

    # Forecast future
    forecast = results.get_forecast(steps=H).predicted_mean
    forecast.index = pd.date_range(series.index[-1] + pd.DateOffset(months=1), periods=H, freq="MS")

    # Calculate RMSE
    if len(series) > H:
        actual = series[-H:]
        pred = results.fittedvalues[-H:]
        rmse = np.sqrt(mean_squared_error(actual, pred))
        r2 = 1 - (np.sum((actual - pred)**2) / np.sum((actual - actual.mean())**2))
    else:
        rmse = np.sqrt(results.mse)
        r2 = np.nan  # not enough data

    # Aggregate to yearly means (2025–2029)
    annual_means = forecast.resample("Y").mean()
    yearly = {str(y.year): round(v, 2) for y, v in annual_means.items() if 2025 <= y.year <= 2029}

    metrics = {
        "R²": round(r2, 4) if not np.isnan(r2) else np.nan,
        "RMSE": round(rmse, 4),
        "AIC": round(results.aic, 1),
        "BIC": round(results.bic, 1),
    }

    return metrics, yearly

# -------- Run for all wells --------
results = []
forecast_years = ["2025", "2026", "2027", "2028", "2029"]

for well in [c for c in df.columns if c.startswith("W")]:
    series = clean_series(df[well])
    if series.isnull().sum() > 0:
        print(f"Skipped {well} due to NaNs.")
        continue
    try:
        metrics, yearly_forecast = sarima_forecast(series)
        row = {"Well": well, **metrics}
        for y in forecast_years:
            row[y] = yearly_forecast.get(y, np.nan)
        results.append(row)
    except Exception as e:
        print(f"Error on {well}: {e}")

# -------- Save and Show --------
summary_df = pd.DataFrame(results)
print("\nSARIMA Forecast Summary (2025–2029):\n")
print(summary_df)

output_file = "arima_forecast_2025_2029.csv"
summary_df.to_csv(output_file, index=False)
print(f"\nSaved results to: {output_file}")
