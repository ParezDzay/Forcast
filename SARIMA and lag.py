import pandas as pd
import numpy as np
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_squared_error
import warnings

warnings.filterwarnings("ignore")

# -------- CONFIG --------
DATA_PATH = "GW data (missing filled).csv"
BEST_LAG_FILE = "best_lag_arima_summary.csv"
FORECAST_HORIZON = 60
FORECAST_YEARS = ["2025", "2026", "2027", "2028", "2029"]
SEASONAL_ORDER = (0, 1, 1, 12)  # Monthly seasonal pattern

# -------- Load data --------
df = pd.read_csv(DATA_PATH)
df.columns = df.columns.str.strip()
df["Date"] = pd.to_datetime(df["Date"])
df = df.set_index("Date").sort_index()

lags_df = pd.read_csv(BEST_LAG_FILE)

# -------- Clean time series --------
def clean_series(series):
    q1, q3 = series.quantile([0.25, 0.75])
    iqr = q3 - q1
    return series.where(series.between(q1 - 3 * iqr, q3 + 3 * iqr)).interpolate(limit_direction="both")

# -------- Forecast using SARIMA --------
results = []

for _, row in lags_df.iterrows():
    well = row["Well"]
    p = int(row["Best_Lag(p)"])

    series = clean_series(df[well])
    if series.isnull().sum() > 0:
        print(f"Skipped {well} due to NaNs.")
        continue

    try:
        model = SARIMAX(series,
                        order=(p, 1, 0),
                        seasonal_order=SEASONAL_ORDER,
                        enforce_stationarity=False,
                        enforce_invertibility=False)
        fitted = model.fit(disp=False)

        forecast = fitted.get_forecast(steps=FORECAST_HORIZON).predicted_mean
        forecast.index = pd.date_range(series.index[-1] + pd.DateOffset(months=1), periods=FORECAST_HORIZON, freq="MS")
        yearly = forecast.resample("Y").mean()
        yearly_forecast = {str(y.year): round(v, 2) for y, v in yearly.items() if y.year in range(2025, 2030)}

        row_data = {
            "Well": well,
            "Order": f"SARIMA({p},1,0)(0,1,1,12)",
            "AIC": round(fitted.aic, 1),
            "RMSE": round(np.sqrt(fitted.mse), 4),
        }
        for y in FORECAST_YEARS:
            row_data[y] = yearly_forecast.get(y, np.nan)

        results.append(row_data)

    except Exception as e:
        print(f"Error on {well}: {e}")

# -------- Output results --------
summary_df = pd.DataFrame(results)
print("\nSARIMA Forecast Summary (2025–2029 using best p):\n")
print(summary_df)

output_file = "sarima_forecast_best_p_2025_2029.csv"
summary_df.to_csv(output_file, index=False)
print(f"\nSaved results to: {output_file}")
