import pandas as pd
import numpy as np
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_squared_error
import warnings

warnings.filterwarnings("ignore")

# -------- CONFIG --------
DATA_PATH = "GW data (missing filled).csv"
BEST_LAGS_CSV = "best_lag_arima_summary.csv"  # From your lag selection
FORECAST_YEARS = [2025, 2026, 2027, 2028, 2029]
FORECAST_MONTHS = len(FORECAST_YEARS) * 12

# -------- Load and prepare data --------
df = pd.read_csv(DATA_PATH)
df.columns = df.columns.str.strip()
df["Date"] = pd.to_datetime(df["Date"])
df = df.set_index("Date").sort_index()

lags_df = pd.read_csv(BEST_LAGS_CSV)
results = []

# -------- Loop over wells --------
for _, row in lags_df.iterrows():
    well = row["Well"]
    p = int(row["Best_Lag(p)"])
    
    series = df[well].copy().dropna()
    if len(series) < 100:
        print(f"Skipped {well}: insufficient data.")
        continue

    try:
        model = SARIMAX(series, order=(p,1,0), seasonal_order=(0,1,1,12), enforce_stationarity=False)
        fitted = model.fit(disp=False)
        
        # Forecast
        forecast = fitted.forecast(steps=FORECAST_MONTHS)
        forecast.index = pd.date_range(series.index[-1] + pd.DateOffset(months=1), periods=FORECAST_MONTHS, freq='MS')
        yearly_forecast = forecast.resample("Y").mean()
        
        row_data = {
            "Well": well,
            "Order": f"SARIMA({p},1,0)(0,1,1,12)",
            "AIC": round(fitted.aic, 1),
            "RMSE": round(np.sqrt(mean_squared_error(series[-FORECAST_MONTHS:], fitted.fittedvalues[-FORECAST_MONTHS:])), 4)
        }

        for year in FORECAST_YEARS:
            val = yearly_forecast[yearly_forecast.index.year == year].mean()
            row_data[str(year)] = round(val, 2) if not pd.isna(val) else None

        results.append(row_data)

    except Exception as e:
        print(f"Error fitting SARIMA for {well}: {e}")

# -------- Save and print results --------
output_df = pd.DataFrame(results)
print("\nSARIMA Forecast Summary (2025–2029):\n")
print(output_df)

output_df.to_csv("sarima_forecast_with_best_lags.csv", index=False)
print("\nSaved to: sarima_forecast_with_best_lags.csv")
