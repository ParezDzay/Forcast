import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import numpy as np

# Load the SARIMA forecast data with coordinates
file_path = "sarima_forecast_with_best_lags.csv"
df = pd.read_csv(file_path)

# Add coordinates for each well
coordinates = {
    'W1': (44.082643, 36.251453),
    'W2': (43.910736, 36.24819),
    'W3': (44.139397, 36.207817),
    'W4': (44.028271, 36.128172),
    'W5': (43.794883, 36.114882),
    'W6': (44.022078, 36.096136),
    'W7': (43.949399, 36.07923),
    'W8': (43.838362, 36.076239),
    'W9': (43.88417, 36.073222),
    'W10': (44.191777, 36.05757),
    'W11': (43.817923, 36.048399),
    'W12': (43.733967, 36.041898),
    'W13': (43.923973, 36.022505),
    'W14': (43.833997, 36.005553),
    'W15': (44.110357, 35.99346),
    'W16': (44.057626, 35.981464),
    'W17': (44.040146, 35.981325),
    'W18': (43.946396, 35.972911),
    'W19': (43.887315, 35.969709),
    'W20': (44.033139, 35.958297)
}

df["Longitude"] = df["Well"].map(lambda w: coordinates[w][0])
df["Latitude"] = df["Well"].map(lambda w: coordinates[w][1])

# Use forecast values for clustering
forecast_cols = ['2025', '2026', '2027', '2028', '2029']
X = df[forecast_cols].values

# Apply KMeans clustering
kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
df['Cluster'] = kmeans.fit_predict(X)

# Plot clustering result on map using real coordinates
plt.figure(figsize=(10, 8))
scatter = plt.scatter(df['Longitude'], df['Latitude'], c=df['Cluster'], cmap='Set1', s=100, edgecolor='k')
plt.title('K-Means Clustering of Wells Based on SARIMA Forecast Patterns')
plt.xlabel('Longitude')
plt.ylabel('Latitude')
plt.grid(True)
plt.colorbar(scatter, label='Cluster')
plt.tight_layout()
plt.show()
