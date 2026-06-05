# Time Series Analysis - Air Passengers Dataset

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_squared_error

# ----------------------------
# 1. Load Dataset
# ----------------------------

df = pd.read_csv("airline-passengers.csv")

# Convert Month column to datetime
df["Month"] = pd.to_datetime(df["Month"])

# Set Month as index
df.set_index("Month", inplace=True)

print("First 5 Rows:")
print(df.head())

print("\nDataset Info:")
print(df.info())

# ----------------------------
# 2. Plot Original Data
# ----------------------------

plt.figure(figsize=(12, 6))
plt.plot(df["Passengers"])
plt.title("Monthly Airline Passengers")
plt.xlabel("Year")
plt.ylabel("Passengers")
plt.grid(True)
plt.show()

# ----------------------------
# 3. Trend & Seasonality Decomposition
# ----------------------------

decomposition = seasonal_decompose(
    df["Passengers"],
    model="multiplicative",
    period=12
)

fig = decomposition.plot()
fig.set_size_inches(12, 8)
plt.show()

# ----------------------------
# 4. Moving Averages
# ----------------------------

df["MA_6"] = df["Passengers"].rolling(window=6).mean()
df["MA_12"] = df["Passengers"].rolling(window=12).mean()

plt.figure(figsize=(12, 6))
plt.plot(df["Passengers"], label="Actual")
plt.plot(df["MA_6"], label="6-Month Moving Average")
plt.plot(df["MA_12"], label="12-Month Moving Average")
plt.title("Moving Average Analysis")
plt.legend()
plt.grid(True)
plt.show()

# ----------------------------
# 5. Train-Test Split
# ----------------------------

train = df.iloc[:-12]
test = df.iloc[-12:]

# ----------------------------
# 6. Build SARIMA Model
# ----------------------------

model = SARIMAX(
    train["Passengers"],
    order=(2, 1, 1),
    seasonal_order=(1, 1, 1, 12)
)

results = model.fit()

# ----------------------------
# 7. Forecast
# ----------------------------

forecast = results.forecast(steps=12)

# ----------------------------
# 8. RMSE Calculation
# ----------------------------

rmse = np.sqrt(
    mean_squared_error(
        test["Passengers"],
        forecast
    )
)

print(f"\nRMSE: {rmse:.2f}")

# ----------------------------
# 9. Plot Forecast vs Actual
# ----------------------------

plt.figure(figsize=(12, 6))

plt.plot(
    train.index,
    train["Passengers"],
    label="Training Data"
)

plt.plot(
    test.index,
    test["Passengers"],
    label="Actual Data"
)

plt.plot(
    test.index,
    forecast,
    label="Forecast"
)

plt.title("SARIMA Forecast vs Actual")
plt.xlabel("Year")
plt.ylabel("Passengers")
plt.legend()
plt.grid(True)
plt.show()

# ----------------------------
# 10. Future Forecast
# ----------------------------

future_forecast = results.forecast(steps=12)

print("\nNext 12 Months Forecast:")
print(future_forecast)