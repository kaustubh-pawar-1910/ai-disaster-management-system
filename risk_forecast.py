import pandas as pd

df = pd.read_csv("final_dataset.csv")

# Group by state-month to get disaster frequency
trend = df.groupby(["state","month"]).size().reset_index(name="count")

# Compute rolling 3-month average risk
trend["forecast_risk"] = (
    trend.groupby("state")["count"]
    .rolling(3, min_periods=1)
    .mean()
    .reset_index(level=0, drop=True)
)

trend.to_csv("risk_forecast.csv", index=False)

print("✅ Future risk data created")
