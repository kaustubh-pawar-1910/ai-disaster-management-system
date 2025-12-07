import pandas as pd

RAW = "risk_new.csv"
OUT = "final_dataset.csv"

df = pd.read_csv(RAW)

print("Rows before cleaning:", len(df))

# ---------------------------------
# CLEAN
# ---------------------------------
df.drop_duplicates(inplace=True)

# Ensure correct types
num_cols = ["month", "year", "casualties",
            "economic_loss_crores", "response_time_hours"]
df[num_cols] = df[num_cols].apply(pd.to_numeric, errors="coerce")

# Remove invalid rows
df.dropna(inplace=True)
df = df[
    (df["month"] >= 1) & (df["month"] <= 12) &
    (df["casualties"] >= 0) &
    (df["economic_loss_crores"] >= 0) &
    (df["response_time_hours"] > 0)
]

print("Rows after cleaning:", len(df))

# ---------------------------------
# FEATURE ENGINEERING
# ---------------------------------

# SEASON
def season(m):
    if m in [12,1,2]:
        return "Winter"
    if m in [3,4,5]:
        return "Summer"
    if m in [6,7,8,9]:
        return "Monsoon"
    return "Post-Monsoon"

df["season"] = df["month"].apply(season)

# RISK SCORE
df["risk_score"] = (
    df["casualties"] * 0.4
    + df["economic_loss_crores"] * 0.4
    + (24 - df["response_time_hours"]) * 0.2
)

df.to_csv(OUT, index=False)

print("\n✅ FINAL DATASET CREATED → final_dataset.csv")
print(df.head())
