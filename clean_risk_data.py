import pandas as pd

RAW_FILE = "risk_raw.csv"
OUT_FILE = "risk_processed.csv"

print("Loading dataset...")
df = pd.read_csv(RAW_FILE)

print("Rows:", len(df))
print("Columns:", list(df.columns))

# -----------------------------
# Clean column names
# -----------------------------
df.columns = df.columns.str.strip()

# -----------------------------
# Select required columns
# -----------------------------
used_cols = [
    "Region",
    "Year",
    "WRI",
    "Exposure",
    "Vulnerability",
    "Susceptibility",
    "Lack of Coping Capabilities",
    "Lack of Adaptive Capacities",
    "WRI Category"
]

df = df[used_cols]

# -----------------------------
# Rename for ML friendliness
# -----------------------------
df.rename(columns={
    "Region":"region",
    "Year":"year",
    "WRI":"wri",
    "Exposure":"exposure",
    "Vulnerability":"vulnerability",
    "Susceptibility":"susceptibility",
    "Lack of Coping Capabilities":"coping",
    "Lack of Adaptive Capacities":"adaptive",
    "WRI Category":"risk_category"
}, inplace=True)

# -----------------------------
# Convert numeric columns
# -----------------------------
numeric_cols = [
    "year",
    "wri",
    "exposure",
    "vulnerability",
    "susceptibility",
    "coping",
    "adaptive"
]

for col in numeric_cols:
    df[col] = pd.to_numeric(df[col], errors="coerce")

# -----------------------------
# Remove incomplete rows
# -----------------------------
df.dropna(inplace=True)

# -----------------------------
# Save cleaned dataset
# -----------------------------
df.to_csv(OUT_FILE, index=False)

print("✅ Processed dataset saved as:", OUT_FILE)
print("Final rows:", len(df))
print(df.head())
