import pandas as pd
import numpy as np

# ----------------------------------------------------
# 1. LOAD RAW EM-DAT DATA (SAFE ENCODING)
# ----------------------------------------------------
RAW_FILE = "emdat_raw.csv"

print(f"Loading raw data from: {RAW_FILE}")

# ✅ FORCE safe decoding to avoid Unicode errors
print(f"Loading raw data from: {RAW_FILE}")

df = pd.read_csv(
    RAW_FILE,
    encoding="latin1",
    engine="python",
    on_bad_lines="skip"
)

print("✅ File loaded successfully!")
print("Rows before cleaning:", len(df))


print("✅ File loaded successfully!")
print("Rows before cleaning:", len(df))

print("\nColumns found in dataset:")
for c in df.columns:
    print("-", c)

print("-" * 60)

# ----------------------------------------------------
# 2. SELECT RELEVANT COLUMNS
# ----------------------------------------------------
expected_cols = [
    "Disaster Type",
    "Disaster Subtype",
    "Country",
    "Region",
    "Location",
    "Start Year",
    "Start Month",
    "Total Deaths",
    "No. Injured",
    "Total Affected",
    "Total Damage",
    "Magnitude",
    "Latitude",
    "Longitude",
]

available_cols = [c for c in expected_cols if c in df.columns]

missing_cols = list(set(expected_cols) - set(available_cols))
if missing_cols:
    print("\n⚠️ Missing columns:")
    for c in missing_cols:
        print(" -", c)

df = df[available_cols].copy()

# ----------------------------------------------------
# 3. RENAME COLUMNS
# ----------------------------------------------------
df = df.rename(columns={
    "Disaster Type": "disaster_type",
    "Disaster Subtype": "disaster_subtype",
    "Country": "country",
    "Region": "region",
    "Location": "location",
    "Start Year": "year",
    "Start Month": "month",
    "Total Deaths": "total_deaths",
    "No. Injured": "injured",
    "Total Affected": "affected",
    "Total Damage": "damage",
    "Magnitude": "magnitude",
    "Latitude": "latitude",
    "Longitude": "longitude",
})

# ----------------------------------------------------
# 4. CLEAN DATA
# ----------------------------------------------------
num_cols = [
    "year", "month",
    "total_deaths", "injured", "affected",
    "damage", "magnitude", "latitude", "longitude"
]

for col in num_cols:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)

text_cols = ["disaster_type", "disaster_subtype", "country", "region", "location"]

for col in text_cols:
    if col in df.columns:
        df[col] = df[col].fillna("Unknown").astype(str)

# Remove bad years
if "year" in df.columns:
    before = len(df)
    df = df[df["year"] > 0]
    print(f"\nRemoved {before - len(df)} rows with invalid year.")

# ----------------------------------------------------
# 5. CREATE REAL SEVERITY SCORE
# ----------------------------------------------------
df["severity_score"] = (
    df.get("total_deaths", 0) * 0.4 +
    df.get("affected", 0) * 0.3 +
    df.get("damage", 0) * 0.2 +
    df.get("magnitude", 0) * 0.1
)

# ----------------------------------------------------
# 6. CREATE LABELS
# ----------------------------------------------------
def label_severity(score):
    if score < 15:
        return "Low"
    elif score < 40:
        return "Medium"
    else:
        return "High"

df["severity"] = df["severity_score"].apply(label_severity)

# ----------------------------------------------------
# 7. FINAL DATASET
# ----------------------------------------------------
final_cols = [
    "disaster_type",
    "disaster_subtype",
    "country",
    "region",
    "location",
    "year",
    "month",
    "total_deaths",
    "injured",
    "affected",
    "damage",
    "magnitude",
    "latitude",
    "longitude",
    "severity_score",
    "severity"
]

df_final = df[final_cols].copy()

# ----------------------------------------------------
# 8. SAVE OUTPUT
# ----------------------------------------------------
OUT_FILE = "emdat_processed.csv"

df_final.to_csv(OUT_FILE, index=False)

print("\n✅ CLEAN DATASET CREATED")
print("Saved as:", OUT_FILE)
print("Final rows:", len(df_final))

print("\nSample rows:")
print(df_final.head())
