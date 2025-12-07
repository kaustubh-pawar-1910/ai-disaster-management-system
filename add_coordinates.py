import pandas as pd

DATA = "final_dataset.csv"
OUT = "mapped_dataset.csv"

# Mini geocode lookup for major Indian cities
city_coords = {
    "Pune": (18.5204, 73.8567),
    "Mumbai": (19.0760, 72.8777),
    "Delhi": (28.6139, 77.2090),
    "Chennai": (13.0827, 80.2707),
    "Bangalore": (12.9716, 77.5946),
    "Hyderabad": (17.3850, 78.4867),
    "Kolkata": (22.5726, 88.3639),
    "Ahmedabad": (23.0225, 72.5714),
    "Jaipur": (26.9124, 75.7873)
}

df = pd.read_csv(DATA)

# Assign coordinates
def geo(city):
    return city_coords.get(city, (20.5937, 78.9629))  # center of India fallback

coords = df["city"].apply(geo)

df["latitude"] = coords.apply(lambda x: x[0])
df["longitude"] = coords.apply(lambda x: x[1])

df.to_csv(OUT, index=False)

print("✅ Mapping complete → mapped_dataset.csv")
print(df.head())
