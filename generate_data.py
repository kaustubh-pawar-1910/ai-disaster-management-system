import pandas as pd
import random

states_cities = {
    "Maharashtra": ["Mumbai", "Pune", "Nagpur", "Nashik"],
    "Gujarat": ["Ahmedabad", "Surat", "Vadodara"],
    "Karnataka": ["Bengaluru", "Mysuru", "Mangaluru"],
    "Tamil Nadu": ["Chennai", "Coimbatore", "Madurai"],
    "West Bengal": ["Kolkata", "Howrah", "Durgapur"],
}

disasters = ["Flood", "Fire", "Cyclone", "Earthquake", "Landslide"]

rows = []

for i in range(1, 801):
    state = random.choice(list(states_cities.keys()))
    city = random.choice(states_cities[state])
    disaster = random.choice(disasters)
    
    month = random.randint(1, 12)
    year = random.randint(2015, 2025)

    casualties = random.randint(0, 40)
    loss = round(random.uniform(0.1, 80), 2)
    response_time = round(random.uniform(1, 24), 1)

    # RULE BASED SEVERITY SCORE
    score = (casualties * 0.4) + (loss * 0.4) + ((24 - response_time) * 0.2)

    if score < 15:
        severity = "Low"
    elif score < 35:
        severity = "Medium"
    else:
        severity = "High"

    rows.append({
        "incident_id": i,
        "state": state,
        "city": city,
        "disaster_type": disaster,
        "month": month,
        "year": year,
        "casualties": casualties,
        "economic_loss_crores": loss,
        "response_time_hours": response_time,
        "severity": severity
    })

df = pd.DataFrame(rows)
df.to_csv("incidents_full.csv", index=False)

print("✅ incidents_full.csv created with", len(df), "rows")
