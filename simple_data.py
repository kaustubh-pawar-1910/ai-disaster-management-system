import pandas as pd

print(">>> Running simple_data.py")

data = [
    {
        "incident_id": 1,
        "state": "Maharashtra",
        "city": "Pune",
        "disaster_type": "Flood",
        "month": 7,
        "year": 2023,
        "casualties": 3,
        "economic_loss_crores": 10.5,
        "response_time_hours": 4.0,
        "severity": "Medium",
    },
    {
        "incident_id": 2,
        "state": "Gujarat",
        "city": "Ahmedabad",
        "disaster_type": "Earthquake",
        "month": 1,
        "year": 2022,
        "casualties": 10,
        "economic_loss_crores": 35.0,
        "response_time_hours": 6.0,
        "severity": "High",
    },
    {
        "incident_id": 3,
        "state": "Karnataka",
        "city": "Bengaluru",
        "disaster_type": "Fire",
        "month": 11,
        "year": 2024,
        "casualties": 1,
        "economic_loss_crores": 2.0,
        "response_time_hours": 2.5,
        "severity": "Low",
    },
]

df = pd.DataFrame(data)

print("\nSample disaster data:")
print(df)

df.to_csv("incidents_small.csv", index=False)
print("\nSaved to incidents_small.csv")
