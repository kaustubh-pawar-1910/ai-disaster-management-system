import pandas as pd
import joblib
import streamlit as st
import plotly.express as px

# ---------------- CONFIG ----------------
DATA_PATH = "final_dataset.csv"
MODEL_PATH = "models/severity_model.pkl"

# ---------------- LOAD ----------------
@st.cache_data(show_spinner=False)
def load_data():
    return pd.read_csv(DATA_PATH)

@st.cache_resource(show_spinner=False)
def load_model():
    return joblib.load(MODEL_PATH)

df = load_data()
model = load_model()

st.set_page_config(page_title="Disaster Management System", layout="wide")

# ---------------- TITLE ----------------
st.title("🌪️ AI-based Smart Disaster Management System")
st.write("Prediction and analytics powered by a trained machine learning model.")

# ---------------- DASHBOARD ----------------
st.subheader("📊 Incident Analytics")

col1, col2 = st.columns(2)

with col1:
    fig1 = px.pie(df, names="severity", title="Severity Distribution")
    st.plotly_chart(fig1, use_container_width=True)

with col2:
    state_counts = df["state"].value_counts().reset_index()
    state_counts.columns = ["state", "count"]
    fig2 = px.bar(
        state_counts,
        x="state",
        y="count",
        title="Incidents by State"
    )
    st.plotly_chart(fig2, use_container_width=True)

type_counts = df["disaster_type"].value_counts().reset_index()
type_counts.columns = ["disaster_type", "count"]
fig3 = px.bar(
    type_counts,
    x="disaster_type",
    y="count",
    title="Incidents by Disaster Type"
)
st.plotly_chart(fig3, use_container_width=True)

st.divider()

# ---------------- PREDICTION FORM ----------------
st.subheader("🤖 Predict Disaster Severity")

state = st.selectbox("State", sorted(df["state"].unique()))
city = st.selectbox("City", sorted(df[df["state"] == state]["city"].unique()))
disaster_type = st.selectbox("Disaster Type", sorted(df["disaster_type"].unique()))

month = st.slider("Month", 1, 12, 6)
year = st.number_input(
    "Year",
    min_value=int(df["year"].min()),
    max_value=int(df["year"].max()),
    value=int(df["year"].max())
)

casualties = st.number_input("Casualties", min_value=0, value=5)
economic_loss = st.number_input(
    "Economic Loss (Crores)",
    min_value=0.0,
    value=10.0,
    step=0.5
)
response_time = st.number_input(
    "Response Time (Hours)",
    min_value=0.1,
    value=5.0,
    step=0.5
)

# same season logic used in dataset prep
def season(m):
    if m in [12, 1, 2]:
        return "Winter"
    if m in [3, 4, 5]:
        return "Summer"
    if m in [6, 7, 8, 9]:
        return "Monsoon"
    return "Post-Monsoon"

season_value = season(month)

if st.button("Predict Severity"):
    input_df = pd.DataFrame([{
        "state": state,
        "city": city,
        "disaster_type": disaster_type,
        "month": month,
        "year": year,
        "casualties": casualties,
        "economic_loss_crores": economic_loss,
        "response_time_hours": response_time,
        "season": season_value
    }])

    pred = model.predict(input_df)[0]
    proba = model.predict_proba(input_df)[0]

    st.success(f"🔮 Predicted Severity: **{pred}**")

    st.write("Prediction probabilities:")
    st.bar_chart(
        {
            cls: float(p)
            for cls, p in zip(model.classes_, proba)
        }
    )

# ---------------- MAP VISUALIZATION ----------------
st.subheader("🗺️ Incident Map")

# Simple city → coordinates lookup
city_coords = {
    "Pune": (18.5204, 73.8567),
    "Mumbai": (19.0760, 72.8777),
    "Delhi": (28.6139, 77.2090),
    "Chennai": (13.0827, 80.2707),
    "Bangalore": (12.9716, 77.5946),
    "Hyderabad": (17.3850, 78.4867),
    "Kolkata": (22.5726, 88.3639),
    "Ahmedabad": (23.0225, 72.5714),
    "Jaipur": (26.9124, 75.7873),
}

def get_coords(city):
    # default: center of India if city not in dict
    return city_coords.get(city, (21.0000, 78.0000))

df_map = df.copy()
df_map[["lat", "lon"]] = df_map["city"].apply(
    lambda c: pd.Series(get_coords(c))
)

fig_map = px.scatter_mapbox(
    df_map,
    lat="lat",
    lon="lon",
    hover_name="city",
    color="severity",
    size="casualties",
    zoom=4,
    height=500,
)

# ---------------- FUTURE RISK FORECAST ----------------
st.subheader("🔮 Disaster Risk Forecast")

forecast = pd.read_csv("risk_forecast.csv")

forecast_chart = forecast.groupby("month")["forecast_risk"].mean()

st.line_chart(forecast_chart)

fig_map.update_layout(mapbox_style="open-street-map")
st.plotly_chart(fig_map, use_container_width=True)

# ---------------- ALERT PANEL ----------------
st.subheader("🚨 Active Alerts")

alerts = df[
    (df["severity"] == "High") |
    (df["casualties"] > 50) |
    (df["response_time_hours"] > 10)
]

st.metric("Total Active Alerts", len(alerts))

st.dataframe(
    alerts[
        ["state", "city", "disaster_type", "casualties",
         "economic_loss_crores", "response_time_hours", "severity"]
    ]
)
# ---------------- MODEL INSIGHTS ----------------
st.subheader("📈 Model Feature Importance")

rf = model.named_steps["model"]
features = model.named_steps["preprocess"].get_feature_names_out()

importance = pd.DataFrame({
    "Feature": features,
    "Importance": rf.feature_importances_
}).sort_values("Importance", ascending=False)

st.bar_chart(importance.head(12).set_index("Feature"))

# ---------------- STATE RISK TABLE ----------------
st.subheader("⚠️ State Risk Prediction")

state_risk = forecast.groupby("state")["forecast_risk"].mean().reset_index()

st.dataframe(
    state_risk.sort_values("forecast_risk", ascending=False)
)
