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
