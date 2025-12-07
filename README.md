# AI-based Smart Disaster Management System

This project is an end-to-end data science application that predicts the severity of disaster incidents and provides an interactive analytics dashboard using Streamlit.

## Project Overview

Emergency agencies require fast assessment tools to understand how severe an incident might be based on factors such as location, casualties, economic loss, and response time.

This system:
- Cleans and processes disaster incident data
- Trains a machine learning classifier to predict severity levels (Low / Medium / High)
- Deploys a Streamlit dashboard for analytics and real-time predictions

## Dataset

The dataset contains incident level information including:

- State  
- City  
- Disaster Type  
- Month / Year  
- Casualties  
- Economic Loss (crores)  
- Response Time (hours)  
- Severity (label)

Additional engineered features:
- Season  
- Risk score

## Tech Stack

- Python 3
- Pandas, NumPy
- scikit-learn (Random Forest pipeline)
- Streamlit
- Plotly
- joblib

## Project Structure

- `final_dataset.csv` – cleaned ML-ready dataset  
- `prepare_final_dataset.py` – data cleaning & feature engineering  
- `train_final_model.py` – ML training and model saving  
- `models/severity_model.pkl` – trained ML model  
- `app.py` – Streamlit dashboard and prediction UI  
- `requirements.txt` – Python dependencies  

## How to Run

```bash
git clone <REPO_URL>
cd disaster_mgmt

python -m venv venv
source venv/bin/activate

pip install -r requirements.txt

python prepare_final_dataset.py
python train_final_model.py

streamlit run app.py
