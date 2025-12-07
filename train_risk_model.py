import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import joblib

DATA_FILE = "risk_processed.csv"

df = pd.read_csv(DATA_FILE)

X = df[[
    "wri",
    "exposure",
    "vulnerability",
    "susceptibility",
    "coping",
    "adaptive",
    "year"
]]

y = df["risk_category"]

encoder = LabelEncoder()
y_encoded = encoder.fit_transform(y)

X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded, test_size=0.25, random_state=42
)

model = RandomForestClassifier(
    n_estimators=200,
    class_weight="balanced",
    random_state=42
)

model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print("\nModel Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

joblib.dump(model, "risk_model.pkl")
joblib.dump(encoder, "label_encoder.pkl")

print("\n✅ Model saved as: risk_model.pkl")
