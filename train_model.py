import os
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import joblib

# 1) Load the full dataset
df = pd.read_csv("incidents_full.csv")

X = df[[
    "state",
    "city",
    "disaster_type",
    "month",
    "year",
    "casualties",
    "economic_loss_crores",
    "response_time_hours",
]]
y = df["severity"]

# 2) Define features
cat_features = ["state", "city", "disaster_type"]
num_features = ["month", "year", "casualties",
                "economic_loss_crores", "response_time_hours"]

preprocessor = ColumnTransformer(
    transformers=[
        ("cat", OneHotEncoder(handle_unknown="ignore"), cat_features),
        ("num", "passthrough", num_features),
    ]
)

model = RandomForestClassifier(
    n_estimators=200,
    random_state=42,
    class_weight="balanced"
)

clf = Pipeline(
    steps=[
        ("preprocess", preprocessor),
        ("model", model),
    ]
)

# 3) Train/test split for evaluation
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# 4) Train the model
clf.fit(X_train, y_train)

# 5) Evaluate
y_pred = clf.predict(X_test)
print("Classification report:")
print(classification_report(y_test, y_pred))
print("Confusion matrix:")
print(confusion_matrix(y_test, y_pred))

# 6) Save model
os.makedirs("models", exist_ok=True)
joblib.dump(clf, "models/severity_model.pkl")
print("\n✅ Model saved to models/severity_model.pkl")
