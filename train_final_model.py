import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, accuracy_score
from sklearn.ensemble import RandomForestClassifier
import joblib

DATA_PATH = "final_dataset.csv"
MODEL_PATH = "models/severity_model.pkl"

# 1) Load data
df = pd.read_csv(DATA_PATH)
print("Rows in final_dataset:", len(df))

# 2) Features & target
cat_features = ["state", "city", "disaster_type", "season"]
num_features = [
    "month", "year",
    "casualties",
    "economic_loss_crores",
    "response_time_hours"
]

X = df[cat_features + num_features]
y = df["severity"]

# 3) Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# 4) Preprocessing + model pipeline
preprocessor = ColumnTransformer(
    transformers=[
        ("cat", OneHotEncoder(handle_unknown="ignore"), cat_features),
        ("num", "passthrough", num_features),
    ]
)

model = RandomForestClassifier(
    n_estimators=300,
    random_state=42,
    class_weight="balanced"
)

clf = Pipeline(
    steps=[
        ("preprocess", preprocessor),
        ("model", model),
    ]
)

# 5) Train model
clf.fit(X_train, y_train)

# 6) Evaluation
y_pred = clf.predict(X_test)
print("\n🎯 Accuracy:", accuracy_score(y_test, y_pred))

print("\n📊 Classification Report:")
print(classification_report(y_test, y_pred))

# 7) Save model
os.makedirs("models", exist_ok=True)
joblib.dump(clf, MODEL_PATH)

print(f"\n✅ Model training complete")
print(f"✅ Saved model to: {MODEL_PATH}")
