# train_model.py

import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score

# Load dataset
df = pd.read_csv("eda_cleaned_sensor_data.csv")
df.columns = df.columns.str.strip()  # Clean any extra whitespace

# Drop timestamp column if any
for col in df.columns:
    if "time" in col.lower() or "date" in col.lower():
        df.drop(col, axis=1, inplace=True)

# Encode categorical columns
for col in df.columns:
    if df[col].dtype == "object":
        df[col] = df[col].astype("category").cat.codes  # Label encoding

# Split into features and target
X = df.drop("failure", axis=1)
y = df["failure"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# Save model
joblib.dump(model, "predictive_maintenance_model.pkl")
print("✅ Model saved as predictive_maintenance_model.pkl")
