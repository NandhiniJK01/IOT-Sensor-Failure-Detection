# preprocess_sensor_data.py

import pandas as pd
from sklearn.preprocessing import LabelEncoder
import joblib

# Load the cleaned data
df = pd.read_csv("eda_cleaned_sensor_data.csv")
df.columns = df.columns.str.strip() 

# Drop any unnecessary columns if needed
# Example: df = df.drop(columns=["Unnamed: 0"])  # Uncomment if this column exists

# Separate features and target
X = df.drop("failure", axis=1)  # Replace 'Failure' with your actual label column name
y = df["failure"]

# Encode the target variable if it's categorical
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

# Save the encoder for use in prediction later
joblib.dump(label_encoder, "label_encoder.pkl")

# Save the processed data
X.to_csv("X_features.csv", index=False)
pd.DataFrame(y_encoded, columns=["failure"]).to_csv("y_labels.csv", index=False)

print("✅ Preprocessing completed and saved to disk.")
