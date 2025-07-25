import pandas as pd

# Load the dataset
df = pd.read_csv('sensor_data.csv')

# Display the first 5 rows
print(df.head())

# Basic info
print(df.info())

# Check for missing values
print(df.isnull().sum())

# Summary statistics
print(df.describe())
