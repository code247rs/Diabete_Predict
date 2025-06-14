

import pandas as pd
from sklearn.preprocessing import StandardScaler
from joblib import dump

# Load the dataset
df = pd.read_csv('/Users/alndsrood/Documents/UKH/Special Topics/projects/Final Project/Dataset of Diabetes .csv')

# Drop rows with too many missing values
df = df.dropna(thresh=10)

# Fill remaining missing values with column means (numeric only)
df.fillna(df.mean(numeric_only=True), inplace=True)

# Standardize numeric columns
cols_to_scale = ['AGE', 'Urea', 'Cr', 'HbA1c', 'Chol', 'TG', 'HDL', 'LDL', 'VLDL', 'BMI']
scaler = StandardScaler()
df[cols_to_scale] = scaler.fit_transform(df[cols_to_scale])

dump(scaler, 'scaler.joblib')


# Encode categorical columns

df['Gender'] = df['Gender'].apply(lambda x: 0 if x == 'M' else 1)
df['CLASS'] = df['CLASS'].map({'N': 0, 'P': 1, 'Y': 2})

df.dropna(subset=['CLASS'], inplace=True)
print("Number of nulls in CLASS:", df['CLASS'].isnull().sum())
print(df.tail(), flush=True)

# Drop unwanted columns
clean_data = df.drop(columns=['ID', 'No_Pation'])

# Save cleaned dataset
clean_data.to_csv("cleaned_data.csv", index=False)

print("✅ Preprocessing complete. Cleaned data saved to cleaned_data.csv.")
