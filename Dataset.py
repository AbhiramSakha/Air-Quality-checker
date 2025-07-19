import pandas as pd
import os

# Modify paths as needed
input_path = os.path.abspath('New_AirQuality_Dataset.csv')
output_path = os.path.abspath('Cleaned_AirQuality.csv')

df = pd.read_csv(input_path, sep=';')
df.columns = df.columns.str.strip()

# Check required columns
for c in ['CO(GT)', 'NO2(GT)', 'RH']:
    if c not in df.columns:
        raise ValueError(f"Missing column: {c}")

df = df[['CO(GT)', 'NO2(GT)', 'RH']].rename(columns={'CO(GT)': 'CO', 'NO2(GT)': 'NO2', 'RH': 'Humidity'})
df[['CO', 'NO2', 'Humidity']] = df[['CO', 'NO2', 'Humidity']].apply(pd.to_numeric, errors='coerce')
df.dropna(inplace=True)

df['PM2.5'] = (df['CO'] * 10 + df['NO2']) / 2

def categorize(row):
    if row['PM2.5'] < 50 and row['CO'] < 2 and row['NO2'] < 40:
        return 'Green'
    elif row['PM2.5'] < 100:
        return 'Amber'
    else:
        return 'Red'

df['Label'] = df.apply(categorize, axis=1)
df.to_csv(output_path, index=False)
print(f"Cleaned data saved to {output_path}")
