import pandas as pd

df = pd.read_csv('C:/Users/91939/OneDrive/Desktop/Ai Quality/New_AirQuality_Dataset.csv', sep=';')

df = df[['CO(GT)', 'NO2(GT)', 'RH']]
df.columns = ['CO', 'NO2', 'Humidity']

df['CO'] = pd.to_numeric(df['CO'], errors='coerce')
df['NO2'] = pd.to_numeric(df['NO2'], errors='coerce')
df['Humidity'] = pd.to_numeric(df['Humidity'], errors='coerce')
df = df.dropna()

df['PM2.5'] = (df['CO'] * 10 + df['NO2']) / 2

def categorize(row):
    if row['PM2.5'] < 50 and row['CO'] < 2 and row['NO2'] < 40:
        return 'Green'
    elif row['PM2.5'] < 100:
        return 'Amber'
    else:
        return 'Red'

df['Label'] = df.apply(categorize, axis=1)

df.to_csv('C:/Users/91939/OneDrive/Desktop/Ai Quality/Cleaned_AirQuality.csv', index=False)
