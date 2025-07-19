from flask import Flask, render_template, request
import pandas as pd
import numpy as np
import os
from sklearn.neighbors import KNeighborsClassifier

app = Flask(__name__)

# Load and clean CSV
csv_path = os.path.join(os.path.dirname(__file__), 'AirQuality.csv')
df = pd.read_csv(csv_path, delimiter=';')

# Strip column names
df.columns = [col.strip() for col in df.columns]

# Rename for simplicity (optional)
df.rename(columns={
    'CO(GT)': 'CO',
    'NO2(GT)': 'NO2',
    'RH': 'Humidity'
}, inplace=True)

# Check for required columns
required_columns = ['CO', 'NO2', 'Humidity']
missing_cols = [col for col in required_columns if col not in df.columns]
if missing_cols:
    raise ValueError(f"Missing required columns in CSV: {missing_cols}")

# Drop rows with missing data
df = df[required_columns].dropna()

# Generate Label column (basic logic: Poor, Moderate, Good based on CO & NO2)
def assign_label(row):
    if row['CO'] > 5 or row['NO2'] > 100:
        return 'Poor'
    elif row['CO'] > 2 or row['NO2'] > 60:
        return 'Moderate'
    else:
        return 'Good'

df['Label'] = df.apply(assign_label, axis=1)

# Prepare features and labels
X = df[['CO', 'NO2', 'Humidity']]
y = df['Label']
y_encoded = pd.get_dummies(y)

# Normalize features
means = X.mean()
stds = X.std().replace(0, 1)
X_normalized = (X - means) / stds

# Train model
model = KNeighborsClassifier(n_neighbors=3)
model.fit(X_normalized, y_encoded)

@app.route('/', methods=['GET', 'POST'])
def index():
    prediction = None
    if request.method == 'POST':
        try:
            co = float(request.form['co'])
            no2 = float(request.form['no2'])
            humidity = float(request.form['humidity'])

            input_data = pd.DataFrame([[co, no2, humidity]], columns=['CO', 'NO2', 'Humidity'])
            input_normalized = (input_data - means) / stds

            prediction_encoded = model.predict(input_normalized)[0]
            prediction = y_encoded.columns[np.argmax(prediction_encoded)]

        except ValueError:
            prediction = "Invalid input. Please enter numeric values."

    return render_template('index.html', prediction=prediction)

if __name__ == '__main__':
    app.run(debug=True)
