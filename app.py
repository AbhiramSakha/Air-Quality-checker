from flask import Flask, render_template, request
import pandas as pd
import numpy as np
import os
from sklearn.neighbors import KNeighborsClassifier

app = Flask(__name__)

# Load dataset
csv_path = os.path.join(os.path.dirname(__file__), 'AirQuality.csv')
df = pd.read_csv(csv_path)

# Clean column names (remove leading/trailing spaces)
df.columns = [col.strip() for col in df.columns]

# Debug print (for logs in Railway or other hosting)
print("Available CSV Columns:", df.columns.tolist())

# Check required columns
required_columns = ['CO', 'NO2', 'Humidity', 'Label']
missing_cols = [col for col in required_columns if col not in df.columns]

if missing_cols:
    raise ValueError(f"Missing required columns in CSV: {missing_cols}")

# Split features and target
X = df[['CO', 'NO2', 'Humidity']]
y = df['Label']

# One-hot encode labels
y_encoded = pd.get_dummies(y)

# Normalize features
means = X.mean()
stds = X.std().replace(0, 1)
X_normalized = (X - means) / stds

# Train KNN model
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
