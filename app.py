from flask import Flask, render_template, request
import pandas as pd
import numpy as np
import os

app = Flask(__name__)

df = pd.read_csv(r'C:\Users\91939\OneDrive\Desktop\Air Quality\AirQuality.csv')

X = df[['CO', 'NO2', 'Humidity']]
y = df['Label']

y_encoded = pd.get_dummies(y)

means = X.mean()
stds = X.std().replace(0, 1)  

X_normalized = (X - means) / stds

from sklearn.neighbors import KNeighborsClassifier
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
