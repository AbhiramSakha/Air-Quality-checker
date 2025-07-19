from flask import Flask, render_template, request
import pandas as pd, numpy as np, os
from sklearn.neighbors import KNeighborsClassifier

app = Flask(__name__)

csv_path = os.path.join(os.path.dirname(__file__), 'Cleaned_AirQuality.csv')
df = pd.read_csv(csv_path)
print("CSV columns:", df.columns.tolist())

# Check columns
for col in ['CO', 'NO2', 'Humidity', 'Label']:
    if col not in df.columns:
        raise ValueError(f"Missing column: {col}")

X = df[['CO', 'NO2', 'Humidity']]
y = pd.get_dummies(df['Label'])
means, stds = X.mean(), X.std().replace(0, 1)
model = KNeighborsClassifier(n_neighbors=3)
model.fit((X - means)/stds, y)

@app.route('/', methods=['GET', 'POST'])
def index():
    prediction = None
    if request.method == 'POST':
        try:
            co, no2, hum = [float(request.form[k]) for k in ('co','no2','humidity')]
            inp = pd.DataFrame([[co, no2, hum]], columns=['CO','NO2','Humidity'])
            pred = model.predict((inp-means)/stds)[0]
            prediction = y.columns[np.argmax(pred)]
        except ValueError:
            prediction = "Invalid input; enter numeric values."
    return render_template('index.html', prediction=prediction)

if __name__=='__main__':
    app.run(debug=True)
