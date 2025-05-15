from flask import Flask, request, render_template
import numpy as np
import joblib

app = Flask(__name__)
model = joblib.load('diabetes_model.pkl')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        avg_sugar = float(request.form['avg_sugar'])
        per_capita = float(request.form['per_capita'])
        total_sugar = float(request.form['total_sugar'])
        import_export_ratio = float(request.form['import_export_ratio'])

        input_data = np.array([[avg_sugar, per_capita, total_sugar, import_export_ratio]])
        prediction = model.predict(input_data)[0]

        return render_template('index.html', prediction_text=f"Predicted Diabetes Prevalence: {prediction:.2f}%")
    except Exception as e:
        return render_template('index.html', prediction_text="Error: " + str(e))

if __name__ == '__main__':
    app.run(debug=True)
