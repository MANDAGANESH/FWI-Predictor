import pickle
from flask import Flask, request, render_template
import numpy as np

application = Flask(__name__)
app = application

# Load the model and scaler
ridge_model = pickle.load(open('models/ridge.pkl', 'rb'))
standard_scaler = pickle.load(open('models/scaler.pkl', 'rb'))

@app.route("/")
def index():
    return render_template('index.html')

@app.route('/predictdata', methods=['GET', 'POST'])
def predict_datapoint():
    if request.method == "POST":
        try:
            # Extract form inputs
            data = [
                float(request.form.get('Temperature')),
                float(request.form.get('RH')),
                float(request.form.get('Ws')),
                float(request.form.get('Rain')),
                float(request.form.get('FFMC')),
                float(request.form.get('DMC')),
                float(request.form.get('DC')),
                float(request.form.get('ISI')),
                float(request.form.get('BUI')),
                # float(request.form.get('Classes')),
                # float(request.form.get('Region'))
            ]

            input_array = np.array(data).reshape(1, -1)

            # Scale input
            scaled_input = standard_scaler.transform(input_array)

            # Predict
            prediction = ridge_model.predict(scaled_input)

            # Render result
            return render_template('home.html', result=round(prediction[0], 2))
        except Exception as e:
            return f"Error occurred: {e}"

    else:
        return render_template('home.html', result=None)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
