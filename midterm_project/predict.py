import pickle
import numpy as np
import pandas as pd
from flask import Flask, request, jsonify


app = Flask("bike_rental_prediction")

with open('gb_model.pkl', 'rb') as f:
    model = pickle.load(f)


@app.route('/predict', methods=['POST'])
def predict():
    """
    Accepts JSON input and returns model predictions.
    """
    # Get JSON data from the request
    json_data = request.get_json()

    X = pd.DataFrame(json_data)

    log_predictions = model.predict(X)

    # Inverse transform (from log space)
    predictions = np.expm1(log_predictions)
    predictions[predictions < 0] = 0

    # Return predictions as JSON
    return jsonify({'predictions': predictions.tolist()})


if __name__ == '__main__':
    app.run(debug=False, host='0.0.0.0', port=9696)


