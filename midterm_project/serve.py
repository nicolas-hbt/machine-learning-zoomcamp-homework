# serve.py
import pandas as pd
import numpy as np
import pickle
from flask import Flask, request, jsonify

app = Flask(__name__)

# Load artifacts on app startup
print("Loading model and processor for Flask app...")
try:
    with open('processor.pkl', 'rb') as f:
        processor = pickle.load(f)
    with open('model.pkl', 'rb') as f:
        model = pickle.load(f)
except FileNotFoundError:
    print("Error: 'processor.pkl' or 'model.pkl' not found. Run train.py first.")
    exit()

print("Artifacts loaded successfully.")


@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get JSON data from the request
        json_data = request.get_json()

        # Expecting a list of records, like:
        # [
        #   {"datetime": "2011-01-20 00:00:00", "season": 1, ...},
        #   {"datetime": "2011-01-20 01:00:00", "season": 1, ...}
        # ]
        input_df = pd.DataFrame(json_data)

        X_processed = processor.transform(input_df, scaled=False)

        log_predictions = model.predict(X_processed)
        predictions = np.expm1(log_predictions)
        predictions[predictions < 0] = 0

        return jsonify({'predictions': predictions.tolist()})

    except Exception as e:
        return jsonify({'error': str(e)}), 400


if __name__ == '__main__':
    # port=5001 to avoid conflicts with other common ports
    app.run(debug=True, port=5001, host='0.0.0.0')

# To run:
# python serve.py
#
# To test (in another terminal):
# curl -X POST http://127.0.0.1:5001/predict -H "Content-Type: application/json" \
# -d '[{"datetime": "2012-12-20 00:00:00", "season": 4, "holiday": 0, "workingday": 1, "weather": 1, "temp": 10.66, "humidity": 56, "windspeed": 26.0027}]'