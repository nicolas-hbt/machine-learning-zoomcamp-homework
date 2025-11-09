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


@app.route('/health', methods=['GET'])
def health():
    """Simple health check endpoint."""
    return "OK", 200


@app.route('/predict', methods=['POST'])
def predict():
    """Accepts JSON input and returns model predictions."""
    try:
        # Get JSON data from the request
        json_data = request.get_json()

        # Convert JSON list of records to a DataFrame
        input_df = pd.DataFrame(json_data)

        # Preprocess the data using the loaded processor
        X_processed = processor.transform(input_df, scaled=False)

        # --- THIS IS THE FIX ---
        # Drop the 'day' column, as the model was not trained on it
        if 'day' in X_processed.columns:
            X_processed = X_processed.drop(columns=['day'])
        # ----------------------

        # Make predictions (log-transformed)
        log_predictions = model.predict(X_processed)

        # Inverse transform (from log space)
        predictions = np.expm1(log_predictions)
        predictions[predictions < 0] = 0

        # Return predictions as JSON
        return jsonify({'predictions': predictions.tolist()})

    except Exception as e:
        return jsonify({'error': str(e)}), 400


if __name__ == '__main__':
    # Run on port 9696 and make it accessible outside the container (host='0.0.0.0')
    app.run(debug=False, host='0.0.0.0', port=9696)