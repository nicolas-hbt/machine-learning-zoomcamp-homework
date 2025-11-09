# predict.py
import pandas as pd
import numpy as np
import pickle
import argparse

print("Loading artifacts...")
try:
    with open('processor.pkl', 'rb') as f:
        processor = pickle.load(f)
    with open('model.pkl', 'rb') as f:
        model = pickle.load(f)
except FileNotFoundError:
    print("Error: 'processor.pkl' or 'model.pkl' not found.")
    print("Please run train.py first to create these files.")
    exit()


def make_predictions(input_file_path):
    """
    Loads data, preprocesses it, and returns predictions.
    """
    input_df = pd.read_csv(input_file_path)

    # Use the *transform* method of our loaded processor
    X_processed = processor.transform(input_df, scaled=False)

    if 'day' in X_processed.columns:
        X_processed = X_processed.drop(columns=['day'])

    # Predict
    log_predictions = model.predict(X_processed)

    # Inverse transform (from log space)
    predictions = np.expm1(log_predictions)

    # Handle any potential negative predictions
    predictions[predictions < 0] = 0

    # Format output
    output_df = pd.DataFrame({
        'datetime': input_df['datetime'],
        'count': predictions
    })

    return output_df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Batch prediction script.")
    parser.add_argument(
        'input_file',
        type=str,
        help="Path to the input CSV file (e.g., 'test.csv')"
    )
    parser.add_argument(
        '--output_file',
        type=str,
        default='predictions.csv',
        help="Path to save the output predictions (default: 'predictions.csv')"
    )
    args = parser.parse_args()

    print(f"Making predictions on {args.input_file}...")
    predictions_df = make_predictions(args.input_file)

    predictions_df.to_csv(args.output_file, index=False)
    print(f"Predictions saved to {args.output_file}")

# To run:
# python predict.py test.csv --output_file submission.csv