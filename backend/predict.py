import pandas as pd
import numpy as np
import json
import joblib
import sys
import os


def predict_with_saved_model(
        data,
        model_path='./output/hepatitis_model.pkl',
        feature_columns_path='./output/feature_columns.json'):
    """
    Make predictions using a saved model
    
    Args:
        data: DataFrame with the same structure as training data
        model_path: Path to the saved model file
        feature_columns_path: Path to the saved feature columns JSON
        
    Returns:
        Predictions
    """
    # Load the model
    model = joblib.load(model_path)

    # Load feature columns
    with open(feature_columns_path, 'r') as f:
        feature_columns = json.load(f)

    # Preprocess the data the same way as during training
    # Extract features from Symptoms text
    if 'Symptoms' in data.columns:
        data['SymptomLength'] = data['Symptoms'].apply(lambda x: len(str(x)))

        # Count specific symptoms
        common_symptoms = [
            'jaundice', 'fatigue', 'nausea', 'vomiting', 'fever', 'pain'
        ]
        for symptom in common_symptoms:
            data[f'has_{symptom}'] = data['Symptoms'].str.lower().str.contains(
                symptom, regex=False).astype(int)

    # Process DiagnosisDate
    if 'DiagnosisDate' in data.columns:
        data['DiagnosisDate'] = pd.to_datetime(data['DiagnosisDate'],
                                               errors='coerce')
        data['DiagnosisYear'] = data['DiagnosisDate'].dt.year
        data['DiagnosisMonth'] = data['DiagnosisDate'].dt.month

    # Select only the features used during training
    X = data[feature_columns['features']]

    # Make predictions
    predictions = model.predict(X)
    probabilities = model.predict_proba(X)

    # Get class names
    class_names = model.named_steps['classifier'].classes_

    # Create a DataFrame with predictions and probabilities
    results = pd.DataFrame({'predicted_class': predictions})

    # Add probability columns for each class
    for i, class_name in enumerate(class_names):
        results[f'probability_{class_name}'] = probabilities[:, i]

    return results


if __name__ == "__main__":
    # Check if input file path is provided
    if len(sys.argv) < 2:
        print("Usage: python predict.py <input_file_path> [output_file_path]")
        sys.exit(1)

    input_file = sys.argv[1]

    # Set default output file path if not provided
    output_file = sys.argv[2] if len(sys.argv) > 2 else 'predictions.csv'

    # Check if model exists
    model_path = './output/hepatitis_model.pkl'
    feature_columns_path = './output/feature_columns.json'

    if not os.path.exists(model_path) or not os.path.exists(
            feature_columns_path):
        print(
            json.dumps(
                {"error": "Model not found. Please train the model first."}))
        sys.exit(1)

    try:
        # Load the data
        data = pd.read_csv(input_file)

        # Make predictions
        results = predict_with_saved_model(data, model_path,
                                           feature_columns_path)

        # Save predictions to CSV
        results.to_csv(output_file, index=False)

        # Print results as JSON for the Node.js server to parse
        print(
            json.dumps({
                "success":
                True,
                "message":
                f"Predictions saved to {output_file}",
                "predictions":
                results.head(10).to_dict(orient='records'),
                "total_predictions":
                len(results)
            }))

    except Exception as e:
        print(json.dumps({"error": str(e)}))
        sys.exit(1)
