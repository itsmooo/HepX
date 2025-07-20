import pandas as pd
import numpy as np
import json
import joblib
import sys
import os


def create_prediction_data(user_data):
    """
    Create a DataFrame from user input data for prediction
    
    Args:
        user_data: Dictionary containing user input (age, gender, symptoms, riskFactors)
        
    Returns:
        DataFrame ready for prediction
    """
    # Count symptoms
    symptoms = user_data['symptoms']
    symptom_count = sum([
        symptoms.get('jaundice', False),
        symptoms.get('dark_urine', False),
        symptoms.get('pain', False),
        symptoms.get('fatigue', 0) > 0,
        symptoms.get('nausea', False),
        symptoms.get('vomiting', False),
        symptoms.get('fever', False),
        symptoms.get('loss_of_appetite', False),
        symptoms.get('joint_pain', False)
    ])
    
    # Create symptoms text
    symptoms_text = create_symptoms_text(symptoms)
    
    # Determine severity based on symptoms and age
    age = int(user_data['age'])
    severity = 1  # Default mild
    if symptom_count >= 5 or age > 60:
        severity = 3  # Severe
    elif symptom_count >= 3 or age > 40:
        severity = 2  # Moderate
    
    # Create a single row DataFrame with expected features
    data = pd.DataFrame([{
        'SymptomCount': symptom_count,
        'Severity': severity,
        'SymptomLength': len(symptoms_text),
        'has_jaundice': 1 if symptoms.get('jaundice') else 0,
        'has_fatigue': 1 if symptoms.get('fatigue', 0) > 0 else 0,
        'has_nausea': 1 if symptoms.get('nausea') else 0,
        'has_vomiting': 1 if symptoms.get('vomiting') else 0,
        'has_fever': 1 if symptoms.get('fever') else 0,
        'has_pain': 1 if symptoms.get('pain') else 0,
        'DiagnosisYear': pd.Timestamp.now().year,
        'DiagnosisMonth': pd.Timestamp.now().month
    }])
    
    return data


def create_symptoms_text(symptoms):
    """
    Create a symptoms text string from the symptoms dictionary
    
    Args:
        symptoms: Dictionary of symptoms
        
    Returns:
        String containing all symptoms
    """
    symptom_list = []
    
    if symptoms.get('jaundice'):
        symptom_list.append('jaundice')
    if symptoms.get('dark_urine'):
        symptom_list.append('dark urine')
    if symptoms.get('pain'):
        symptom_list.append('abdominal pain')
    if symptoms.get('fatigue', 0) > 0:
        symptom_list.append('fatigue')
    if symptoms.get('nausea'):
        symptom_list.append('nausea')
    if symptoms.get('vomiting'):
        symptom_list.append('vomiting')
    if symptoms.get('fever'):
        symptom_list.append('fever')
    if symptoms.get('loss_of_appetite'):
        symptom_list.append('loss of appetite')
    if symptoms.get('joint_pain'):
        symptom_list.append('joint pain')
    
    return ', '.join(symptom_list) if symptom_list else 'no symptoms'


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

    # Fill missing columns with 0 (in case any features are missing)
    for feature in feature_columns['features']:
        if feature not in data.columns:
            data[feature] = 0

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
        print("Usage: python predict.py <input_file_path>")
        sys.exit(1)

    input_file = sys.argv[1]

    # Check if model exists
    model_path = './output/hepatitis_model.pkl'
    feature_columns_path = './output/feature_columns.json'

    if not os.path.exists(model_path) or not os.path.exists(feature_columns_path):
        print(json.dumps({"error": "Model not found. Please train the model first."}))
        sys.exit(1)

    try:
        # Load the JSON data from the server
        with open(input_file, 'r') as f:
            user_data = json.load(f)
        
        # Create prediction data
        data = create_prediction_data(user_data)

        # Make predictions
        results = predict_with_saved_model(data, model_path, feature_columns_path)

        # Create response for the server
        prediction_result = {
            "success": True,
            "message": "Prediction completed successfully",
            "predictions": results.to_dict(orient='records'),
            "total_predictions": len(results)
        }

        # Print results as JSON for the Node.js server to parse
        print(json.dumps(prediction_result))

    except Exception as e:
        print(json.dumps({"error": str(e)}))
        sys.exit(1)
