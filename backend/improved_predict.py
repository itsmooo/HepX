import pandas as pd
import numpy as np
import json
import pickle
import sys
import os
import tensorflow as tf
from tensorflow.keras.models import load_model
from sklearn.preprocessing import StandardScaler, LabelEncoder

def create_prediction_data(user_data):
    """
    Create a DataFrame from user input data for prediction with improved model format
    
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
    severity = 'Mild'  # Default mild
    if symptom_count >= 5 or age > 60:
        severity = 'Severe'
    elif symptom_count >= 3 or age > 40:
        severity = 'Moderate'
    
    # Create input data matching the improved model's expected format
    input_data = {
        'Symptoms': symptoms_text,
        'SymptomCount': symptom_count,
        'Severity': severity,
        'DiagnosisDate': '2023-10-15',  # Default date
        'Treatment': 'Supportive care'  # Default treatment
    }
    
    return input_data

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
        symptom_list.append('dark-colored urine')
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

def preprocess_input_for_improved_model(input_data, preprocessing_artifacts_path):
    """
    Preprocess input data for the improved model
    
    Args:
        input_data: Dictionary with input data
        preprocessing_artifacts_path: Path to preprocessing artifacts
        
    Returns:
        Preprocessed input ready for prediction
    """
    # Load preprocessing artifacts
    with open(preprocessing_artifacts_path, 'rb') as f:
        artifacts = pickle.load(f)
    
    scaler = artifacts['scaler']
    label_encoder_target = artifacts['label_encoder_target']
    label_encoders_other = artifacts['label_encoders_other']
    feature_columns = artifacts['feature_columns']
    
    # Create DataFrame
    input_df = pd.DataFrame([input_data])
    processed_input = input_df.copy()

    # Feature engineering (same as training)
    symptoms_list = [
        'ascites', 'swelling', 'spider angiomas', 'itchy skin', 'jaundice',
        'dark-colored urine', 'bleeding easily', 'weight loss', 'fatigue',
        'confusion', 'not wanting to eat', 'bruising easily', 'nausea',
        'vomiting', 'fever', 'abdominal pain', 'joint pain', 'loss of appetite'
    ]

    for symptom in symptoms_list:
        processed_input[f'has_{symptom}'] = processed_input['Symptoms'].str.contains(
            symptom, case=False).astype(int)

    # Create symptom combinations
    processed_input['has_jaundice_fatigue'] = (
        processed_input['has_jaundice'] & processed_input['has_fatigue']
    ).astype(int)
    
    processed_input['has_jaundice_dark_urine'] = (
        processed_input['has_jaundice'] & processed_input['has_dark-colored urine']
    ).astype(int)

    # Date features
    processed_input['DiagnosisDate'] = pd.to_datetime(processed_input['DiagnosisDate'])
    processed_input['DiagnosisYear'] = processed_input['DiagnosisDate'].dt.year
    processed_input['DiagnosisMonth'] = processed_input['DiagnosisDate'].dt.month
    processed_input['DiagnosisDay'] = processed_input['DiagnosisDate'].dt.day
    processed_input['DiagnosisDayOfWeek'] = processed_input['DiagnosisDate'].dt.dayofweek

    # Severity encoding
    severity_mapping = {'Mild': 1, 'Moderate': 2, 'Severe': 3}
    processed_input['Severity_encoded'] = processed_input['Severity'].map(severity_mapping).fillna(2)

    # Treatment encoding
    for col in label_encoders_other:
        if col in processed_input.columns:
            try:
                processed_input[col + '_encoded'] = label_encoders_other[col].transform(
                    processed_input[col].astype(str))
            except ValueError:
                processed_input[col + '_encoded'] = -1

    # Select features and scale
    input_features = processed_input.reindex(columns=feature_columns, fill_value=0)
    input_scaled = scaler.transform(input_features)
    
    return input_scaled, label_encoder_target

def predict_with_improved_model(
        data,
        model_path='./improved_hepatitis_outputs/models/best_improved_model.keras',
        preprocessing_artifacts_path='./improved_hepatitis_outputs/models/preprocessing_artifacts.pkl'):
    """
    Make predictions using the improved model
    
    Args:
        data: Dictionary with input data
        model_path: Path to the improved model file
        preprocessing_artifacts_path: Path to preprocessing artifacts
        
    Returns:
        Predictions with probabilities
    """
    # Load the improved model
    model = load_model(model_path)

    # Preprocess input
    input_scaled, label_encoder = preprocess_input_for_improved_model(data, preprocessing_artifacts_path)
    
    # Make prediction
    prediction_proba = model.predict(input_scaled, verbose=0)[0][0]
    prediction = (prediction_proba > 0.5).astype(int)
    
    # Decode prediction
    predicted_type = label_encoder.inverse_transform([prediction])[0]
    confidence = prediction_proba if prediction == 1 else 1 - prediction_proba
    
    # Create results
    results = {
        'predicted_class': predicted_type,
        'confidence': float(confidence),
        'probability_hepatitis_a': float(1 - prediction_proba),
        'probability_hepatitis_c': float(prediction_proba)
    }
    
    return results

if __name__ == "__main__":
    # Check if input file path is provided
    if len(sys.argv) < 2:
        print("Usage: python improved_predict.py <input_file_path>")
        sys.exit(1)

    input_file = sys.argv[1]

    # Check if improved model exists
    model_path = './improved_hepatitis_outputs/models/best_improved_model.keras'
    preprocessing_artifacts_path = './improved_hepatitis_outputs/models/preprocessing_artifacts.pkl'

    if not os.path.exists(model_path) or not os.path.exists(preprocessing_artifacts_path):
        print(json.dumps({"error": "Improved model not found. Please run improved_training.py first."}))
        sys.exit(1)

    try:
        # Load the JSON data from the server
        with open(input_file, 'r') as f:
            user_data = json.load(f)
        
        # Create prediction data
        data = create_prediction_data(user_data)

        # Make predictions with improved model
        results = predict_with_improved_model(data, model_path, preprocessing_artifacts_path)

        # Create response for the server
        prediction_result = {
            "success": True,
            "message": "Improved model prediction completed successfully",
            "predictions": [results],
            "total_predictions": 1
        }

        # Print results as JSON for the Node.js server to parse
        print(json.dumps(prediction_result))

    except Exception as e:
        print(json.dumps({"error": str(e)}))
        sys.exit(1) 