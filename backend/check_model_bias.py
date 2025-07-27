import os
import pickle
import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt

def check_model_bias():
    """Check if the model is biased towards Hepatitis A"""
    print("🔍 CHECKING MODEL BIAS")
    print("=" * 50)
    
    # Load the model
    model_path = 'improved_hepatitis_outputs/models/best_improved_model.keras'
    model = load_model(model_path)
    print(f"✅ Model loaded from: {model_path}")
    
    # Load preprocessing artifacts
    artifacts_path = 'improved_hepatitis_outputs/models/preprocessing_artifacts.pkl'
    with open(artifacts_path, 'rb') as f:
        artifacts = pickle.load(f)
    
    scaler = artifacts['scaler']
    label_encoder = artifacts['label_encoder_target']
    feature_cols = artifacts['feature_columns']
    
    print(f"📊 Label encoder classes: {label_encoder.classes_}")
    print(f"📊 Feature columns: {len(feature_cols)}")
    
    # Test with different symptom combinations
    test_cases = [
        {
            'name': 'Hepatitis A symptoms',
            'symptoms': ['fatigue', 'jaundice', 'nausea', 'abdominal pain', 'fever'],
            'expected': 'Hepatitis A'
        },
        {
            'name': 'Hepatitis C symptoms',
            'symptoms': ['fatigue', 'jaundice', 'dark-colored urine', 'weight loss', 'spider angiomas'],
            'expected': 'Hepatitis C'
        },
        {
            'name': 'Severe symptoms',
            'symptoms': ['fatigue', 'jaundice', 'dark-colored urine', 'abdominal pain', 'weight loss', 'nausea', 'vomiting', 'fever'],
            'expected': 'Hepatitis C'
        },
        {
            'name': 'Mild symptoms',
            'symptoms': ['fatigue', 'jaundice'],
            'expected': 'Hepatitis A'
        }
    ]
    
    print(f"\n🧪 Testing {len(test_cases)} different symptom combinations:")
    print("-" * 60)
    
    all_predictions = []
    all_confidences = []
    
    for i, case in enumerate(test_cases, 1):
        print(f"\n🔍 Test Case {i}: {case['name']}")
        print(f"   Symptoms: {case['symptoms']}")
        print(f"   Expected: {case['expected']}")
        
        # Create input data
        input_data = {
            'Symptoms': ', '.join(case['symptoms']),
            'SymptomCount': len(case['symptoms']),
            'Severity': 'Moderate',
            'DiagnosisDate': '2023-10-15',
            'Treatment': 'Supportive care'
        }
        
        input_df = pd.DataFrame([input_data])
        processed_input = input_df.copy()
        
        # Feature engineering
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
        processed_input['Severity'] = processed_input['Severity'].map(severity_mapping).fillna(2)
        
        # Treatment encoding
        label_encoders_other = artifacts['label_encoders_other']
        for col in label_encoders_other:
            if col in processed_input.columns:
                try:
                    processed_input[col + '_encoded'] = label_encoders_other[col].transform(
                        processed_input[col].astype(str))
                except ValueError:
                    processed_input[col + '_encoded'] = -1
        
        # Add missing PatientID_encoded feature
        if 'PatientID_encoded' not in processed_input.columns:
            processed_input['PatientID_encoded'] = 0
        
        # Select features and scale
        input_features = processed_input.reindex(columns=feature_cols, fill_value=0)
        input_scaled = scaler.transform(input_features)
        
        # Make prediction
        prediction_proba = model.predict(input_scaled, verbose=0)[0][0]
        prediction = (prediction_proba > 0.5).astype(int)
        predicted_type = label_encoder.inverse_transform([prediction])[0]
        confidence = prediction_proba if prediction == 1 else 1 - prediction_proba
        
        print(f"   Predicted: {predicted_type}")
        print(f"   Confidence: {confidence:.1%}")
        print(f"   Raw probability: {prediction_proba:.3f}")
        
        all_predictions.append(predicted_type)
        all_confidences.append(confidence)
        
        if predicted_type == case['expected']:
            print(f"   ✅ CORRECT")
        else:
            print(f"   ❌ INCORRECT")
    
    # Analyze results
    print(f"\n📊 ANALYSIS RESULTS:")
    print(f"=" * 50)
    
    unique_predictions = set(all_predictions)
    print(f"Unique predictions: {unique_predictions}")
    
    if len(unique_predictions) == 1:
        print(f"⚠️ MODEL BIAS DETECTED: Always predicts {list(unique_predictions)[0]}")
    else:
        print(f"✅ Model shows variety in predictions")
    
    avg_confidence = np.mean(all_confidences)
    print(f"Average confidence: {avg_confidence:.1%}")
    
    # Check training data balance
    print(f"\n📋 CHECKING TRAINING DATA:")
    print(f"=" * 50)
    
    try:
        # Load original dataset to check balance
        df = pd.read_csv('data/hepatitis_dataset_R.csv')
        target_dist = df['HepatitisType'].value_counts()
        print(f"Training data distribution:")
        for hep_type, count in target_dist.items():
            percentage = count / len(df) * 100
            print(f"   {hep_type}: {count} ({percentage:.1f}%)")
        
        if len(target_dist) == 2:
            balance_ratio = min(target_dist) / max(target_dist)
            print(f"Balance ratio: {balance_ratio:.2f}")
            if balance_ratio < 0.3:
                print(f"⚠️ SEVERE IMBALANCE: One class dominates the data")
            elif balance_ratio < 0.7:
                print(f"⚠️ MODERATE IMBALANCE: Data is not well balanced")
            else:
                print(f"✅ Good balance in training data")
        
    except Exception as e:
        print(f"❌ Could not load training data: {e}")

if __name__ == "__main__":
    check_model_bias() 