import os
import pickle
import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt

def analyze_confidence_issues():
    """Analyze why model confidence isn't higher"""
    print("🔍 ANALYZING CONFIDENCE ISSUES")
    print("=" * 50)
    
    # Load the model
    model_path = 'improved_hepatitis_outputs/models/best_improved_model.keras'
    model = load_model(model_path)
    
    # Load preprocessing artifacts
    artifacts_path = 'improved_hepatitis_outputs/models/preprocessing_artifacts.pkl'
    with open(artifacts_path, 'rb') as f:
        artifacts = pickle.load(f)
    
    scaler = artifacts['scaler']
    label_encoder = artifacts['label_encoder_target']
    feature_cols = artifacts['feature_columns']
    
    print(f"📊 Model loaded successfully")
    print(f"📊 Feature columns: {len(feature_cols)}")
    
    # Test with very clear symptom patterns
    test_cases = [
        {
            'name': 'CLEAR Hepatitis A pattern',
            'symptoms': ['fatigue', 'jaundice', 'nausea', 'vomiting', 'abdominal pain', 'fever', 'loss of appetite'],
            'expected': 'Hepatitis A',
            'description': 'Classic acute hepatitis symptoms'
        },
        {
            'name': 'CLEAR Hepatitis C pattern',
            'symptoms': ['fatigue', 'jaundice', 'dark-colored urine', 'weight loss', 'spider angiomas', 'ascites', 'bleeding easily'],
            'expected': 'Hepatitis C',
            'description': 'Classic chronic hepatitis symptoms'
        },
        {
            'name': 'MIXED symptoms (confusing)',
            'symptoms': ['fatigue', 'jaundice', 'nausea', 'weight loss'],
            'expected': 'Unclear',
            'description': 'Mixed acute and chronic symptoms'
        },
        {
            'name': 'MINIMAL symptoms',
            'symptoms': ['fatigue'],
            'expected': 'Unclear',
            'description': 'Too few symptoms to be confident'
        }
    ]
    
    print(f"\n🧪 Testing confidence with different symptom patterns:")
    print("-" * 70)
    
    for i, case in enumerate(test_cases, 1):
        print(f"\n🔍 Test Case {i}: {case['name']}")
        print(f"   Description: {case['description']}")
        print(f"   Symptoms: {case['symptoms']}")
        
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
        
        # Analyze confidence level
        if confidence > 0.9:
            print(f"   ✅ EXCELLENT confidence (>90%)")
        elif confidence > 0.8:
            print(f"   ✅ GOOD confidence (80-90%)")
        elif confidence > 0.7:
            print(f"   ⚠️ MODERATE confidence (70-80%)")
        elif confidence > 0.6:
            print(f"   ⚠️ LOW confidence (60-70%)")
        else:
            print(f"   ❌ VERY LOW confidence (<60%)")
    
    print(f"\n📊 CONFIDENCE ANALYSIS:")
    print(f"=" * 50)
    print(f"🔍 Why confidence might not be 90%+:")
    print(f"   1. **Symptom overlap**: Hepatitis A and C share many symptoms")
    print(f"   2. **Insufficient symptoms**: Need more specific symptom combinations")
    print(f"   3. **Mixed patterns**: Symptoms from both acute and chronic patterns")
    print(f"   4. **Model uncertainty**: Real medical diagnosis is complex")
    print(f"   5. **Feature noise**: Some features might not be strongly predictive")
    
    print(f"\n💡 How to get 90%+ confidence:")
    print(f"   1. **Use more specific symptoms**: Add unique symptoms for each type")
    print(f"   2. **Include symptom combinations**: Use patterns the model learned")
    print(f"   3. **Add severity information**: Severe vs mild cases")
    print(f"   4. **Use more symptoms**: More data = higher confidence")
    print(f"   5. **Retrain with better features**: Improve the model itself")

if __name__ == "__main__":
    analyze_confidence_issues() 