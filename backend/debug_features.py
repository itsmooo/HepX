import os
import pickle
import pandas as pd
from tensorflow.keras.models import load_model

def debug_feature_mismatch():
    """Debug the feature mismatch issue"""
    print("🔍 DEBUGGING FEATURE MISMATCH")
    print("=" * 50)
    
    # Load preprocessing artifacts
    artifacts_path = 'improved_hepatitis_outputs/models/preprocessing_artifacts.pkl'
    with open(artifacts_path, 'rb') as f:
        artifacts = pickle.load(f)
    
    feature_cols = artifacts['feature_columns']
    print(f"📊 Model expects {len(feature_cols)} features:")
    for i, feature in enumerate(feature_cols, 1):
        print(f"   {i:2d}. {feature}")
    
    print("\n" + "=" * 50)
    print("TESTING SYMPTOM PROCESSING")
    print("=" * 50)
    
    # Test with your symptoms
    test_symptoms = ['fatigue', 'poor appetite', 'nausea', 'muscle aches', 'joint pain', 
                    'dark urine', 'jaundice', 'abdominal pain', 'fever', 'itchy skin', 
                    'clay-colored stool', 'weight loss', 'confusion']
    
    print(f"🔍 Testing symptoms: {test_symptoms}")
    
    # Create input data
    input_data = {
        'Symptoms': ', '.join(test_symptoms),
        'SymptomCount': len(test_symptoms),
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
    
    print(f"\n🔧 Creating symptom features...")
    for symptom in symptoms_list:
        has_symptom = processed_input['Symptoms'].str.contains(symptom, case=False).astype(int)
        processed_input[f'has_{symptom}'] = has_symptom
        if has_symptom.iloc[0] == 1:
            print(f"   ✅ {symptom}: {has_symptom.iloc[0]}")
    
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
    
    # Remove original Severity column
    if 'Severity' in processed_input.columns:
        processed_input = processed_input.drop('Severity', axis=1)
    
    # Treatment encoding
    label_encoders_other = artifacts['label_encoders_other']
    for col in label_encoders_other:
        if col in processed_input.columns:
            try:
                processed_input[col + '_encoded'] = label_encoders_other[col].transform(
                    processed_input[col].astype(str))
            except ValueError:
                processed_input[col + '_encoded'] = -1
    
    print(f"\n📋 Available columns after processing:")
    for col in processed_input.columns:
        print(f"   - {col}")
    
    print(f"\n🔍 Checking feature availability:")
    available_features = [col for col in feature_cols if col in processed_input.columns]
    missing_features = [col for col in feature_cols if col not in processed_input.columns]
    
    print(f"   ✅ Available: {len(available_features)}/{len(feature_cols)}")
    print(f"   ❌ Missing: {len(missing_features)}")
    
    if missing_features:
        print(f"   Missing features: {missing_features}")
    
    # Show what we have vs what we need
    print(f"\n📊 Feature comparison:")
    for feature in feature_cols:
        if feature in processed_input.columns:
            value = processed_input[feature].iloc[0]
            print(f"   ✅ {feature}: {value}")
        else:
            print(f"   ❌ {feature}: MISSING")

if __name__ == "__main__":
    debug_feature_mismatch() 