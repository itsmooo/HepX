import os
import pickle
import numpy as np
import pandas as pd
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Deep Learning Libraries
import tensorflow as tf
from tensorflow.keras.models import load_model
from sklearn.preprocessing import StandardScaler, LabelEncoder

class InteractiveHepatitisTester:
    """Interactive testing for the trained hepatitis model"""
    
    def __init__(self, model_dir='improved_hepatitis_outputs'):
        self.model_dir = model_dir
        self.model = None
        self.scaler = None
        self.label_encoder = None
        self.feature_cols = []
        self.label_encoders_other = {}
        
        print("🧪 INTERACTIVE HEPATITIS MODEL TESTER")
        print("=" * 50)
        
    def load_model_and_artifacts(self):
        """Load the trained model and preprocessing artifacts"""
        try:
            # Load model
            model_path = f'{self.model_dir}/models/best_improved_model.keras'
            if not os.path.exists(model_path):
                model_path = f'{self.model_dir}/models/improved_hepatitis_model.keras'
            
            if os.path.exists(model_path):
                self.model = load_model(model_path)
                print(f"✅ Model loaded from: {model_path}")
            else:
                print(f"❌ Model not found in: {self.model_dir}/models/")
                return False
            
            # Load preprocessing artifacts
            artifacts_path = f'{self.model_dir}/models/preprocessing_artifacts.pkl'
            if os.path.exists(artifacts_path):
                with open(artifacts_path, 'rb') as f:
                    artifacts = pickle.load(f)
                
                self.scaler = artifacts['scaler']
                self.label_encoder = artifacts['label_encoder_target']
                self.feature_cols = artifacts['feature_columns']
                self.label_encoders_other = artifacts['label_encoders_other']
                
                print(f"✅ Preprocessing artifacts loaded")
                print(f"📊 Feature columns: {len(self.feature_cols)}")
                return True
            else:
                print(f"❌ Preprocessing artifacts not found: {artifacts_path}")
                return False
                
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            return False
    
    def preprocess_input(self, symptoms, severity='Moderate', diagnosis_date='2023-10-15', treatment='Supportive care'):
        """Preprocess input symptoms for prediction"""
        # Create input data
        input_data = {
            'Symptoms': ', '.join(symptoms),
            'SymptomCount': len(symptoms),
            'Severity': severity,
            'DiagnosisDate': diagnosis_date,
            'Treatment': treatment
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
        
        # Severity encoding - keep both original and encoded
        severity_mapping = {'Mild': 1, 'Moderate': 2, 'Severe': 3}
        processed_input['Severity_encoded'] = processed_input['Severity'].map(severity_mapping).fillna(2)
        # Keep the original Severity column as the model expects it
        processed_input['Severity'] = processed_input['Severity'].map(severity_mapping).fillna(2)
        
        # Treatment encoding
        for col in self.label_encoders_other:
            if col in processed_input.columns:
                try:
                    processed_input[col + '_encoded'] = self.label_encoders_other[col].transform(
                        processed_input[col].astype(str))
                except ValueError:
                    processed_input[col + '_encoded'] = -1
        
        # Add missing PatientID_encoded feature (default to 0 for new patients)
        if 'PatientID_encoded' not in processed_input.columns:
            processed_input['PatientID_encoded'] = 0
        
        # Select features and scale - ensure we only use the feature columns
        available_features = [col for col in self.feature_cols if col in processed_input.columns]
        missing_features = [col for col in self.feature_cols if col not in processed_input.columns]
        
        if missing_features:
            print(f"⚠️ Missing features: {missing_features}")
            for feature in missing_features:
                processed_input[feature] = 0
        
        input_features = processed_input[self.feature_cols]
        input_scaled = self.scaler.transform(input_features)
        
        return input_scaled
    
    def predict(self, symptoms, severity='Moderate'):
        """Make prediction for given symptoms"""
        if self.model is None:
            return "Error: Model not loaded", 0.0
        
        try:
            # Preprocess input
            input_scaled = self.preprocess_input(symptoms, severity)
            
            # Make prediction
            prediction_proba = self.model.predict(input_scaled, verbose=0)[0][0]
            prediction = (prediction_proba > 0.5).astype(int)
            
            # Decode prediction
            predicted_type = self.label_encoder.inverse_transform([prediction])[0]
            confidence = prediction_proba if prediction == 1 else 1 - prediction_proba
            
            return predicted_type, confidence
            
        except Exception as e:
            return f"Error: {e}", 0.0
    
    def interactive_testing(self):
        """Start interactive testing mode"""
        if not self.load_model_and_artifacts():
            print("❌ Cannot start testing - model not loaded")
            return
        
        print("\n🧪 INTERACTIVE TESTING MODE")
        print("Enter symptoms and see predictions!")
        print("Type 'quit' to exit")
        print("Type 'help' for symptom suggestions")
        print("-" * 50)
        
        while True:
            print("\n📝 Enter symptoms (comma-separated):")
            user_input = input("Symptoms: ").strip()
            
            if user_input.lower() == 'quit':
                print("👋 Goodbye!")
                break
            
            if user_input.lower() == 'help':
                print("\n💡 Common symptoms you can try:")
                print("   - fatigue, jaundice, nausea")
                print("   - fatigue, jaundice, dark-colored urine, weight loss")
                print("   - fatigue, abdominal pain, vomiting")
                print("   - jaundice, fever, loss of appetite")
                print("   - fatigue, nausea, vomiting, fever")
                continue
            
            if not user_input:
                print("❌ Please enter symptoms")
                continue
            
            # Parse symptoms
            symptoms = [s.strip() for s in user_input.split(',')]
            
            print(f"\n🔍 Analyzing symptoms: {symptoms}")
            
            # Get prediction
            prediction, confidence = self.predict(symptoms)
            
            print(f"🎯 Prediction: {prediction}")
            print(f"📊 Confidence: {confidence:.1%}")
            print(f"📋 Symptom count: {len(symptoms)}")
            
            # Show interpretation
            if confidence > 0.8:
                print("✅ High confidence prediction")
            elif confidence > 0.6:
                print("⚠️ Moderate confidence prediction")
            else:
                print("❓ Low confidence prediction - consider more symptoms")

def main():
    print("🚀 Starting Interactive Hepatitis Model Tester...")
    
    tester = InteractiveHepatitisTester()
    tester.interactive_testing()

if __name__ == "__main__":
    main() 