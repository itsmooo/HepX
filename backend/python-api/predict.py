import pandas as pd
import numpy as np
import json
import pickle
import sys
import os
import tensorflow as tf
from tensorflow.keras.models import load_model
from sklearn.preprocessing import StandardScaler, LabelEncoder
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn
from typing import Optional, List
from contextlib import asynccontextmanager

# Global model variables (loaded once at startup)
model = None
artifacts = None

# Pydantic models for request/response
class Symptoms(BaseModel):
    jaundice: bool = False
    dark_urine: bool = False
    pain: int = 0  # Changed from bool to int to handle numeric values from frontend
    fatigue: int = 0
    nausea: bool = False
    vomiting: bool = False
    fever: bool = False
    loss_of_appetite: bool = False
    joint_pain: bool = False

class PredictionRequest(BaseModel):
    age: str
    gender: str
    symptoms: Symptoms
    riskFactors: List[str] = []

class PredictionResponse(BaseModel):
    success: bool
    prediction: dict

def load_model_and_artifacts():
    """Load the model and preprocessing artifacts at startup"""
    global model, artifacts
    
    # Enhanced model paths
    model_path = './enhanced_hepatitis_outputs/models/best_enhanced_model.keras'
    artifacts_path = './enhanced_hepatitis_outputs/models/preprocessing_artifacts.pkl'
    
    # Check if enhanced model exists
    if not os.path.exists(model_path):
        raise Exception(f"Enhanced model not found at: {model_path}. Please run enhanced_training_with_plots.py first.")
    
    if not os.path.exists(artifacts_path):
        raise Exception(f"Preprocessing artifacts not found at: {artifacts_path}. Please run enhanced_training_with_plots.py first.")
    
    # Load model
    model = load_model(model_path)
    
    # Load preprocessing artifacts
    with open(artifacts_path, 'rb') as f:
        artifacts = pickle.load(f)
    
    print(f"Model loaded from: {model_path}")
    print(f"Artifacts loaded from: {artifacts_path}")
    print("Enhanced model and artifacts loaded successfully!")

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan event handler for FastAPI"""
    # Startup
    try:
        load_model_and_artifacts()
        print("✅ Enhanced model and artifacts loaded successfully!")
    except Exception as e:
        print(f"❌ Error loading enhanced model: {e}")
        print("💡 Please run enhanced_training_with_plots.py first to generate the enhanced model files.")
    
    yield
    
    # Shutdown
    print("🛑 Shutting down Python API service...")

# Create FastAPI app with lifespan and CORS middleware
app = FastAPI(
    title="Hepatitis Prediction API", 
    version="1.0.0",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure this for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
async def root():
    """Health check endpoint"""
    return {"status": "Hepatitis Prediction API is running", "version": "1.0.0"}

@app.get("/health")
async def health_check():
    """Detailed health check"""
    global model, artifacts
    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "artifacts_loaded": artifacts is not None
    }

@app.post("/predict", response_model=PredictionResponse)
async def predict_hepatitis(request: PredictionRequest):
    """Main prediction endpoint"""
    global model, artifacts
    
    if model is None or artifacts is None:
        raise HTTPException(status_code=500, detail="Enhanced model not loaded")
    
    try:
        # Convert request to the format expected by enhanced model
        symptoms_text = create_symptoms_text(request.symptoms)
        symptom_count = count_symptoms(request.symptoms)
        severity = determine_severity(request.age, symptom_count)
        
        # Create input data matching enhanced model format
        input_data = {
            'Symptoms': symptoms_text,
            'SymptomCount': symptom_count,
            'Severity': severity,
            'DiagnosisDate': '2023-10-15',  # Default date
            'Treatment': 'Supportive care'   # Default treatment
        }
        
        # Use rule-based prediction instead of biased model
        result = predict_with_rule_based_logic(request)
        
        return {
            "success": True,
            "prediction": result
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

def create_symptoms_text(symptoms):
    """Create symptoms text string from symptoms object"""
    symptom_list = []
    
    if symptoms.jaundice:
        symptom_list.append('jaundice')
    if symptoms.dark_urine:
        symptom_list.append('dark-colored urine')
    if symptoms.pain > 0:
        symptom_list.append('abdominal pain')
    if symptoms.fatigue > 0:
        symptom_list.append('fatigue')
    if symptoms.nausea:
        symptom_list.append('nausea')
    if symptoms.vomiting:
        symptom_list.append('vomiting')
    if symptoms.fever:
        symptom_list.append('fever')
    if symptoms.loss_of_appetite:
        symptom_list.append('not wanting to eat')
    if symptoms.joint_pain:
        symptom_list.append('joint pain')
    
    return ', '.join(symptom_list) if symptom_list else 'no symptoms'

def count_symptoms(symptoms):
    """Count total symptoms"""
    return sum([
        symptoms.jaundice,
        symptoms.dark_urine,
        symptoms.pain > 0,
        symptoms.fatigue > 0,
        symptoms.nausea,
        symptoms.vomiting,
        symptoms.fever,
        symptoms.loss_of_appetite,
        symptoms.joint_pain
    ])

def determine_severity(age_str, symptom_count):
    """Determine severity based on age and symptoms"""
    # Convert age string to numeric value
    if age_str == "under18":
        age = 17
    elif age_str == "18-30":
        age = 24
    elif age_str == "31-45":
        age = 38
    elif age_str == "46-60":
        age = 53
    elif age_str == "over60":
        age = 65
    else:
        age = 30  # Default age
    
    if symptom_count >= 5 or age > 60:
        return 'Severe'
    elif symptom_count >= 3 or age > 40:
        return 'Moderate'
    else:
        return 'Mild'

def predict_with_rule_based_logic(request):
    """Rule-based prediction logic for balanced results"""
    symptoms = request.symptoms
    age = request.age
    risk_factors = request.riskFactors or []
    
    # Calculate symptoms and risk scores
    symptom_score = calculate_symptom_score(symptoms)
    risk_score = calculate_risk_score(age, risk_factors)
    
    # Decision logic
    hepatitis_a_prob = 0.5  # Start with 50/50
    hepatitis_c_prob = 0.5
    
    # Acute presentation strongly suggests Hepatitis A
    acute_symptoms = sum([
        symptoms.jaundice,
        symptoms.fever,
        symptoms.nausea,
        symptoms.vomiting
    ])
    
    if acute_symptoms >= 3:
        hepatitis_a_prob += 0.3
        hepatitis_c_prob -= 0.3
    elif acute_symptoms >= 2:
        hepatitis_a_prob += 0.2
        hepatitis_c_prob -= 0.2
    
    # Chronic, subtle symptoms suggest Hepatitis C
    chronic_symptoms = sum([
        symptoms.fatigue > 2,
        symptoms.pain > 2,
        symptoms.joint_pain
    ])
    
    if chronic_symptoms >= 2 and acute_symptoms <= 1:
        hepatitis_c_prob += 0.25
        hepatitis_a_prob -= 0.25
    
    # Risk factors adjustment
    if "recent_travel" in risk_factors:
        hepatitis_a_prob += 0.2
        hepatitis_c_prob -= 0.2
    
    if any(rf in risk_factors for rf in ["blood_transfusion_history", "unsafe_injection_history"]):
        hepatitis_c_prob += 0.3
        hepatitis_a_prob -= 0.3
    
    # Age adjustment
    if age in ["under18", "18-30"]:
        hepatitis_a_prob += 0.1
        hepatitis_c_prob -= 0.1
    elif age == "over60":
        hepatitis_c_prob += 0.2
        hepatitis_a_prob -= 0.2
    
    # Ensure probabilities are between 0.05 and 0.95
    hepatitis_a_prob = max(0.05, min(0.95, hepatitis_a_prob))
    hepatitis_c_prob = 1 - hepatitis_a_prob
    
    # Determine predicted class
    predicted_class = "Hepatitis A" if hepatitis_a_prob > hepatitis_c_prob else "Hepatitis C"
    confidence = max(hepatitis_a_prob, hepatitis_c_prob)
    
    # Create symptoms text
    symptoms_analyzed = create_symptoms_text(symptoms)
    severity_assessed = determine_severity_from_rule(age, symptom_score)
    symptom_count = count_symptoms(symptoms)
    
    return {
        'predicted_class': predicted_class,
        'confidence': float(confidence),
        'probability_Hepatitis A': float(hepatitis_a_prob),
        'probability_Hepatitis C': float(hepatitis_c_prob),
        'symptoms_analyzed': symptoms_analyzed,
        'severity_assessed': severity_assessed,
        'symptom_count': symptom_count
    }

def calculate_symptom_score(symptoms):
    """Calculate symptom score for rule-based logic"""
    score = 0
    
    # Acute symptoms (more associated with Hepatitis A)
    if symptoms.jaundice: score += 2
    if symptoms.fever: score += 2
    if symptoms.nausea: score += 1
    if symptoms.vomiting: score += 1
    if symptoms.dark_urine: score += 1
    if symptoms.loss_of_appetite: score += 1
    
    # Chronic symptoms
    if symptoms.fatigue > 0: score += 2 if symptoms.fatigue > 3 else 1
    if symptoms.pain > 0: score += 2 if symptoms.pain > 5 else 1
    if symptoms.joint_pain: score += 1
    
    return score

def calculate_risk_score(age, risk_factors):
    """Calculate risk score for rule-based logic"""
    risk_score = 0
    
    # Age-based risk
    if age in ["under18", "18-30"]: risk_score += 1  # Higher A risk
    if age == "over60": risk_score += 3  # Higher C risk
    if age == "46-60": risk_score += 2  # Moderate C risk
    
    # Risk factors
    if "recent_travel" in risk_factors: risk_score -= 2  # A indicator
    if "blood_transfusion_history" in risk_factors: risk_score += 3  # C indicator
    if "unsafe_injection_history" in risk_factors: risk_score += 3  # C indicator
    if "infected_contact" in risk_factors: risk_score += 1  # Either, but slightly A
    
    return risk_score

def determine_severity_from_rule(age, symptom_score):
    """Determine severity from rule-based assessment"""
    if symptom_score >= 8: return 'Severe'
    if symptom_score >= 4: return 'Moderate'
    return 'Mild'

def predict_with_enhanced_model(input_data):
    """Make prediction using enhanced model format"""
    global model, artifacts
    
    if model is None or artifacts is None:
        raise Exception("Enhanced model not loaded")
    
    # Extract artifacts
    scaler = artifacts['scaler']
    label_encoder_target = artifacts['label_encoder_target']
    label_encoders_other = artifacts['label_encoders_other']
    feature_columns = artifacts['feature_columns']
    symptoms_list = artifacts.get('symptoms_list', [
        'ascites', 'swelling', 'spider angiomas', 'itchy skin', 'jaundice',
        'dark-colored urine', 'bleeding easily', 'weight loss', 'fatigue',
        'confusion', 'not wanting to eat', 'bruising easily'
    ])
    
    # Create DataFrame
    input_df = pd.DataFrame([input_data])
    processed_input = input_df.copy()
    
    # Feature engineering (same as enhanced training)
    for symptom in symptoms_list:
        processed_input[f'has_{symptom}'] = processed_input['Symptoms'].str.contains(
            symptom, case=False).astype(int)
    
    # Date features
    processed_input['DiagnosisDate'] = pd.to_datetime(processed_input['DiagnosisDate'])
    processed_input['DiagnosisYear'] = processed_input['DiagnosisDate'].dt.year
    processed_input['DiagnosisMonth'] = processed_input['DiagnosisDate'].dt.month
    
    # Encode categorical variables
    for col in label_encoders_other:
        if col in processed_input.columns:
            try:
                processed_input[col + '_encoded'] = label_encoders_other[col].transform(
                    processed_input[col].astype(str))
            except ValueError:
                processed_input[col + '_encoded'] = -1  # Handle unseen labels
    
    # Handle Severity column specifically - convert to numeric
    if 'Severity' in processed_input.columns:
        severity_mapping = {'Mild': 1, 'Moderate': 2, 'Severe': 3}
        processed_input['Severity'] = processed_input['Severity'].map(severity_mapping).fillna(1)
    
    # Select features and scale
    input_features = processed_input.reindex(columns=feature_columns, fill_value=0)
    input_scaled = scaler.transform(input_features)
    
    # Debug feature values
    print(f"DEBUG - Input features shape: {input_features.shape}")
    print(f"DEBUG - Non-zero features: {input_features.columns[input_features.iloc[0] != 0].tolist()}")
    print(f"DEBUG - Feature values: {input_features.iloc[0][input_features.iloc[0] != 0].to_dict()}")
    
    # Make prediction
    prediction_proba = model.predict(input_scaled, verbose=0)[0][0]
    prediction = (prediction_proba > 0.5).astype(int)
    predicted_type = label_encoder_target.inverse_transform([prediction])[0]
    
    # Debug output
    print(f"DEBUG - Raw prediction probability: {prediction_proba}")
    print(f"DEBUG - Prediction class (0=A, 1=C): {prediction}")
    print(f"DEBUG - Predicted type: {predicted_type}")
    
    # Calculate confidence
    confidence = max(prediction_proba, 1 - prediction_proba)
    
    # Create results
    results = {
        'predicted_class': predicted_type,
        'confidence': float(confidence),
        'probability_Hepatitis A': float(1 - prediction_proba),  # Class 0
        'probability_Hepatitis C': float(prediction_proba),      # Class 1
        'symptoms_analyzed': input_data['Symptoms'],
        'severity_assessed': input_data['Severity'],
        'symptom_count': input_data['SymptomCount']
    }
    
    return results

# CLI prediction function for testing
def predict_cli(symptoms_text, symptom_count=0, severity='Mild', diagnosis_date='2023-10-15', treatment='Supportive care'):
    """CLI prediction function for testing"""
    input_data = {
        'Symptoms': symptoms_text,
        'SymptomCount': symptom_count,
        'Severity': severity,
        'DiagnosisDate': diagnosis_date,
        'Treatment': treatment
    }
    
    try:
        result = predict_with_enhanced_model(input_data)
        return result
    except Exception as e:
        return {'error': str(e)}

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Hepatitis Prediction API')
    parser.add_argument('--server', action='store_true', help='Run as FastAPI server')
    parser.add_argument('--port', type=int, default=8000, help='Port for server')
    parser.add_argument('--test', action='store_true', help='Test prediction with sample data')
    
    args = parser.parse_args()
    
    if args.test:
        # Test prediction
        print("🧪 Testing prediction with sample data...")
        test_result = predict_cli(
            symptoms_text="fatigue, jaundice, dark-colored urine",
            symptom_count=3,
            severity="Moderate"
        )
        print("Test result:", json.dumps(test_result, indent=2))
    
    elif args.server:
        # Run as server
        print(f"🚀 Starting Hepatitis Prediction API server on port {args.port}")
        uvicorn.run(app, host="0.0.0.0", port=args.port)
    
    else:
        # Default: run as server
        print("🚀 Starting Hepatitis Prediction API server on port 8000")
        uvicorn.run(app, host="0.0.0.0", port=8000)
