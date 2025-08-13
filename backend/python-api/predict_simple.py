import pandas as pd
import numpy as np
import json
import pickle
import sys
import os
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
    pain: int = 0
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
    """Load the simplified model and preprocessing artifacts at startup"""
    global model, artifacts
    
    # Simplified model paths
    model_path = './simple_model/hepatitis_model.pkl'
    artifacts_path = './simple_model/preprocessing_artifacts.pkl'
    
    # Check if simplified model exists
    if not os.path.exists(model_path):
        raise Exception(f"Simplified model not found at: {model_path}. Please run quick_retrain.py first.")
    
    if not os.path.exists(artifacts_path):
        raise Exception(f"Preprocessing artifacts not found at: {artifacts_path}. Please run quick_retrain.py first.")
    
    # Load model
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    
    # Load preprocessing artifacts
    with open(artifacts_path, 'rb') as f:
        artifacts = pickle.load(f)
    
    print(f"Model loaded from: {model_path}")
    print(f"Artifacts loaded from: {artifacts_path}")
    print("Simplified model and artifacts loaded successfully!")

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan event handler for FastAPI"""
    # Startup
    try:
        load_model_and_artifacts()
        print("✅ Simplified model and artifacts loaded successfully!")
    except Exception as e:
        print(f"❌ Error loading simplified model: {e}")
        print("💡 Please run quick_retrain.py first to generate the simplified model files.")
    
    yield
    
    # Shutdown
    print("🛑 Shutting down Python API service...")

# Create FastAPI app with lifespan and CORS middleware
app = FastAPI(
    title="Hepatitis Prediction API (Simplified)", 
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
    return {"status": "Hepatitis Prediction API (Simplified) is running", "version": "1.0.0"}

@app.get("/health")
async def health_check():
    """Detailed health check"""
    global model, artifacts
    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "artifacts_loaded": artifacts is not None,
        "model_type": "simplified_random_forest"
    }

@app.post("/predict", response_model=PredictionResponse)
async def predict_hepatitis(request: PredictionRequest):
    """Main prediction endpoint"""
    global model, artifacts
    
    if model is None or artifacts is None:
        raise HTTPException(status_code=500, detail="Simplified model not loaded")
    
    try:
        # Create input features matching the simplified model
        input_features = create_input_features(request)
        
        # Make prediction using simplified model
        result = predict_with_simplified_model(input_features)
        
        return {
            "success": True,
            "prediction": result
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

def create_input_features(request: PredictionRequest):
    """Create input features from request data"""
    symptoms = request.symptoms
    
    # Count symptoms (pain and fatigue are numeric, others boolean)
    symptom_count = sum([
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
    
    # Determine severity
    severity = determine_severity_numeric(request.age, symptom_count)
    
    # Create feature vector matching training data
    features = {
        'SymptomCount': symptom_count,
        'Severity_numeric': severity,
        'has_jaundice': int(symptoms.jaundice),
        'has_dark_urine': int(symptoms.dark_urine),
        'has_fatigue': int(symptoms.fatigue > 0),
        'has_nausea': int(symptoms.nausea),
        'has_vomiting': int(symptoms.vomiting),
        'has_fever': int(symptoms.fever),
        'has_abdominal_pain': int(symptoms.pain > 0),
        'has_loss_appetite': int(symptoms.loss_of_appetite),
        'has_joint_pain': int(symptoms.joint_pain),
        'has_muscle_aches': 0,  # Not collected from frontend
        'has_headache': 0,      # Not collected from frontend
        'has_clay_stools': 0,   # Not collected from frontend
        'has_weight_loss': 0,   # Not collected from frontend
        'has_itchy_skin': 0     # Not collected from frontend
    }
    
    return features

def determine_severity_numeric(age_str, symptom_count):
    """Determine numeric severity based on age and symptoms"""
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
        return 3  # Severe
    elif symptom_count >= 3 or age > 40:
        return 2  # Moderate
    else:
        return 1  # Mild

def predict_with_simplified_model(input_features):
    """Make prediction using simplified model"""
    global model, artifacts
    
    if model is None or artifacts is None:
        raise Exception("Simplified model not loaded")
    
    # Extract artifacts
    scaler = artifacts['scaler']
    label_encoder = artifacts['label_encoder']
    feature_columns = artifacts['feature_columns']
    
    # Create DataFrame with correct feature order
    input_df = pd.DataFrame([input_features])
    input_df = input_df[feature_columns]  # Ensure correct order
    
    # Scale features
    input_scaled = scaler.transform(input_df)
    
    # Make prediction
    prediction_proba = model.predict_proba(input_scaled)[0]
    prediction = model.predict(input_scaled)[0]
    predicted_type = label_encoder.inverse_transform([prediction])[0]
    
    # Calculate confidence
    confidence = max(prediction_proba)
    
    # Create results
    results = {
        'predicted_class': predicted_type,
        'confidence': float(confidence),
        'probability_Hepatitis A': float(prediction_proba[0]),  # Class 0
        'probability_Hepatitis C': float(prediction_proba[1]),  # Class 1
        'symptoms_analyzed': create_symptoms_text(input_features),
        'severity_assessed': map_severity_numeric_to_text(input_features['Severity_numeric']),
        'symptom_count': input_features['SymptomCount']
    }
    
    return results

def create_symptoms_text(features):
    """Create symptoms text from features"""
    symptoms = []
    if features['has_jaundice']: symptoms.append('jaundice')
    if features['has_dark_urine']: symptoms.append('dark-colored urine')
    if features['has_abdominal_pain']: symptoms.append('abdominal pain')
    if features['has_fatigue']: symptoms.append('fatigue')
    if features['has_nausea']: symptoms.append('nausea')
    if features['has_vomiting']: symptoms.append('vomiting')
    if features['has_fever']: symptoms.append('fever')
    if features['has_loss_appetite']: symptoms.append('loss of appetite')
    if features['has_joint_pain']: symptoms.append('joint pain')
    
    return ', '.join(symptoms) if symptoms else 'no symptoms'

def map_severity_numeric_to_text(severity_num):
    """Map numeric severity back to text"""
    mapping = {1: 'Mild', 2: 'Moderate', 3: 'Severe'}
    return mapping.get(severity_num, 'Moderate')

# CLI prediction function for testing
def predict_cli(symptoms_dict, age="31-45", gender="male", risk_factors=None):
    """CLI prediction function for testing"""
    if risk_factors is None:
        risk_factors = []
    
    # Convert symptoms dict to Symptoms object
    symptoms = Symptoms(**symptoms_dict)
    
    # Create request
    request = PredictionRequest(
        age=age,
        gender=gender,
        symptoms=symptoms,
        riskFactors=risk_factors
    )
    
    try:
        input_features = create_input_features(request)
        result = predict_with_simplified_model(input_features)
        return result
    except Exception as e:
        return {'error': str(e)}

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Hepatitis Prediction API (Simplified)')
    parser.add_argument('--server', action='store_true', help='Run as FastAPI server')
    parser.add_argument('--port', type=int, default=8001, help='Port for server')
    parser.add_argument('--test', action='store_true', help='Test prediction with sample data')
    
    args = parser.parse_args()
    
    if args.test:
        # Test prediction
        print("🧪 Testing prediction with sample data...")
        
        # Test Hepatitis A symptoms
        test_result_a = predict_cli({
            "jaundice": True,
            "dark_urine": True,
            "pain": 3,
            "fatigue": 2,
            "nausea": True,
            "vomiting": True,
            "fever": True,
            "loss_of_appetite": True,
            "joint_pain": False
        }, age="18-30", risk_factors=["recent_travel"])
        
        print("Hepatitis A symptoms test:", json.dumps(test_result_a, indent=2))
        
        # Test Hepatitis C symptoms
        test_result_c = predict_cli({
            "jaundice": False,
            "dark_urine": False,
            "pain": 2,
            "fatigue": 4,
            "nausea": False,
            "vomiting": False,
            "fever": False,
            "loss_of_appetite": False,
            "joint_pain": True
        }, age="over60", risk_factors=["unsafe_injection_history"])
        
        print("Hepatitis C symptoms test:", json.dumps(test_result_c, indent=2))
    
    elif args.server:
        # Run as server
        print(f"🚀 Starting Hepatitis Prediction API (Simplified) server on port {args.port}")
        uvicorn.run(app, host="0.0.0.0", port=args.port)
    
    else:
        # Default: run as server
        print("🚀 Starting Hepatitis Prediction API (Simplified) server on port 8001")
        uvicorn.run(app, host="0.0.0.0", port=8001)
