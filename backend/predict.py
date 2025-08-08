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

# Global model variables (loaded once at startup)
model = None
artifacts = None

def load_model_and_artifacts():
    """Load the model and preprocessing artifacts at startup"""
    global model, artifacts
    
    model_path = './improved_hepatitis_outputs/models/best_improved_model.keras'
    preprocessing_artifacts_path = './improved_hepatitis_outputs/models/preprocessing_artifacts.pkl'
    
    if not os.path.exists(model_path) or not os.path.exists(preprocessing_artifacts_path):
        raise Exception("Model files not found. Please run improved_training.py first.")
    
    # Load model
    model = load_model(model_path)
    
    # Load preprocessing artifacts
    with open(preprocessing_artifacts_path, 'rb') as f:
        artifacts = pickle.load(f)
    
    print("Model and artifacts loaded successfully!")

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan event handler for FastAPI"""
    # Startup
    try:
        load_model_and_artifacts()
        print("✅ Model and artifacts loaded successfully!")
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        print("💡 Please run improved_training.py first to generate the model files.")
    
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
        raise HTTPException(status_code=500, detail="Model not loaded")
    
    try:
        # Convert request to dictionary format for existing functions
        user_data = {
            'age': request.age,
            'gender': request.gender,
            'symptoms': {
                'jaundice': request.symptoms.jaundice,
                'dark_urine': request.symptoms.dark_urine,
                'pain': request.symptoms.pain,  # This is already an int from frontend
                'fatigue': request.symptoms.fatigue,  # This is already an int from frontend
                'nausea': request.symptoms.nausea,
                'vomiting': request.symptoms.vomiting,
                'fever': request.symptoms.fever,
                'loss_of_appetite': request.symptoms.loss_of_appetite,
                'joint_pain': request.symptoms.joint_pain
            },
            'riskFactors': request.riskFactors
        }
        
        # Create prediction data
        data = create_prediction_data(user_data)
        
        # Make prediction
        results = predict_with_improved_model_api(data)
        
        return {
            "success": True,
            "prediction": {
                "success": True,
                "message": "Prediction completed successfully",
                "predictions": [results],
                "total_predictions": 1
            }
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

def create_prediction_data(user_data):
    """
    Create a DataFrame from user input data for prediction with improved model format
    
    Args:
        user_data: Dictionary containing user input (age, gender, symptoms, riskFactors)
        
    Returns:
        Dictionary ready for prediction
    """
    # Count symptoms
    symptoms = user_data['symptoms']
    symptom_count = sum([
        symptoms.get('jaundice', False),
        symptoms.get('dark_urine', False),
        symptoms.get('pain', 0) > 0,  # Convert numeric pain to boolean
        symptoms.get('fatigue', 0) > 0,  # Convert numeric fatigue to boolean
        symptoms.get('nausea', False),
        symptoms.get('vomiting', False),
        symptoms.get('fever', False),
        symptoms.get('loss_of_appetite', False),
        symptoms.get('joint_pain', False)
    ])
    
    # Create symptoms text
    symptoms_text = create_symptoms_text(symptoms)
    
    # Determine severity based on symptoms and age
    # Convert age string to numeric value
    age_str = user_data['age']
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
        symptom_list.append('dark-colored urine')  # Match the feature engineering exactly
    if symptoms.get('pain', 0) > 0:  # Handle numeric pain value
        symptom_list.append('abdominal pain')
    if symptoms.get('fatigue', 0) > 0:  # Handle numeric fatigue value
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
    
    # Remove the original Severity column to avoid conflicts
    if 'Severity' in processed_input.columns:
        processed_input = processed_input.drop('Severity', axis=1)

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

def predict_with_improved_model_api(data):
    """
    Make predictions using the pre-loaded model (for API use)
    
    Args:
        data: Dictionary with input data
        
    Returns:
        Predictions with probabilities
    """
    global model, artifacts
    
    if model is None or artifacts is None:
        raise Exception("Model not loaded")
    
    # Preprocess input using global artifacts
    input_scaled, label_encoder = preprocess_input_for_improved_model_api(data)
    
    # Make prediction
    prediction_proba = model.predict(input_scaled, verbose=0)[0][0]
    prediction = (prediction_proba > 0.5).astype(int)
    
    # Decode prediction
    predicted_type = label_encoder.inverse_transform([prediction])[0]
    
    # FIXED: Calculate confidence as the maximum probability between the two classes
    # This ensures confidence is always the higher probability value
    confidence = max(prediction_proba, 1 - prediction_proba)
    
    # Create results with corrected probability assignments
    results = {
        'predicted_class': predicted_type,
        'confidence': float(confidence),
        'probability_Hepatitis A': float(1 - prediction_proba),  # Class 0
        'probability_Hepatitis C': float(prediction_proba)       # Class 1
    }
    
    return results

def preprocess_input_for_improved_model_api(input_data):
    """
    Preprocess input data for the improved model using global artifacts
    
    Args:
        input_data: Dictionary with input data
        
    Returns:
        Preprocessed input ready for prediction
    """
    global artifacts
    
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
    
    # Remove the original Severity column to avoid conflicts
    if 'Severity' in processed_input.columns:
        processed_input = processed_input.drop('Severity', axis=1)

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
    Make predictions using the improved model (for CLI use)
    
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
    
    # FIXED: Calculate confidence as the maximum probability between the two classes
    confidence = max(prediction_proba, 1 - prediction_proba)
    
    # Create results with corrected probability assignments
    results = {
        'predicted_class': predicted_type,
        'confidence': float(confidence),
        'probability_hepatitis_a': float(1 - prediction_proba),  # Class 0
        'probability_hepatitis_c': float(prediction_proba)       # Class 1
    }
    
    return results


if __name__ == "__main__":
    # Check command line arguments
    if len(sys.argv) < 2:
        print("🚀 HepX Hepatitis Prediction API")
        print("=" * 40)
        print()
        print("Usage:")
        print("  🐍 API Server Mode:")
        print("    python predict.py --server [--port PORT]")
        print("    python predict.py --server --port 8001")
        print()
        print("  📁 CLI Prediction Mode:")
        print("    python predict.py <input_file_path>")
        print("    python predict.py test_data.json")
        print()
        print("Examples:")
        print("  # Start API server on default port 8000")
        print("  python predict.py --server")
        print()
        print("  # Start API server on custom port")
        print("  # python predict.py --server --port 8001")
        print()
        print("  # Make prediction from file")
        print("  # python predict.py prediction_data.json")
        print()
        print("📖 API Documentation: http://localhost:8000/docs")
        print("🔗 Health Check: http://localhost:8000/health")
        sys.exit(1)
    
    # Check if running as API server
    if sys.argv[1] == "--server":
        port = 8000  # default port
        if len(sys.argv) > 2 and sys.argv[2] == "--port":
            if len(sys.argv) > 3:
                port = int(sys.argv[3])
        
        print(f"🚀 Starting Hepatitis Prediction API server on port {port}")
        print(f"📖 API Documentation: http://localhost:{port}/docs")
        print(f"🔗 Health Check: http://localhost:{port}/health")
        print("🔄 Press Ctrl+C to stop the server")
        print()
        
        uvicorn.run(app, host="0.0.0.0", port=port)
    else:
        # CLI mode - original functionality
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
