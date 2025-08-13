import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
import pickle
import os

def create_simplified_features(df):
    """Create simplified features that match our frontend input"""
    
    # Create binary features for key symptoms
    df['has_jaundice'] = df['Symptoms'].str.contains('jaundice', case=False).astype(int)
    df['has_dark_urine'] = df['Symptoms'].str.contains('dark.?urine', case=False).astype(int)
    df['has_fatigue'] = df['Symptoms'].str.contains('fatigue|weakness', case=False).astype(int)
    df['has_nausea'] = df['Symptoms'].str.contains('nausea', case=False).astype(int)
    df['has_vomiting'] = df['Symptoms'].str.contains('vomiting', case=False).astype(int)
    df['has_fever'] = df['Symptoms'].str.contains('fever', case=False).astype(int)
    df['has_abdominal_pain'] = df['Symptoms'].str.contains('abdominal|pain', case=False).astype(int)
    df['has_loss_appetite'] = df['Symptoms'].str.contains('appetite', case=False).astype(int)
    df['has_joint_pain'] = df['Symptoms'].str.contains('joint', case=False).astype(int)
    df['has_muscle_aches'] = df['Symptoms'].str.contains('muscle', case=False).astype(int)
    df['has_headache'] = df['Symptoms'].str.contains('headache', case=False).astype(int)
    df['has_clay_stools'] = df['Symptoms'].str.contains('clay', case=False).astype(int)
    df['has_weight_loss'] = df['Symptoms'].str.contains('weight.?loss', case=False).astype(int)
    df['has_itchy_skin'] = df['Symptoms'].str.contains('itchy', case=False).astype(int)
    
    # Map severity to numeric
    severity_map = {'Mild': 1, 'Moderate': 2, 'Severe': 3}
    if 'Severity' in df.columns:
        df['Severity_numeric'] = df['Severity'].map(severity_map).fillna(2)
    else:
        df['Severity_numeric'] = 2  # Default to moderate
    
    # Feature columns to use for training
    feature_cols = [
        'SymptomCount', 'Severity_numeric',
        'has_jaundice', 'has_dark_urine', 'has_fatigue', 'has_nausea', 
        'has_vomiting', 'has_fever', 'has_abdominal_pain', 'has_loss_appetite',
        'has_joint_pain', 'has_muscle_aches', 'has_headache', 'has_clay_stools',
        'has_weight_loss', 'has_itchy_skin'
    ]
    
    return df[feature_cols], feature_cols

def train_simplified_model():
    """Train a simplified but effective model"""
    
    print("🚀 Training simplified hepatitis prediction model...")
    
    # Load data
    df = pd.read_csv('../data/hepatitis_dataset_5k.csv')
    print(f"📊 Loaded {len(df)} samples")
    print(f"Distribution: {df['HepatitisType'].value_counts().to_dict()}")
    
    # Create features
    X, feature_cols = create_simplified_features(df)
    
    # Encode target
    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(df['HepatitisType'])
    
    print(f"Features created: {len(feature_cols)}")
    print(f"Label classes: {label_encoder.classes_}")
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train Random Forest (more robust than neural network for this data)
    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        random_state=42,
        class_weight='balanced'  # Handle any slight imbalances
    )
    
    model.fit(X_train_scaled, y_train)
    
    # Evaluate
    y_pred = model.predict(X_test_scaled)
    accuracy = accuracy_score(y_test, y_pred)
    
    print(f"\n✅ Model trained successfully!")
    print(f"Accuracy: {accuracy:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=label_encoder.classes_))
    
    # Save model and artifacts
    os.makedirs('simple_model', exist_ok=True)
    
    # Save Random Forest model
    with open('simple_model/hepatitis_model.pkl', 'wb') as f:
        pickle.dump(model, f)
    
    # Save preprocessing artifacts
    artifacts = {
        'scaler': scaler,
        'label_encoder': label_encoder,
        'feature_columns': feature_cols,
        'model_type': 'random_forest'
    }
    
    with open('simple_model/preprocessing_artifacts.pkl', 'wb') as f:
        pickle.dump(artifacts, f)
    
    print(f"\n💾 Model saved to: simple_model/")
    return model, artifacts

if __name__ == "__main__":
    train_simplified_model()
