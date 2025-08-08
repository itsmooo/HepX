#!/usr/bin/env python3
"""
Test script to verify improved model prediction functionality
"""

import json
import sys
import os
from improved_predict import create_prediction_data, predict_with_improved_model

def test_improved_prediction():
    """Test the improved prediction functionality"""
    print("🧪 Testing Improved Model Prediction")
    print("=" * 50)
    
    # Test case 1: Typical Hepatitis A symptoms
    test_case_1 = {
        'age': '31-45',  # Fixed: use string age
        'gender': 'Male',
        'symptoms': {
            'jaundice': True,
            'fatigue': 2,
            'nausea': True,
            'vomiting': False,
            'fever': True,
            'pain': True,  # Fixed: use boolean
            'dark_urine': False,
            'loss_of_appetite': True,
            'joint_pain': False
        },
        'riskFactors': {
            'alcohol_abuse': False,
            'drug_use': False,
            'blood_transfusion': False,
            'tattoos': False,
            'travel': True
        }
    }
    
    # Test case 2: Typical Hepatitis C symptoms
    test_case_2 = {
        'age': '46-60',  # Fixed: use string age
        'gender': 'Female',
        'symptoms': {
            'jaundice': True,
            'fatigue': 3,
            'nausea': True,
            'vomiting': True,
            'fever': False,
            'pain': True,  # Fixed: use boolean
            'dark_urine': True,
            'loss_of_appetite': True,
            'joint_pain': True
        },
        'riskFactors': {
            'alcohol_abuse': False,
            'drug_use': False,
            'blood_transfusion': True,
            'tattoos': True,
            'travel': False
        }
    }
    
    test_cases = [
        ("Hepatitis A Case", test_case_1),
        ("Hepatitis C Case", test_case_2)
    ]
    
    for case_name, test_data in test_cases:
        print(f"\n🔍 Testing: {case_name}")
        print(f"   Age: {test_data['age']}")
        print(f"   Gender: {test_data['gender']}")
        print(f"   Symptoms: {test_data['symptoms']}")
        
        try:
            # Create prediction data
            data = create_prediction_data(test_data)
            print(f"   Processed data: {data}")
            
            # Make prediction
            results = predict_with_improved_model(data)
            
            print(f"   ✅ Prediction: {results['predicted_class']}")
            print(f"   📊 Confidence: {results['confidence']:.1%}")
            print(f"   📈 Hepatitis A probability: {results['probability_hepatitis_a']:.1%}")
            print(f"   📈 Hepatitis C probability: {results['probability_hepatitis_c']:.1%}")
            
        except Exception as e:
            print(f"   ❌ Error: {e}")
    
    print("\n" + "=" * 50)
    print("🎉 Improved Model Test Completed!")

if __name__ == "__main__":
    test_improved_prediction() 