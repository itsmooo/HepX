import os
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Deep Learning Libraries
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.utils import plot_model

# Machine Learning Libraries
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.metrics import (confusion_matrix, classification_report,
                           roc_curve, auc, precision_score,
                           recall_score, f1_score, accuracy_score)
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression

class EnhancedHepatitisSystem:
    """
    Enhanced Hepatitis Risk Prediction System with Maximum Accuracy
    Features:
    - Advanced feature engineering with hepatitis-specific signatures
    - Strict preprocessing pipeline with artifact management
    - Dynamic threshold optimization
    - Comprehensive model comparison
    - Interactive testing capabilities
    """
    
    def __init__(self, output_dir='enhanced_hepatitis_outputs'):
        self.output_dir = output_dir
        self.create_directory_structure()
        
        # Initialize containers
        self.df = None
        self.df_processed = None
        self.X_train_scaled = None
        self.X_val_scaled = None
        self.X_test_scaled = None
        self.y_train = None
        self.y_val = None
        self.y_test = None
        self.model = None
        self.history = None
        self.scaler = None
        self.label_encoder = None
        self.feature_cols = []
        self.target_col = None
        self.results = {}
        self.label_encoders_other = {}
        self.imputers = {}
        self.optimal_threshold = 0.5
        self.required_columns = None
        
        print("🚀 ENHANCED HEPATITIS PREDICTION SYSTEM")
        print("=" * 60)
        print("Maximum Accuracy with Advanced Features")
        print("=" * 60)

    def create_directory_structure(self):
        """Create comprehensive directory structure"""
        directories = [
            self.output_dir,
            f'{self.output_dir}/figures',
            f'{self.output_dir}/models',
            f'{self.output_dir}/data',
            f'{self.output_dir}/evaluation',
            f'{self.output_dir}/preprocessing'
        ]
        for directory in directories:
            os.makedirs(directory, exist_ok=True)
        print(f"📁 Created directory structure in: {self.output_dir}")

    def create_sample_dataset(self):
        """Create a comprehensive sample hepatitis dataset for demonstration"""
        print("\n" + "="*50)
        print("CREATING SAMPLE HEPATITIS DATASET")
        print("="*50)
        
        np.random.seed(42)
        n_samples = 1000
        
        # Define hepatitis types and their characteristics
        hepatitis_types = ['Hepatitis A', 'Hepatitis C']
        
        # Common symptoms for each type
        hep_a_symptoms = [
            'fever, abdominal pain, nausea, fatigue',
            'jaundice, dark-colored urine, fever, muscle aches',
            'nausea, vomiting, fever, loss of appetite',
            'abdominal pain, fatigue, headache, fever'
        ]
        
        hep_c_symptoms = [
            'weight loss, dark-colored urine, jaundice, fatigue',
            'jaundice, bleeding easily, swelling, confusion',
            'fatigue, weight loss, bruising easily, spider angiomas',
            'dark-colored urine, jaundice, ascites, itchy skin'
        ]
        
        data = []
        
        for i in range(n_samples):
            # Randomly assign hepatitis type
            hep_type = np.random.choice(hepatitis_types)
            
            # Generate symptoms based on type
            if hep_type == 'Hepatitis A':
                symptoms = np.random.choice(hep_a_symptoms)
                severity = np.random.choice(['Mild', 'Moderate'], p=[0.7, 0.3])
                alt = np.random.normal(80, 20)  # Elevated ALT
                ast = np.random.normal(70, 15)  # Elevated AST
            else:  # Hepatitis C
                symptoms = np.random.choice(hep_c_symptoms)
                severity = np.random.choice(['Moderate', 'Severe'], p=[0.6, 0.4])
                alt = np.random.normal(120, 30)  # Higher ALT
                ast = np.random.normal(100, 25)  # Higher AST
            
            # Generate other fields
            patient_id = f"PAT{i+1:04d}"
            diagnosis_date = pd.Timestamp('2023-01-01') + pd.Timedelta(days=np.random.randint(0, 365))
            treatment = np.random.choice(['Supportive care', 'Antiviral therapy', 'Rest and hydration'])
            
            data.append({
                'PatientID': patient_id,
                'Symptoms': symptoms,
                'Severity': severity,
                'DiagnosisDate': diagnosis_date.strftime('%Y-%m-%d'),
                'Treatment': treatment,
                'ALT': max(10, alt),  # Ensure positive values
                'AST': max(10, ast),  # Ensure positive values
                'HepatitisType': hep_type
            })
        
        # Create DataFrame
        df = pd.DataFrame(data)
        
        # Save the dataset
        df.to_csv(f'{self.output_dir}/data/hepatitis_dataset_sample.csv', index=False)
        print(f"✅ Created sample dataset with {len(df)} records")
        print(f"📊 Dataset saved to: {self.output_dir}/data/hepatitis_dataset_sample.csv")
        
        return df

    def load_and_explore_data(self, file_path=None):
        """Load and comprehensively explore the dataset"""
        print("\n" + "="*50)
        print("DATA LOADING AND EXPLORATION")
        print("="*50)
        
        try:
            if file_path and os.path.exists(file_path):
                self.df = pd.read_csv(file_path)
                print(f"✅ Dataset loaded from: {file_path}")
            else:
                # Try to load from default location
                default_path = 'data/hepatitis_dataset_R.csv'
                if os.path.exists(default_path):
                    self.df = pd.read_csv(default_path)
                    print(f"✅ Dataset loaded from: {default_path}")
                else:
                    print("📝 No dataset found, creating sample dataset...")
                    self.df = self.create_sample_dataset()
            
            print(f"📊 Dataset shape: {self.df.shape}")
            print(f"📋 Columns: {list(self.df.columns)}")
            
            # Store required columns for later use
            self.required_columns = list(self.df.columns)
            
            # Detailed data exploration
            print(f"\n📈 Data Types:")
            print(self.df.dtypes)
            
            print(f"\n🔍 Missing Values:")
            missing_data = self.df.isnull().sum()
            print(missing_data[missing_data > 0])
            
            # Check target distribution
            print(f"\n🎯 Target Distribution:")
            target_dist = self.df['HepatitisType'].value_counts()
            print(target_dist)
            print(f"Class balance: {target_dist.iloc[0]/(target_dist.sum()):.1%} vs {target_dist.iloc[1]/(target_dist.sum()):.1%}")
            
            # Save exploration results
            exploration_results = {
                'shape': self.df.shape,
                'columns': list(self.df.columns),
                'missing_values': missing_data.to_dict(),
                'target_distribution': target_dist.to_dict()
            }
            
            with open(f'{self.output_dir}/data/exploration_results.pkl', 'wb') as f:
                pickle.dump(exploration_results, f)
            
            return self.df
            
        except Exception as e:
            print(f"❌ Error loading dataset: {e}")
            return None

    def advanced_feature_engineering(self):
        """Enhanced feature engineering with hepatitis-specific signatures"""
        print("\n" + "="*50)
        print("ADVANCED FEATURE ENGINEERING")
        print("="*50)
        
        if self.df is None:
            print("❌ No data loaded. Please load data first.")
            return None
            
        self.df_processed = self.df.copy()
        
        # Step 1: Advanced missing value handling
        print("🔧 Step 1: Advanced missing value handling...")
        self.imputers = {}
        
        for col in self.df_processed.columns:
            if self.df_processed[col].dtype == 'object':
                # Use mode for categorical
                imputer = SimpleImputer(strategy='most_frequent')
                self.df_processed[[col]] = imputer.fit_transform(self.df_processed[[col]])
                self.imputers[col] = imputer
            elif pd.api.types.is_numeric_dtype(self.df_processed[col]):
                # Use median for numerical
                imputer = SimpleImputer(strategy='median')
                self.df_processed[[col]] = imputer.fit_transform(self.df_processed[[col]])
                self.imputers[col] = imputer
            else:
                # For other types (like datetime), just fill with most frequent
                imputer = SimpleImputer(strategy='most_frequent')
                self.df_processed[[col]] = imputer.fit_transform(self.df_processed[[col]])
                self.imputers[col] = imputer

        # Step 2: Create target variable
        print("🎯 Step 2: Creating target variable...")
        self.target_col = 'HepatitisType'
        self.label_encoder = LabelEncoder()
        self.df_processed[self.target_col] = self.label_encoder.fit_transform(
            self.df_processed[self.target_col].astype(str))
        print(f"   Encoded target: {self.label_encoder.classes_}")
        
        # Step 3: Enhanced symptom feature engineering with specific patterns
        print("🛠️ Step 3: Enhanced symptom feature engineering...")
        
        # Comprehensive symptom list with specific patterns for A vs C
        symptoms_list = [
            # Hepatitis A specific symptoms (acute, gastrointestinal)
            'clay-colored stool', 'clay- or gray-colored stool', 'gray-colored stool',
            'sudden nausea', 'vomiting', 'diarrhea', 'abdominal pain', 'joint pain',
            'low-grade fever', 'fever', 'loss of appetite', 'unusual tiredness',
            'weakness', 'intense itching',
            
            # Hepatitis C specific symptoms (chronic, liver damage)
            'ascites', 'fluid buildup', 'swelling', 'spider angiomas', 'spiderlike blood vessels',
            'itchy skin', 'bleeding easily', 'bruising easily', 'weight loss',
            'confusion', 'drowsiness', 'slurred speech', 'hepatic encephalopathy',
            'dark-colored urine', 'dark urine', 'not wanting to eat',
            
            # Common symptoms (need careful differentiation)
            'jaundice', 'yellowing', 'fatigue', 'nausea'
        ]
        
        for symptom in symptoms_list:
            self.df_processed[f'has_{symptom.replace(" ", "_").replace("-", "_")}'] = self.df_processed['Symptoms'].str.contains(
                symptom, case=False, na=False).astype(int)

        # Step 4: Hepatitis-specific signature combinations (IMPROVED)
        print("🧬 Step 4: Creating hepatitis-specific signatures...")
        
        # Hepatitis A signature (acute onset, gastrointestinal, clay stool)
        self.df_processed['hepatitis_a_signature'] = (
            (self.df_processed['has_clay_colored_stool'] | 
             self.df_processed['has_clay_or_gray_colored_stool'] |
             self.df_processed['has_gray_colored_stool']) &
            (self.df_processed['has_abdominal_pain'] | 
             self.df_processed['has_sudden_nausea'] |
             self.df_processed['has_vomiting'] |
             self.df_processed['has_diarrhea']) &
            ~self.df_processed['has_weight_loss'] &
            ~self.df_processed['has_ascites'] &
            ~self.df_processed['has_spider_angiomas']
        ).astype(int)
        
        # Hepatitis C signature (chronic symptoms, liver damage indicators)
        self.df_processed['hepatitis_c_signature'] = (
            (self.df_processed['has_weight_loss'] |
             self.df_processed['has_ascites'] |
             self.df_processed['has_spider_angiomas']) &
            (self.df_processed['has_dark_colored_urine'] |
             self.df_processed['has_dark_urine']) &
            (self.df_processed['has_bleeding_easily'] |
             self.df_processed['has_bruising_easily'])
        ).astype(int)
        
        # Additional specific combinations for better differentiation
        self.df_processed['acute_gastrointestinal_cluster'] = (
            self.df_processed['has_sudden_nausea'] & 
            self.df_processed['has_vomiting'] &
            self.df_processed['has_diarrhea'] &
            self.df_processed['has_abdominal_pain']
        ).astype(int)
        
        self.df_processed['chronic_liver_damage_cluster'] = (
            self.df_processed['has_ascites'] &
            self.df_processed['has_spider_angiomas'] &
            self.df_processed['has_confusion']
        ).astype(int)
        
        self.df_processed['hepatitis_a_acute_pattern'] = (
            self.df_processed['has_clay_colored_stool'] |
            self.df_processed['has_clay_or_gray_colored_stool'] |
            self.df_processed['has_gray_colored_stool']
        ).astype(int)
        
        self.df_processed['hepatitis_c_chronic_pattern'] = (
            self.df_processed['has_weight_loss'] &
            self.df_processed['has_dark_colored_urine']
        ).astype(int)

        # Step 5: Temporal and severity features
        print("📅 Step 5: Temporal and severity feature engineering...")
        
        # Date features
        if 'DiagnosisDate' in self.df_processed.columns:
            self.df_processed['DiagnosisDate'] = pd.to_datetime(self.df_processed['DiagnosisDate'])
            self.df_processed['DiagnosisYear'] = self.df_processed['DiagnosisDate'].dt.year
            self.df_processed['DiagnosisMonth'] = self.df_processed['DiagnosisDate'].dt.month
            self.df_processed['DiagnosisDay'] = self.df_processed['DiagnosisDate'].dt.day
            self.df_processed['DiagnosisDayOfWeek'] = self.df_processed['DiagnosisDate'].dt.dayofweek
            self.df_processed['DiagnosisQuarter'] = self.df_processed['DiagnosisDate'].dt.quarter

        # Severity encoding with enhanced mapping
        print("⚠️ Step 6: Enhanced severity encoding...")
        if 'Severity' in self.df_processed.columns:
            severity_mapping = {'Mild': 1, 'Moderate': 2, 'Severe': 3}
            self.df_processed['Severity_encoded'] = self.df_processed['Severity'].map(severity_mapping).fillna(2)

        # Lab value ratios (if available)
        if 'ALT' in self.df_processed.columns and 'AST' in self.df_processed.columns:
            self.df_processed['alt_ast_ratio'] = (
                self.df_processed['ALT'] / (self.df_processed['AST'] + 1e-6)
            )

        # Step 7: Categorical encoding
        print("🏷️ Step 7: Advanced categorical encoding...")
        self.label_encoders_other = {}
        categorical_cols = self.df_processed.select_dtypes(include=['object']).columns
        
        for col in categorical_cols:
            if col not in [self.target_col, 'Symptoms', 'DiagnosisDate', 'SymptomOnsetDate']:
                le = LabelEncoder()
                # Handle unknown categories
                unique_values = list(self.df_processed[col].unique()) + ['UNK']
                le.fit(unique_values)
                self.df_processed[col + '_encoded'] = le.transform(
                    self.df_processed[col].astype(str))
                self.label_encoders_other[col] = le

        # Step 8: Symptom count and complexity features
        print("📊 Step 8: Symptom complexity features...")
        symptom_cols = [col for col in self.df_processed.columns if col.startswith('has_')]
        self.df_processed['total_symptoms'] = self.df_processed[symptom_cols].sum(axis=1)
        if 'Severity_encoded' in self.df_processed.columns:
            self.df_processed['symptom_severity_score'] = (
                self.df_processed['total_symptoms'] * self.df_processed['Severity_encoded']
            )

        # Step 9: Feature selection with improved logic
        print("🎯 Step 9: Intelligent feature selection...")
        self.feature_cols = []
        for col in self.df_processed.columns:
            if (col.endswith('_encoded') or
                col.startswith('has_') or
                col.endswith('_signature') or
                col.endswith('_cluster') or
                col.endswith('_pattern') or
                col in ['total_symptoms', 'symptom_severity_score', 'alt_ast_ratio'] or
                (self.df_processed[col].dtype in ['int64', 'float64'] and
                 col not in [self.target_col, 'PatientID'] and
                 not col.startswith('Diagnosis') and
                 col != 'Symptoms')):
                self.feature_cols.append(col)

        print(f"   Selected {len(self.feature_cols)} features for modeling")

        # Step 10: Data preparation
        X = self.df_processed[self.feature_cols]
        y = self.df_processed[self.target_col]

        print(f"📈 Final feature matrix shape: {X.shape}")
        print(f"📈 Target distribution (encoded):")
        print(y.value_counts())

        # Step 11: Stratified data splitting
        print("✂️ Step 11: Stratified data splitting...")
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        X_train, X_val, y_train, y_val = train_test_split(
            X_train, y_train, test_size=0.2, random_state=42, stratify=y_train
        )

        # Step 12: Feature scaling
        print("⚖️ Step 12: Robust feature scaling...")
        self.scaler = StandardScaler()
        self.X_train_scaled = self.scaler.fit_transform(X_train)
        self.X_val_scaled = self.scaler.transform(X_val)
        self.X_test_scaled = self.scaler.transform(X_test)

        self.y_train = y_train
        self.y_val = y_val
        self.y_test = y_test

        print(f"   Training set: {self.X_train_scaled.shape}")
        print(f"   Validation set: {self.X_val_scaled.shape}")
        print(f"   Test set: {self.X_test_scaled.shape}")

        # Step 13: Save comprehensive preprocessing artifacts
        print("💾 Step 13: Saving preprocessing artifacts...")
        preprocessing_artifacts = {
            'scaler': self.scaler,
            'label_encoder_target': self.label_encoder,
            'label_encoders_other': self.label_encoders_other,
            'imputers': self.imputers,
            'feature_columns': self.feature_cols,
            'target_column': self.target_col,
            'symptoms_list': symptoms_list,
            'required_columns': self.required_columns
        }

        with open(f'{self.output_dir}/models/preprocessing_artifacts.pkl', 'wb') as f:
            pickle.dump(preprocessing_artifacts, f)

        print(f"✅ Advanced feature engineering completed")
        return X, y

    def build_enhanced_neural_network(self):
        """Build an enhanced neural network architecture"""
        print("\n" + "="*50)
        print("BUILDING ENHANCED NEURAL NETWORK")
        print("="*50)
        
        input_dim = self.X_train_scaled.shape[1]
        print(f"📊 Input dimensions: {input_dim}")
        
        # Enhanced architecture with regularization
        model = Sequential([
            # Input layer with batch normalization
            Dense(256, activation='relu', input_shape=(input_dim,)),
            BatchNormalization(),
            Dropout(0.3),
            
            # Hidden layers with progressive reduction
            Dense(128, activation='relu'),
            BatchNormalization(),
            Dropout(0.3),
            
            Dense(64, activation='relu'),
            BatchNormalization(),
            Dropout(0.2),
            
            Dense(32, activation='relu'),
            Dropout(0.2),
            
            # Output layer
            Dense(1, activation='sigmoid')
        ])
        
        # Compile with advanced optimizer
        model.compile(
            optimizer=Adam(learning_rate=0.001, beta_1=0.9, beta_2=0.999),
            loss='binary_crossentropy',
            metrics=['accuracy', 'precision', 'recall']
        )
        
        print("✅ Enhanced neural network built successfully")
        print(f"📊 Total parameters: {model.count_params():,}")
        
        return model

    def train_enhanced_model(self):
        """Train the enhanced model with advanced callbacks"""
        print("\n" + "="*50)
        print("TRAINING ENHANCED MODEL")
        print("="*50)
        
        # Build model
        self.model = self.build_enhanced_neural_network()
        
        # Advanced callbacks
        callbacks = [
            EarlyStopping(
                monitor='val_loss',
                patience=15,
                restore_best_weights=True,
                verbose=1
            ),
            ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=8,
                min_lr=1e-7,
                verbose=1
            ),
            ModelCheckpoint(
                f'{self.output_dir}/models/best_enhanced_model.keras',
                monitor='val_accuracy',
                save_best_only=True,
                verbose=1
            )
        ]
        
        # Train model
        print("🚀 Starting training...")
        self.history = self.model.fit(
            self.X_train_scaled, self.y_train,
            validation_data=(self.X_val_scaled, self.y_val),
            epochs=100,
            batch_size=32,
            callbacks=callbacks,
            verbose=1
        )
        
        print("✅ Model training completed")
        return self.model, self.history

    def optimize_threshold(self):
        """Optimize classification threshold for maximum accuracy"""
        print("\n" + "="*50)
        print("OPTIMIZING CLASSIFICATION THRESHOLD")
        print("="*50)
        
        # Get validation predictions
        val_proba = self.model.predict(self.X_val_scaled, verbose=0)
        
        # Test different thresholds
        thresholds = np.arange(0.1, 0.9, 0.01)
        best_threshold = 0.5
        best_f1 = 0
        
        threshold_results = []
        
        for threshold in thresholds:
            val_pred = (val_proba > threshold).astype(int)
            f1 = f1_score(self.y_val, val_pred)
            accuracy = accuracy_score(self.y_val, val_pred)
            precision = precision_score(self.y_val, val_pred)
            recall = recall_score(self.y_val, val_pred)
            
            threshold_results.append({
                'threshold': threshold,
                'f1_score': f1,
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall
            })
            
            if f1 > best_f1:
                best_f1 = f1
                best_threshold = threshold
        
        self.optimal_threshold = best_threshold
        
        # Test optimal threshold
        val_pred_optimal = (val_proba > best_threshold).astype(int)
        
        results = {
            'optimal_threshold': best_threshold,
            'f1_score': f1_score(self.y_val, val_pred_optimal),
            'accuracy': accuracy_score(self.y_val, val_pred_optimal),
            'precision': precision_score(self.y_val, val_pred_optimal),
            'recall': recall_score(self.y_val, val_pred_optimal)
        }
        
        print(f"🎯 Optimal threshold: {best_threshold:.4f}")
        print(f"📊 F1-Score: {results['f1_score']:.4f}")
        print(f"📊 Accuracy: {results['accuracy']:.4f}")
        print(f"📊 Precision: {results['precision']:.4f}")
        print(f"📊 Recall: {results['recall']:.4f}")
        
        self.results['optimized_threshold'] = results
        return best_threshold, results

    def comprehensive_evaluation(self):
        """Comprehensive model evaluation"""
        print("\n" + "="*50)
        print("COMPREHENSIVE MODEL EVALUATION")
        print("="*50)
        
        # Test set predictions
        test_proba = self.model.predict(self.X_test_scaled, verbose=0)
        test_pred = (test_proba > self.optimal_threshold).astype(int)
        
        # Calculate metrics
        test_accuracy = accuracy_score(self.y_test, test_pred)
        test_precision = precision_score(self.y_test, test_pred)
        test_recall = recall_score(self.y_test, test_pred)
        test_f1 = f1_score(self.y_test, test_pred)
        
        print(f"🎯 TEST SET RESULTS:")
        print(f"   Accuracy: {test_accuracy:.4f}")
        print(f"   Precision: {test_precision:.4f}")
        print(f"   Recall: {test_recall:.4f}")
        print(f"   F1-Score: {test_f1:.4f}")
        
        # Classification report
        print(f"\n📊 DETAILED CLASSIFICATION REPORT:")
        target_names = self.label_encoder.classes_
        print(classification_report(self.y_test, test_pred, target_names=target_names))
        
        # Confusion matrix
        cm = confusion_matrix(self.y_test, test_pred)
        print(f"\n🔍 CONFUSION MATRIX:")
        print(cm)
        
        # Store results
        self.results['final_evaluation'] = {
            'test_accuracy': test_accuracy,
            'test_precision': test_precision,
            'test_recall': test_recall,
            'test_f1': test_f1,
            'confusion_matrix': cm.tolist(),
            'classification_report': classification_report(self.y_test, test_pred, target_names=target_names, output_dict=True)
        }
        
        return self.results['final_evaluation']

    def predict_single_case(self, symptoms, threshold=None):
        """Enhanced prediction with optimized threshold"""
        if self.model is None:
            try:
                self.model = load_model(f'{self.output_dir}/models/best_enhanced_model.keras')
                print("✅ Loaded saved model for prediction")
            except Exception as e:
                return {
                    'error': f"Model not available: {e}",
                    'prediction': None,
                    'confidence': 0.0
                }

        # Use optimized threshold if available
        threshold = threshold or getattr(self, 'optimal_threshold', 0.5)

        # Prepare input data with all required columns
        input_data = {
            'Symptoms': ', '.join(symptoms) if isinstance(symptoms, list) else symptoms,
            'Severity': 'Moderate',  # Default
            'DiagnosisDate': datetime.now().strftime('%Y-%m-%d'),
            'Treatment': 'Supportive care',  # Default
            'PatientID': 'TEST0001',  # Example
            'ALT': 40,  # Example default values
            'AST': 35,  # Example default values
        }

        try:
            # Preprocess input
            processed_input = self.preprocess_single_input(input_data)

            if processed_input is None:
                return {
                    'error': 'Preprocessing failed',
                    'prediction': None,
                    'confidence': 0.0
                }

            # Make prediction
            prediction_proba = self.model.predict(processed_input, verbose=0)[0][0]
            prediction = (prediction_proba > threshold).astype(int)

            # Decode prediction
            predicted_type = self.label_encoder.inverse_transform([prediction])[0]
            confidence = max(prediction_proba, 1 - prediction_proba)

            return {
                'prediction': predicted_type,
                'confidence': float(confidence),
                'probability': float(prediction_proba),
                'threshold_used': float(threshold),
                'symptoms_analyzed': len(symptoms) if isinstance(symptoms, list) else len(symptoms.split(',')),
                'error': None
            }

        except Exception as e:
            return {
                'error': f"Prediction failed: {e}",
                'prediction': None,
                'confidence': 0.0
            }

    def preprocess_single_input(self, input_data):
        """Preprocess a single input with strict pipeline matching"""
        try:
            # Load preprocessing artifacts
            with open(f'{self.output_dir}/models/preprocessing_artifacts.pkl', 'rb') as f:
                artifacts = pickle.load(f)

            # Create DataFrame with all required columns
            input_df = pd.DataFrame(columns=artifacts['required_columns'])

            # Fill in provided data
            for col in input_data:
                if col in input_df.columns:
                    input_df[col] = [input_data[col]]

            # Fill missing columns with defaults
            for col in input_df.columns:
                if col not in input_data:
                    if col == 'Severity':
                        input_df[col] = 'Moderate'
                    elif col == 'DiagnosisDate':
                        input_df[col] = datetime.now().strftime('%Y-%m-%d')
                    elif col == 'Treatment':
                        input_df[col] = 'Supportive care'
                    elif col in ['ALT', 'AST']:
                        input_df[col] = 30  # Default normal values
                    else:
                        input_df[col] = ''  # Empty string for other columns

            # Apply feature engineering
            processed_input = self.apply_feature_engineering(input_df, artifacts)

            # Apply categorical encoding
            for col, encoder in artifacts['label_encoders_other'].items():
                if col in processed_input.columns:
                    # Handle unseen categories safely
                    processed_input[col] = processed_input[col].astype(str).apply(
                        lambda x: x if x in encoder.classes_ else 'UNK'
                    )
                    processed_input[col + '_encoded'] = encoder.transform(processed_input[col])

            # Select features and handle missing ones
            final_features = processed_input.reindex(
                columns=artifacts['feature_columns'], 
                fill_value=0
            )

            # Scale features
            input_scaled = artifacts['scaler'].transform(final_features)

            return input_scaled

        except Exception as e:
            print(f"❌ Error in preprocessing: {e}")
            return None

    def apply_feature_engineering(self, input_df, artifacts):
        """Apply the same feature engineering to new data"""
        processed_df = input_df.copy()
        
        symptoms_list = artifacts['symptoms_list']

        # Apply symptom extraction
        for symptom in symptoms_list:
            processed_df[f'has_{symptom.replace(" ", "_").replace("-", "_")}'] = processed_df['Symptoms'].str.contains(
                symptom, case=False, na=False).astype(int)

        # Apply signature combinations
        processed_df['hepatitis_a_signature'] = (
            (processed_df['has_clay_colored_stool'] | 
             processed_df['has_clay_or_gray_colored_stool'] |
             processed_df['has_gray_colored_stool']) &
            (processed_df['has_abdominal_pain'] | 
             processed_df['has_sudden_nausea'] |
             processed_df['has_vomiting'] |
             processed_df['has_diarrhea']) &
            ~processed_df['has_weight_loss'] &
            ~processed_df['has_ascites'] &
            ~processed_df['has_spider_angiomas']
        ).astype(int)

        processed_df['hepatitis_c_signature'] = (
            (processed_df['has_weight_loss'] |
             processed_df['has_ascites'] |
             processed_df['has_spider_angiomas']) &
            (processed_df['has_dark_colored_urine'] |
             processed_df['has_dark_urine']) &
            (processed_df['has_bleeding_easily'] |
             processed_df['has_bruising_easily'])
        ).astype(int)

        processed_df['acute_gastrointestinal_cluster'] = (
            processed_df['has_sudden_nausea'] & 
            processed_df['has_vomiting'] &
            processed_df['has_diarrhea'] &
            processed_df['has_abdominal_pain']
        ).astype(int)

        processed_df['chronic_liver_damage_cluster'] = (
            processed_df['has_ascites'] &
            processed_df['has_spider_angiomas'] &
            processed_df['has_confusion']
        ).astype(int)

        processed_df['hepatitis_a_acute_pattern'] = (
            processed_df['has_clay_colored_stool'] |
            processed_df['has_clay_or_gray_colored_stool'] |
            processed_df['has_gray_colored_stool']
        ).astype(int)

        processed_df['hepatitis_c_chronic_pattern'] = (
            processed_df['has_weight_loss'] &
            processed_df['has_dark_colored_urine']
        ).astype(int)

        # Date features
        if 'DiagnosisDate' in processed_df.columns:
            processed_df['DiagnosisDate'] = pd.to_datetime(processed_df['DiagnosisDate'])
            processed_df['DiagnosisYear'] = processed_df['DiagnosisDate'].dt.year
            processed_df['DiagnosisMonth'] = processed_df['DiagnosisDate'].dt.month
            processed_df['DiagnosisDay'] = processed_df['DiagnosisDate'].dt.day
            processed_df['DiagnosisDayOfWeek'] = processed_df['DiagnosisDate'].dt.dayofweek
            processed_df['DiagnosisQuarter'] = processed_df['DiagnosisDate'].dt.quarter

        # Severity encoding
        severity_mapping = {'Mild': 1, 'Moderate': 2, 'Severe': 3}
        if 'Severity' in processed_df.columns:
            processed_df['Severity_encoded'] = processed_df['Severity'].map(severity_mapping).fillna(2)

        # Lab ratios
        if 'ALT' in processed_df.columns and 'AST' in processed_df.columns:
            processed_df['alt_ast_ratio'] = (
                processed_df['ALT'] / (processed_df['AST'] + 1e-6)
            )

        # Symptom complexity features
        symptom_cols = [col for col in processed_df.columns if col.startswith('has_')]
        processed_df['total_symptoms'] = processed_df[symptom_cols].sum(axis=1)
        if 'Severity_encoded' in processed_df.columns:
            processed_df['symptom_severity_score'] = (
                processed_df['total_symptoms'] * processed_df['Severity_encoded']
            )

        return processed_df

    def interactive_testing(self):
        """Interactive testing interface"""
        print("\n" + "="*50)
        print("INTERACTIVE TESTING INTERFACE")
        print("="*50)
        
        test_cases = [
            {
                'name': 'Hepatitis A Case (Acute)',
                'symptoms': 'clay-colored stool, sudden nausea, vomiting, diarrhea, abdominal pain, low-grade fever'
            },
            {
                'name': 'Hepatitis C Case (Chronic)',
                'symptoms': 'weight loss, dark-colored urine, jaundice, ascites, spider angiomas, bleeding easily'
            },
            {
                'name': 'Mixed Symptoms (Challenging)',
                'symptoms': 'fatigue, jaundice, nausea, abdominal pain'
            },
            {
                'name': 'Hepatitis A Specific',
                'symptoms': 'clay-colored stool, joint pain, sudden nausea, vomiting, abdominal pain'
            },
            {
                'name': 'Hepatitis C Specific',
                'symptoms': 'weight loss, dark-colored urine, confusion, bleeding easily, spider angiomas'
            }
        ]
        
        print("🧪 Testing with predefined cases:")
        
        for i, case in enumerate(test_cases, 1):
            print(f"\n--- Test Case {i}: {case['name']} ---")
            print(f"Symptoms: {case['symptoms']}")
            
            result = self.predict_single_case(case['symptoms'])
            
            if result['error']:
                print(f"❌ Error: {result['error']}")
            else:
                print(f"🎯 Prediction: {result['prediction']}")
                print(f"📊 Confidence: {result['confidence']:.2%}")
                print(f"📈 Probability: {result['probability']:.4f}")
                print(f"⚖️ Threshold: {result['threshold_used']:.4f}")

    def run_complete_analysis(self):
        """Run the complete analysis pipeline"""
        print("\n🚀 STARTING COMPLETE ANALYSIS PIPELINE")
        print("=" * 60)
        
        try:
            # Step 1: Load and explore data
            self.load_and_explore_data()
            
            # Step 2: Advanced feature engineering
            X, y = self.advanced_feature_engineering()
            
            # Step 3: Train enhanced model
            self.train_enhanced_model()
            
            # Step 4: Optimize threshold
            self.optimize_threshold()
            
            # Step 5: Comprehensive evaluation
            self.comprehensive_evaluation()
            
            print("\n✅ COMPLETE ANALYSIS PIPELINE FINISHED")
            print("=" * 60)
            
            return True
            
        except Exception as e:
            print(f"❌ Error in complete analysis: {e}")
            return False

def main():
    """Main execution function"""
    print("🚀 INITIALIZING ENHANCED HEPATITIS SYSTEM...")
    print("=" * 60)
    
    # Initialize system
    enhanced_system = EnhancedHepatitisSystem()
    
    try:
        # Run complete analysis
        print("\n🔄 RUNNING COMPLETE ANALYSIS...")
        success = enhanced_system.run_complete_analysis()
        
        if success:
            # Interactive testing
            print("\n🔄 STARTING INTERACTIVE TESTING...")
            enhanced_system.interactive_testing()
            
            print("\n" + "🎉" * 20)
            print("ENHANCED SYSTEM EXECUTION COMPLETED!")
            print("🎉" * 20)
            print(f"\nFinal Results:")
            if 'optimized_threshold' in enhanced_system.results:
                print(f"- Model accuracy: {enhanced_system.results['optimized_threshold']['accuracy']:.1%}")
                print(f"- Optimal threshold: {enhanced_system.optimal_threshold:.4f}")
            print(f"- All outputs saved in: {enhanced_system.output_dir}")
        else:
            print("❌ Analysis pipeline failed")
            
    except Exception as e:
        print(f"❌ Error in main execution: {e}")
        raise e

if __name__ == "__main__":
    main()