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
from sklearn.metrics import (confusion_matrix, classification_report,
                           roc_curve, auc, precision_score,
                           recall_score, f1_score, accuracy_score)
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression

class ImprovedHepatitisSystem:
    """
    Improved Hepatitis Risk Prediction System with Better Accuracy
    """

    def __init__(self, output_dir='improved_hepatitis_outputs'):
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

        print("🚀 IMPROVED HEPATITIS PREDICTION SYSTEM")
        print("=" * 60)
        print("Focusing on Higher Accuracy")
        print("=" * 60)

    def create_directory_structure(self):
        """Create directory structure"""
        directories = [
            self.output_dir,
            f'{self.output_dir}/figures',
            f'{self.output_dir}/models',
            f'{self.output_dir}/data',
            f'{self.output_dir}/evaluation'
        ]
        for directory in directories:
            os.makedirs(directory, exist_ok=True)
        print(f"📁 Created directory structure in: {self.output_dir}")

    def load_and_explore_data(self, file_path='data/hepatitis_dataset_R.csv'):
        """Load and explore the dataset"""
        print("\n" + "="*50)
        print("DATA LOADING AND EXPLORATION")
        print("="*50)

        try:
            self.df = pd.read_csv(file_path)
            print(f"✅ Dataset loaded successfully")
            print(f"📊 Dataset shape: {self.df.shape}")
            print(f"📋 Columns: {list(self.df.columns)}")
            
            # Check target distribution
            print(f"\n🎯 Target Distribution:")
            target_dist = self.df['HepatitisType'].value_counts()
            print(target_dist)
            print(f"Class balance: {target_dist[0]/(target_dist[0]+target_dist[1]):.1%} vs {target_dist[1]/(target_dist[0]+target_dist[1]):.1%}")
            
            return self.df
        except Exception as e:
            print(f"❌ Error loading dataset: {e}")
            return None

    def advanced_feature_engineering(self):
        """Advanced feature engineering for better accuracy"""
        print("\n" + "="*50)
        print("ADVANCED FEATURE ENGINEERING")
        print("="*50)

        if self.df is None:
            print("❌ No data loaded. Please load data first.")
            return None

        self.df_processed = self.df.copy()

        # Step 1: Handle missing values
        print("🔧 Step 1: Handling missing values...")
        for col in self.df_processed.columns:
            if self.df_processed[col].dtype == 'object':
                mode_value = self.df_processed[col].mode()
                fill_value = mode_value[0] if not mode_value.empty else 'Unknown'
                self.df_processed[col] = self.df_processed[col].fillna(fill_value)
            else:
                self.df_processed[col] = self.df_processed[col].fillna(self.df_processed[col].median())

        # Step 2: Create target variable
        print("🎯 Step 2: Creating target variable...")
        self.target_col = 'HepatitisType'
        self.label_encoder = LabelEncoder()
        self.df_processed[self.target_col] = self.label_encoder.fit_transform(
            self.df_processed[self.target_col].astype(str))
        print(f"   Encoded target: {self.label_encoder.classes_}")

        # Step 3: Advanced symptom feature engineering
        print("🛠️ Step 3: Advanced symptom feature engineering...")
        
        # Extract symptoms from text
        symptoms_list = [
            'ascites', 'swelling', 'spider angiomas', 'itchy skin', 'jaundice',
            'dark-colored urine', 'bleeding easily', 'weight loss', 'fatigue',
            'confusion', 'not wanting to eat', 'bruising easily', 'nausea',
            'vomiting', 'fever', 'abdominal pain', 'joint pain', 'loss of appetite'
        ]

        for symptom in symptoms_list:
            self.df_processed[f'has_{symptom}'] = self.df_processed['Symptoms'].str.contains(
                symptom, case=False).astype(int)

        # Create symptom combinations
        self.df_processed['has_jaundice_fatigue'] = (
            self.df_processed['has_jaundice'] & self.df_processed['has_fatigue']
        ).astype(int)
        
        self.df_processed['has_jaundice_dark_urine'] = (
            self.df_processed['has_jaundice'] & self.df_processed['has_dark-colored urine']
        ).astype(int)

        # Step 4: Date features
        print("📅 Step 4: Date feature engineering...")
        self.df_processed['DiagnosisDate'] = pd.to_datetime(self.df_processed['DiagnosisDate'])
        self.df_processed['DiagnosisYear'] = self.df_processed['DiagnosisDate'].dt.year
        self.df_processed['DiagnosisMonth'] = self.df_processed['DiagnosisDate'].dt.month
        self.df_processed['DiagnosisDay'] = self.df_processed['DiagnosisDate'].dt.day
        self.df_processed['DiagnosisDayOfWeek'] = self.df_processed['DiagnosisDate'].dt.dayofweek

        # Step 5: Severity encoding
        print("⚠️ Step 5: Severity encoding...")
        severity_mapping = {'Mild': 1, 'Moderate': 2, 'Severe': 3}
        self.df_processed['Severity_encoded'] = self.df_processed['Severity'].map(severity_mapping).fillna(2)

        # Step 6: Treatment encoding
        print("💊 Step 6: Treatment encoding...")
        self.label_encoders_other = {}
        categorical_cols = self.df_processed.select_dtypes(include=['object']).columns
        
        for col in categorical_cols:
            if col not in [self.target_col, 'Symptoms']:
                le = LabelEncoder()
                self.df_processed[col + '_encoded'] = le.fit_transform(
                    self.df_processed[col].astype(str))
                self.label_encoders_other[col] = le

        # Step 7: Feature selection
        print("📊 Step 7: Feature selection...")
        self.feature_cols = []
        for col in self.df_processed.columns:
            if (col.endswith('_encoded') or
                col.startswith('has_') or
                (self.df_processed[col].dtype in ['int64', 'float64'] and
                 col not in [self.target_col, 'DiagnosisDate', 'PatientID', 'Symptoms'])):
                self.feature_cols.append(col)

        print(f"   Selected {len(self.feature_cols)} features for modeling")

        # Step 8: Prepare data
        X = self.df_processed[self.feature_cols]
        y = self.df_processed[self.target_col]

        print(f"📈 Target distribution (encoded):")
        print(y.value_counts())

        # Step 9: Split data with stratification
        print("✂️ Step 9: Splitting data...")
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        X_train, X_val, y_train, y_val = train_test_split(
            X_train, y_train, test_size=0.2, random_state=42, stratify=y_train
        )

        # Step 10: Scale features
        print("⚖️ Step 10: Feature scaling...")
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

        # Save preprocessing artifacts
        preprocessing_artifacts = {
            'scaler': self.scaler,
            'label_encoder_target': self.label_encoder,
            'label_encoders_other': self.label_encoders_other,
            'feature_columns': self.feature_cols,
            'target_column': self.target_col
        }

        with open(f'{self.output_dir}/models/preprocessing_artifacts.pkl', 'wb') as f:
            pickle.dump(preprocessing_artifacts, f)

        print(f"💾 Preprocessing artifacts saved")
        return X, y

    def compare_models(self):
        """Compare different models to find the best one"""
        print("\n" + "="*50)
        print("MODEL COMPARISON")
        print("="*50)

        models = {
            'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
            'SVM': SVC(probability=True, random_state=42),
            'Logistic Regression': LogisticRegression(random_state=42)
        }

        results = {}
        
        for name, model in models.items():
            print(f"\n🔍 Testing {name}...")
            
            # Cross-validation
            cv_scores = cross_val_score(model, self.X_train_scaled, self.y_train, cv=5)
            print(f"   CV Accuracy: {cv_scores.mean():.3f} (+/- {cv_scores.std() * 2:.3f})")
            
            # Train and evaluate
            model.fit(self.X_train_scaled, self.y_train)
            y_pred = model.predict(self.X_test_scaled)
            accuracy = accuracy_score(self.y_test, y_pred)
            
            results[name] = {
                'model': model,
                'cv_accuracy': cv_scores.mean(),
                'test_accuracy': accuracy
            }
            
            print(f"   Test Accuracy: {accuracy:.3f}")

        # Find best model
        best_model_name = max(results.keys(), key=lambda x: results[x]['test_accuracy'])
        best_model = results[best_model_name]['model']
        
        print(f"\n🏆 Best Model: {best_model_name}")
        print(f"   Test Accuracy: {results[best_model_name]['test_accuracy']:.3f}")
        
        return best_model, results

    def build_improved_neural_network(self):
        """Build an improved neural network"""
        print("\n" + "="*50)
        print("IMPROVED NEURAL NETWORK ARCHITECTURE")
        print("="*50)

        if self.X_train_scaled is None:
            print("❌ No preprocessed data available.")
            return None

        print("🏗️ Building Improved Neural Network...")

        # Build improved architecture
        self.model = Sequential([
            Dense(256, activation='relu', input_shape=(self.X_train_scaled.shape[1],)),
            BatchNormalization(),
            Dropout(0.4),
            
            Dense(128, activation='relu'),
            BatchNormalization(),
            Dropout(0.3),
            
            Dense(64, activation='relu'),
            BatchNormalization(),
            Dropout(0.2),
            
            Dense(32, activation='relu'),
            Dropout(0.1),
            
            Dense(1, activation='sigmoid')
        ])

        # Compile with better optimizer
        self.model.compile(
            optimizer=Adam(learning_rate=0.0005),
            loss='binary_crossentropy',
            metrics=['accuracy', tf.keras.metrics.Precision(), tf.keras.metrics.Recall()]
        )

        print("✅ Improved model architecture built")
        print(f"📊 Total parameters: {self.model.count_params():,}")
        
        return self.model

    def train_improved_model(self):
        """Train the improved model"""
        print("\n" + "="*50)
        print("IMPROVED MODEL TRAINING")
        print("="*50)

        if self.model is None:
            print("❌ No model built.")
            return None

        print("🚀 Starting improved training...")

        # Better callbacks
        callbacks = [
            EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True),
            ReduceLROnPlateau(monitor='val_loss', factor=0.3, patience=8, min_lr=1e-7),
            ModelCheckpoint(
                filepath=f'{self.output_dir}/models/best_improved_model.keras',
                monitor='val_accuracy',
                save_best_only=True
            )
        ]

        # Train with more epochs and better batch size
        start_time = datetime.now()
        
        self.history = self.model.fit(
            self.X_train_scaled, self.y_train,
            validation_data=(self.X_val_scaled, self.y_val),
            epochs=200,
            batch_size=16,
            callbacks=callbacks,
            verbose=1
        )

        end_time = datetime.now()
        training_time = end_time - start_time

        print(f"✅ Training completed in {training_time}")
        print(f"📊 Total epochs: {len(self.history.history['loss'])}")
        
        # Save model
        self.model.save(f'{self.output_dir}/models/improved_hepatitis_model.keras')
        print(f"💾 Model saved")

        return self.history

    def comprehensive_evaluation(self):
        """Comprehensive model evaluation"""
        print("\n" + "="*50)
        print("COMPREHENSIVE EVALUATION")
        print("="*50)

        if self.model is None:
            print("❌ No trained model available.")
            return None

        print("📊 Evaluating model performance...")

        # Get predictions
        y_pred_proba = self.model.predict(self.X_test_scaled, verbose=0)
        y_pred = (y_pred_proba > 0.5).astype(int)

        # Calculate metrics
        accuracy = accuracy_score(self.y_test, y_pred)
        precision = precision_score(self.y_test, y_pred)
        recall = recall_score(self.y_test, y_pred)
        f1 = f1_score(self.y_test, y_pred)

        # ROC AUC
        fpr, tpr, _ = roc_curve(self.y_test, y_pred_proba)
        roc_auc = auc(fpr, tpr)

        print("🎯 PERFORMANCE METRICS:")
        print(f"   Accuracy:  {accuracy:.1%}")
        print(f"   Precision: {precision:.1%}")
        print(f"   Recall:    {recall:.1%}")
        print(f"   F1-Score:  {f1:.1%}")
        print(f"   AUC-ROC:   {roc_auc:.3f}")

        # Save results
        self.results = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'auc_roc': roc_auc
        }

        # Generate classification report
        y_test_decoded = self.label_encoder.inverse_transform(self.y_test)
        y_pred_decoded = self.label_encoder.inverse_transform(y_pred.flatten())

        class_report = classification_report(
            y_test_decoded, y_pred_decoded,
            target_names=self.label_encoder.classes_,
            output_dict=True
        )

        # Save report
        report_df = pd.DataFrame(class_report).transpose()
        report_df.to_csv(f'{self.output_dir}/evaluation/improved_classification_report.csv')

        # Plot confusion matrix
        cm = confusion_matrix(y_test_decoded, y_pred_decoded)
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=self.label_encoder.classes_,
                    yticklabels=self.label_encoder.classes_)
        plt.title('Improved Model - Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.savefig(f'{self.output_dir}/figures/improved_confusion_matrix.png')
        plt.close()

        print(f"💾 Evaluation results saved")

        return self.results

    def run_improved_analysis(self):
        """Run the complete improved analysis"""
        print("\n" + "="*60)
        print("RUNNING IMPROVED ANALYSIS PIPELINE")
        print("="*60)

        try:
            # Step 1: Load data
            self.load_and_explore_data()

            # Step 2: Advanced feature engineering
            self.advanced_feature_engineering()

            # Step 3: Compare models
            best_model, results = self.compare_models()

            # Step 4: Build improved neural network
            self.build_improved_neural_network()

            # Step 5: Train improved model
            self.train_improved_model()

            # Step 6: Comprehensive evaluation
            self.comprehensive_evaluation()

            print("\n" + "🎉" * 20)
            print("IMPROVED ANALYSIS COMPLETED!")
            print("🎉" * 20)
            print(f"\nAll outputs saved in: {self.output_dir}")

        except Exception as e:
            print(f"❌ Error in improved analysis: {e}")
            raise e

    def interactive_testing(self):
        """Interactive testing of the trained model"""
        print("\n" + "="*50)
        print("INTERACTIVE MODEL TESTING")
        print("="*50)

        if self.model is None:
            try:
                self.model = load_model(f'{self.output_dir}/models/best_improved_model.keras')
                print("✅ Loaded saved model for testing")
            except Exception as e:
                print(f"❌ Error loading model: {e}")
                return None

        print("\n🧪 Interactive Testing Mode")
        print("Enter symptoms and see predictions!")
        print("Type 'quit' to exit testing mode")
        print("-" * 50)

        while True:
            print("\n📝 Enter symptoms (comma-separated):")
            symptoms_input = input("Symptoms: ").strip()
            
            if symptoms_input.lower() == 'quit':
                break
            
            if not symptoms_input:
                print("❌ Please enter symptoms")
                continue

            # Parse symptoms
            symptoms = [s.strip() for s in symptoms_input.split(',')]
            
            print(f"🔍 Analyzing symptoms: {symptoms}")
            
            # Get prediction
            try:
                prediction, confidence = self.predict_single_case(symptoms)
                print(f"🎯 Prediction: {prediction}")
                print(f"📊 Confidence: {confidence:.1%}")
                
                # Show symptom analysis
                print(f"📋 Symptom count: {len(symptoms)}")
                print(f"🔍 Detected symptoms: {symptoms}")
                
            except Exception as e:
                print(f"❌ Error making prediction: {e}")

    def predict_single_case(self, symptoms):
        """Predict for a single case with symptoms"""
        if self.model is None:
            return "Error: Model not available", 0.0

        # Create input data
        input_data = {
            'Symptoms': ', '.join(symptoms),
            'SymptomCount': len(symptoms),
            'Severity': 'Moderate',  # Default severity
            'DiagnosisDate': '2023-10-15',  # Default date
            'Treatment': 'Supportive care'  # Default treatment
        }

        # Preprocess input
        processed_input = self.preprocess_single_input(input_data)
        
        # Make prediction
        prediction_proba = self.model.predict(processed_input, verbose=0)[0][0]
        prediction = (prediction_proba > 0.5).astype(int)
        
        # Decode prediction
        predicted_type = self.label_encoder.inverse_transform([prediction])[0]
        confidence = prediction_proba if prediction == 1 else 1 - prediction_proba
        
        return predicted_type, confidence

    def preprocess_single_input(self, input_data):
        """Preprocess a single input case"""
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

        # Severity encoding - fix the mapping issue
        severity_mapping = {'Mild': 1, 'Moderate': 2, 'Severe': 3}
        processed_input['Severity_encoded'] = processed_input['Severity'].map(severity_mapping).fillna(2)
        
        # Remove the original Severity column to avoid conflicts
        if 'Severity' in processed_input.columns:
            processed_input = processed_input.drop('Severity', axis=1)

        # Treatment encoding
        for col in self.label_encoders_other:
            if col in processed_input.columns:
                try:
                    processed_input[col + '_encoded'] = self.label_encoders_other[col].transform(
                        processed_input[col].astype(str))
                except ValueError:
                    processed_input[col + '_encoded'] = -1

        # Select features and scale
        input_features = processed_input.reindex(columns=self.feature_cols, fill_value=0)
        input_scaled = self.scaler.transform(input_features)
        
        return input_scaled

    def test_accuracy_with_sample_cases(self):
        """Test accuracy with predefined sample cases"""
        print("\n" + "="*50)
        print("ACCURACY TESTING WITH SAMPLE CASES")
        print("="*50)

        if self.model is None:
            try:
                self.model = load_model(f'{self.output_dir}/models/best_improved_model.keras')
                print("✅ Loaded saved model for testing")
            except Exception as e:
                print(f"❌ Error loading model: {e}")
                return None

        # Define test cases
        test_cases = [
            {
                'symptoms': ['fatigue', 'jaundice', 'nausea', 'abdominal pain'],
                'expected': 'Hepatitis A',
                'description': 'Typical Hepatitis A symptoms'
            },
            {
                'symptoms': ['fatigue', 'jaundice', 'dark-colored urine', 'weight loss'],
                'expected': 'Hepatitis C',
                'description': 'Typical Hepatitis C symptoms'
            },
            {
                'symptoms': ['fatigue', 'jaundice'],
                'expected': 'Hepatitis A',
                'description': 'Mild symptoms'
            },
            {
                'symptoms': ['fatigue', 'jaundice', 'dark-colored urine', 'abdominal pain', 'weight loss', 'nausea'],
                'expected': 'Hepatitis C',
                'description': 'Severe symptoms'
            },
            {
                'symptoms': ['fatigue', 'nausea', 'vomiting'],
                'expected': 'Hepatitis A',
                'description': 'Gastrointestinal symptoms'
            }
        ]

        correct_predictions = 0
        total_cases = len(test_cases)

        print(f"\n🧪 Testing {total_cases} sample cases...")
        print("-" * 60)

        for i, case in enumerate(test_cases, 1):
            print(f"\n🔍 Test Case {i}: {case['description']}")
            print(f"   Symptoms: {case['symptoms']}")
            print(f"   Expected: {case['expected']}")
            
            try:
                prediction, confidence = self.predict_single_case(case['symptoms'])
                is_correct = prediction == case['expected']
                
                if is_correct:
                    correct_predictions += 1
                    result = "✅ CORRECT"
                else:
                    result = "❌ INCORRECT"
                
                print(f"   Predicted: {prediction}")
                print(f"   Confidence: {confidence:.1%}")
                print(f"   Result: {result}")
                
            except Exception as e:
                print(f"   ❌ Error: {e}")

        accuracy = correct_predictions / total_cases
        print(f"\n📊 Sample Case Testing Results:")
        print(f"   Correct predictions: {correct_predictions}/{total_cases}")
        print(f"   Accuracy: {accuracy:.1%}")

        return accuracy

def main():
    print("🚀 INITIALIZING IMPROVED HEPATITIS SYSTEM...")
    improved_system = ImprovedHepatitisSystem()
    
    # Run the complete analysis
    improved_system.run_improved_analysis()
    
    # Test accuracy with sample cases
    print("\n" + "="*60)
    print("TESTING MODEL ACCURACY")
    print("="*60)
    improved_system.test_accuracy_with_sample_cases()
    
    # Interactive testing
    print("\n" + "="*60)
    print("INTERACTIVE TESTING")
    print("="*60)
    improved_system.interactive_testing()
    
    print("\n🎉 IMPROVED SYSTEM EXECUTION COMPLETED!")

if __name__ == "__main__":
    main() 