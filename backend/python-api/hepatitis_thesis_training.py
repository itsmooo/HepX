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
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import (confusion_matrix, classification_report,
                           roc_curve, auc, precision_score,
                           recall_score, f1_score, accuracy_score)

class HepatitisThesisSystem:
    """
    Comprehensive Hepatitis Risk Prediction System for Academic Thesis
    """

    def __init__(self, output_dir='hepatitis_thesis_outputs'):
        self.output_dir = output_dir
        self.create_directory_structure()

        # Initialize data containers
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

        print("🎓 HEPATITIS RISK PREDICTION THESIS SYSTEM")
        print("=" * 60)
        print("Following Academic Research Standards")
        print("=" * 60)

    def create_directory_structure(self):
        """Create comprehensive directory structure for thesis outputs"""
        directories = [
            self.output_dir,
            f'{self.output_dir}/figures',
            f'{self.output_dir}/models',
            f'{self.output_dir}/data',
            f'{self.output_dir}/reports',
            f'{self.output_dir}/evaluation',
            f'{self.output_dir}/test_results'
        ]

        for directory in directories:
            os.makedirs(directory, exist_ok=True)

        print(f"📁 Created directory structure in: {self.output_dir}")

    def load_and_explore_data(self, file_path='data/hepatitis_dataset_R.csv'):
        """
        Load and perform initial exploration of the hepatitis dataset
        """
        print("\n" + "="*50)
        print("CHAPTER 5.2.1: DATA LOADING AND EXPLORATION")
        print("="*50)

        try:
            # Load the dataset
            self.df = pd.read_csv(file_path)

            # Filter only Hepatitis A and C
            self.df = self.df[self.df['HepatitisType'].isin(['Hepatitis A', 'Hepatitis C'])]

            print(f"✅ Dataset loaded successfully")
            print(f"📊 Dataset shape: {self.df.shape}")
            print(f"📋 Columns: {list(self.df.columns)}")

            # Save dataset info
            dataset_info = {
                'total_samples': len(self.df),
                'total_features': len(self.df.columns),
                'missing_values': self.df.isnull().sum().sum(),
                'columns': list(self.df.columns),
                'data_types': self.df.dtypes.to_dict()
            }

            with open(f'{self.output_dir}/data/dataset_info.pkl', 'wb') as f:
                pickle.dump(dataset_info, f)

            # Save sample data
            self.df.head(10).to_csv(f'{self.output_dir}/data/dataset_sample.csv', index=False)

            print(f"💾 Dataset information saved to: {self.output_dir}/data/")

            return self.df

        except Exception as e:
            print(f"❌ Error loading dataset: {e}")
            return None

    def preprocess_data(self):
        """
        Comprehensive data preprocessing following academic standards
        """
        print("\n" + "="*50)
        print("CHAPTER 5.2.2: DATA PREPROCESSING")
        print("="*50)

        if self.df is None:
            print("❌ No data loaded. Please load data first.")
            return None

        self.df_processed = self.df.copy()

        # Step 1: Handle missing values
        print("🔧 Step 1: Handling missing values...")
        missing_before = self.df_processed.isnull().sum().sum()

        for col in self.df_processed.columns:
            if self.df_processed[col].dtype == 'object':
                mode_value = self.df_processed[col].mode()
                fill_value = mode_value[0] if not mode_value.empty else 'Unknown'
                self.df_processed[col] = self.df_processed[col].fillna(fill_value)
            else:
                self.df_processed[col] = self.df_processed[col].fillna(self.df_processed[col].median())

        missing_after = self.df_processed.isnull().sum().sum()
        print(f"   Missing values before: {missing_before}")
        print(f"   Missing values after: {missing_after}")

        # Step 2: Create target variable
        print("🎯 Step 2: Creating and encoding target variable...")
        self.target_col = 'HepatitisType'
        self.label_encoder = LabelEncoder()
        self.df_processed[self.target_col] = self.label_encoder.fit_transform(
            self.df_processed[self.target_col].astype(str))
        print(f"   Encoded target variable '{self.target_col}'")

        # Step 3: Feature engineering
        print("🛠️ Step 3: Feature engineering...")
        symptoms_list = [
            'ascites', 'swelling', 'spider angiomas', 'itchy skin', 'jaundice',
            'dark-colored urine', 'bleeding easily', 'weight loss', 'fatigue',
            'confusion', 'not wanting to eat', 'bruising easily'
        ]

        for symptom in symptoms_list:
            self.df_processed[f'has_{symptom}'] = self.df_processed['Symptoms'].str.contains(
                symptom, case=False).astype(int)

        self.df_processed['DiagnosisDate'] = pd.to_datetime(self.df_processed['DiagnosisDate'])
        self.df_processed['DiagnosisYear'] = self.df_processed['DiagnosisDate'].dt.year
        self.df_processed['DiagnosisMonth'] = self.df_processed['DiagnosisDate'].dt.month

        # Step 4: Encode other categorical variables
        print("🏷️ Step 4: Encoding other categorical variables...")
        self.label_encoders_other = {}
        categorical_cols = self.df_processed.select_dtypes(include=['object']).columns

        for col in categorical_cols:
            if col not in [self.target_col, 'Symptoms']:
                le = LabelEncoder()
                self.df_processed[col + '_encoded'] = le.fit_transform(
                    self.df_processed[col].astype(str))
                self.label_encoders_other[col] = le

        print(f"   Encoded {len(self.label_encoders_other)} other categorical variables")

        # Step 5: Select features
        print("📊 Step 5: Feature selection...")
        self.feature_cols = []
        for col in self.df_processed.columns:
            if (col.endswith('_encoded') or
                col.startswith('has_') or
                (self.df_processed[col].dtype in ['int64', 'float64'] and
                 col not in [self.target_col, 'DiagnosisDate', 'PatientID', 'Symptoms'])):
                self.feature_cols.append(col)

        print(f"   Selected {len(self.feature_cols)} features for modeling")

        # Step 6: Prepare data for modeling
        X = self.df_processed[self.feature_cols]
        y = self.df_processed[self.target_col]

        print(f"📈 Target distribution (encoded):")
        print(y.value_counts())

        # Step 7: Split the data
        print("✂️ Step 7: Splitting data...")
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=42, stratify=y
        )
        X_train, X_val, y_train, y_val = train_test_split(
            X_train, y_train, test_size=0.2, random_state=42, stratify=y_train
        )

        # Step 8: Scale features
        print("⚖️ Step 8: Feature scaling...")
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
            'target_column': self.target_col,
            'data_splits': {
                'train_size': len(self.X_train_scaled),
                'val_size': len(self.X_val_scaled),
                'test_size': len(self.X_test_scaled)
            }
        }

        with open(f'{self.output_dir}/models/preprocessing_artifacts.pkl', 'wb') as f:
            pickle.dump(preprocessing_artifacts, f)

        print(f"💾 Preprocessing artifacts saved")

        return X, y

    def build_model_architecture(self):
        """
        Build the deep neural network following EfficientNet principles
        """
        print("\n" + "="*50)
        print("CHAPTER 5.2.3: MODEL ARCHITECTURE DESIGN")
        print("="*50)

        if self.X_train_scaled is None:
            print("❌ No preprocessed data available. Please preprocess data first.")
            return None

        print("🏗️ Building Deep Neural Network Architecture...")

        # Build model architecture
        self.model = Sequential([
            Dense(128, activation='relu', input_shape=(self.X_train_scaled.shape[1],), name='input_dense'),
            BatchNormalization(name='input_batch_norm'),
            Dropout(0.3, name='input_dropout'),

            Dense(64, activation='relu', name='hidden_dense_1'),
            BatchNormalization(name='hidden_batch_norm_1'),
            Dropout(0.2, name='hidden_dropout_1'),

            Dense(32, activation='relu', name='hidden_dense_2'),
            Dropout(0.1, name='hidden_dropout_2'),

            Dense(1, activation='sigmoid', name='output_dense')
        ], name='HepatitisRiskPredictor')

        # Compile model
        self.model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss='binary_crossentropy',
            metrics=['accuracy', tf.keras.metrics.Precision(), tf.keras.metrics.Recall()]
        )

        print("✅ Model architecture built successfully")
        print("\n📋 Model Summary:")
        self.model.summary()

        # Save model architecture
        model_config = {
            'architecture': 'Deep Neural Network',
            'layers': [
                {'type': 'Dense', 'units': 128, 'activation': 'relu'},
                {'type': 'BatchNormalization'},
                {'type': 'Dropout', 'rate': 0.3},
                {'type': 'Dense', 'units': 64, 'activation': 'relu'},
                {'type': 'BatchNormalization'},
                {'type': 'Dropout', 'rate': 0.2},
                {'type': 'Dense', 'units': 32, 'activation': 'relu'},
                {'type': 'Dropout', 'rate': 0.1},
                {'type': 'Dense', 'units': 1, 'activation': 'sigmoid'}
            ],
            'total_parameters': self.model.count_params(),
            'optimizer': 'Adam',
            'learning_rate': 0.001,
            'loss_function': 'binary_crossentropy'
        }

        with open(f'{self.output_dir}/models/model_architecture.pkl', 'wb') as f:
            pickle.dump(model_config, f)

        # Generate architecture diagram
        try:
            plot_model(
                self.model,
                to_file=f'{self.output_dir}/figures/model_architecture.png',
                show_shapes=True,
                show_layer_names=True,
                dpi=300
            )
            print(f"📊 Architecture diagram saved: {self.output_dir}/figures/model_architecture.png")
        except Exception as e:
            print(f"⚠️ Could not save architecture diagram: {e}")

        return self.model

    def train_model(self):
        """
        Train the deep learning model with proper callbacks and monitoring
        """
        print("\n" + "="*50)
        print("CHAPTER 5.2.4: MODEL TRAINING PROCESS")
        print("="*50)

        if self.model is None:
            print("❌ No model built. Please build model first.")
            return None

        print("🚀 Starting model training...")

        # Define callbacks
        callbacks = [
            EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True),
            ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-7),
            ModelCheckpoint(
                filepath=f'{self.output_dir}/models/best_model.keras',
                monitor='val_loss',
                save_best_only=True
            )
        ]

        # Train the model
        start_time = datetime.now()

        self.history = self.model.fit(
            self.X_train_scaled, self.y_train,
            validation_data=(self.X_val_scaled, self.y_val),
            epochs=100,
            batch_size=32,
            callbacks=callbacks,
            verbose=1
        )

        end_time = datetime.now()
        training_time = end_time - start_time

        print(f"✅ Training completed in {training_time}")
        print(f"📊 Total epochs: {len(self.history.history['loss'])}")

        # Save training history
        training_results = {
            'history': self.history.history,
            'training_time': str(training_time),
            'total_epochs': len(self.history.history['loss']),
            'best_val_accuracy': max(self.history.history['val_accuracy']),
            'best_val_loss': min(self.history.history['val_loss'])
        }

        with open(f'{self.output_dir}/models/training_history.pkl', 'wb') as f:
            pickle.dump(training_results, f)

        # Save the final model
        self.model.save(f'{self.output_dir}/models/hepatitis_model.keras')
        print(f"💾 Model saved to: {self.output_dir}/models/hepatitis_model.keras")

        return self.history

    def evaluate_model_performance(self):
        """
        Comprehensive evaluation following academic standards
        """
        print("\n" + "="*50)
        print("CHAPTER 5.2.5: MODEL EVALUATION")
        print("="*50)

        if self.model is None or self.history is None:
            print("❌ No trained model available. Please train model first.")
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

        # Store results
        self.results = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'auc_roc': roc_auc,
            'y_pred': y_pred,
            'y_pred_proba': y_pred_proba,
            'fpr': fpr,
            'tpr': tpr
        }

        print("🎯 PERFORMANCE METRICS:")
        print(f"   Accuracy:  {accuracy:.1%}")
        print(f"   Precision: {precision:.1%}")
        print(f"   Recall:    {recall:.1%}")
        print(f"   F1-Score:  {f1:.1%}")
        print(f"   AUC-ROC:   {roc_auc:.3f}")

        # Generate classification report
        y_test_decoded = self.label_encoder.inverse_transform(self.y_test)
        y_pred_decoded = self.label_encoder.inverse_transform(y_pred.flatten())

        class_report = classification_report(
            y_test_decoded, y_pred_decoded,
            target_names=self.label_encoder.classes_,
            output_dict=True
        )

        # Save classification report
        report_df = pd.DataFrame(class_report).transpose()
        report_df.to_csv(f'{self.output_dir}/evaluation/classification_report.csv')

        # Plot confusion matrix
        cm = confusion_matrix(y_test_decoded, y_pred_decoded)
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=self.label_encoder.classes_,
                    yticklabels=self.label_encoder.classes_)
        plt.title('Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.savefig(f'{self.output_dir}/figures/confusion_matrix.png')
        plt.close()

        print(f"💾 Evaluation results saved to: {self.output_dir}/evaluation/")

        return self.results

    def test_model_with_sample_cases(self):
        """
        Test the trained model with carefully designed test cases
        """
        print("\n" + "="*50)
        print("CHAPTER 5.2.6: MODEL TESTING WITH SAMPLE CASES")
        print("="*50)

        if self.model is None:
            try:
                self.model = load_model(f'{self.output_dir}/models/best_model.keras')
                print("✅ Loaded saved model for testing")
            except Exception as e:
                print(f"❌ Error loading model: {e}")
                return None

        # Define test cases
        test_cases = [
            # Case 1: Typical Hepatitis A case
            {
                'symptoms': ['fatigue', 'jaundice', 'nausea', 'abdominal pain'],
                'symptom_count': 4,
                'severity': 'Moderate',
                'diagnosis_date': '2023-05-15',
                'treatment': 'Supportive care',
                'expected_type': 'Hepatitis A'
            },
            # Case 2: Typical Hepatitis C case
            {
                'symptoms': ['fatigue', 'jaundice', 'dark-colored urine', 'abdominal pain', 'weight loss'],
                'symptom_count': 5,
                'severity': 'Severe',
                'diagnosis_date': '2023-06-20',
                'treatment': 'Antiviral therapy',
                'expected_type': 'Hepatitis C'
            },
            # Case 3: Borderline case
            {
                'symptoms': ['fatigue', 'jaundice'],
                'symptom_count': 2,
                'severity': 'Mild',
                'diagnosis_date': '2023-04-10',
                'treatment': 'Supportive care',
                'expected_type': 'Hepatitis A'
            }
        ]

        test_results = []
        
        print("\n🧪 Testing model with sample cases...")
        for i, case in enumerate(test_cases, 1):
            print(f"\n🔍 Test Case {i}:")
            print(f"   Symptoms: {case['symptoms']}")
            print(f"   Expected Type: {case['expected_type']}")

            predicted_type = self.predict_hepatitis_type(
                symptoms=case['symptoms'],
                symptom_count=case['symptom_count'],
                severity=case['severity'],
                diagnosis_date_str=case['diagnosis_date'],
                treatment=case['treatment']
            )

            is_correct = predicted_type == case['expected_type']
            result = "✅ CORRECT" if is_correct else "❌ INCORRECT"
            
            print(f"   Predicted Type: {predicted_type}")
            print(f"   Result: {result}")

            test_results.append({
                'case_id': i,
                'symptoms': ', '.join(case['symptoms']),
                'expected_type': case['expected_type'],
                'predicted_type': predicted_type,
                'is_correct': is_correct
            })

        # Save test results
        test_results_df = pd.DataFrame(test_results)
        test_results_df.to_csv(f'{self.output_dir}/test_results/sample_case_results.csv', index=False)
        
        # Calculate accuracy
        accuracy = test_results_df['is_correct'].mean()
        print(f"\n📊 Test Case Accuracy: {accuracy:.1%}")
        
        return test_results

    def predict_hepatitis_type(self, symptoms, symptom_count, severity, diagnosis_date_str, treatment):
        """
        Predict Hepatitis type for new input data
        """
        if self.model is None:
            try:
                self.model = load_model(f'{self.output_dir}/models/best_model.keras')
                with open(f'{self.output_dir}/models/preprocessing_artifacts.pkl', 'rb') as f:
                    artifacts = pickle.load(f)
                self.scaler = artifacts['scaler']
                self.label_encoder = artifacts['label_encoder_target']
                self.feature_cols = artifacts['feature_columns']
                self.label_encoders_other = artifacts['label_encoders_other']
            except Exception as e:
                print(f"❌ Error loading model or artifacts: {e}")
                return "Error: Model not available"

        # Create input DataFrame
        input_data = {
            'Symptoms': ', '.join(symptoms),
            'SymptomCount': symptom_count,
            'Severity': severity,
            'DiagnosisDate': diagnosis_date_str,
            'Treatment': treatment
        }
        input_df = pd.DataFrame([input_data])

        # Preprocess input
        processed_input = input_df.copy()

        # Feature engineering
        symptoms_list = [
            'ascites', 'swelling', 'spider angiomas', 'itchy skin', 'jaundice',
            'dark-colored urine', 'bleeding easily', 'weight loss', 'fatigue',
            'confusion', 'not wanting to eat', 'bruising easily'
        ]
        for symptom in symptoms_list:
            processed_input[f'has_{symptom}'] = processed_input['Symptoms'].str.contains(
                symptom, case=False).astype(int)

        processed_input['DiagnosisDate'] = pd.to_datetime(processed_input['DiagnosisDate'])
        processed_input['DiagnosisYear'] = processed_input['DiagnosisDate'].dt.year
        processed_input['DiagnosisMonth'] = processed_input['DiagnosisDate'].dt.month

        # Encode categorical variables
        for col in self.label_encoders_other:
            if col in processed_input.columns:
                try:
                    processed_input[col + '_encoded'] = self.label_encoders_other[col].transform(
                        processed_input[col].astype(str))
                except ValueError:
                    processed_input[col + '_encoded'] = -1  # Handle unseen labels

        # Handle Severity column specifically - convert to numeric
        if 'Severity' in processed_input.columns:
            severity_mapping = {'Mild': 1, 'Moderate': 2, 'Severe': 3}
            processed_input['Severity'] = processed_input['Severity'].map(severity_mapping).fillna(1)

        # Select features and scale
        input_features = processed_input.reindex(columns=self.feature_cols, fill_value=0)
        input_scaled = self.scaler.transform(input_features)

        # Make prediction
        prediction_proba = self.model.predict(input_scaled, verbose=0)[0][0]
        prediction = (prediction_proba > 0.5).astype(int)
        predicted_type = self.label_encoder.inverse_transform([prediction])[0]

        return predicted_type

    def run_complete_thesis_analysis(self):
        """
        Execute the complete thesis analysis pipeline
        """
        print("\n" + "="*60)
        print("EXECUTING COMPLETE THESIS ANALYSIS PIPELINE")
        print("="*60)

        try:
            # Step 1: Data Loading and Exploration
            self.load_and_explore_data()

            # Step 2: Data Preprocessing
            self.preprocess_data()

            # Step 3: Model Architecture Design
            self.build_model_architecture()

            # Step 4: Model Training
            self.train_model()

            # Step 5: Model Evaluation
            self.evaluate_model_performance()

            # Step 6: Test with sample cases
            self.test_model_with_sample_cases()

            print("\n" + "🎉" * 20)
            print("THESIS ANALYSIS COMPLETED SUCCESSFULLY!")
            print("🎉" * 20)
            print(f"\nAll outputs saved in: {self.output_dir}")

        except Exception as e:
            print(f"❌ Error in thesis analysis pipeline: {e}")
            raise e

def main():
    print("🎓 INITIALIZING HEPATITIS THESIS SYSTEM...")
    thesis_system = HepatitisThesisSystem(output_dir='hepatitis_thesis_outputs')
    thesis_system.run_complete_thesis_analysis()

    # Example prediction
    symptoms = ['fatigue', 'jaundice', 'dark-colored urine']
    predicted_type = thesis_system.predict_hepatitis_type(
        symptoms=symptoms,
        symptom_count=len(symptoms),
        severity='Moderate',
        diagnosis_date_str='2023-10-15',
        treatment='Supportive care'
    )
    print(f"\nExample Prediction for symptoms {symptoms}: {predicted_type}")

    print("\n🎉 THESIS SYSTEM EXECUTION COMPLETED!")

if __name__ == "__main__":
    main() 