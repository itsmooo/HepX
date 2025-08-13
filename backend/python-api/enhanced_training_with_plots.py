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

# Machine Learning Libraries
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.metrics import (confusion_matrix, classification_report,
                           precision_score, recall_score, f1_score, accuracy_score)
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression

# Hyperparameter tuning
import optuna

class EnhancedHepatitisSystemWithPlots:
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
        
        # Set style for plots
        plt.style.use('seaborn-v0_8')
        sns.set_palette("husl")
        
        print("🚀 ENHANCED HEPATITIS PREDICTION SYSTEM WITH PLOTS")
        print("=" * 70)
        print("Maximum Accuracy with Advanced Features and Comprehensive Visualization")
        print("=" * 70)

    def create_directory_structure(self):
        """Create comprehensive directory structure"""
        directories = [
            self.output_dir,
            f'{self.output_dir}/figures',
            f'{self.output_dir}/models',
            f'{self.output_dir}/data',
            f'{self.output_dir}/evaluation',
            f'{self.output_dir}/preprocessing',
            f'{self.output_dir}/hyperparameter_tuning'
        ]
        for directory in directories:
            os.makedirs(directory, exist_ok=True)
        print(f"📁 Created directory structure in: {self.output_dir}")

    def load_and_explore_data(self, file_path='../data/hepatitis_dataset_5k.csv'):
        """Load and comprehensively explore the dataset"""
        print("\n" + "="*50)
        print("DATA LOADING AND EXPLORATION")
        print("="*50)
        
        try:
            if os.path.exists(file_path):
                self.df = pd.read_csv(file_path)
                print(f"✅ Dataset loaded from: {file_path}")
            else:
                print(f"❌ Dataset not found at: {file_path}")
                return None
            
            print(f"📊 Dataset shape: {self.df.shape}")
            print(f"📋 Columns: {list(self.df.columns)}")
            
            # Store required columns for later use
            self.required_columns = list(self.df.columns)
            
            # Check target distribution
            if 'HepatitisType' in self.df.columns:
                print(f"\n🎯 Target Distribution:")
                target_dist = self.df['HepatitisType'].value_counts()
                print(target_dist)
                print(f"Class balance: {target_dist.iloc[0]/(target_dist.sum()):.1%} vs {target_dist.iloc[1]/(target_dist.sum()):.1%}")
                
                # Plot target distribution
                self.plot_target_distribution(target_dist)
            
            return self.df
            
        except Exception as e:
            print(f"❌ Error loading dataset: {e}")
            return None

    def plot_target_distribution(self, target_dist):
        """Plot target distribution"""
        plt.figure(figsize=(10, 6))
        colors = ['#FF6B6B', '#4ECDC4']
        bars = plt.bar(target_dist.index, target_dist.values, color=colors, alpha=0.8)
        
        # Add value labels on bars
        for bar, value in zip(bars, target_dist.values):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01*max(target_dist.values),
                    f'{value}', ha='center', va='bottom', fontweight='bold')
        
        plt.title('Hepatitis Type Distribution in Dataset', fontsize=16, fontweight='bold')
        plt.xlabel('Hepatitis Type', fontsize=12)
        plt.ylabel('Number of Cases', fontsize=12)
        plt.xticks(rotation=45)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        # Save plot
        plt.savefig(f'{self.output_dir}/figures/target_distribution.png', dpi=300, bbox_inches='tight')
        plt.show()
        print(f"📊 Target distribution plot saved to: {self.output_dir}/figures/target_distribution.png")

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
        
        # Step 3: Enhanced symptom feature engineering
        print("🛠️ Step 3: Enhanced symptom feature engineering...")
        
        # Comprehensive symptom list
        symptoms_list = [
            'ascites', 'swelling', 'spider angiomas', 'itchy skin', 'jaundice',
            'dark-colored urine', 'bleeding easily', 'weight loss', 'fatigue',
            'confusion', 'not wanting to eat', 'bruising easily', 'nausea',
            'vomiting', 'fever', 'abdominal pain', 'joint pain', 'loss of appetite'
        ]
        
        for symptom in symptoms_list:
            self.df_processed[f'has_{symptom.replace(" ", "_").replace("-", "_")}'] = self.df_processed['Symptoms'].str.contains(
                symptom, case=False, na=False).astype(int)

        # Step 4: Hepatitis-specific signature combinations
        print("🧬 Step 4: Creating hepatitis-specific signatures...")
        
        # Hepatitis A signature (acute onset, gastrointestinal)
        self.df_processed['hepatitis_a_signature'] = (
            (self.df_processed['has_abdominal_pain'] | 
             self.df_processed['has_nausea'] |
             self.df_processed['has_vomiting']) &
            ~self.df_processed['has_weight_loss'] &
            ~self.df_processed['has_ascites']
        ).astype(int)
        
        # Hepatitis C signature (chronic symptoms, liver damage indicators)
        self.df_processed['hepatitis_c_signature'] = (
            (self.df_processed['has_weight_loss'] |
             self.df_processed['has_ascites'] |
             self.df_processed['has_spider_angiomas']) &
            (self.df_processed['has_dark_colored_urine'] |
             self.df_processed['has_bleeding_easily'])
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

        # Severity encoding
        if 'Severity' in self.df_processed.columns:
            severity_mapping = {'Mild': 1, 'Moderate': 2, 'Severe': 3}
            self.df_processed['Severity_encoded'] = self.df_processed['Severity'].map(severity_mapping).fillna(2)

        # Step 6: Categorical encoding
        print("🏷️ Step 6: Advanced categorical encoding...")
        self.label_encoders_other = {}
        categorical_cols = self.df_processed.select_dtypes(include=['object']).columns
        
        for col in categorical_cols:
            if col not in [self.target_col, 'Symptoms', 'DiagnosisDate']:
                le = LabelEncoder()
                # Handle unknown categories
                unique_values = list(self.df_processed[col].unique()) + ['UNK']
                le.fit(unique_values)
                self.df_processed[col + '_encoded'] = le.transform(
                    self.df_processed[col].astype(str))
                self.label_encoders_other[col] = le

        # Step 7: Feature selection
        print("🎯 Step 7: Intelligent feature selection...")
        self.feature_cols = []
        for col in self.df_processed.columns:
            if (col.endswith('_encoded') or
                col.startswith('has_') or
                col.endswith('_signature') or
                (self.df_processed[col].dtype in ['int64', 'float64'] and
                 col not in [self.target_col, 'PatientID'] and
                 not col.startswith('Diagnosis') and
                 col != 'Symptoms')):
                self.feature_cols.append(col)

        print(f"   Selected {len(self.feature_cols)} features for modeling")

        # Step 8: Data preparation
        X = self.df_processed[self.feature_cols]
        y = self.df_processed[self.target_col]

        print(f"📈 Final feature matrix shape: {X.shape}")
        print(f"📈 Target distribution (encoded):")
        print(y.value_counts())

        # Step 9: Stratified data splitting
        print("✂️ Step 9: Stratified data splitting...")
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        X_train, X_val, y_train, y_val = train_test_split(
            X_train, y_train, test_size=0.2, random_state=42, stratify=y_train
        )

        # Step 10: Feature scaling
        print("⚖️ Step 10: Robust feature scaling...")
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

        # Step 11: Save preprocessing artifacts
        print("💾 Step 11: Saving preprocessing artifacts...")
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

    def hyperparameter_tuning(self):
        """Perform hyperparameter tuning using Optuna"""
        print("\n" + "="*50)
        print("HYPERPARAMETER TUNING WITH OPTUNA")
        print("="*50)
        
        def objective(trial):
            # Define hyperparameter search space
            n_layers = trial.suggest_int('n_layers', 2, 5)
            units = []
            dropout_rates = []
            
            for i in range(n_layers):
                units.append(trial.suggest_int(f'units_layer_{i}', 32, 256))
                dropout_rates.append(trial.suggest_float(f'dropout_layer_{i}', 0.1, 0.5))
            
            learning_rate = trial.suggest_float('learning_rate', 1e-5, 1e-2, log=True)
            batch_size = trial.suggest_categorical('batch_size', [16, 32, 64, 128])
            
            # Build model
            model = Sequential()
            model.add(Dense(units[0], activation='relu', input_shape=(self.X_train_scaled.shape[1],)))
            model.add(BatchNormalization())
            model.add(Dropout(dropout_rates[0]))
            
            for i in range(1, n_layers):
                model.add(Dense(units[i], activation='relu'))
                model.add(BatchNormalization())
                model.add(Dropout(dropout_rates[i]))
            
            model.add(Dense(1, activation='sigmoid'))
            
            # Compile model
            model.compile(
                optimizer=Adam(learning_rate=learning_rate),
                loss='binary_crossentropy',
                metrics=['accuracy']
            )
            
            # Train model
            history = model.fit(
                self.X_train_scaled, self.y_train,
                validation_data=(self.X_val_scaled, self.y_val),
                epochs=50,
                batch_size=batch_size,
                verbose=0,
                callbacks=[EarlyStopping(patience=10, restore_best_weights=True)]
            )
            
            # Return validation accuracy
            return history.history['val_accuracy'][-1]
        
        # Create study
        study = optuna.create_study(direction='maximize')
        study.optimize(objective, n_trials=20)
        
        print(f"🎯 Best trial: {study.best_trial.value:.4f}")
        print(f"🎯 Best parameters: {study.best_trial.params}")
        
        # Save tuning results
        tuning_results = {
            'best_params': study.best_trial.params,
            'best_value': study.best_trial.value,
            'all_trials': [(trial.params, trial.value) for trial in study.trials]
        }
        
        with open(f'{self.output_dir}/hyperparameter_tuning/tuning_results.pkl', 'wb') as f:
            pickle.dump(tuning_results, f)
        
        # Plot optimization history
        self.plot_optimization_history(study)
        
        return study.best_trial.params

    def plot_optimization_history(self, study):
        """Plot hyperparameter optimization history"""
        plt.figure(figsize=(12, 8))
        
        # Plot optimization history
        plt.subplot(2, 2, 1)
        optuna.visualization.matplotlib.plot_optimization_history(study)
        plt.title('Optimization History', fontweight='bold')
        
        # Plot parameter importance
        plt.subplot(2, 2, 2)
        optuna.visualization.matplotlib.plot_param_importances(study)
        plt.title('Parameter Importance', fontweight='bold')
        
        # Plot parallel coordinate
        plt.subplot(2, 2, 3)
        optuna.visualization.matplotlib.plot_parallel_coordinate(study)
        plt.title('Parallel Coordinate', fontweight='bold')
        
        # Plot slice
        plt.subplot(2, 2, 4)
        optuna.visualization.matplotlib.plot_slice(study)
        plt.title('Parameter Slice', fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(f'{self.output_dir}/figures/hyperparameter_tuning_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()
        print(f"📊 Hyperparameter tuning plots saved to: {self.output_dir}/figures/hyperparameter_tuning_analysis.png")

    def build_enhanced_neural_network(self, best_params=None):
        """Build an enhanced neural network architecture"""
        print("\n" + "="*50)
        print("BUILDING ENHANCED NEURAL NETWORK")
        print("="*50)
        
        input_dim = self.X_train_scaled.shape[1]
        print(f"📊 Input dimensions: {input_dim}")
        
        if best_params is None:
            # Default architecture
            best_params = {
                'n_layers': 4,
                'units_layer_0': 128,
                'units_layer_1': 64,
                'units_layer_2': 32,
                'dropout_layer_0': 0.3,
                'dropout_layer_1': 0.2,
                'dropout_layer_2': 0.1,
                'learning_rate': 0.001,
                'batch_size': 32
            }
        
        # Build model with best parameters
        model = Sequential()
        model.add(Dense(best_params['units_layer_0'], activation='relu', input_shape=(input_dim,)))
        model.add(BatchNormalization())
        model.add(Dropout(best_params['dropout_layer_0']))
        
        for i in range(1, best_params['n_layers']):
            units_key = f'units_layer_{i}'
            dropout_key = f'dropout_layer_{i}'
            if units_key in best_params and dropout_key in best_params:
                model.add(Dense(best_params[units_key], activation='relu'))
                model.add(BatchNormalization())
                model.add(Dropout(best_params[dropout_key]))
        
        model.add(Dense(1, activation='sigmoid'))
        
        # Compile with best learning rate
        model.compile(
            optimizer=Adam(learning_rate=best_params['learning_rate']),
            loss='binary_crossentropy',
            metrics=['accuracy', 'precision', 'recall']
        )
        
        print("✅ Enhanced neural network built successfully")
        print(f"📊 Total parameters: {model.count_params():,}")
        
        return model, best_params

    def train_enhanced_model(self, best_params=None):
        """Train the enhanced model with advanced callbacks"""
        print("\n" + "="*50)
        print("TRAINING ENHANCED MODEL")
        print("="*50)
        
        # Build model
        self.model, self.best_params = self.build_enhanced_neural_network(best_params)
        
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
            batch_size=self.best_params.get('batch_size', 32),
            callbacks=callbacks,
            verbose=1
        )
        
        print("✅ Model training completed")
        return self.model, self.history

    def plot_training_curves(self):
        """Plot training accuracy and loss curves"""
        print("\n📊 Plotting training curves...")
        
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))
        
        # Accuracy plot
        axes[0].plot(self.history.history['accuracy'], label='Training Accuracy', linewidth=2)
        axes[0].plot(self.history.history['val_accuracy'], label='Validation Accuracy', linewidth=2)
        axes[0].set_title('Model Training Accuracy', fontsize=16, fontweight='bold')
        axes[0].set_xlabel('Epoch', fontsize=12)
        axes[0].set_ylabel('Accuracy', fontsize=12)
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # Loss plot
        axes[1].plot(self.history.history['loss'], label='Training Loss', linewidth=2)
        axes[1].plot(self.history.history['val_loss'], label='Validation Loss', linewidth=2)
        axes[1].set_title('Model Training Loss', fontsize=16, fontweight='bold')
        axes[1].set_xlabel('Epoch', fontsize=12)
        axes[1].set_ylabel('Loss', fontsize=12)
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f'{self.output_dir}/figures/training_curves.png', dpi=300, bbox_inches='tight')
        plt.show()
        print(f"📊 Training curves plot saved to: {self.output_dir}/figures/training_curves.png")

    def plot_confusion_matrix(self):
        """Plot confusion matrix of deep neural network predictions"""
        print("\n📊 Plotting confusion matrix...")
        
        # Get predictions
        y_pred_proba = self.model.predict(self.X_test_scaled, verbose=0)
        y_pred = (y_pred_proba > 0.5).astype(int)
        
        # Calculate confusion matrix
        cm = confusion_matrix(self.y_test, y_pred)
        
        # Plot confusion matrix
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                    xticklabels=self.label_encoder.classes_,
                    yticklabels=self.label_encoder.classes_)
        plt.title('Confusion Matrix of Deep Neural Network Predictions', fontsize=16, fontweight='bold')
        plt.xlabel('Predicted Label', fontsize=12)
        plt.ylabel('True Label', fontsize=12)
        plt.tight_layout()
        
        plt.savefig(f'{self.output_dir}/figures/confusion_matrix.png', dpi=300, bbox_inches='tight')
        plt.show()
        print(f"📊 Confusion matrix plot saved to: {self.output_dir}/figures/confusion_matrix.png")
        
        return cm

    def plot_precision_recall_f1(self):
        """Plot precision, recall, and F1-score per class"""
        print("\n📊 Plotting precision, recall, and F1-score...")
        
        # Get predictions
        y_pred_proba = self.model.predict(self.X_test_scaled, verbose=0)
        y_pred = (y_pred_proba > 0.5).astype(int)
        
        # Calculate metrics
        precision = precision_score(self.y_test, y_pred, average=None)
        recall = recall_score(self.y_test, y_pred, average=None)
        f1 = f1_score(self.y_test, y_pred, average=None)
        
        # Create plot
        x = np.arange(len(self.label_encoder.classes_))
        width = 0.25
        
        fig, ax = plt.subplots(figsize=(10, 6))
        bars1 = ax.bar(x - width, precision, width, label='Precision', alpha=0.8)
        bars2 = ax.bar(x, recall, width, label='Recall', alpha=0.8)
        bars3 = ax.bar(x + width, f1, width, label='F1-Score', alpha=0.8)
        
        ax.set_xlabel('Hepatitis Type', fontsize=12)
        ax.set_ylabel('Score', fontsize=12)
        ax.set_title('Precision, Recall, and F1-Score per Class', fontsize=16, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(self.label_encoder.classes_)
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Add value labels on bars
        for bars in [bars1, bars2, bars3]:
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                       f'{height:.3f}', ha='center', va='bottom')
        
        plt.tight_layout()
        plt.savefig(f'{self.output_dir}/figures/precision_recall_f1.png', dpi=300, bbox_inches='tight')
        plt.show()
        print(f"📊 Precision, recall, and F1-score plot saved to: {self.output_dir}/figures/precision_recall_f1.png")
        
        return precision, recall, f1

    def compare_models(self):
        """Compare different models and plot accuracy comparison"""
        print("\n📊 Comparing different models...")
        
        models = {
            'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
            'SVM': SVC(probability=True, random_state=42),
            'Logistic Regression': LogisticRegression(random_state=42),
            'Deep Neural Network': self.model
        }
        
        results = {}
        
        for name, model in models.items():
            if name == 'Deep Neural Network':
                # Use pre-trained neural network
                y_pred_proba = model.predict(self.X_test_scaled, verbose=0)
                y_pred = (y_pred_proba > 0.5).astype(int)
                accuracy = accuracy_score(self.y_test, y_pred)
            else:
                # Train and evaluate traditional ML models
                model.fit(self.X_train_scaled, self.y_train)
                y_pred = model.predict(self.X_test_scaled)
                accuracy = accuracy_score(self.y_test, y_pred)
            
            results[name] = accuracy
        
        # Plot accuracy comparison
        plt.figure(figsize=(12, 6))
        bars = plt.bar(results.keys(), results.values(), color=['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4'], alpha=0.8)
        
        # Add value labels on bars
        for bar, value in zip(bars, results.values()):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f'{value:.3f}', ha='center', va='bottom', fontweight='bold')
        
        plt.title('Accuracy Comparison Across Models', fontsize=16, fontweight='bold')
        plt.xlabel('Model', fontsize=12)
        plt.ylabel('Accuracy', fontsize=12)
        plt.ylim(0, 1)
        plt.grid(True, alpha=0.3)
        plt.xticks(rotation=45)
        plt.tight_layout()
        
        plt.savefig(f'{self.output_dir}/figures/model_comparison.png', dpi=300, bbox_inches='tight')
        plt.show()
        print(f"📊 Model comparison plot saved to: {self.output_dir}/figures/model_comparison.png")
        
        # Save results
        with open(f'{self.output_dir}/evaluation/model_comparison_results.pkl', 'wb') as f:
            pickle.dump(results, f)
        
        return results

    def comprehensive_evaluation(self):
        """Comprehensive model evaluation with all plots"""
        print("\n" + "="*50)
        print("COMPREHENSIVE MODEL EVALUATION")
        print("="*50)
        
        # Plot training curves
        self.plot_training_curves()
        
        # Plot confusion matrix
        cm = self.plot_confusion_matrix()
        
        # Plot precision, recall, F1-score
        precision, recall, f1 = self.plot_precision_recall_f1()
        
        # Compare models
        model_comparison = self.compare_models()
        
        # Calculate final metrics
        y_pred_proba = self.model.predict(self.X_test_scaled, verbose=0)
        y_pred = (y_pred_proba > 0.5).astype(int)
        
        test_accuracy = accuracy_score(self.y_test, y_pred)
        test_precision = precision_score(self.y_test, y_pred, average='weighted')
        test_recall = recall_score(self.y_test, y_pred, average='weighted')
        test_f1 = f1_score(self.y_test, y_pred, average='weighted')
        
        print(f"\n🎯 FINAL MODEL PERFORMANCE:")
        print(f"   Accuracy: {test_accuracy:.4f}")
        print(f"   Precision: {test_precision:.4f}")
        print(f"   Recall: {test_recall:.4f}")
        print(f"   F1-Score: {test_f1:.4f}")
        
        # Store results
        self.results['final_evaluation'] = {
            'test_accuracy': test_accuracy,
            'test_precision': test_precision,
            'test_recall': test_recall,
            'test_f1': test_f1,
            'confusion_matrix': cm.tolist(),
            'model_comparison': model_comparison
        }
        
        # Save final results
        with open(f'{self.output_dir}/evaluation/final_results.pkl', 'wb') as f:
            pickle.dump(self.results, f)
        
        print(f"\n✅ All plots and results saved to: {self.output_dir}")
        return self.results['final_evaluation']

    def run_complete_analysis(self):
        """Run the complete analysis pipeline"""
        print("\n🚀 STARTING COMPLETE ANALYSIS PIPELINE")
        print("=" * 70)
        
        try:
            # Step 1: Load and explore data
            self.load_and_explore_data()
            
            # Step 2: Advanced feature engineering
            X, y = self.advanced_feature_engineering()
            
            # Step 3: Hyperparameter tuning
            best_params = self.hyperparameter_tuning()
            
            # Step 4: Train enhanced model
            self.train_enhanced_model(best_params)
            
            # Step 5: Comprehensive evaluation with plots
            self.comprehensive_evaluation()
            
            print("\n✅ COMPLETE ANALYSIS PIPELINE FINISHED")
            print("=" * 70)
            print(f"📊 All plots saved in: {self.output_dir}/figures/")
            print(f"💾 All results saved in: {self.output_dir}/evaluation/")
            print(f"🤖 Best model saved in: {self.output_dir}/models/")
            
            return True
            
        except Exception as e:
            print(f"❌ Error in complete analysis: {e}")
            import traceback
            traceback.print_exc()
            return False

def main():
    """Main execution function"""
    print("🚀 INITIALIZING ENHANCED HEPATITIS SYSTEM WITH PLOTS...")
    print("=" * 70)
    
    # Initialize system
    enhanced_system = EnhancedHepatitisSystemWithPlots()
    
    try:
        # Run complete analysis
        print("\n🔄 RUNNING COMPLETE ANALYSIS...")
        success = enhanced_system.run_complete_analysis()
        
        if success:
            print("\n" + "🎉" * 20)
            print("ENHANCED SYSTEM WITH PLOTS EXECUTION COMPLETED!")
            print("🎉" * 20)
            print(f"\nFinal Results:")
            if 'final_evaluation' in enhanced_system.results:
                final_results = enhanced_system.results['final_evaluation']
                print(f"- Model accuracy: {final_results['test_accuracy']:.1%}")
                print(f"- Model precision: {final_results['test_precision']:.1%}")
                print(f"- Model recall: {final_results['test_recall']:.1%}")
                print(f"- Model F1-score: {final_results['test_f1']:.1%}")
            print(f"- All outputs saved in: {enhanced_system.output_dir}")
        else:
            print("❌ Analysis pipeline failed")
            
    except Exception as e:
        print(f"❌ Error in main execution: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
