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
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.metrics import (confusion_matrix, classification_report,
                           precision_score, recall_score, f1_score, accuracy_score)
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier

# Hyperparameter tuning
import optuna
from sklearn.model_selection import StratifiedKFold

class EnhancedMLModelTraining:
    def __init__(self, output_dir='enhanced_ml_outputs'):
        self.output_dir = output_dir
        self.create_directory_structure()
        
        # Initialize containers
        self.df = None
        self.X_train_scaled = None
        self.X_val_scaled = None
        self.X_test_scaled = None
        self.y_train = None
        self.y_val = None
        self.y_test = None
        self.models = {}
        self.scaler = None
        self.label_encoder = None
        self.feature_cols = []
        self.results = {}
        self.best_model = None
        self.best_model_name = None
        
        # Set style for plots
        plt.style.use('seaborn-v0_8')
        sns.set_palette("husl")
        
        print("🚀 ENHANCED MACHINE LEARNING MODEL TRAINING SYSTEM")
        print("=" * 70)
        print("Training Real ML Models: Neural Networks, Random Forest, SVM, etc.")
        print("=" * 70)

    def create_directory_structure(self):
        """Create comprehensive directory structure"""
        directories = [
            self.output_dir,
            f'{self.output_dir}/figures',
            f'{self.output_dir}/models',
            f'{self.output_dir}/data',
            f'{self.output_dir}/evaluation',
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
        """Create features based on discovered symptom patterns"""
        print("\n" + "="*50)
        print("ADVANCED FEATURE ENGINEERING")
        print("="*50)
        
        self.df_processed = self.df.copy()
        
        # Create target variable
        self.label_encoder = LabelEncoder()
        self.df_processed['target'] = self.label_encoder.fit_transform(self.df_processed['HepatitisType'])
        print(f"🎯 Target classes: {self.label_encoder.classes_}")
        
        # Based on our data analysis, create binary features for key symptoms
        print("🔍 Creating symptom-based features...")
        
        # Hepatitis C exclusive indicators (found in data analysis)
        hep_c_symptoms = [
            'ascites', 'cirrhosis', 'spider', 'bruising', 'confusion', 
            'swelling', 'spleen', 'chronic'
        ]
        
        # Hepatitis A indicators
        hep_a_symptoms = [
            'weakness', 'clay', 'irritability', 'headache'
        ]
        
        # Common symptoms with different frequencies
        common_symptoms = [
            'jaundice', 'fever', 'nausea', 'vomiting', 'fatigue', 
            'joint', 'muscle', 'dark.*urine', 'abdominal.*pain', 'appetite'
        ]
        
        # Create binary features for all symptoms
        all_symptoms = hep_c_symptoms + hep_a_symptoms + common_symptoms
        
        for symptom in all_symptoms:
            feature_name = f'has_{symptom.replace(".*", "").replace(" ", "_")}'
            self.df_processed[feature_name] = self.df_processed['Symptoms'].str.contains(
                symptom, case=False, na=False).astype(int)
        
        # Create composite features
        print("🧬 Creating composite features...")
        
        # Hepatitis C severity score
        hep_c_features = [f'has_{s}' for s in hep_c_symptoms]
        self.df_processed['hep_c_score'] = self.df_processed[hep_c_features].sum(axis=1)
        
        # Hepatitis A indicator score
        hep_a_features = [f'has_{s}' for s in hep_a_symptoms]
        self.df_processed['hep_a_score'] = self.df_processed[hep_a_features].sum(axis=1)
        
        # Symptom intensity features
        self.df_processed['symptom_count_normalized'] = self.df_processed['SymptomCount'] / 15.0  # Normalize
        self.df_processed['high_symptom_count'] = (self.df_processed['SymptomCount'] >= 10).astype(int)
        self.df_processed['low_symptom_count'] = (self.df_processed['SymptomCount'] <= 3).astype(int)
        
        # Severity encoding
        severity_mapping = {'Mild': 1, 'Moderate': 2, 'Severe': 3}
        if 'Severity' in self.df_processed.columns:
            # Map text severity to numeric, handle numeric severity
            def map_severity(val):
                if isinstance(val, str):
                    return severity_mapping.get(val, 2)  # Default to moderate
                else:
                    return val  # Already numeric
            
            self.df_processed['severity_numeric'] = self.df_processed['Severity'].apply(map_severity)
        
        # Select feature columns
        self.feature_cols = [col for col in self.df_processed.columns 
                           if col.startswith('has_') or 
                              col.endswith('_score') or 
                              col.endswith('_count') or 
                              col in ['symptom_count_normalized', 'severity_numeric']]
        
        print(f"✅ Created {len(self.feature_cols)} features for modeling")
        print(f"📊 Feature examples: {self.feature_cols[:5]}...")
        
        return self.df_processed

    def prepare_data_splits(self):
        """Prepare train/validation/test splits"""
        print("\n" + "="*50)
        print("DATA PREPARATION AND SPLITTING")
        print("="*50)
        
        # Prepare features and target
        X = self.df_processed[self.feature_cols]
        y = self.df_processed['target']
        
        print(f"📊 Feature matrix shape: {X.shape}")
        print(f"📊 Target distribution: {y.value_counts().to_dict()}")
        
        # Stratified split
        X_temp, self.X_test, y_temp, self.y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        self.X_train, self.X_val, self.y_train, self.y_val = train_test_split(
            X_temp, y_temp, test_size=0.2, random_state=42, stratify=y_temp
        )
        
        # Scale features
        print("⚖️ Scaling features...")
        self.scaler = StandardScaler()
        self.X_train_scaled = self.scaler.fit_transform(self.X_train)
        self.X_val_scaled = self.scaler.transform(self.X_val)
        self.X_test_scaled = self.scaler.transform(self.X_test)
        
        print(f"✅ Training set: {self.X_train_scaled.shape}")
        print(f"✅ Validation set: {self.X_val_scaled.shape}")
        print(f"✅ Test set: {self.X_test_scaled.shape}")

    def train_traditional_ml_models(self):
        """Train multiple traditional ML models"""
        print("\n" + "="*50)
        print("TRAINING TRADITIONAL ML MODELS")
        print("="*50)
        
        # Define models to train
        models_to_train = {
            'Random Forest': RandomForestClassifier(
                n_estimators=200, max_depth=15, random_state=42, 
                class_weight='balanced', n_jobs=-1
            ),
            'Gradient Boosting': GradientBoostingClassifier(
                n_estimators=200, max_depth=8, learning_rate=0.1, random_state=42
            ),
            'SVM': SVC(
                C=1.0, kernel='rbf', probability=True, random_state=42, class_weight='balanced'
            ),
            'Logistic Regression': LogisticRegression(
                C=1.0, random_state=42, class_weight='balanced', max_iter=1000
            ),
            'Decision Tree': DecisionTreeClassifier(
                max_depth=12, random_state=42, class_weight='balanced'
            )
        }
        
        model_results = {}
        
        for name, model in models_to_train.items():
            print(f"\n🔄 Training {name}...")
            
            # Train model
            model.fit(self.X_train_scaled, self.y_train)
            
            # Predict on validation set
            y_pred = model.predict(self.X_val_scaled)
            y_pred_proba = model.predict_proba(self.X_val_scaled)[:, 1] if hasattr(model, 'predict_proba') else None
            
            # Calculate metrics
            accuracy = accuracy_score(self.y_val, y_pred)
            precision = precision_score(self.y_val, y_pred, average='weighted')
            recall = recall_score(self.y_val, y_pred, average='weighted')
            f1 = f1_score(self.y_val, y_pred, average='weighted')
            
            model_results[name] = {
                'model': model,
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1_score': f1,
                'predictions': y_pred,
                'probabilities': y_pred_proba
            }
            
            print(f"   ✅ {name} - Accuracy: {accuracy:.4f}, F1: {f1:.4f}")
        
        self.models.update(model_results)
        return model_results

    def build_neural_network(self):
        """Build and train a neural network"""
        print("\n" + "="*50)
        print("BUILDING NEURAL NETWORK")
        print("="*50)
        
        input_dim = self.X_train_scaled.shape[1]
        print(f"📊 Input dimensions: {input_dim}")
        
        # Build model
        model = Sequential([
            Dense(128, activation='relu', input_shape=(input_dim,)),
            BatchNormalization(),
            Dropout(0.3),
            
            Dense(64, activation='relu'),
            BatchNormalization(),
            Dropout(0.2),
            
            Dense(32, activation='relu'),
            BatchNormalization(),
            Dropout(0.1),
            
            Dense(1, activation='sigmoid')
        ])
        
        # Compile model
        model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss='binary_crossentropy',
            metrics=['accuracy']
        )
        
        print(f"🧠 Neural network built with {model.count_params():,} parameters")
        
        # Callbacks
        callbacks = [
            EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True),
            ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=8, min_lr=1e-6),
            ModelCheckpoint(f'{self.output_dir}/models/best_neural_network.keras', 
                          monitor='val_accuracy', save_best_only=True)
        ]
        
        # Train model
        print("🚀 Training neural network...")
        history = model.fit(
            self.X_train_scaled, self.y_train,
            validation_data=(self.X_val_scaled, self.y_val),
            epochs=100,
            batch_size=32,
            callbacks=callbacks,
            verbose=1
        )
        
        # Evaluate
        y_pred_proba = model.predict(self.X_val_scaled, verbose=0)
        y_pred = (y_pred_proba > 0.5).astype(int).flatten()
        
        accuracy = accuracy_score(self.y_val, y_pred)
        precision = precision_score(self.y_val, y_pred, average='weighted')
        recall = recall_score(self.y_val, y_pred, average='weighted')
        f1 = f1_score(self.y_val, y_pred, average='weighted')
        
        self.models['Neural Network'] = {
            'model': model,
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'predictions': y_pred,
            'probabilities': y_pred_proba.flatten(),
            'history': history
        }
        
        print(f"✅ Neural Network - Accuracy: {accuracy:.4f}, F1: {f1:.4f}")
        return model, history

    def find_best_model(self):
        """Find the best performing model"""
        print("\n" + "="*50)
        print("FINDING BEST MODEL")
        print("="*50)
        
        best_f1 = 0
        best_name = None
        
        print("📊 Model Performance Summary:")
        print("-" * 60)
        print(f"{'Model':<20} {'Accuracy':<12} {'Precision':<12} {'Recall':<12} {'F1-Score':<12}")
        print("-" * 60)
        
        for name, results in self.models.items():
            accuracy = results['accuracy']
            precision = results['precision']
            recall = results['recall']
            f1 = results['f1_score']
            
            print(f"{name:<20} {accuracy:<12.4f} {precision:<12.4f} {recall:<12.4f} {f1:<12.4f}")
            
            if f1 > best_f1:
                best_f1 = f1
                best_name = name
        
        print("-" * 60)
        print(f"🏆 Best Model: {best_name} (F1-Score: {best_f1:.4f})")
        
        self.best_model = self.models[best_name]['model']
        self.best_model_name = best_name
        
        return best_name, self.best_model

    def evaluate_best_model_on_test(self):
        """Evaluate the best model on test set"""
        print(f"\n🔍 EVALUATING BEST MODEL ({self.best_model_name}) ON TEST SET")
        print("="*50)
        
        # Predict on test set
        if self.best_model_name == 'Neural Network':
            y_test_pred_proba = self.best_model.predict(self.X_test_scaled, verbose=0)
            y_test_pred = (y_test_pred_proba > 0.5).astype(int).flatten()
        else:
            y_test_pred = self.best_model.predict(self.X_test_scaled)
            y_test_pred_proba = self.best_model.predict_proba(self.X_test_scaled)[:, 1]
        
        # Calculate final metrics
        test_accuracy = accuracy_score(self.y_test, y_test_pred)
        test_precision = precision_score(self.y_test, y_test_pred, average='weighted')
        test_recall = recall_score(self.y_test, y_test_pred, average='weighted')
        test_f1 = f1_score(self.y_test, y_test_pred, average='weighted')
        
        print(f"🎯 FINAL TEST RESULTS:")
        print(f"   Accuracy:  {test_accuracy:.4f}")
        print(f"   Precision: {test_precision:.4f}")
        print(f"   Recall:    {test_recall:.4f}")
        print(f"   F1-Score:  {test_f1:.4f}")
        
        # Store results
        self.results['final_test'] = {
            'accuracy': test_accuracy,
            'precision': test_precision,
            'recall': test_recall,
            'f1_score': test_f1,
            'predictions': y_test_pred,
            'probabilities': y_test_pred_proba,
            'best_model_name': self.best_model_name
        }
        
        return test_accuracy, test_precision, test_recall, test_f1

    def plot_model_comparison(self):
        """Plot comparison of all models"""
        print("\n📊 Plotting model comparison...")
        
        model_names = list(self.models.keys())
        accuracies = [self.models[name]['accuracy'] for name in model_names]
        f1_scores = [self.models[name]['f1_score'] for name in model_names]
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Accuracy comparison
        bars1 = ax1.bar(model_names, accuracies, alpha=0.8, color='skyblue')
        ax1.set_title('Model Accuracy Comparison', fontsize=14, fontweight='bold')
        ax1.set_ylabel('Accuracy', fontsize=12)
        ax1.set_ylim(0, 1)
        ax1.tick_params(axis='x', rotation=45)
        ax1.grid(True, alpha=0.3)
        
        # Add value labels
        for bar, acc in zip(bars1, accuracies):
            ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f'{acc:.3f}', ha='center', va='bottom', fontweight='bold')
        
        # F1-Score comparison
        bars2 = ax2.bar(model_names, f1_scores, alpha=0.8, color='lightcoral')
        ax2.set_title('Model F1-Score Comparison', fontsize=14, fontweight='bold')
        ax2.set_ylabel('F1-Score', fontsize=12)
        ax2.set_ylim(0, 1)
        ax2.tick_params(axis='x', rotation=45)
        ax2.grid(True, alpha=0.3)
        
        # Add value labels
        for bar, f1 in zip(bars2, f1_scores):
            ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f'{f1:.3f}', ha='center', va='bottom', fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(f'{self.output_dir}/figures/model_comparison.png', dpi=300, bbox_inches='tight')
        plt.show()
        print(f"📊 Model comparison plot saved to: {self.output_dir}/figures/model_comparison.png")

    def plot_confusion_matrix(self):
        """Plot confusion matrix for best model"""
        print("\n📊 Plotting confusion matrix for best model...")
        
        if 'final_test' in self.results:
            y_pred = self.results['final_test']['predictions']
            cm = confusion_matrix(self.y_test, y_pred)
            
            plt.figure(figsize=(8, 6))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                        xticklabels=self.label_encoder.classes_,
                        yticklabels=self.label_encoder.classes_)
            plt.title(f'Confusion Matrix - {self.best_model_name}', fontsize=16, fontweight='bold')
            plt.xlabel('Predicted Label', fontsize=12)
            plt.ylabel('True Label', fontsize=12)
            plt.tight_layout()
            
            plt.savefig(f'{self.output_dir}/figures/confusion_matrix.png', dpi=300, bbox_inches='tight')
            plt.show()
            print(f"📊 Confusion matrix plot saved to: {self.output_dir}/figures/confusion_matrix.png")

    def save_models_and_artifacts(self):
        """Save the best model and preprocessing artifacts"""
        print("\n💾 Saving models and artifacts...")
        
        # Save the best model
        if self.best_model_name == 'Neural Network':
            self.best_model.save(f'{self.output_dir}/models/best_model.keras')
        else:
            with open(f'{self.output_dir}/models/best_model.pkl', 'wb') as f:
                pickle.dump(self.best_model, f)
        
        # Save preprocessing artifacts
        artifacts = {
            'scaler': self.scaler,
            'label_encoder_target': self.label_encoder,
            'feature_columns': self.feature_cols,
            'target_column': 'HepatitisType',
            'model_type': self.best_model_name,
            'created_date': datetime.now().isoformat(),
            'performance': self.results['final_test']
        }
        
        with open(f'{self.output_dir}/models/preprocessing_artifacts.pkl', 'wb') as f:
            pickle.dump(artifacts, f)
        
        # Save all results
        with open(f'{self.output_dir}/evaluation/training_results.pkl', 'wb') as f:
            pickle.dump(self.results, f)
        
        print(f"✅ Best model saved: {self.output_dir}/models/")
        print(f"✅ Artifacts saved: {self.output_dir}/models/preprocessing_artifacts.pkl")
        print(f"✅ Results saved: {self.output_dir}/evaluation/")

    def run_complete_training(self):
        """Run the complete ML training pipeline"""
        print("\n🚀 STARTING COMPLETE ML TRAINING PIPELINE")
        print("=" * 70)
        
        try:
            # Step 1: Load and explore data
            self.load_and_explore_data()
            
            # Step 2: Feature engineering
            self.advanced_feature_engineering()
            
            # Step 3: Prepare data splits
            self.prepare_data_splits()
            
            # Step 4: Train traditional ML models
            self.train_traditional_ml_models()
            
            # Step 5: Train neural network
            self.build_neural_network()
            
            # Step 6: Find best model
            self.find_best_model()
            
            # Step 7: Evaluate on test set
            self.evaluate_best_model_on_test()
            
            # Step 8: Generate plots
            self.plot_model_comparison()
            self.plot_confusion_matrix()
            
            # Step 9: Save everything
            self.save_models_and_artifacts()
            
            print("\n✅ COMPLETE ML TRAINING PIPELINE FINISHED")
            print("=" * 70)
            print(f"🏆 Best Model: {self.best_model_name}")
            if 'final_test' in self.results:
                final = self.results['final_test']
                print(f"📊 Final Test Accuracy: {final['accuracy']:.1%}")
                print(f"📊 Final Test F1-Score: {final['f1_score']:.1%}")
            print(f"💾 All outputs saved in: {self.output_dir}")
            
            return True
            
        except Exception as e:
            print(f"❌ Error in training pipeline: {e}")
            import traceback
            traceback.print_exc()
            return False

def main():
    """Main execution function"""
    print("🚀 INITIALIZING ENHANCED ML MODEL TRAINING SYSTEM...")
    print("=" * 70)
    
    # Initialize system
    ml_trainer = EnhancedMLModelTraining()
    
    try:
        # Run complete training
        print("\n🔄 RUNNING COMPLETE ML TRAINING...")
        success = ml_trainer.run_complete_training()
        
        if success:
            print("\n" + "🎉" * 20)
            print("ENHANCED ML MODEL TRAINING COMPLETED!")
            print("🎉" * 20)
        else:
            print("❌ Training pipeline failed")
            
    except Exception as e:
        print(f"❌ Error in main execution: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
