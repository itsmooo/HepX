import os
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Machine Learning Libraries for evaluation
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import (confusion_matrix, classification_report,
                           precision_score, recall_score, f1_score, accuracy_score)

class EnhancedRuleBasedHepatitisSystem:
    def __init__(self, output_dir='enhanced_rule_based_outputs'):
        self.output_dir = output_dir
        self.create_directory_structure()
        
        # Initialize containers
        self.df = None
        self.df_processed = None
        self.X_test = None
        self.y_test = None
        self.label_encoder = None
        self.results = {}
        
        # Set style for plots
        plt.style.use('seaborn-v0_8')
        sns.set_palette("husl")
        
        print("🚀 ENHANCED RULE-BASED HEPATITIS PREDICTION SYSTEM")
        print("=" * 70)
        print("Medical Logic with Comprehensive Evaluation and Visualization")
        print("=" * 70)

    def create_directory_structure(self):
        """Create comprehensive directory structure"""
        directories = [
            self.output_dir,
            f'{self.output_dir}/figures',
            f'{self.output_dir}/models',
            f'{self.output_dir}/data',
            f'{self.output_dir}/evaluation',
            f'{self.output_dir}/analysis'
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

    def analyze_symptom_patterns(self):
        """Analyze symptom patterns for rule development"""
        print("\n" + "="*50)
        print("SYMPTOM PATTERN ANALYSIS")
        print("="*50)
        
        # Extract symptoms by hepatitis type
        hepatitis_a_symptoms = self.df[self.df['HepatitisType'] == 'Hepatitis A']['Symptoms']
        hepatitis_c_symptoms = self.df[self.df['HepatitisType'] == 'Hepatitis C']['Symptoms']
        
        # Key symptoms to analyze
        key_symptoms = [
            'jaundice', 'fever', 'nausea', 'vomiting', 'dark.*urine', 'fatigue', 
            'joint.*pain', 'abdominal.*pain', 'appetite', 'weight.*loss', 'muscle'
        ]
        
        # Create symptom frequency analysis
        symptom_analysis = {}
        
        for symptom in key_symptoms:
            a_freq = hepatitis_a_symptoms.str.contains(symptom, case=False, na=False).mean()
            c_freq = hepatitis_c_symptoms.str.contains(symptom, case=False, na=False).mean()
            
            symptom_analysis[symptom] = {
                'Hepatitis A': a_freq,
                'Hepatitis C': c_freq,
                'Difference (A-C)': a_freq - c_freq
            }
        
        # Convert to DataFrame for plotting
        symptom_df = pd.DataFrame(symptom_analysis).T
        
        # Plot symptom frequency comparison
        self.plot_symptom_frequency(symptom_df)
        
        return symptom_analysis

    def plot_symptom_frequency(self, symptom_df):
        """Plot symptom frequency comparison"""
        plt.figure(figsize=(14, 8))
        
        x = np.arange(len(symptom_df))
        width = 0.35
        
        bars1 = plt.bar(x - width/2, symptom_df['Hepatitis A'], width, 
                       label='Hepatitis A', alpha=0.8, color='#FF6B6B')
        bars2 = plt.bar(x + width/2, symptom_df['Hepatitis C'], width,
                       label='Hepatitis C', alpha=0.8, color='#4ECDC4')
        
        plt.xlabel('Symptoms', fontsize=12)
        plt.ylabel('Frequency', fontsize=12)
        plt.title('Symptom Frequency Comparison: Hepatitis A vs C', fontsize=16, fontweight='bold')
        plt.xticks(x, symptom_df.index, rotation=45, ha='right')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Add value labels on bars
        for bars in [bars1, bars2]:
            for bar in bars:
                height = bar.get_height()
                plt.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                       f'{height:.2f}', ha='center', va='bottom', fontsize=8)
        
        plt.tight_layout()
        plt.savefig(f'{self.output_dir}/figures/symptom_frequency_comparison.png', dpi=300, bbox_inches='tight')
        plt.show()
        print(f"📊 Symptom frequency plot saved to: {self.output_dir}/figures/symptom_frequency_comparison.png")

    def rule_based_prediction_engine(self, symptoms_text, age=None, risk_factors=None):
        """Data-driven rule-based prediction engine based on actual symptom patterns"""
        if risk_factors is None:
            risk_factors = []
        
        # Initialize probabilities
        hepatitis_a_prob = 0.5
        hepatitis_c_prob = 0.5
        
        # Convert symptoms text to binary indicators
        symptoms_lower = symptoms_text.lower()
        
        # HEPATITIS C EXCLUSIVE INDICATORS (Based on data analysis)
        # These symptoms appear ONLY in Hepatitis C (0% in Hepatitis A)
        hep_c_exclusive = [
            'ascites' in symptoms_lower,
            'cirrhosis' in symptoms_lower, 
            'spider' in symptoms_lower,  # spider-like blood vessels
            'bruising' in symptoms_lower and 'easy' in symptoms_lower,
            'confusion' in symptoms_lower,
            'swelling' in symptoms_lower and 'legs' in symptoms_lower,
            'spleen' in symptoms_lower and 'enlarged' in symptoms_lower,
            'chronic' in symptoms_lower
        ]
        exclusive_count = sum(hep_c_exclusive)
        
        # HEPATITIS A INDICATORS 
        # "weakness" appears 38% in A vs 0% in C - strong A indicator
        hep_a_indicators = [
            'weakness' in symptoms_lower,
            'clay' in symptoms_lower and 'stool' in symptoms_lower,  # More common in A
            'irritability' in symptoms_lower,  # More common in A
            'headache' in symptoms_lower  # More common in A
        ]
        hep_a_count = sum(hep_a_indicators)
        
        # RULE 1: Hepatitis C exclusive symptoms (VERY STRONG)
        if exclusive_count >= 1:
            # Any exclusive symptom = very likely Hepatitis C
            hepatitis_c_prob += 0.4 * exclusive_count  # +40% per exclusive symptom
            hepatitis_a_prob -= 0.4 * exclusive_count
        
        # RULE 2: Hepatitis A indicators
        if hep_a_count >= 2:
            hepatitis_a_prob += 0.25
            hepatitis_c_prob -= 0.25
        elif hep_a_count >= 1:
            hepatitis_a_prob += 0.15
            hepatitis_c_prob -= 0.15
        
        # RULE 3: Symptom intensity (Hepatitis C has higher symptom counts)
        symptom_count = len([s for s in symptoms_lower.split(',') if s.strip()])
        if symptom_count >= 10:  # Very high symptom count
            hepatitis_c_prob += 0.20
            hepatitis_a_prob -= 0.20
        elif symptom_count >= 7:  # High symptom count
            hepatitis_c_prob += 0.10
            hepatitis_a_prob -= 0.10
        elif symptom_count <= 3:  # Low symptom count
            hepatitis_a_prob += 0.10
            hepatitis_c_prob -= 0.10
        
        # RULE 4: Common symptoms (slight preferences based on data)
        # Hepatitis C has slightly higher rates for fever, nausea, vomiting
        common_hep_c = [
            'fever' in symptoms_lower,
            'nausea' in symptoms_lower,
            'vomiting' in symptoms_lower,
            'joint' in symptoms_lower and 'pain' in symptoms_lower,
            'muscle' in symptoms_lower
        ]
        common_c_count = sum(common_hep_c)
        
        if common_c_count >= 3:
            hepatitis_c_prob += 0.10
            hepatitis_a_prob -= 0.10
        
        # RULE 5: Age-based adjustments (if available)
        if age:
            if 'under18' in str(age) or '18-30' in str(age):
                hepatitis_a_prob += 0.05  # Slightly more A in younger
                hepatitis_c_prob -= 0.05
            elif 'over60' in str(age):
                hepatitis_c_prob += 0.10  # More C in older (chronic nature)
                hepatitis_a_prob -= 0.10
        
        # RULE 6: Risk factors
        for risk_factor in risk_factors:
            if 'travel' in risk_factor.lower():
                hepatitis_a_prob += 0.15
                hepatitis_c_prob -= 0.15
            elif 'blood' in risk_factor.lower() or 'injection' in risk_factor.lower():
                hepatitis_c_prob += 0.20
                hepatitis_a_prob -= 0.20
        
        # Ensure probabilities are within bounds
        hepatitis_a_prob = max(0.05, min(0.95, hepatitis_a_prob))
        hepatitis_c_prob = 1 - hepatitis_a_prob
        
        # Determine prediction
        predicted_class = 'Hepatitis A' if hepatitis_a_prob > hepatitis_c_prob else 'Hepatitis C'
        confidence = max(hepatitis_a_prob, hepatitis_c_prob)
        
        return {
            'predicted_class': predicted_class,
            'probability_Hepatitis A': hepatitis_a_prob,
            'probability_Hepatitis C': hepatitis_c_prob,
            'confidence': confidence,
            'exclusive_indicators': exclusive_count,
            'hep_a_indicators': hep_a_count,
            'symptom_count': symptom_count
        }

    def evaluate_rule_based_system(self):
        """Evaluate the rule-based system on the test set"""
        print("\n" + "="*50)
        print("RULE-BASED SYSTEM EVALUATION")
        print("="*50)
        
        # Create test split
        self.label_encoder = LabelEncoder()
        y = self.label_encoder.fit_transform(self.df['HepatitisType'])
        
        # Split data for evaluation
        X_train, self.X_test, y_train, self.y_test = train_test_split(
            self.df, y, test_size=0.2, random_state=42, stratify=y
        )
        
        print(f"📊 Test set size: {len(self.X_test)} samples")
        
        # Apply rule-based predictions
        predictions = []
        probabilities_a = []
        probabilities_c = []
        confidences = []
        
        for idx, row in self.X_test.iterrows():
            # Extract age from symptom count and severity if available
            age_hint = None
            if 'SymptomCount' in row and row['SymptomCount'] > 6:
                age_hint = 'over60'  # More symptoms might indicate older patient
            
            result = self.rule_based_prediction_engine(
                row['Symptoms'], 
                age=age_hint,
                risk_factors=[]  # Risk factors not available in original dataset
            )
            
            predictions.append(result['predicted_class'])
            probabilities_a.append(result['probability_Hepatitis A'])
            probabilities_c.append(result['probability_Hepatitis C'])
            confidences.append(result['confidence'])
        
        # Convert predictions to encoded format
        pred_encoded = self.label_encoder.transform(predictions)
        
        # Calculate metrics
        accuracy = accuracy_score(self.y_test, pred_encoded)
        precision = precision_score(self.y_test, pred_encoded, average='weighted')
        recall = recall_score(self.y_test, pred_encoded, average='weighted')
        f1 = f1_score(self.y_test, pred_encoded, average='weighted')
        
        print(f"\n🎯 RULE-BASED SYSTEM PERFORMANCE:")
        print(f"   Accuracy: {accuracy:.4f}")
        print(f"   Precision: {precision:.4f}")
        print(f"   Recall: {recall:.4f}")
        print(f"   F1-Score: {f1:.4f}")
        print(f"   Mean Confidence: {np.mean(confidences):.4f}")
        
        # Store predictions for plotting
        self.predictions = pred_encoded
        self.probabilities = {
            'Hepatitis A': probabilities_a,
            'Hepatitis C': probabilities_c
        }
        self.confidences = confidences
        
        # Store results
        self.results['evaluation'] = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'mean_confidence': np.mean(confidences)
        }
        
        return accuracy, precision, recall, f1

    def plot_confusion_matrix(self):
        """Plot confusion matrix"""
        print("\n📊 Plotting confusion matrix...")
        
        cm = confusion_matrix(self.y_test, self.predictions)
        
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                    xticklabels=self.label_encoder.classes_,
                    yticklabels=self.label_encoder.classes_)
        plt.title('Confusion Matrix - Rule-Based Hepatitis Prediction', fontsize=16, fontweight='bold')
        plt.xlabel('Predicted Label', fontsize=12)
        plt.ylabel('True Label', fontsize=12)
        plt.tight_layout()
        
        plt.savefig(f'{self.output_dir}/figures/confusion_matrix.png', dpi=300, bbox_inches='tight')
        plt.show()
        print(f"📊 Confusion matrix plot saved to: {self.output_dir}/figures/confusion_matrix.png")
        
        return cm

    def plot_confidence_distribution(self):
        """Plot confidence distribution"""
        print("\n📊 Plotting confidence distribution...")
        
        plt.figure(figsize=(12, 6))
        
        # Subplot 1: Overall confidence distribution
        plt.subplot(1, 2, 1)
        plt.hist(self.confidences, bins=20, alpha=0.7, color='skyblue', edgecolor='black')
        plt.title('Confidence Distribution', fontsize=14, fontweight='bold')
        plt.xlabel('Confidence', fontsize=12)
        plt.ylabel('Frequency', fontsize=12)
        plt.grid(True, alpha=0.3)
        
        # Subplot 2: Confidence by class
        plt.subplot(1, 2, 2)
        true_classes = self.label_encoder.inverse_transform(self.y_test)
        
        # Separate confidences by true class
        conf_a = [conf for conf, true_class in zip(self.confidences, true_classes) if true_class == 'Hepatitis A']
        conf_c = [conf for conf, true_class in zip(self.confidences, true_classes) if true_class == 'Hepatitis C']
        
        plt.hist([conf_a, conf_c], bins=15, alpha=0.7, 
                label=['Hepatitis A', 'Hepatitis C'], color=['#FF6B6B', '#4ECDC4'])
        plt.title('Confidence by True Class', fontsize=14, fontweight='bold')
        plt.xlabel('Confidence', fontsize=12)
        plt.ylabel('Frequency', fontsize=12)
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f'{self.output_dir}/figures/confidence_distribution.png', dpi=300, bbox_inches='tight')
        plt.show()
        print(f"📊 Confidence distribution plot saved to: {self.output_dir}/figures/confidence_distribution.png")

    def plot_probability_calibration(self):
        """Plot probability calibration"""
        print("\n📊 Plotting probability calibration...")
        
        plt.figure(figsize=(10, 6))
        
        # Create probability bins
        bins = np.linspace(0, 1, 11)
        bin_centers = (bins[:-1] + bins[1:]) / 2
        
        # Calculate empirical probabilities
        empirical_probs = []
        
        for i in range(len(bins)-1):
            # Find predictions in this probability bin
            in_bin = ((np.array(self.probabilities['Hepatitis A']) >= bins[i]) & 
                     (np.array(self.probabilities['Hepatitis A']) < bins[i+1]))
            
            if np.sum(in_bin) > 0:
                # Calculate actual rate of Hepatitis A in this bin
                actual_rate = np.mean(self.y_test[in_bin] == 0)  # 0 = Hepatitis A
                empirical_probs.append(actual_rate)
            else:
                empirical_probs.append(np.nan)
        
        # Plot calibration curve
        plt.plot([0, 1], [0, 1], 'k--', alpha=0.5, label='Perfect Calibration')
        plt.plot(bin_centers, empirical_probs, 'o-', color='red', label='Rule-Based System')
        
        plt.xlabel('Mean Predicted Probability', fontsize=12)
        plt.ylabel('Fraction of Positives', fontsize=12)
        plt.title('Probability Calibration Curve', fontsize=16, fontweight='bold')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        plt.savefig(f'{self.output_dir}/figures/probability_calibration.png', dpi=300, bbox_inches='tight')
        plt.show()
        print(f"📊 Probability calibration plot saved to: {self.output_dir}/figures/probability_calibration.png")

    def save_rule_based_model(self):
        """Save the rule-based model artifacts"""
        print("\n💾 Saving rule-based model artifacts...")
        
        # Create model artifacts
        model_artifacts = {
            'model_type': 'rule_based',
            'label_encoder': self.label_encoder,
            'prediction_function': 'rule_based_prediction_engine',
            'version': '1.0',
            'created_date': datetime.now().isoformat(),
            'performance': self.results['evaluation']
        }
        
        # Save artifacts
        with open(f'{self.output_dir}/models/rule_based_model.pkl', 'wb') as f:
            pickle.dump(model_artifacts, f)
        
        # Save results
        with open(f'{self.output_dir}/evaluation/evaluation_results.pkl', 'wb') as f:
            pickle.dump(self.results, f)
        
        print(f"✅ Model artifacts saved to: {self.output_dir}/models/")
        print(f"✅ Evaluation results saved to: {self.output_dir}/evaluation/")

    def run_complete_analysis(self):
        """Run the complete rule-based analysis pipeline"""
        print("\n🚀 STARTING COMPLETE RULE-BASED ANALYSIS PIPELINE")
        print("=" * 70)
        
        try:
            # Step 1: Load and explore data
            self.load_and_explore_data()
            
            # Step 2: Analyze symptom patterns
            self.analyze_symptom_patterns()
            
            # Step 3: Evaluate rule-based system
            self.evaluate_rule_based_system()
            
            # Step 4: Generate evaluation plots
            self.plot_confusion_matrix()
            self.plot_confidence_distribution()
            self.plot_probability_calibration()
            
            # Step 5: Save model artifacts
            self.save_rule_based_model()
            
            print("\n✅ COMPLETE RULE-BASED ANALYSIS PIPELINE FINISHED")
            print("=" * 70)
            print(f"📊 All plots saved in: {self.output_dir}/figures/")
            print(f"💾 All results saved in: {self.output_dir}/evaluation/")
            print(f"🤖 Rule-based model saved in: {self.output_dir}/models/")
            
            return True
            
        except Exception as e:
            print(f"❌ Error in complete analysis: {e}")
            import traceback
            traceback.print_exc()
            return False

def main():
    """Main execution function"""
    print("🚀 INITIALIZING ENHANCED RULE-BASED HEPATITIS SYSTEM...")
    print("=" * 70)
    
    # Initialize system
    rule_system = EnhancedRuleBasedHepatitisSystem()
    
    try:
        # Run complete analysis
        print("\n🔄 RUNNING COMPLETE RULE-BASED ANALYSIS...")
        success = rule_system.run_complete_analysis()
        
        if success:
            print("\n" + "🎉" * 20)
            print("ENHANCED RULE-BASED SYSTEM EXECUTION COMPLETED!")
            print("🎉" * 20)
            print(f"\nFinal Results:")
            if 'evaluation' in rule_system.results:
                results = rule_system.results['evaluation']
                print(f"- Model accuracy: {results['accuracy']:.1%}")
                print(f"- Model precision: {results['precision']:.1%}")
                print(f"- Model recall: {results['recall']:.1%}")
                print(f"- Model F1-score: {results['f1_score']:.1%}")
                print(f"- Mean confidence: {results['mean_confidence']:.1%}")
            print(f"- All outputs saved in: {rule_system.output_dir}")
        else:
            print("❌ Analysis pipeline failed")
            
    except Exception as e:
        print(f"❌ Error in main execution: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
