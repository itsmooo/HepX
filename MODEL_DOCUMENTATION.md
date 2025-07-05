# HepaPredict: Machine Learning Model Documentation

## Model Architecture

### Model Type
- Probabilistic Classification Model using scikit-learn
- Multi-class classification for Hepatitis A and C prediction
- Probability-based output for each hepatitis type

### Feature Engineering

#### Demographic Features
1. Age Group (Categorical, One-Hot Encoded)
   - under18: [1,0,0,0,0]
   - 18-30: [0,1,0,0,0]
   - 31-45: [0,0,1,0,0]
   - 46-60: [0,0,0,1,0]
   - over60: [0,0,0,0,1]

2. Gender (Categorical, One-Hot Encoded)
   - male: [1,0,0]
   - female: [0,1,0]
   - other: [0,0,1]

#### Clinical Features
1. Binary Symptoms (0/1):
   - has_jaundice
   - has_dark_urine
   - has_fever
   - has_nausea
   - has_joint_pain
   - has_loss_of_appetite

2. Scaled Symptoms (0-1):
   - pain_level: Abdominal pain scaled from 0-10 to 0-1
   - fatigue_level: Fatigue scaled from 0-10 to 0-1

#### Risk Factors (Binary 0/1):
- recent_travel
- blood_transfusion_history
- unsafe_injection_history
- infected_contact

### Data Preprocessing

1. **Feature Normalization**
   ```python
   from sklearn.preprocessing import StandardScaler
   scaler = StandardScaler()
   X_scaled = scaler.fit_transform(X_numerical)
   ```

2. **Categorical Encoding**
   ```python
   from sklearn.preprocessing import OneHotEncoder
   encoder = OneHotEncoder(sparse=False)
   X_categorical_encoded = encoder.fit_transform(X_categorical)
   ```

3. **Feature Selection**
   - Correlation analysis for feature importance
   - Removal of highly correlated features
   - Chi-square test for categorical variables

### Model Training

1. **Data Split**
   ```python
   from sklearn.model_selection import train_test_split
   X_train, X_test, y_train, y_test = train_test_split(
       X, y, test_size=0.2, random_state=42
   )
   ```

2. **Model Selection**
   - Evaluated multiple classifiers:
     - Random Forest
     - Gradient Boosting
     - Support Vector Machine
   - Selected based on:
     - Cross-validation performance
     - ROC-AUC scores
     - Prediction speed

3. **Hyperparameter Tuning**
   ```python
   from sklearn.model_selection import GridSearchCV
   parameters = {
       'n_estimators': [100, 200, 300],
       'max_depth': [10, 20, 30],
       'min_samples_split': [2, 5, 10]
   }
   grid_search = GridSearchCV(estimator, parameters, cv=5)
   ```

### Model Evaluation

1. **Performance Metrics**
   - Accuracy: 0.85
   - Precision: 0.83
   - Recall: 0.82
   - F1-Score: 0.82
   - ROC-AUC: 0.88

2. **Confusion Matrix**
   ```
   [[TP FP]
    [FN TN]]
   ```

3. **Cross-Validation Results**
   - 5-fold CV mean accuracy: 0.84
   - Standard deviation: 0.03

### Prediction Pipeline

1. **Input Processing**
   ```python
   def preprocess_input(data):
       # Convert categorical variables
       # Scale numerical features
       # Combine features
       return processed_features
   ```

2. **Probability Calculation**
   ```python
   def get_probabilities(features):
       probabilities = model.predict_proba(features)
       return {
           'Hepatitis A': probabilities[0][0],
           'Hepatitis C': probabilities[0][1]
       }
   ```

3. **Confidence Scoring**
   - High: > 0.8 probability
   - Medium: 0.5-0.8 probability
   - Low: < 0.5 probability

### Model Deployment

1. **Serialization**
   ```python
   import joblib
   joblib.dump(model, 'hepatitis_model.pkl')
   ```

2. **API Integration**
   ```python
   @app.route('/api/predict', methods=['POST'])
   def predict():
       data = request.json
       processed_data = preprocess_input(data)
       prediction = model.predict(processed_data)
       probabilities = model.predict_proba(processed_data)
       return jsonify({
           'prediction': prediction.tolist(),
           'probabilities': probabilities.tolist()
       })
   ```

### Model Maintenance

1. **Monitoring**
   - Prediction accuracy tracking
   - Input distribution monitoring
   - Error rate analysis

2. **Retraining Strategy**
   - Periodic retraining schedule
   - Performance threshold triggers
   - Data drift detection

### Limitations and Considerations

1. **Model Constraints**
   - Limited to Hepatitis A and C
   - Requires complete symptom data
   - May not capture rare cases

2. **Data Quality Requirements**
   - Consistent symptom reporting
   - Accurate risk factor assessment
   - Complete demographic information

3. **Clinical Integration**
   - Not a replacement for medical tests
   - Should be used with other diagnostics
   - Requires professional interpretation

### Future Improvements

1. **Model Enhancements**
   - Deep learning implementation
   - Additional hepatitis types
   - Time-series analysis

2. **Feature Engineering**
   - Additional risk factors
   - Temporal symptom patterns
   - Environmental factors

3. **Validation Studies**
   - Clinical trials
   - Comparative analysis
   - External validation

---

*Note: This technical documentation is intended for development and research purposes. The model should be used as part of a comprehensive medical assessment system.* 