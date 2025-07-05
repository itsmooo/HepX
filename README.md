# HepaPredict: Hepatitis Type Prediction System

## Project Overview
HepaPredict is an advanced web-based application designed to predict the likelihood of different types of hepatitis based on patient symptoms and risk factors. The system utilizes machine learning to provide preliminary assessments, helping healthcare providers and patients make informed decisions about further medical evaluation.

## Types of Hepatitis Predicted
The system focuses on predicting two main types of hepatitis:

1. **Hepatitis A**
   - Acute viral infection
   - Transmitted through contaminated food/water
   - Usually self-limiting and resolves within 6 months

2. **Hepatitis C**
   - Chronic viral infection
   - Transmitted through blood contact
   - Can lead to long-term liver complications

## Features and Symptoms Analyzed

### Basic Information
- Age Group Categories:
  - Under 18
  - 18-30
  - 31-45
  - 46-60
  - Over 60
- Gender (Male/Female/Other)

### Primary Symptoms
1. **Jaundice (Yellowing of skin/eyes)**
   - Key indicator for liver dysfunction
   - Measured as: Present/Absent

2. **Dark Urine**
   - Early sign of liver problems
   - Measured as: Present/Absent

3. **Abdominal Pain**
   - Severity scale: 0-10
   - Location and intensity considered

4. **Fatigue Level**
   - Severity scale: 0-10
   - Impact on daily activities

5. **Additional Symptoms**
   - Fever (Present/Absent)
   - Nausea (Present/Absent)
   - Joint Pain (Present/Absent)
   - Loss of Appetite (Present/Absent)

### Risk Factors
- Recent Travel to High-Risk Areas
- History of Blood Transfusion
- History of Unsafe Injection Practices
- Contact with Infected Person

## Technical Architecture

### Frontend Technology Stack
- **Framework**: Next.js 15.3.1
- **UI Components**: Custom components using Tailwind CSS
- **State Management**: React Hooks
- **Animations**: Framer Motion
- **Form Handling**: Custom form implementation with validation

### Backend Technology Stack
- **Server**: Node.js with Express
- **Machine Learning**: Python with scikit-learn
- **API Integration**: RESTful API endpoints
- **Data Processing**: Pandas for data preprocessing

### Machine Learning Model
- **Model Type**: Probabilistic Classification Model
- **Features Used**:
  - Demographic data (age, gender)
  - Clinical symptoms (binary and scaled values)
  - Risk factors (binary indicators)
- **Output**: Probability scores for Hepatitis A and C

## Data Flow
1. **User Input Collection**
   - Multi-step form interface
   - Real-time validation
   - Progress tracking

2. **Data Processing**
   - Symptom normalization
   - Feature scaling
   - Boolean to numeric conversion

3. **Prediction Generation**
   - Model inference
   - Probability calculation
   - Confidence scoring

4. **Result Presentation**
   - Visual probability displays
   - Downloadable reports
   - Next steps recommendations

## Security Features
- Data encryption in transit
- Secure API endpoints
- No personal health information storage
- HIPAA-aware design principles

## Deployment Requirements
- Node.js >= 14.0.0
- Python >= 3.8
- NPM or Yarn package manager
- Required Python packages:
  - scikit-learn
  - pandas
  - numpy
- Required Node.js packages:
  - next
  - react
  - tailwindcss
  - framer-motion

## Installation and Setup
1. Clone the repository
2. Install frontend dependencies:
   ```bash
   cd frontend
   npm install
   ```
3. Install backend dependencies:
   ```bash
   cd backend
   pip install -r requirements.txt
   ```
4. Start the development servers:
   - Frontend: `npm run dev`
   - Backend: `python server.py`

## Usage Guidelines
1. Navigate to the web interface
2. Complete the multi-step assessment form
3. Review the prediction results
4. Download the detailed report if needed
5. Follow the recommended next steps

## Limitations and Disclaimers
- Not a substitute for professional medical diagnosis
- Requires validation with clinical tests
- Should be used as a preliminary screening tool only
- Accuracy depends on input data quality

## Future Enhancements
1. Integration of additional hepatitis types
2. Mobile application development
3. Enhanced risk factor analysis
4. Integration with electronic health records
5. Multi-language support

## Contributors
- Development Team
- Medical Advisors
- UI/UX Designers
- Machine Learning Engineers

## License
This project is proprietary and confidential. All rights reserved.

## Contact
For technical support or inquiries, please contact the development team.

---

*Note: This documentation is intended for academic and research purposes as part of a graduation project. The system should not be used as a standalone diagnostic tool.* 