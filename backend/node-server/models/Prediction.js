import mongoose from 'mongoose'

const predictionSchema = new mongoose.Schema({
  user: {
    type: mongoose.Schema.Types.ObjectId,
    ref: 'User',
    required: false,  // Allow guest predictions
    default: null
  },
  age: {
    type: String,
    required: true,
    enum: ['under18', '18-30', '31-45', '46-60', 'over60']
  },
  gender: {
    type: String,
    required: true,
    enum: ['male', 'female', 'other']
  },
  symptoms: {
    jaundice: { type: Boolean, default: false },
    dark_urine: { type: Boolean, default: false },
    pain: { type: Number, default: 0 },
    fatigue: { type: Number, default: 0 },
    nausea: { type: Boolean, default: false },
    vomiting: { type: Boolean, default: false },
    fever: { type: Boolean, default: false },
    loss_of_appetite: { type: Boolean, default: false },
    joint_pain: { type: Boolean, default: false }
  },
  riskFactors: [{
    type: String,
    enum: ['recent_travel', 'blood_transfusion_history', 'unsafe_injection_history', 'infected_contact']
  }],
  prediction: {
    predicted_class: {
      type: String,
      required: true
    },
    confidence: {
      type: Number,
      required: true
    },
    probability_Hepatitis_A: {
      type: Number,
      required: true
    },
    probability_Hepatitis_C: {
      type: Number,
      required: true
    }
  },
  status: {
    type: String,
    enum: ['pending', 'completed', 'failed'],
    default: 'completed'
  },
  notes: {
    type: String,
    trim: true
  }
}, {
  timestamps: true
})

// Index for faster queries
predictionSchema.index({ user: 1, createdAt: -1 })
predictionSchema.index({ 'prediction.predicted_class': 1 })

const Prediction = mongoose.model('Prediction', predictionSchema)

export default Prediction
