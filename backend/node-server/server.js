import express from "express"
import cors from "cors"
import session from "express-session"
import passport from "passport"
import jwt from "jsonwebtoken"
import { Strategy as GoogleStrategy } from "passport-google-oauth20"

import { spawn, execSync } from "child_process"
import fs from "fs"
import path from "path"
import { fileURLToPath } from "url"
import { dirname } from "path"
import multer from "multer"
import fetch from "node-fetch"
import dotenv from "dotenv"
import connectDB from "./config/db.js"
import { register, login, getMe, updateProfile, changePassword, logout } from "./routes/auth.js"
import { protect, authorize } from "./middleware/auth.js"
import {
  getAllUsers,
  getUserById,
  createUser,
  updateUser,
  deleteUser,
  getAllPredictions,
  getPredictionById,
  updatePrediction,
  deletePrediction,
  getDashboardStats
} from "./routes/admin.js"
import Prediction from "./models/Prediction.js"
import User from "./models/User.js"

// Load env vars
dotenv.config()

const __filename = fileURLToPath(import.meta.url)
const __dirname = dirname(__filename)

const app = express()
const PORT = process.env.PORT || 5000

// Connect to MongoDB
connectDB()

// Middleware
app.use(cors())
app.use(express.json())
app.use(express.static(path.join(__dirname, "output")))

// Session and Passport setup for OAuth
app.set('trust proxy', 1)
app.use(session({
  secret: process.env.SESSION_SECRET || 'session-secret',
  resave: false,
  saveUninitialized: true,
  cookie: {
    sameSite: 'lax',
    secure: false
  }
}))
app.use(passport.initialize())
app.use(passport.session())

passport.serializeUser((user, done) => {
  done(null, user._id)
})
passport.deserializeUser(async (id, done) => {
  try {
    const user = await User.findById(id).select('-password')
    done(null, user)
  } catch (e) {
    done(e)
  }
})

// OAuth strategies
const HAS_GOOGLE = Boolean(process.env.GOOGLE_CLIENT_ID && process.env.GOOGLE_CLIENT_SECRET)
const BASE_URL = process.env.BACKEND_BASE_URL || `http://localhost:${PORT}`

if (HAS_GOOGLE) {
  passport.use(new GoogleStrategy({
    clientID: process.env.GOOGLE_CLIENT_ID,
    clientSecret: process.env.GOOGLE_CLIENT_SECRET,
    callbackURL: `${BASE_URL}/api/auth/google/callback`
  }, async (accessToken, refreshToken, profile, done) => {
    try {
      const email = profile.emails?.[0]?.value
      let user = await User.findOne({ provider: 'google', providerId: profile.id })
      if (!user && email) {
        user = await User.findOne({ email })
      }
      if (!user) {
        user = await User.create({
          firstName: profile.name?.givenName || 'Google',
          lastName: profile.name?.familyName || 'User',
          email: email || `${profile.id}@google.local`,
          password: Math.random().toString(36).slice(2),
          provider: 'google',
          providerId: profile.id,
          avatar: profile.photos?.[0]?.value || null
        })
      } else {
        user.provider = 'google'
        user.providerId = profile.id
        user.avatar = profile.photos?.[0]?.value || user.avatar
        await user.save()
      }
      return done(null, user)
    } catch (e) {
      return done(e)
    }
  }))
}



// Set up storage for uploaded files
const storage = multer.diskStorage({
  destination: (req, file, cb) => {
    cb(null, path.join(__dirname, "uploads"))
  },
  filename: (req, file, cb) => {
    cb(null, "hepatitis_dataset.csv")
  },
})

const upload = multer({ storage: storage })

// Ensure directories exist
const outputDir = path.join(__dirname, "output")
const uploadsDir = path.join(__dirname, "uploads")

if (!fs.existsSync(outputDir)) {
  fs.mkdirSync(outputDir, { recursive: true })
}

if (!fs.existsSync(uploadsDir)) {
  fs.mkdirSync(uploadsDir, { recursive: true })
}

// Auth Routes
app.post("/api/auth/register", register)
app.post("/api/auth/login", login)
app.get("/api/auth/me", protect, getMe)
app.put("/api/auth/profile", protect, updateProfile)
app.put("/api/auth/change-password", protect, changePassword)
app.post("/api/auth/logout", protect, logout)

// OAuth routes (issue JWT after successful OAuth login)
if (HAS_GOOGLE) {
  app.get('/api/auth/google', passport.authenticate('google', { scope: ['profile', 'email'] }))
  app.get('/api/auth/google/callback', passport.authenticate('google', { failureRedirect: '/login' }), (req, res) => {
    const token = jwtSignForUser(req.user)
    res.redirect(`/oauth-success?token=${token}`)
  })
} else {
  app.get('/api/auth/google', (req, res) => res.status(503).json({ success: false, message: 'Google OAuth is not configured' }))
}



// Admin Routes
app.get("/api/admin/dashboard", protect, authorize('admin'), getDashboardStats)
app.get("/api/admin/users", protect, authorize('admin'), getAllUsers)
app.get("/api/admin/users/:id", protect, authorize('admin'), getUserById)
app.post("/api/admin/users", protect, authorize('admin'), createUser)
app.put("/api/admin/users/:id", protect, authorize('admin'), updateUser)
app.delete("/api/admin/users/:id", protect, authorize('admin'), deleteUser)
app.get("/api/admin/predictions", protect, authorize('admin'), getAllPredictions)
app.get("/api/admin/predictions/:id", protect, authorize('admin'), getPredictionById)
app.put("/api/admin/predictions/:id", protect, authorize('admin'), updatePrediction)
app.delete("/api/admin/predictions/:id", protect, authorize('admin'), deletePrediction)

// API Routes
app.get("/api/status", (req, res) => {
  res.json({ status: "API is running" })
})

// Health check endpoint for Python service
app.get("/api/python-status", async (req, res) => {
  try {
    const PYTHON_API_URL = process.env.PYTHON_API_URL || 'http://localhost:8000'
    
    const response = await fetch(`${PYTHON_API_URL}/health`, {
      method: 'GET',
      timeout: 5000 // 5 second timeout
    })
    
    if (response.ok) {
      const healthData = await response.json()
      res.json({
        status: "Python service is healthy",
        python_service: healthData,
        api_url: PYTHON_API_URL
      })
    } else {
      res.status(503).json({
        status: "Python service is unhealthy",
        error: `HTTP ${response.status}`,
        api_url: PYTHON_API_URL
      })
    }
  } catch (error) {
    res.status(503).json({
      status: "Python service is not available",
      error: error.message,
      api_url: process.env.PYTHON_API_URL || 'http://localhost:8000',
      suggestion: "Start Python service with: python predict.py --server"
    })
  }
})



// Route to get model results
app.get("/api/model-results", (req, res) => {
  try {
    const resultsPath = path.join(outputDir, "model_results.json")

    if (!fs.existsSync(resultsPath)) {
      return res.status(404).json({ error: "Model results not found. Train the model first." })
    }

    const results = JSON.parse(fs.readFileSync(resultsPath, "utf8"))
    res.json(results)
  } catch (error) {
    console.error("Error in /api/model-results:", error)
    res.status(500).json({ error: "Failed to retrieve model results" })
  }
})

// Route to get visualization images
app.get("/api/visualizations", (req, res) => {
  try {
    const images = [
      "hepatitis_types_distribution.png",
      "severity_distribution.png",
      "symptom_count_distribution.png",
      "confusion_matrix.png",
      "feature_importance.png",
      "actual_vs_predicted.png",
    ]

    const availableImages = images.filter((img) => fs.existsSync(path.join(outputDir, img)))

    res.json({
      images: availableImages.map((img) => `/api/visualization/${img}`),
    })
  } catch (error) {
    console.error("Error in /api/visualizations:", error)
    res.status(500).json({ error: "Failed to retrieve visualizations" })
  }
})

// Route to get a specific visualization
app.get("/api/visualization/:filename", (req, res) => {
  try {
    const { filename } = req.params
    const filePath = path.join(outputDir, filename)

    if (!fs.existsSync(filePath)) {
      return res.status(404).json({ error: "Visualization not found" })
    }

    res.sendFile(filePath)
  } catch (error) {
    console.error("Error in /api/visualization/:filename:", error)
    res.status(500).json({ error: "Failed to retrieve visualization" })
  }
})

// Route to handle prediction (supports both authenticated and guest users)
app.post("/api/predict", async (req, res) => {
  try {
    const { age, gender, symptoms, riskFactors } = req.body

    // Validate required fields
    if (!age || !gender) {
      return res.status(400).json({
        success: false,
        message: "Age and gender are required",
      })
    }

    // Try to get authenticated user (if token is provided)
    let authenticatedUser = null;
    try {
      if (req.headers.authorization && req.headers.authorization.startsWith('Bearer')) {
        const token = req.headers.authorization.split(' ')[1];
        const decoded = jwt.verify(token, process.env.JWT_SECRET || 'your-secret-key');
        authenticatedUser = await User.findById(decoded.id).select('-password');
      }
    } catch (error) {
      // Token invalid or expired, proceed as guest
      console.log('No valid token provided, proceeding as guest user');
    }

    // Use improved model prediction
    const predictionResult = await performImprovedPrediction(age, gender, symptoms, riskFactors)
    
    // Extract prediction data from the result
    const predictionData = predictionResult.prediction || predictionResult
    
    // Create prediction record in database (with user if authenticated, null if guest)
    const prediction = new Prediction({
      user: authenticatedUser ? authenticatedUser._id : null,
      age,
      gender,
      symptoms: {
        jaundice: symptoms?.jaundice || false,
        dark_urine: symptoms?.dark_urine || false,
        pain: symptoms?.pain || 0,
        fatigue: symptoms?.fatigue || 0,
        nausea: symptoms?.nausea || false,
        vomiting: symptoms?.vomiting || false,
        fever: symptoms?.fever || false,
        loss_of_appetite: symptoms?.loss_of_appetite || false,
        joint_pain: symptoms?.joint_pain || false
      },
      riskFactors: mapRiskFactors(riskFactors) || [],
      prediction: {
        predicted_class: predictionData.predicted_class || 'Unknown',
        confidence: predictionData.confidence || 0,
        probability_Hepatitis_A: predictionData['probability_Hepatitis A'] || 0,
        probability_Hepatitis_C: predictionData['probability_Hepatitis C'] || 0
      },
      status: 'completed'
    })

    // Save prediction to database
    const savedPrediction = await prediction.save()

    // Update user's predictions array if authenticated
    if (authenticatedUser) {
      await User.findByIdAndUpdate(
        authenticatedUser._id,
        { $push: { predictions: savedPrediction._id } }
      )
    }
        
    // Log prediction
    const logEntry = {
      timestamp: new Date().toISOString(),
      userData: { age, gender, symptoms, riskFactors },
      prediction: predictionResult,
    }
        
    fs.appendFileSync(
      path.join(__dirname, "prediction_logs.log"),
      JSON.stringify(logEntry) + "\n"
    )

    res.json({
      success: true,
      prediction: predictionResult.prediction,
      predictionId: savedPrediction._id
    })
  } catch (error) {
    console.error("Error in /api/predict:", error)
    res.status(500).json({
      success: false,
      message: "Prediction failed",
      error: error.message,
    })
  }
})

// Risk factor mapping from frontend IDs to backend enum values
function mapRiskFactors(frontendRiskFactors) {
  const riskFactorMapping = {
    'recentTravel': 'recent_travel',
    'bloodTransfusion': 'blood_transfusion_history', 
    'unsafeInjection': 'unsafe_injection_history',
    'contactWithInfected': 'infected_contact'
  }
  
  return (frontendRiskFactors || []).map(factor => 
    riskFactorMapping[factor] || factor
  )
}

// Enhanced rule-based prediction function (temporary fix for biased model)
async function performImprovedPrediction(age, gender, symptoms, riskFactors) {
  try {
    // Map risk factors from frontend format to backend format
    const mappedRiskFactors = mapRiskFactors(riskFactors)
    
    // Calculate symptoms and risk scores
    const symptomScore = calculateSymptomScore(symptoms)
    const riskScore = calculateRiskScore(age, mappedRiskFactors)
    
    // Determine prediction based on symptoms and risk factors
    const prediction = determineHepatitisType(symptoms, symptomScore, riskScore, age, mappedRiskFactors)
    
    console.log('Enhanced rule-based prediction:', prediction)
    
    return { prediction }
    
  } catch (error) {
    console.error('Error in enhanced prediction:', error)
    throw error
  }
}

function calculateSymptomScore(symptoms) {
  let score = 0
  
  // Acute symptoms (more associated with Hepatitis A)
  if (symptoms.jaundice) score += 2
  if (symptoms.fever) score += 2
  if (symptoms.nausea) score += 1
  if (symptoms.vomiting) score += 1
  if (symptoms.darkUrine || symptoms.dark_urine) score += 1
  if (symptoms.appetite || symptoms.loss_of_appetite) score += 1
  
  // Chronic symptoms (can be either, but fatigue/pain without acute symptoms lean toward C)
  if (symptoms.fatigue > 0) score += symptoms.fatigue > 3 ? 2 : 1
  if (symptoms.abdominalPain || symptoms.pain > 0) {
    const painLevel = symptoms.abdominalPain || symptoms.pain
    score += painLevel > 5 ? 2 : 1
  }
  if (symptoms.jointPain || symptoms.joint_pain) score += 1
  
  return score
}

function calculateRiskScore(age, riskFactors) {
  let riskScore = 0
  
  // Age-based risk
  if (age === 'under18' || age === '18-30') riskScore += 1 // Higher A risk
  if (age === 'over60') riskScore += 3 // Higher C risk
  if (age === '46-60') riskScore += 2 // Moderate C risk
  
  // Risk factors
  if (riskFactors.includes('recent_travel')) riskScore -= 2 // A indicator
  if (riskFactors.includes('blood_transfusion_history')) riskScore += 3 // C indicator
  if (riskFactors.includes('unsafe_injection_history')) riskScore += 3 // C indicator
  if (riskFactors.includes('infected_contact')) riskScore += 1 // Either, but slightly A
  
  return riskScore
}

function determineHepatitisType(symptoms, symptomScore, riskScore, age, riskFactors) {
  // Decision logic
  let hepatitisA_prob = 0.5 // Start with 50/50
  let hepatitisC_prob = 0.5
  
  // Acute presentation strongly suggests Hepatitis A
  const acuteSymptoms = (symptoms.jaundice ? 1 : 0) + 
                       (symptoms.fever ? 1 : 0) + 
                       (symptoms.nausea ? 1 : 0) + 
                       (symptoms.vomiting ? 1 : 0)
  
  if (acuteSymptoms >= 3) {
    hepatitisA_prob += 0.3
    hepatitisC_prob -= 0.3
  } else if (acuteSymptoms >= 2) {
    hepatitisA_prob += 0.2
    hepatitisC_prob -= 0.2
  }
  
  // Chronic, subtle symptoms suggest Hepatitis C
  const chronicSymptoms = (symptoms.fatigue > 2 ? 1 : 0) + 
                          ((symptoms.abdominalPain || symptoms.pain) > 2 ? 1 : 0) + 
                          (symptoms.jointPain || symptoms.joint_pain ? 1 : 0)
  
  if (chronicSymptoms >= 2 && acuteSymptoms <= 1) {
    hepatitisC_prob += 0.25
    hepatitisA_prob -= 0.25
  }
  
  // Risk factors adjustment
  if (riskFactors.includes('recent_travel')) {
    hepatitisA_prob += 0.2
    hepatitisC_prob -= 0.2
  }
  
  if (riskFactors.includes('blood_transfusion_history') || 
      riskFactors.includes('unsafe_injection_history')) {
    hepatitisC_prob += 0.3
    hepatitisA_prob -= 0.3
  }
  
  // Age adjustment
  if (age === 'under18' || age === '18-30') {
    hepatitisA_prob += 0.1
    hepatitisC_prob -= 0.1
  } else if (age === 'over60') {
    hepatitisC_prob += 0.2
    hepatitisA_prob -= 0.2
  }
  
  // Ensure probabilities are between 0 and 1
  hepatitisA_prob = Math.max(0.05, Math.min(0.95, hepatitisA_prob))
  hepatitisC_prob = 1 - hepatitisA_prob
  
  // Determine predicted class
  const predicted_class = hepatitisA_prob > hepatitisC_prob ? 'Hepatitis A' : 'Hepatitis C'
  const confidence = Math.max(hepatitisA_prob, hepatitisC_prob)
  
  // Create symptoms text
  const symptomsAnalyzed = createSymptomsText(symptoms)
  const severityAssessed = determineSeverity(symptomScore)
  
  return {
    predicted_class,
    confidence,
    'probability_Hepatitis A': hepatitisA_prob,
    'probability_Hepatitis C': hepatitisC_prob,
    symptoms_analyzed: symptomsAnalyzed,
    severity_assessed: severityAssessed,
    symptom_count: countSymptoms(symptoms)
  }
}

function createSymptomsText(symptoms) {
  const symptomList = []
  
  if (symptoms.jaundice) symptomList.push('jaundice')
  if (symptoms.darkUrine || symptoms.dark_urine) symptomList.push('dark-colored urine')
  if (symptoms.abdominalPain || symptoms.pain > 0) symptomList.push('abdominal pain')
  if (symptoms.fatigue > 0) symptomList.push('fatigue')
  if (symptoms.nausea) symptomList.push('nausea')
  if (symptoms.vomiting) symptomList.push('vomiting')
  if (symptoms.fever) symptomList.push('fever')
  if (symptoms.appetite || symptoms.loss_of_appetite) symptomList.push('loss of appetite')
  if (symptoms.jointPain || symptoms.joint_pain) symptomList.push('joint pain')
  
  return symptomList.length > 0 ? symptomList.join(', ') : 'no symptoms'
}

function determineSeverity(symptomScore) {
  if (symptomScore >= 8) return 'Severe'
  if (symptomScore >= 4) return 'Moderate'
  return 'Mild'
}

function countSymptoms(symptoms) {
  return [
    symptoms.jaundice,
    symptoms.darkUrine || symptoms.dark_urine,
    (symptoms.abdominalPain || symptoms.pain) > 0,
    symptoms.fatigue > 0,
    symptoms.nausea,
    symptoms.vomiting,
    symptoms.fever,
    symptoms.appetite || symptoms.loss_of_appetite,
    symptoms.jointPain || symptoms.joint_pain
  ].filter(Boolean).length
}

// Error handling middleware
app.use((err, req, res, next) => {
  console.error(err.stack)
  res.status(500).json({
    success: false,
    message: "Something went wrong!",
    error: process.env.NODE_ENV === "development" ? err.message : "Internal server error",
  })
})

// 404 handler
app.use((req, res) => {
  res.status(404).json({
    success: false,
    message: "Route not found",
  })
})

app.listen(PORT, () => {
  console.log(`Server running on port ${PORT}`)
})

function jwtSignForUser(user) {
  return jwt.sign({ id: user._id }, process.env.JWT_SECRET || 'your-secret-key', { expiresIn: '30d' })
}