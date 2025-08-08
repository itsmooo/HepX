import express from "express"
import cors from "cors"
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

// Route to handle prediction
app.post("/api/predict", protect, async (req, res) => {
  try {
    const { age, gender, symptoms, riskFactors } = req.body

    // Validate required fields
    if (!age || !gender) {
      return res.status(400).json({
        success: false,
        message: "Age and gender are required",
      })
    }

    // Use improved model prediction
    const predictionResult = await performImprovedPrediction(age, gender, symptoms, riskFactors)
    
    // Extract prediction data from the result
    const predictionData = predictionResult.prediction?.predictions?.[0] || predictionResult
    
    // Create prediction record in database
    const prediction = new Prediction({
      user: req.user._id,
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
      riskFactors: riskFactors || [],
      prediction: {
        predicted_class: predictionData.predicted_class || 'Unknown',
        confidence: predictionData.confidence || 0,
        probability_Hepatitis_A: predictionData.probability_Hepatitis_A || 0,
        probability_Hepatitis_C: predictionData.probability_Hepatitis_C || 0
      },
      status: 'completed'
    })

    // Save prediction to database
    const savedPrediction = await prediction.save()

    // Update user's predictions array
    await User.findByIdAndUpdate(
      req.user._id,
      { $push: { predictions: savedPrediction._id } }
    )
        
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
      prediction: predictionResult,
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

// Improved model-based prediction function using HTTP API
async function performImprovedPrediction(age, gender, symptoms, riskFactors) {
  try {
    // Get Python API URL from environment or use default
    const PYTHON_API_URL = process.env.PYTHON_API_URL || 'http://localhost:8000'
    
    // Prepare prediction data for the Python API
    const predictionData = {
      age: age,
      gender: gender,
      symptoms: {
        jaundice: symptoms.jaundice || false,
        dark_urine: symptoms.dark_urine || false,
        pain: symptoms.pain || false,
        fatigue: symptoms.fatigue || 0,
        nausea: symptoms.nausea || false,
        vomiting: symptoms.vomiting || false,
        fever: symptoms.fever || false,
        loss_of_appetite: symptoms.loss_of_appetite || false,
        joint_pain: symptoms.joint_pain || false
      },
      riskFactors: riskFactors || []
    }
    
    console.log('Sending prediction request to Python API:', PYTHON_API_URL)
    console.log('Prediction data:', JSON.stringify(predictionData, null, 2))
    
    // Make HTTP request to Python API
    const response = await fetch(`${PYTHON_API_URL}/predict`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify(predictionData)
    })
    
    if (!response.ok) {
      const errorData = await response.json()
      throw new Error(`Python API error: ${response.status} - ${errorData.detail || 'Unknown error'}`)
    }
    
    const result = await response.json()
    console.log('Prediction result from Python API:', result)
    
    return result
    
  } catch (error) {
    console.error('Error in improved prediction:', error)
    
    // If Python API is not available, provide a helpful error message
    if (error.code === 'ECONNREFUSED' || error.message.includes('fetch')) {
      throw new Error('Python prediction service is not available. Please start the Python API server using: python predict.py --server')
    }
    
    throw error
  }
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