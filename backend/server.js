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
import { protect } from "./middleware/auth.js"

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

// API Routes
app.get("/api/status", (req, res) => {
  res.json({ status: "API is running" })
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

    // Simple rule-based prediction (fallback when ML model fails)
    const prediction = performSimplePrediction(age, gender, symptoms, riskFactors)
    
    // Log prediction
    const logEntry = {
      timestamp: new Date().toISOString(),
      userData: { age, gender, symptoms, riskFactors },
      prediction: prediction,
    }
    
    fs.appendFileSync(
      path.join(__dirname, "prediction_logs.log"),
      JSON.stringify(logEntry) + "\n"
    )

    res.json({
      success: true,
      prediction: prediction,
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

// Simple rule-based prediction function
function performSimplePrediction(age, gender, symptoms, riskFactors) {
  // Count symptoms
  const symptomCount = Object.values(symptoms).filter(s => s === true || (typeof s === 'number' && s > 0)).length
  
  // Calculate severity
  let severity = 1 // mild
  if (symptomCount >= 5 || age > 60) {
    severity = 3 // severe
  } else if (symptomCount >= 3 || age > 40) {
    severity = 2 // moderate
  }
  
  // Simple rules for hepatitis type prediction
  let predictedClass = "Hepatitis A"
  let probabilityA = 0.6
  let probabilityC = 0.4
  
  // Adjust based on symptoms
  if (symptoms.jaundice) {
    probabilityA += 0.1
    probabilityC -= 0.1
  }
  
  if (symptoms.fatigue > 0) {
    probabilityA += 0.05
    probabilityC += 0.05
  }
  
  if (symptoms.pain) {
    probabilityA += 0.05
    probabilityC += 0.05
  }
  
  // Normalize probabilities
  const total = probabilityA + probabilityC
  probabilityA = probabilityA / total
  probabilityC = probabilityC / total
  
  return {
    success: true,
    message: "Prediction completed successfully",
    predictions: [{
      predicted_class: predictedClass,
      "probability_Hepatitis A": probabilityA,
      "probability_Hepatitis C": probabilityC
    }],
    total_predictions: 1
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