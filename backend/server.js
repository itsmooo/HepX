import express from "express"
import cors from "cors"
import { spawn } from "child_process"
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

// Download dataset from Vercel Blob
async function downloadDataset() {
  try {
    const url =
      "https://hebbkx1anhila5yf.public.blob.vercel-storage.com/hepatitis_dataset_R-bHcoR7i8zhDoLlixNRjHOWUwDx82VG.csv"
    const response = await fetch(url)

    if (!response.ok) {
      throw new Error(`Failed to download dataset: ${response.statusText}`)
    }

    const fileStream = fs.createWriteStream(path.join(uploadsDir, "hepatitis_dataset.csv"))
    await new Promise((resolve, reject) => {
      response.body.pipe(fileStream)
      response.body.on("error", reject)
      fileStream.on("finish", resolve)
    })

    console.log("Dataset downloaded successfully")
    return true
  } catch (error) {
    console.error("Error downloading dataset:", error)
    return false
  }
}

// Run Python script
function runPythonScript() {
  return new Promise((resolve, reject) => {
    const datasetPath = path.join(uploadsDir, "hepatitis_dataset.csv")

    // Check if dataset exists
    if (!fs.existsSync(datasetPath)) {
      return reject(new Error("Dataset file not found"))
    }

    const pythonProcess = spawn("python", [path.join(__dirname, "model_training.py")])

    let pythonOutput = ""
    let pythonError = ""

    pythonProcess.stdout.on("data", (data) => {
      console.log(`Python stdout: ${data}`)
      pythonOutput += data.toString()
    })

    pythonProcess.stderr.on("data", (data) => {
      console.error(`Python stderr: ${data}`)
      pythonError += data.toString()
    })

    pythonProcess.on("close", (code) => {
      if (code !== 0) {
        console.error(`Python process exited with code ${code}`)
        reject(new Error(`Python process exited with code ${code}: ${pythonError}`))
      } else {
        console.log("Python process completed successfully")
        resolve(pythonOutput)
      }
    })
  })
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

// Route to trigger model training
app.post("/api/train-model", async (req, res) => {
  try {
    // Download dataset if not already done
    const downloadSuccess = await downloadDataset()
    if (!downloadSuccess) {
      return res.status(500).json({ error: "Failed to download dataset" })
    }

    // Run Python script
    await runPythonScript()

    // Check if results file exists
    const resultsPath = path.join(outputDir, "model_results.json")
    if (!fs.existsSync(resultsPath)) {
      return res.status(500).json({ error: "Model training failed to produce results" })
    }

    // Read results
    const results = JSON.parse(fs.readFileSync(resultsPath, "utf8"))

    res.json({
      success: true,
      message: "Model trained successfully",
      results,
    })
  } catch (error) {
    console.error("Error in /api/train-model:", error)
    res.status(500).json({
      error: "Failed to train model",
      details: error.message,
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

    // Check if model file exists
    const modelPath = path.join(outputDir, "hepatitis_model.pkl")
    if (!fs.existsSync(modelPath)) {
      return res.status(500).json({
        success: false,
        message: "Model not found. Please train the model first.",
      })
    }

    // Prepare data for Python script
    const predictionData = {
      age,
      gender,
      symptoms: {
        jaundice: symptoms.jaundice || false,
        dark_urine: symptoms.dark_urine || false,
        pain: symptoms.pain || false,
        fatigue: symptoms.fatigue || 0,
        nausea: symptoms.nausea || false,
        vomiting: symptoms.vomiting || false,
        fever: symptoms.fever || false,
        loss_of_appetite: symptoms.loss_of_appetite || false,
        joint_pain: symptoms.joint_pain || false,
      },
      riskFactors: riskFactors || [],
    }

    // Save prediction data to file
    const predictionDataPath = path.join(uploadsDir, `prediction_${Date.now()}.json`)
    fs.writeFileSync(predictionDataPath, JSON.stringify(predictionData))

    // Run Python prediction script
    const pythonProcess = spawn("python", [
      path.join(__dirname, "predict.py"),
      predictionDataPath,
    ])

    let pythonOutput = ""
    let pythonError = ""

    pythonProcess.stdout.on("data", (data) => {
      pythonOutput += data.toString()
    })

    pythonProcess.stderr.on("data", (data) => {
      pythonError += data.toString()
    })

    pythonProcess.on("close", (code) => {
      if (code !== 0) {
        console.error(`Python prediction process exited with code ${code}`)
        return res.status(500).json({
          success: false,
          message: "Prediction failed",
          error: pythonError,
        })
      }

      try {
        // Parse Python output
        const predictionResult = JSON.parse(pythonOutput)
        
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
        })
      } catch (parseError) {
        console.error("Error parsing prediction result:", parseError)
        res.status(500).json({
          success: false,
          message: "Error processing prediction result",
        })
      }
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