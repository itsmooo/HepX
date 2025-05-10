import express from "express"
import cors from "cors"
import { spawn } from "child_process"
import fs from "fs"
import path from "path"
import { fileURLToPath } from "url"
import { dirname } from "path"
import multer from "multer"
import fetch from "node-fetch"

const __filename = fileURLToPath(import.meta.url)
const __dirname = dirname(__filename)

const app = express()
const PORT = process.env.PORT || 5000

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
    console.error(`Error in /api/visualization/${req.params.filename}:`, error)
    res.status(500).json({ error: "Failed to retrieve visualization" })
  }
})

// Route to check if model exists
app.get("/api/model-status", (req, res) => {
  try {
    const modelPath = path.join(outputDir, "hepatitis_model.pkl")
    const featureColumnsPath = path.join(outputDir, "feature_columns.json")

    const modelExists = fs.existsSync(modelPath)
    const featureColumnsExist = fs.existsSync(featureColumnsPath)

    res.json({
      modelExists,
      featureColumnsExist,
      modelReady: modelExists && featureColumnsExist,
    })
  } catch (error) {
    console.error("Error in /api/model-status:", error)
    res.status(500).json({ error: "Failed to check model status" })
  }
})

// Route to download the trained model
app.get("/api/download-model", (req, res) => {
  try {
    const modelPath = path.join(outputDir, "hepatitis_model.pkl")

    if (!fs.existsSync(modelPath)) {
      return res.status(404).json({ error: "Model file not found. Train the model first." })
    }

    res.download(modelPath, "hepatitis_model.pkl")
  } catch (error) {
    console.error("Error in /api/download-model:", error)
    res.status(500).json({ error: "Failed to download model" })
  }
})

// Route to make predictions on a new dataset
app.post("/api/predict", upload.single("file"), (req, res) => {
  try {
    if (!req.file) {
      return res.status(400).json({ error: "No file uploaded" })
    }

    const inputFile = req.file.path
    const outputFile = path.join(uploadsDir, `predictions_${Date.now()}.csv`)

    // Run the prediction script
    const pythonProcess = spawn("python", [path.join(__dirname, "predict.py"), inputFile, outputFile])

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
        return res.status(500).json({
          error: "Prediction failed",
          details: pythonError,
        })
      }

      try {
        // Parse the JSON output from the Python script
        const results = JSON.parse(pythonOutput)

        // Add download link for the predictions CSV
        if (fs.existsSync(outputFile)) {
          results.downloadUrl = `/api/download-predictions/${path.basename(outputFile)}`
        }

        res.json(results)
      } catch (parseError) {
        console.error("Error parsing Python output:", parseError)
        res.status(500).json({
          error: "Failed to parse prediction results",
          details: pythonOutput,
        })
      }
    })
  } catch (error) {
    console.error("Error in /api/predict:", error)
    res.status(500).json({ error: "Prediction failed", details: error.message })
  }
})

// Route to download prediction results
app.get("/api/download-predictions/:filename", (req, res) => {
  try {
    const { filename } = req.params
    const filePath = path.join(uploadsDir, filename)

    if (!fs.existsSync(filePath)) {
      return res.status(404).json({ error: "Prediction file not found" })
    }

    res.download(filePath, "hepatitis_predictions.csv")
  } catch (error) {
    console.error(`Error in /api/download-predictions/${req.params.filename}:`, error)
    res.status(500).json({ error: "Failed to download predictions" })
  }
})

// Start server
app.listen(PORT, () => {
  console.log(`Server running on port ${PORT}`)
})
