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

// Route to handle predictions
app.post("/api/predict", express.json(), async (req, res) => {
  try {
    const { age, gender, symptoms, riskFactors } = req.body;
    
    if (!age || !gender || !symptoms) {
      return res.status(400).json({
        error: 'Missing required fields'
      });
    }

    // Create input data for prediction with proper feature names
    const inputData = {
      has_jaundice: symptoms.jaundice ? 1 : 0,
      has_dark_urine: symptoms.dark_urine ? 1 : 0,
      has_pain: symptoms.pain ? 1 : 0,
      has_fatigue: symptoms.fatigue ? 1 : 0,
      has_nausea: symptoms.nausea ? 1 : 0,
      has_vomiting: symptoms.vomiting ? 1 : 0,
      has_fever: symptoms.fever ? 1 : 0,
      has_loss_of_appetite: symptoms.loss_of_appetite ? 1 : 0,
      has_joint_pain: symptoms.joint_pain ? 1 : 0,
      SymptomCount: Object.values(symptoms).filter(Boolean).length,
      Severity: 1.0,
      SymptomLength: Object.values(symptoms).join(' ').length,
      DiagnosisDate: new Date().toISOString().split('T')[0]
    };
    
    // Save input data to CSV with headers
    const headers = Object.keys(inputData);
    const values = Object.values(inputData);
    
    const csvData = [
      headers.join(','),
      values.join(',')
    ].join('\n');
    
    fs.writeFileSync(path.join(uploadsDir, 'hepatitis_dataset.csv'), csvData);
    
    // Run prediction
    const pythonProcess = spawn('python', [path.join(__dirname, 'predict.py'), path.join(uploadsDir, 'hepatitis_dataset.csv')]);
    
    let pythonOutput = '';
    let pythonError = '';
    
    pythonProcess.stdout.on('data', (data) => {
      console.log(`Python stdout: ${data}`);
      pythonOutput += data.toString();
    });
    
    pythonProcess.stderr.on('data', (data) => {
      console.error(`Python stderr: ${data}`);
      pythonError += data.toString();
    });
    
    pythonProcess.on('close', (code) => {
      if (code !== 0) {
        console.error(`Python process exited with code ${code}`);
        return res.status(500).json({ 
          error: pythonError,
          details: "Prediction failed. Please check the server logs for more information."
        });
      }
      
      try {
        const result = JSON.parse(pythonOutput);
        if (result.error) {
          return res.status(500).json({ 
            error: result.error,
            details: "Error during prediction process"
          });
        }
        res.json({
          success: true,
          prediction: result
        });
      } catch (error) {
        console.error('Error parsing Python output:', error);
        res.status(500).json({ 
          error: 'Failed to parse prediction results',
          details: error.message
        });
      }
    });
  } catch (error) {
    console.error('Error in /api/predict:', error);
    res.status(500).json({
      error: error.message || 'Internal server error'
    });
  }
});

// Add health check endpoint
app.get("/api/health", (req, res) => {
  console.log("Health check requested");
  res.json({ status: "Server is running" });
});

// Add logging middleware
app.use((req, res, next) => {
  console.log(`${new Date().toISOString()} - ${req.method} ${req.url}`);
  next();
});

// Error handling middleware
app.use((err, req, res, next) => {
  console.error('Server error:', err);
  res.status(500).json({ error: 'Internal server error' });
});

// Start server
const server = app.listen(PORT, () => {
  console.log(`Server running on port ${PORT}`);
  console.log(`Health check available at http://localhost:${PORT}/api/health`);
  console.log(`Prediction endpoint available at http://localhost:${PORT}/api/predict`);
});

// Handle server errors
server.on('error', (error) => {
  console.error('Server error:', error);
});

// Handle process termination
process.on('SIGTERM', () => {
  console.log('SIGTERM signal received: closing HTTP server');
  server.close(() => {
    console.log('HTTP server closed');
  });
});

process.on('SIGINT', () => {
  console.log('SIGINT signal received: closing HTTP server');
  server.close(() => {
    console.log('HTTP server closed');
  });
});