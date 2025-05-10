import { type NextRequest, NextResponse } from "next/server"

// Define the backend API URL
const API_URL = process.env.NEXT_PUBLIC_API_URL || "http://localhost:5000/api"

export async function POST(request: NextRequest) {
  try {
    const formData = await request.formData()

    // Create a new FormData to send to the backend
    const backendFormData = new FormData()

    // Add the file if it exists
    const file = formData.get("file") as File | null
    if (file) {
      backendFormData.append("file", file)
    } else {
      // If no file is provided, we're using the symptoms form or chatbot
      const age = formData.get("age") as string
      const gender = formData.get("gender") as string
      const symptoms = JSON.parse(formData.get("symptoms") as string)
      const riskFactors = JSON.parse(formData.get("riskFactors") as string)

      // Create a CSV-like structure that matches the expected format for the model
      const csvData = createCSVFromSymptoms(age, gender, symptoms, riskFactors)

      // Create a file from the CSV data
      const csvFile = new File([csvData], "symptoms.csv", { type: "text/csv" })
      backendFormData.append("file", csvFile)
    }

    // Send the request to the backend which uses the trained model
    const response = await fetch(`${API_URL}/predict`, {
      method: "POST",
      body: backendFormData,
    })

    if (!response.ok) {
      const errorData = await response.json()
      return NextResponse.json({ error: errorData.error || "Failed to get prediction" }, { status: response.status })
    }

    const data = await response.json()

    // Process the prediction from the trained model
    if (data.predictions && data.predictions.length > 0) {
      // Find the prediction with the highest probability
      const prediction = data.predictions[0]

      // Determine the result based on the predicted class
      let result
      if (prediction.predicted_class.toLowerCase().includes("b")) {
        result = "hepatitisB"
      } else if (prediction.predicted_class.toLowerCase().includes("c")) {
        result = "hepatitisC"
      } else {
        result = "unlikely"
      }

      return NextResponse.json({
        ...data,
        result,
      })
    }

    // If for some reason we don't get predictions from the model
    return NextResponse.json(
      {
        error: "No predictions returned from model",
        result: "unknown",
      },
      { status: 500 },
    )
  } catch (error) {
    console.error("Error processing prediction:", error)
    return NextResponse.json({ error: "Failed to process prediction" }, { status: 500 })
  }
}

// Helper function to create CSV data from symptoms
function createCSVFromSymptoms(age: string, gender: string, symptoms: Record<string, any>, riskFactors: string[]) {
  // Create headers that match the expected format for the model
  const headers = [
    "PatientID",
    "Age",
    "Gender",
    "HepatitisType", // This will be what we're predicting
    "Symptoms",
    "SymptomCount",
    "Severity",
    "DiagnosisDate",
    "Treatment",
  ]

  // Calculate symptom count and severity
  const symptomCount = Object.values(symptoms).filter(
    (val) => (typeof val === "boolean" && val === true) || (typeof val === "number" && val > 50),
  ).length

  // Calculate average severity for numeric symptoms
  const numericSymptoms = Object.entries(symptoms)
    .filter(([_, val]) => typeof val === "number")
    .map(([_, val]) => val as number)

  const avgSeverity =
    numericSymptoms.length > 0
      ? numericSymptoms.reduce((sum, val) => sum + val, 0) / numericSymptoms.length / 20 // Scale to 0-5
      : 0

  // Create a description of symptoms
  const symptomDescription = Object.entries(symptoms)
    .map(([key, val]) => {
      if (typeof val === "boolean" && val === true) {
        return key
      } else if (typeof val === "number" && val > 0) {
        return `${key} (${val}%)`
      }
      return null
    })
    .filter(Boolean)
    .join(", ")

  // Add risk factors to the description
  const fullDescription =
    riskFactors.length > 0 ? `${symptomDescription}. Risk factors: ${riskFactors.join(", ")}` : symptomDescription

  // Create a row with the data
  const row = [
    `P${Date.now()}`, // Generate a unique patient ID
    age,
    gender,
    "", // HepatitisType is what we're predicting
    fullDescription,
    symptomCount.toString(),
    avgSeverity.toFixed(1),
    new Date().toISOString().split("T")[0], // Today's date
    "", // No treatment yet
  ]

  // Combine headers and row into CSV
  return headers.join(",") + "\n" + row.join(",")
}
