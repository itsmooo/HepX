"use client"

import type React from "react"
import { useState } from "react"
import { Card, CardContent, CardDescription, CardFooter, CardHeader, CardTitle } from "@/components/ui/card"
import { Button } from "@/components/ui/button"
import { Slider } from "@/components/ui/slider"
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select"
import { Switch } from "@/components/ui/switch"
import { Label } from "@/components/ui/label"
import { useToast } from "../hooks/use-toast"
import { Loader2 } from "lucide-react"

const symptoms = [
  { id: "fatigue", label: "Fatigue", type: "slider" },
  { id: "nausea", label: "Nausea", type: "slider" },
  { id: "abdominalPain", label: "Abdominal Pain", type: "slider" },
  { id: "jaundice", label: "Yellowing of Skin/Eyes (Jaundice)", type: "switch" },
  { id: "darkUrine", label: "Dark Urine", type: "switch" },
  { id: "jointPain", label: "Joint Pain", type: "switch" },
  { id: "appetite", label: "Loss of Appetite", type: "switch" },
  { id: "fever", label: "Fever", type: "slider" },
]

const riskFactors = [
  { id: "recentTravel", label: "Recent Travel to High-Risk Areas" },
  { id: "bloodTransfusion", label: "History of Blood Transfusion" },
  { id: "unsafeInjection", label: "History of Unsafe Injection Practices" },
  { id: "contactWithInfected", label: "Contact with Infected Person" },
]

type FormState = {
  symptoms: Record<string, number | boolean>
  age: string
  gender: string
  riskFactors: string[]
  hepatitisType: string // Added for direct selection
}

type PredictionResult = {
  result: "hepatitisB" | "hepatitisC" | "unlikely" | null
  score: number
  details: {
    symptomScore: number
    riskScore: number
  }
  predictions?: Array<{
    predicted_class: string
    [key: string]: any
  }>
  downloadUrl?: string
}

const SymptomsForm = () => {
  const { toast } = useToast()
  const [formState, setFormState] = useState<FormState>({
    symptoms: {
      fatigue: 0,
      nausea: 0,
      abdominalPain: 0,
      jaundice: false,
      darkUrine: false,
      jointPain: false,
      appetite: false,
      fever: 0,
    },
    age: "",
    gender: "",
    riskFactors: [],
    hepatitisType: "", // Default to empty
  })
  const [showResults, setShowResults] = useState(false)
  const [predictionResult, setPredictionResult] = useState<PredictionResult | null>(null)
  const [loading, setLoading] = useState(false)

  const handleSymptomChange = (id: string, value: number | boolean) => {
    setFormState((prev) => ({
      ...prev,
      symptoms: {
        ...prev.symptoms,
        [id]: value,
      },
    }))
  }

  const handleRiskFactorToggle = (id: string) => {
    setFormState((prev) => {
      const riskFactors = [...prev.riskFactors]
      if (riskFactors.includes(id)) {
        return {
          ...prev,
          riskFactors: riskFactors.filter((factorId) => factorId !== id),
        }
      } else {
        return {
          ...prev,
          riskFactors: [...riskFactors, id],
        }
      }
    })
  }

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault()

    // Validation
    if (!formState.age || !formState.gender || !formState.hepatitisType) {
      toast({
        title: "Missing Information",
        description: "Please select your age group, gender, and hepatitis type to continue.",
        variant: "destructive",
      })
      return
    }

    try {
      setLoading(true)

      // Instead of calling the API, we'll use the selected hepatitis type directly
      const result: PredictionResult = {
        result: formState.hepatitisType as "hepatitisB" | "hepatitisC" | "unlikely",
        score: 0,
        details: {
          symptomScore: 0,
          riskScore: 0,
        },
      }

      // Calculate symptom score for display purposes
      const symptomScore = Object.values(formState.symptoms).filter(
        (val) => (typeof val === "boolean" && val === true) || (typeof val === "number" && val > 50),
      ).length

      // Calculate risk score for display purposes
      const riskScore = formState.riskFactors.length

      result.score = symptomScore + riskScore
      result.details.symptomScore = symptomScore
      result.details.riskScore = riskScore

      setPredictionResult(result)
      setShowResults(true)

      // Scroll to results
      setTimeout(() => {
        const resultsElement = document.getElementById("prediction-results")
        if (resultsElement) {
          resultsElement.scrollIntoView({ behavior: "smooth" })
        }
      }, 100)
    } catch (error) {
      toast({
        title: "Error",
        description: "There was an error processing your symptoms. Please try again.",
        variant: "destructive",
      })
    } finally {
      setLoading(false)
    }
  }

  const resetForm = () => {
    setFormState({
      symptoms: {
        fatigue: 0,
        nausea: 0,
        abdominalPain: 0,
        jaundice: false,
        darkUrine: false,
        jointPain: false,
        appetite: false,
        fever: 0,
      },
      age: "",
      gender: "",
      riskFactors: [],
      hepatitisType: "",
    })
    setShowResults(false)
    setPredictionResult(null)
  }

  const downloadResults = () => {
    if (predictionResult?.downloadUrl) {
      const API_URL = process.env.NEXT_PUBLIC_API_URL || "http://localhost:5000/api"
      window.open(`${API_URL}${predictionResult.downloadUrl}`, "_blank")
    } else {
      // If no download URL is available, create a simple text file with the results
      const resultText = `
HepaPredict Results
-------------------
Date: ${new Date().toLocaleString()}

Patient Information:
- Age Group: ${formState.age}
- Gender: ${formState.gender}

Prediction Result: ${
        predictionResult?.result === "hepatitisB"
          ? "Potential Signs of Hepatitis B"
          : predictionResult?.result === "hepatitisC"
            ? "Potential Signs of Hepatitis C"
            : "Low Risk of Hepatitis"
      }

Risk Assessment:
- Total Score: ${predictionResult?.score || 0}
- Symptom Score: ${predictionResult?.details?.symptomScore || 0}
- Risk Factor Score: ${predictionResult?.details?.riskScore || 0}

Reported Symptoms:
${Object.entries(formState.symptoms)
  .map(([key, value]) => {
    if (typeof value === "boolean") {
      return value ? `- ${key}: Yes` : `- ${key}: No`
    } else {
      return `- ${key}: ${value}%`
    }
  })
  .join("\n")}

Risk Factors:
${formState.riskFactors.length > 0 ? formState.riskFactors.map((factor) => `- ${factor}`).join("\n") : "- None reported"}

Important Note:
This prediction is not a medical diagnosis. It's based on the information provided and is intended to guide your next steps. 
For accurate diagnosis, please consult with a healthcare professional.
      `

      const blob = new Blob([resultText], { type: "text/plain" })
      const url = URL.createObjectURL(blob)
      const a = document.createElement("a")
      a.href = url
      a.download = "HepaPredict_Results.txt"
      document.body.appendChild(a)
      a.click()
      document.body.removeChild(a)
      URL.revokeObjectURL(url)
    }
  }

  return (
    <section id="predict" className="py-16 px-6 sm:px-10">
      <div className="max-w-4xl mx-auto">
        <div className="text-center mb-10">
          <h2 className="text-3xl font-bold text-foreground mb-3">Check Your Symptoms</h2>
          <p className="text-lg text-muted-foreground max-w-2xl mx-auto">
            Answer these questions about your symptoms and risk factors to get insights about potential hepatitis types.
          </p>
        </div>

        <Card className="mb-10">
          <CardHeader>
            <CardTitle>Symptom Checker</CardTitle>
            <CardDescription>Rate your symptoms and provide some basic information</CardDescription>
          </CardHeader>
          <CardContent>
            <form onSubmit={handleSubmit}>
              <div className="grid gap-6">
                <div className="space-y-4">
                  <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
                    <div>
                      <Label htmlFor="age">Age Group</Label>
                      <Select
                        value={formState.age}
                        onValueChange={(value) => setFormState((prev) => ({ ...prev, age: value }))}
                      >
                        <SelectTrigger id="age" className="w-full mt-1">
                          <SelectValue placeholder="Select age group" />
                        </SelectTrigger>
                        <SelectContent>
                          <SelectItem value="under18">Under 18</SelectItem>
                          <SelectItem value="18-30">18-30</SelectItem>
                          <SelectItem value="31-45">31-45</SelectItem>
                          <SelectItem value="46-60">46-60</SelectItem>
                          <SelectItem value="over60">Over 60</SelectItem>
                        </SelectContent>
                      </Select>
                    </div>
                    <div>
                      <Label htmlFor="gender">Gender</Label>
                      <Select
                        value={formState.gender}
                        onValueChange={(value) => setFormState((prev) => ({ ...prev, gender: value }))}
                      >
                        <SelectTrigger id="gender" className="w-full mt-1">
                          <SelectValue placeholder="Select gender" />
                        </SelectTrigger>
                        <SelectContent>
                          <SelectItem value="male">Male</SelectItem>
                          <SelectItem value="female">Female</SelectItem>
                          <SelectItem value="other">Other</SelectItem>
                        </SelectContent>
                      </Select>
                    </div>
                    <div>
                      <Label htmlFor="hepatitisType">Hepatitis Type</Label>
                      <Select
                        value={formState.hepatitisType}
                        onValueChange={(value) => setFormState((prev) => ({ ...prev, hepatitisType: value }))}
                      >
                        <SelectTrigger id="hepatitisType" className="w-full mt-1">
                          <SelectValue placeholder="Select hepatitis type" />
                        </SelectTrigger>
                        <SelectContent>
                          <SelectItem value="hepatitisB">Hepatitis B</SelectItem>
                          <SelectItem value="hepatitisC">Hepatitis C</SelectItem>
                          <SelectItem value="unlikely">No Hepatitis</SelectItem>
                        </SelectContent>
                      </Select>
                    </div>
                  </div>

                  <div className="mt-8">
                    <h3 className="text-lg font-semibold mb-4">Symptoms</h3>
                    <div className="space-y-6">
                      {symptoms.map((symptom) => (
                        <div key={symptom.id} className="space-y-2">
                          {symptom.type === "slider" ? (
                            <>
                              <div className="flex justify-between">
                                <Label>{symptom.label}</Label>
                                <span className="text-sm text-muted-foreground">
                                  {formState.symptoms[symptom.id] as number}%
                                </span>
                              </div>
                              <Slider
                                value={[formState.symptoms[symptom.id] as number]}
                                min={0}
                                max={100}
                                step={1}
                                onValueChange={(value) => handleSymptomChange(symptom.id, value[0])}
                              />
                              <div className="flex justify-between text-xs text-muted-foreground">
                                <span>None</span>
                                <span>Mild</span>
                                <span>Moderate</span>
                                <span>Severe</span>
                              </div>
                            </>
                          ) : (
                            <div className="flex items-center justify-between">
                              <Label htmlFor={symptom.id} className="flex-1">
                                {symptom.label}
                              </Label>
                              <Switch
                                id={symptom.id}
                                checked={formState.symptoms[symptom.id] as boolean}
                                onCheckedChange={(checked) => handleSymptomChange(symptom.id, checked)}
                              />
                            </div>
                          )}
                        </div>
                      ))}
                    </div>
                  </div>

                  <div className="mt-8">
                    <h3 className="text-lg font-semibold mb-4">Risk Factors</h3>
                    <div className="space-y-4">
                      {riskFactors.map((factor) => (
                        <div key={factor.id} className="flex items-center justify-between">
                          <Label htmlFor={factor.id} className="flex-1">
                            {factor.label}
                          </Label>
                          <Switch
                            id={factor.id}
                            checked={formState.riskFactors.includes(factor.id)}
                            onCheckedChange={() => handleRiskFactorToggle(factor.id)}
                          />
                        </div>
                      ))}
                    </div>
                  </div>
                </div>
              </div>

              <div className="mt-8 flex justify-end">
                <Button type="submit" className="hepa-button" disabled={loading}>
                  {loading ? (
                    <>
                      <Loader2 className="mr-2 h-4 w-4 animate-spin" /> Processing...
                    </>
                  ) : (
                    "Submit Information"
                  )}
                </Button>
              </div>
            </form>
          </CardContent>
        </Card>

        {showResults && predictionResult && (
          <div id="prediction-results" className="animate-fade-in">
            <Card className="border-t-4 border-t-primary">
              <CardHeader>
                <CardTitle>Your Prediction Results</CardTitle>
                <CardDescription>Based on the information you provided</CardDescription>
              </CardHeader>
              <CardContent>
                <div className="mb-6 p-4 rounded-lg bg-muted">
                  <div className="text-center">
                    <h3 className="text-xl font-bold mb-2">
                      {predictionResult.result === "hepatitisB" && "Potential Signs of Hepatitis B"}
                      {predictionResult.result === "hepatitisC" && "Potential Signs of Hepatitis C"}
                      {predictionResult.result === "unlikely" && "Low Risk of Hepatitis"}
                    </h3>

                    <p className="text-muted-foreground mb-4">
                      {predictionResult.result === "hepatitisB" &&
                        "Your symptoms suggest possible signs of Hepatitis B. This does not replace professional medical advice."}
                      {predictionResult.result === "hepatitisC" &&
                        "Your symptoms suggest possible signs of Hepatitis C. This does not replace professional medical advice."}
                      {predictionResult.result === "unlikely" &&
                        "Your symptoms suggest a lower likelihood of hepatitis. However, if symptoms persist, please consult a doctor."}
                    </p>

                    <div
                      className={`inline-block px-4 py-2 rounded-full font-semibold text-white ${
                        predictionResult.result === "unlikely" ? "bg-green-500" : "bg-amber-500"
                      }`}
                    >
                      {predictionResult.result === "hepatitisB" && "Hepatitis B Possible"}
                      {predictionResult.result === "hepatitisC" && "Hepatitis C Possible"}
                      {predictionResult.result === "unlikely" && "Low Risk Detected"}
                    </div>
                  </div>
                </div>

                <div className="bg-primary/10 p-4 rounded-lg">
                  <h4 className="font-semibold mb-2">Important Note:</h4>
                  <p>
                    This prediction is not a medical diagnosis. It's based on the information you provided and is
                    intended to guide your next steps. For accurate diagnosis, please consult with a healthcare
                    professional.
                  </p>
                </div>

                <div className="mt-6">
                  <h4 className="font-semibold mb-3">Next Steps:</h4>
                  <ul className="space-y-2">
                    <li className="flex items-start">
                      <span className="mr-2 text-primary">✓</span>
                      <span>Consult with a healthcare provider for proper testing and diagnosis</span>
                    </li>
                    <li className="flex items-start">
                      <span className="mr-2 text-primary">✓</span>
                      <span>Mention these specific symptoms to your doctor</span>
                    </li>
                    <li className="flex items-start">
                      <span className="mr-2 text-primary">✓</span>
                      <span>Stay hydrated and get plenty of rest in the meantime</span>
                    </li>
                    <li className="flex items-start">
                      <span className="mr-2 text-primary">✓</span>
                      <span>Learn more about hepatitis in our education section below</span>
                    </li>
                  </ul>
                </div>
              </CardContent>
              <CardFooter className="flex justify-between">
                <Button variant="outline" onClick={resetForm}>
                  Start Over
                </Button>

                <Button className="bg-foreground hover:bg-foreground/80 text-background" onClick={downloadResults}>
                  Download Results
                </Button>
              </CardFooter>
            </Card>
          </div>
        )}
      </div>
    </section>
  )
}

export default SymptomsForm




