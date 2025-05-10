"use client"
import { useState } from "react"
import Header from "./common/header"
import Hero from "./common/hero"
import SymptomsForm from "./common/symptoms-form"
import Education from "./common/education"
import Footer from "./common/footer"
import ModelStatus from "./common/model-status"
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { Progress } from "@/components/ui/progress"
import { Alert, AlertDescription, AlertTitle } from "@/components/ui/alert"
import { CheckCircle2, AlertCircle } from "lucide-react"
import { useToast } from "./hooks/use-toast"

export default function Home() {
  const { toast } = useToast()
  const [isTraining, setIsTraining] = useState(false)
  const [trainProgress, setTrainProgress] = useState(0)
  const [error, setError] = useState<string | null>(null)

  const API_URL = process.env.NEXT_PUBLIC_API_URL || "http://localhost:5000/api"

  const trainModel = async () => {
    try {
      setIsTraining(true)
      setTrainProgress(0)
      setError(null)

      // Simulate progress while training
      const progressInterval = setInterval(() => {
        setTrainProgress((prev) => {
          const newProgress = prev + Math.random() * 10
          return newProgress >= 95 ? 95 : newProgress
        })
      }, 1000)

      const response = await fetch(`${API_URL}/train-model`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
      })

      clearInterval(progressInterval)

      if (!response.ok) {
        const errorData = await response.json()
        throw new Error(errorData.error || "Failed to train model")
      }

      const data = await response.json()
      setTrainProgress(100)

      toast({
        title: "Model Trained Successfully",
        description: `Model accuracy: ${(data.results.accuracy * 100).toFixed(2)}%`,
      })
    } catch (err: any) {
      setError(err.message || "An unexpected error occurred")
      setTrainProgress(0)

      toast({
        title: "Training Failed",
        description: err.message || "An unexpected error occurred",
        variant: "destructive",
      })
    } finally {
      setIsTraining(false)
    }
  }

  return (
    <div className="min-h-screen flex flex-col">
      <Header />
      <main className="flex-grow">
        <Hero />

        <section className="py-8 px-6 sm:px-10 bg-gray-50">
          <div className="max-w-4xl mx-auto">
            <Card>
              <CardHeader>
                <CardTitle>Hepatitis Prediction Model</CardTitle>
                <CardDescription>
                  This application uses a machine learning model to predict hepatitis types based on symptoms
                </CardDescription>
              </CardHeader>
              <CardContent>
                <ModelStatus onTrainModel={trainModel} isTraining={isTraining} />

                {isTraining && (
                  <div className="mt-4">
                    <p className="mb-2 text-sm font-medium">Training Progress</p>
                    <Progress value={trainProgress} className="h-2" />
                    <p className="mt-1 text-xs text-gray-500">
                      {trainProgress < 100 ? "Processing data and training model..." : "Training complete!"}
                    </p>
                  </div>
                )}

                {error && (
                  <Alert variant="destructive" className="mt-4">
                    <AlertCircle className="h-4 w-4" />
                    <AlertTitle>Error</AlertTitle>
                    <AlertDescription>{error}</AlertDescription>
                  </Alert>
                )}

                {trainProgress === 100 && !isTraining && (
                  <Alert className="mt-4 bg-green-50 border-green-200">
                    <CheckCircle2 className="h-4 w-4 text-green-600" />
                    <AlertTitle className="text-green-800">Model Trained Successfully</AlertTitle>
                    <AlertDescription className="text-green-700">
                      The model is now ready to make predictions.
                    </AlertDescription>
                  </Alert>
                )}
              </CardContent>
            </Card>
          </div>
        </section>

        <SymptomsForm />
        <Education />
      </main>
      <Footer />
    </div>
  )
}
