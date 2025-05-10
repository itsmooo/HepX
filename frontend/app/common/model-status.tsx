"use client"

import { useEffect, useState } from "react"
import { Alert, AlertDescription, AlertTitle } from "@/components/ui/alert"
import { Button } from "@/components/ui/button"
import { AlertCircle, CheckCircle2, Loader2 } from "lucide-react"

interface ModelStatusProps {
  onTrainModel: () => void
  isTraining: boolean
}

export default function ModelStatus({ onTrainModel, isTraining }: ModelStatusProps) {
  const [modelStatus, setModelStatus] = useState<{
    modelExists: boolean
    featureColumnsExist: boolean
    modelReady: boolean
  } | null>(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)

  const API_URL = process.env.NEXT_PUBLIC_API_URL || "http://localhost:5000/api"

  useEffect(() => {
    checkModelStatus()
  }, [isTraining])

  const checkModelStatus = async () => {
    try {
      setLoading(true)
      setError(null)

      const response = await fetch(`${API_URL}/model-status`)

      if (!response.ok) {
        throw new Error("Failed to check model status")
      }

      const data = await response.json()
      setModelStatus(data)
    } catch (error) {
      console.error("Error checking model status:", error)
      setError("Could not connect to the backend server")
    } finally {
      setLoading(false)
    }
  }

  if (loading) {
    return (
      <Alert className="bg-gray-50 border-gray-200">
        <Loader2 className="h-4 w-4 animate-spin" />
        <AlertTitle>Checking Model Status</AlertTitle>
        <AlertDescription>Connecting to the backend server...</AlertDescription>
      </Alert>
    )
  }

  if (error) {
    return (
      <Alert variant="destructive">
        <AlertCircle className="h-4 w-4" />
        <AlertTitle>Connection Error</AlertTitle>
        <AlertDescription>
          {error}. Please make sure the backend server is running.
          <Button variant="outline" size="sm" className="ml-2" onClick={checkModelStatus}>
            Retry
          </Button>
        </AlertDescription>
      </Alert>
    )
  }

  if (!modelStatus?.modelReady) {
    return (
      <Alert variant="destructive">
        <AlertCircle className="h-4 w-4" />
        <AlertTitle>Model Not Found</AlertTitle>
        <AlertDescription>
          The hepatitis prediction model is not ready. Please train the model first.
          <Button variant="outline" size="sm" className="ml-2" onClick={onTrainModel} disabled={isTraining}>
            {isTraining ? (
              <>
                <Loader2 className="mr-2 h-4 w-4 animate-spin" /> Training...
              </>
            ) : (
              "Train Model"
            )}
          </Button>
        </AlertDescription>
      </Alert>
    )
  }

  return (
    <Alert className="bg-green-50 border-green-200">
      <CheckCircle2 className="h-4 w-4 text-green-600" />
      <AlertTitle className="text-green-800">Model Ready</AlertTitle>
      <AlertDescription className="text-green-700">
        The hepatitis prediction model is trained and ready to use.
      </AlertDescription>
    </Alert>
  )
}
