"use client"

import type React from "react"

import { useState } from "react"
import { Button } from "@/components/ui/button"
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { Input } from "@/components/ui/input"
import { Label } from "@/components/ui/label"
import { Alert, AlertDescription, AlertTitle } from "@/components/ui/alert"
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from "@/components/ui/table"
import { Badge } from "@/components/ui/badge"
import { Progress } from "@/components/ui/progress"
import { AlertCircle, CheckCircle2, Download, Upload } from "lucide-react"

interface PredictionFormProps {
  apiUrl: string
}

export default function PredictionForm({ apiUrl }: PredictionFormProps) {
  const [file, setFile] = useState<File | null>(null)
  const [loading, setLoading] = useState(false)
  const [progress, setProgress] = useState(0)
  const [error, setError] = useState<string | null>(null)
  const [results, setResults] = useState<any>(null)
  const [modelStatus, setModelStatus] = useState<{
    modelExists: boolean
    featureColumnsExist: boolean
    modelReady: boolean
  } | null>(null)

  // Check if model exists
  const checkModelStatus = async () => {
    try {
      const response = await fetch(`${apiUrl}/model-status`)
      if (response.ok) {
        const data = await response.json()
        setModelStatus(data)
      }
    } catch (err) {
      console.error("Error checking model status:", err)
    }
  }

  // Call checkModelStatus on component mount
  useState(() => {
    checkModelStatus()
  })

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files.length > 0) {
      setFile(e.target.files[0])
    }
  }

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault()
    if (!file) {
      setError("Please select a file to upload")
      return
    }

    try {
      setLoading(true)
      setProgress(0)
      setError(null)

      // Simulate progress
      const progressInterval = setInterval(() => {
        setProgress((prev) => {
          const newProgress = prev + Math.random() * 10
          return newProgress >= 95 ? 95 : newProgress
        })
      }, 500)

      const formData = new FormData()
      formData.append("file", file)

      const response = await fetch(`${apiUrl}/predict`, {
        method: "POST",
        body: formData,
      })

      clearInterval(progressInterval)

      if (response.ok) {
        const data = await response.json()
        setResults(data)
        setProgress(100)
      } else {
        const errorData = await response.json()
        setError(errorData.error || "Failed to make predictions")
        setProgress(0)
      }
    } catch (err: any) {
      setError(err.message || "An unexpected error occurred")
      setProgress(0)
    } finally {
      setLoading(false)
    }
  }

  const downloadModel = () => {
    window.open(`${apiUrl}/download-model`, "_blank")
  }

  const downloadPredictions = () => {
    if (results && results.downloadUrl) {
      window.open(`${apiUrl}${results.downloadUrl}`, "_blank")
    }
  }

  return (
    <div className="space-y-6">
      <Card>
        <CardHeader>
          <CardTitle>Make Predictions</CardTitle>
          <CardDescription>Upload a dataset to make predictions using the trained model</CardDescription>
        </CardHeader>
        <CardContent>
          {!modelStatus?.modelReady && (
            <Alert variant="destructive" className="mb-4">
              <AlertCircle className="h-4 w-4" />
              <AlertTitle>Model Not Found</AlertTitle>
              <AlertDescription>Please train the model first before making predictions.</AlertDescription>
            </Alert>
          )}

          {modelStatus?.modelReady && (
            <Alert className="mb-4 bg-green-50 border-green-200">
              <CheckCircle2 className="h-4 w-4 text-green-600" />
              <AlertTitle className="text-green-800">Model Ready</AlertTitle>
              <AlertDescription className="text-green-700">
                The trained model is ready for making predictions.
                <Button variant="outline" size="sm" className="ml-2" onClick={downloadModel}>
                  <Download className="h-4 w-4 mr-1" /> Download Model
                </Button>
              </AlertDescription>
            </Alert>
          )}

          <form onSubmit={handleSubmit} className="space-y-4">
            <div className="space-y-2">
              <Label htmlFor="file">Upload Dataset (CSV)</Label>
              <Input
                id="file"
                type="file"
                accept=".csv"
                onChange={handleFileChange}
                disabled={loading || !modelStatus?.modelReady}
              />
              <p className="text-sm text-gray-500">The CSV file should have the same structure as the training data.</p>
            </div>

            <Button type="submit" disabled={loading || !file || !modelStatus?.modelReady} className="w-full">
              {loading ? "Processing..." : "Make Predictions"}
              {!loading && <Upload className="ml-2 h-4 w-4" />}
            </Button>
          </form>

          {loading && (
            <div className="mt-4">
              <p className="mb-2 text-sm font-medium">Processing</p>
              <Progress value={progress} className="h-2" />
              <p className="mt-1 text-xs text-gray-500">
                {progress < 100 ? "Processing data and making predictions..." : "Processing complete!"}
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

          {results && (
            <div className="mt-6 space-y-4">
              <Alert className="bg-green-50 border-green-200">
                <CheckCircle2 className="h-4 w-4 text-green-600" />
                <AlertTitle className="text-green-800">Predictions Complete</AlertTitle>
                <AlertDescription className="text-green-700">
                  Successfully generated {results.total_predictions} predictions.
                  <Button variant="outline" size="sm" className="ml-2" onClick={downloadPredictions}>
                    <Download className="h-4 w-4 mr-1" /> Download CSV
                  </Button>
                </AlertDescription>
              </Alert>

              <div>
                <h3 className="text-lg font-medium mb-2">Sample Predictions</h3>
                <div className="overflow-x-auto">
                  <Table>
                    <TableHeader>
                      <TableRow>
                        <TableHead>#</TableHead>
                        <TableHead>Predicted Class</TableHead>
                        {results.predictions[0] &&
                          Object.keys(results.predictions[0])
                            .filter((key) => key.startsWith("probability_"))
                            .map((key) => <TableHead key={key}>{key.replace("probability_", "")}</TableHead>)}
                      </TableRow>
                    </TableHeader>
                    <TableBody>
                      {results.predictions.map((pred: any, idx: number) => (
                        <TableRow key={idx}>
                          <TableCell>{idx + 1}</TableCell>
                          <TableCell>
                            <Badge>{pred.predicted_class}</Badge>
                          </TableCell>
                          {Object.entries(pred)
                            .filter(([key]) => key.startsWith("probability_"))
                            .map(([key, value]) => (
                              <TableCell key={key}>
                                {typeof value === "number" ? (value as number).toFixed(4) : String(value)}
                              </TableCell>
                            ))}
                        </TableRow>
                      ))}
                    </TableBody>
                  </Table>
                </div>
              </div>
            </div>
          )}
        </CardContent>
      </Card>
    </div>
  )
}
