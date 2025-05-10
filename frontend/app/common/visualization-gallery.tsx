"use client"

import type React from "react"

import { useState } from "react"
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs"
import Image from "next/image"
import { BarChartIcon as ChartIcon, BarChart3, PieChart, Activity } from "lucide-react"

interface VisualizationGalleryProps {
  visualizations: string[]
  apiUrl: string
}

export default function VisualizationGallery({ visualizations, apiUrl }: VisualizationGalleryProps) {
  const [selectedViz, setSelectedViz] = useState(visualizations[0] || "")

  // Map visualization filenames to more readable titles and descriptions
  const vizInfo: Record<string, { title: string; description: string; icon: React.ReactNode }> = {
    "hepatitis_types_distribution.png": {
      title: "Hepatitis Types Distribution",
      description: "Distribution of different hepatitis types in the dataset",
      icon: <PieChart className="h-5 w-5" />,
    },
    "severity_distribution.png": {
      title: "Severity Distribution",
      description: "Distribution of severity scores across all patients",
      icon: <BarChart3 className="h-5 w-5" />,
    },
    "symptom_count_distribution.png": {
      title: "Symptom Count Distribution",
      description: "Distribution of the number of symptoms reported by patients",
      icon: <BarChart3 className="h-5 w-5" />,
    },
    "confusion_matrix.png": {
      title: "Confusion Matrix",
      description: "Visualization of model prediction accuracy for each class",
      icon: <ChartIcon className="h-5 w-5" />,
    },
    "feature_importance.png": {
      title: "Feature Importance",
      description: "Ranking of features by their importance in the model",
      icon: <BarChart3 className="h-5 w-5" />,
    },
    "actual_vs_predicted.png": {
      title: "Actual vs Predicted",
      description: "Comparison between actual and predicted hepatitis types",
      icon: <Activity className="h-5 w-5" />,
    },
  }

  // Extract filename from URL
  const getFilename = (url: string) => {
    return url.split("/").pop() || ""
  }

  if (visualizations.length === 0) {
    return (
      <Card>
        <CardHeader>
          <CardTitle>Visualizations</CardTitle>
          <CardDescription>No visualizations available yet. Train the model first.</CardDescription>
        </CardHeader>
        <CardContent>
          <div className="h-[400px] flex items-center justify-center bg-gray-100 rounded-md">
            <p className="text-gray-500">No visualizations to display</p>
          </div>
        </CardContent>
      </Card>
    )
  }

  return (
    <div className="space-y-6">
      <Card>
        <CardHeader>
          <CardTitle>Model Visualizations</CardTitle>
          <CardDescription>Visual representations of the dataset and model performance</CardDescription>
        </CardHeader>
        <CardContent>
          <Tabs defaultValue={getFilename(visualizations[0])} onValueChange={setSelectedViz} className="w-full">
            <TabsList className="grid grid-cols-3 md:grid-cols-6 w-full">
              {visualizations.map((viz) => {
                const filename = getFilename(viz)
                const info = vizInfo[filename] || {
                  title: filename,
                  description: "Visualization",
                  icon: <ChartIcon className="h-5 w-5" />,
                }

                return (
                  <TabsTrigger key={viz} value={filename} className="flex flex-col items-center gap-1 py-2 h-auto">
                    {info.icon}
                    <span className="text-xs truncate max-w-full">{info.title.split(" ")[0]}</span>
                  </TabsTrigger>
                )
              })}
            </TabsList>

            {visualizations.map((viz) => {
              const filename = getFilename(viz)
              const info = vizInfo[filename] || {
                title: filename,
                description: "Visualization",
                icon: null,
              }

              return (
                <TabsContent key={viz} value={filename} className="mt-6">
                  <div className="space-y-4">
                    <div>
                      <h3 className="text-lg font-medium flex items-center gap-2">
                        {info.icon}
                        {info.title}
                      </h3>
                      <p className="text-gray-600">{info.description}</p>
                    </div>

                    <div className="relative h-[500px] w-full bg-gray-50 rounded-lg border overflow-hidden">
                      <div className="absolute inset-0 flex items-center justify-center">
                        <Image
                          src={`${apiUrl}/visualization/${filename}`}
                          alt={info.title}
                          fill
                          style={{ objectFit: "contain" }}
                          className="p-4"
                        />
                      </div>
                    </div>
                  </div>
                </TabsContent>
              )
            })}
          </Tabs>
        </CardContent>
      </Card>
    </div>
  )
}
