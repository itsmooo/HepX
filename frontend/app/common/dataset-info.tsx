"use client"

import { useState, useEffect } from "react"
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from "@/components/ui/table"
import { Badge } from "@/components/ui/badge"
import { Skeleton } from "@/components/ui/skeleton"
import { InfoIcon } from "lucide-react"

export default function DatasetInfo() {
  const [loading, setLoading] = useState(true)

  // This would normally fetch dataset info from the API
  // For demo purposes, we'll use hardcoded data
  useEffect(() => {
    // Simulate loading
    const timer = setTimeout(() => {
      setLoading(false)
    }, 1500)

    return () => clearTimeout(timer)
  }, [])

  const datasetSchema = [
    { name: "PatientID", type: "string", example: "P1000", description: "Unique identifier for each patient" },
    { name: "HepatitisType", type: "string", example: "Hepatitis C", description: "Type of hepatitis diagnosed" },
    {
      name: "Symptoms",
      type: "string",
      example: "Fluid buildup in the stomach area, called ascites...",
      description: "Description of patient symptoms",
    },
    { name: "SymptomCount", type: "number", example: "5", description: "Number of symptoms reported" },
    { name: "Severity", type: "number", example: "4", description: "Severity rating of the condition (1-5)" },
    { name: "DiagnosisDate", type: "date", example: "2022-07-15", description: "Date when diagnosis was made" },
    {
      name: "Treatment",
      type: "string",
      example: "Guidelines & manuals...",
      description: "Treatment plan or recommendations",
    },
  ]

  const datasetStats = {
    totalRecords: 1000,
    hepatitisTypes: [
      { type: "Hepatitis A", count: 250 },
      // { type: "Hepatitis B", count: 350 },
      { type: "Hepatitis C", count: 300 },
      { type: "Autoimmune Hepatitis", count: 100 },
    ],
    averageSeverity: 3.2,
    averageSymptomCount: 4.5,
    dateRange: "2020-01-15 to 2023-12-31",
  }

  if (loading) {
    return (
      <Card>
        <CardHeader>
          <Skeleton className="h-8 w-3/4" />
          <Skeleton className="h-4 w-1/2" />
        </CardHeader>
        <CardContent>
          <div className="space-y-2">
            <Skeleton className="h-4 w-full" />
            <Skeleton className="h-4 w-full" />
            <Skeleton className="h-4 w-3/4" />
          </div>
        </CardContent>
      </Card>
    )
  }

  return (
    <div className="space-y-6">
      <Card>
        <CardHeader>
          <CardTitle>Dataset Overview</CardTitle>
          <CardDescription>Information about the hepatitis dataset used for model training</CardDescription>
        </CardHeader>
        <CardContent>
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4 mb-6">
            <div className="bg-blue-50 p-4 rounded-lg border border-blue-100">
              <h3 className="font-medium text-blue-800 mb-2 flex items-center gap-2">
                <InfoIcon className="h-4 w-4" />
                Dataset Statistics
              </h3>
              <ul className="space-y-2 text-sm">
                <li className="flex justify-between">
                  <span className="text-gray-600">Total Records:</span>
                  <span className="font-medium">{datasetStats.totalRecords}</span>
                </li>
                <li className="flex justify-between">
                  <span className="text-gray-600">Average Severity:</span>
                  <span className="font-medium">{datasetStats.averageSeverity}</span>
                </li>
                <li className="flex justify-between">
                  <span className="text-gray-600">Average Symptom Count:</span>
                  <span className="font-medium">{datasetStats.averageSymptomCount}</span>
                </li>
                <li className="flex justify-between">
                  <span className="text-gray-600">Date Range:</span>
                  <span className="font-medium">{datasetStats.dateRange}</span>
                </li>
              </ul>
            </div>

            <div className="bg-blue-50 p-4 rounded-lg border border-blue-100">
              <h3 className="font-medium text-blue-800 mb-2 flex items-center gap-2">
                <InfoIcon className="h-4 w-4" />
                Hepatitis Type Distribution
              </h3>
              <ul className="space-y-2 text-sm">
                {datasetStats.hepatitisTypes.map((item, index) => (
                  <li key={index} className="flex justify-between">
                    <span className="text-gray-600">{item.type}:</span>
                    <div className="flex items-center gap-2">
                      <div className="w-24 bg-gray-200 rounded-full h-2">
                        <div
                          className="h-2 rounded-full bg-blue-600"
                          style={{ width: `${(item.count / datasetStats.totalRecords) * 100}%` }}
                        ></div>
                      </div>
                      <span className="font-medium">{item.count}</span>
                    </div>
                  </li>
                ))}
              </ul>
            </div>
          </div>

          <h3 className="text-lg font-medium mb-3">Dataset Schema</h3>
          <div className="overflow-x-auto">
            <Table>
              <TableHeader>
                <TableRow>
                  <TableHead>Column</TableHead>
                  <TableHead>Type</TableHead>
                  <TableHead>Example</TableHead>
                  <TableHead>Description</TableHead>
                </TableRow>
              </TableHeader>
              <TableBody>
                {datasetSchema.map((column, index) => (
                  <TableRow key={index}>
                    <TableCell className="font-medium">{column.name}</TableCell>
                    <TableCell>
                      <Badge variant="outline">{column.type}</Badge>
                    </TableCell>
                    <TableCell className="text-sm text-gray-600 max-w-[200px] truncate">{column.example}</TableCell>
                    <TableCell>{column.description}</TableCell>
                  </TableRow>
                ))}
              </TableBody>
            </Table>
          </div>
        </CardContent>
      </Card>

      <Card>
        <CardHeader>
          <CardTitle>Data Processing</CardTitle>
          <CardDescription>How the data is processed before model training</CardDescription>
        </CardHeader>
        <CardContent>
          <div className="space-y-4">
            <div>
              <h3 className="font-medium mb-2">Feature Engineering</h3>
              <p className="text-gray-600">The raw dataset is processed to extract meaningful features:</p>
              <ul className="list-disc list-inside mt-2 space-y-1 text-gray-600">
                <li>Text analysis on symptom descriptions</li>
                <li>Date-based features from diagnosis dates</li>
                <li>Categorical encoding of hepatitis types</li>
                <li>Normalization of numeric features</li>
              </ul>
            </div>

            <div>
              <h3 className="font-medium mb-2">Data Splitting</h3>
              <p className="text-gray-600">
                The dataset is split into training (70%) and testing (30%) sets to evaluate model performance.
              </p>
            </div>

            <div>
              <h3 className="font-medium mb-2">Model Selection</h3>
              <p className="text-gray-600">
                A Random Forest classifier was chosen for this task due to its ability to handle both categorical and
                numerical data, as well as its robustness against overfitting.
              </p>
            </div>
          </div>
        </CardContent>
      </Card>
    </div>
  )
}
