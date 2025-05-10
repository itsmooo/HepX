"use client"

import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from "@/components/ui/table"
import { Badge } from "@/components/ui/badge"
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, ResponsiveContainer } from "recharts"
import { ChartContainer, ChartTooltip, ChartTooltipContent } from "@/components/ui/chart"

interface ModelResultsProps {
  results: {
    accuracy: number
    classification_report: {
      [key: string]: any
    }
    feature_importances: {
      Feature: string
      Importance: number
    }[]
    confusion_matrix: number[][]
    y_test: string[]
    y_pred: string[]
  }
}

export default function ModelResults({ results }: ModelResultsProps) {
  // Format feature importance data for chart
  const featureImportanceData = results.feature_importances.slice(0, 10).map((item) => ({
    feature: item.Feature,
    importance: Number.parseFloat((item.Importance * 100).toFixed(2)),
  }))

  // Get class metrics from classification report
  const classMetrics = Object.entries(results.classification_report)
    .filter(([key]) => !["accuracy", "macro avg", "weighted avg"].includes(key))
    .map(([className, metrics]: [string, any]) => ({
      className,
      precision: Number.parseFloat((metrics.precision * 100).toFixed(2)),
      recall: Number.parseFloat((metrics.recall * 100).toFixed(2)),
      f1Score: Number.parseFloat((metrics.f1_score * 100).toFixed(2)),
      support: metrics.support,
    }))

  return (
    <div className="space-y-6">
      <Card>
        <CardHeader>
          <CardTitle>Model Performance</CardTitle>
          <CardDescription>Overall accuracy and class-specific metrics</CardDescription>
        </CardHeader>
        <CardContent>
          <div className="mb-6">
            <h3 className="text-lg font-medium mb-2">Overall Accuracy</h3>
            <div className="flex items-center gap-2">
              <div className="w-full bg-gray-200 rounded-full h-4">
                <div className="h-4 rounded-full bg-blue-600" style={{ width: `${results.accuracy * 100}%` }}></div>
              </div>
              <span className="font-medium">{(results.accuracy * 100).toFixed(2)}%</span>
            </div>
          </div>

          <h3 className="text-lg font-medium mb-2">Class Metrics</h3>
          <div className="overflow-x-auto">
            <Table>
              <TableHeader>
                <TableRow>
                  <TableHead>Class</TableHead>
                  <TableHead>Precision</TableHead>
                  <TableHead>Recall</TableHead>
                  <TableHead>F1 Score</TableHead>
                  <TableHead>Support</TableHead>
                </TableRow>
              </TableHeader>
              <TableBody>
                {classMetrics.map((metric, index) => (
                  <TableRow key={index}>
                    <TableCell>
                      <Badge variant="outline">{metric.className}</Badge>
                    </TableCell>
                    <TableCell>{metric.precision}%</TableCell>
                    <TableCell>{metric.recall}%</TableCell>
                    <TableCell>{metric.f1Score}%</TableCell>
                    <TableCell>{metric.support}</TableCell>
                  </TableRow>
                ))}
              </TableBody>
            </Table>
          </div>
        </CardContent>
      </Card>

      <Card>
        <CardHeader>
          <CardTitle>Feature Importance</CardTitle>
          <CardDescription>Top 10 most influential features in the model</CardDescription>
        </CardHeader>
        <CardContent>
          <div className="h-[400px]">
            <ChartContainer
              config={{
                importance: {
                  label: "Importance (%)",
                  color: "hsl(var(--chart-1))",
                },
              }}
            >
              <ResponsiveContainer width="100%" height="100%">
                <BarChart
                  data={featureImportanceData}
                  layout="vertical"
                  margin={{ top: 20, right: 30, left: 100, bottom: 5 }}
                >
                  <CartesianGrid strokeDasharray="3 3" />
                  <XAxis type="number" label={{ value: "Importance (%)", position: "insideBottom", offset: -5 }} />
                  <YAxis type="category" dataKey="feature" width={90} tick={{ fontSize: 12 }} />
                  <ChartTooltip content={<ChartTooltipContent />} />
                  <Bar dataKey="importance" fill="var(--color-importance)" />
                </BarChart>
              </ResponsiveContainer>
            </ChartContainer>
          </div>
        </CardContent>
      </Card>

      <Card>
        <CardHeader>
          <CardTitle>Prediction Analysis</CardTitle>
          <CardDescription>Comparison of actual vs predicted values</CardDescription>
        </CardHeader>
        <CardContent>
          <div className="mb-4">
            <h3 className="text-lg font-medium mb-2">Prediction Summary</h3>
            <p className="text-gray-600">
              The model made {results.y_test.length} predictions with an accuracy of{" "}
              {(results.accuracy * 100).toFixed(2)}%.{" "}
              {results.y_test.filter((val, idx) => val === results.y_pred[idx]).length} predictions were correct.
            </p>
          </div>

          <div className="overflow-x-auto">
            <Table>
              <TableHeader>
                <TableRow>
                  <TableHead>Sample ID</TableHead>
                  <TableHead>Actual</TableHead>
                  <TableHead>Predicted</TableHead>
                  <TableHead>Correct</TableHead>
                </TableRow>
              </TableHeader>
              <TableBody>
                {results.y_test.slice(0, 10).map((actual, idx) => (
                  <TableRow key={idx}>
                    <TableCell>#{idx + 1}</TableCell>
                    <TableCell>
                      <Badge variant="outline">{actual}</Badge>
                    </TableCell>
                    <TableCell>
                      <Badge variant="outline">{results.y_pred[idx]}</Badge>
                    </TableCell>
                    <TableCell>
                      {actual === results.y_pred[idx] ? (
                        <Badge className="bg-green-100 text-green-800 hover:bg-green-100">Correct</Badge>
                      ) : (
                        <Badge variant="destructive">Incorrect</Badge>
                      )}
                    </TableCell>
                  </TableRow>
                ))}
              </TableBody>
            </Table>
          </div>
        </CardContent>
      </Card>
    </div>
  )
}
