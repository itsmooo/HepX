"use client";

import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from "@/components/ui/card";
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "@/components/ui/table";
import { Badge } from "@/components/ui/badge";
import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  ResponsiveContainer,
  Tooltip,
  Cell,
} from "recharts";
import {
  ChartContainer,
  ChartTooltip,
  ChartTooltipContent,
} from "@/components/ui/chart";
import { motion } from "framer-motion";
import {
  Info,
  BarChart3,
  ListFilter,
  CheckCircle2,
  XCircle,
} from "lucide-react";

interface ModelResultsProps {
  results: {
    accuracy: number;
    classification_report: {
      [key: string]: any;
    };
    feature_importances: {
      Feature: string;
      Importance: number;
    }[];
    confusion_matrix: number[][];
    y_test: string[];
    y_pred: string[];
  };
}

export default function ModelResults({ results }: ModelResultsProps) {
  // Format feature importance data for chart
  const featureImportanceData = results.feature_importances
    .slice(0, 10)
    .map((item) => ({
      feature: item.Feature,
      importance: Number.parseFloat((item.Importance * 100).toFixed(2)),
    }));

  // Get class metrics from classification report
  const classMetrics = Object.entries(results.classification_report)
    .filter(([key]) => !["accuracy", "macro avg", "weighted avg"].includes(key))
    .map(([className, metrics]: [string, any]) => ({
      className,
      precision: Number.parseFloat((metrics.precision * 100).toFixed(2)),
      recall: Number.parseFloat((metrics.recall * 100).toFixed(2)),
      f1Score: Number.parseFloat((metrics.f1_score * 100).toFixed(2)),
      support: metrics.support,
    }));

  const containerVariants = {
    hidden: { opacity: 0 },
    visible: {
      opacity: 1,
      transition: {
        staggerChildren: 0.2,
      },
    },
  };

  const itemVariants = {
    hidden: { y: 20, opacity: 0 },
    visible: {
      y: 0,
      opacity: 1,
      transition: { type: "spring", stiffness: 100 },
    },
  };

  // Custom colors for the chart
  const barColors = ["#3b82f6", "#60a5fa", "#93c5fd", "#bfdbfe", "#dbeafe"];

  return (
    <motion.div
      className="space-y-6"
      initial="hidden"
      animate="visible"
      variants={containerVariants}
    >
      <motion.div variants={itemVariants}>
        <Card className="overflow-hidden border-blue-200 dark:border-blue-900 shadow-lg">
          <CardHeader className="bg-gradient-to-r from-blue-50 to-blue-100 dark:from-blue-900/30 dark:to-blue-800/20 border-b border-blue-200 dark:border-blue-800">
            <div className="flex items-center gap-2">
              <Info className="h-5 w-5 text-blue-600 dark:text-blue-400" />
              <CardTitle>Model Performance</CardTitle>
            </div>
            <CardDescription>
              Overall accuracy and class-specific metrics
            </CardDescription>
          </CardHeader>
          <CardContent className="p-6">
            <div className="mb-8">
              <h3 className="text-lg font-medium mb-3 flex items-center gap-2 text-slate-900 dark:text-white">
                <CheckCircle2 className="h-5 w-5 text-blue-600 dark:text-blue-400" />
                Overall Accuracy
              </h3>
              <div className="flex items-center gap-4">
                <div className="w-full bg-slate-100 dark:bg-slate-800 rounded-full h-5">
                  <div
                    className="h-5 rounded-full bg-gradient-to-r from-blue-600 to-blue-400"
                    style={{ width: `${results.accuracy * 100}%` }}
                  ></div>
                </div>
                <span className="font-medium text-lg text-blue-600 dark:text-blue-400 min-w-[80px] text-right">
                  {(results.accuracy * 100).toFixed(2)}%
                </span>
              </div>
            </div>

            <h3 className="text-lg font-medium mb-4 flex items-center gap-2 text-slate-900 dark:text-white">
              <ListFilter className="h-5 w-5 text-blue-600 dark:text-blue-400" />
              Class Metrics
            </h3>
            <div className="overflow-x-auto rounded-xl border border-slate-200 dark:border-slate-800">
              <Table>
                <TableHeader className="bg-slate-50 dark:bg-slate-900">
                  <TableRow>
                    <TableHead className="font-semibold">Class</TableHead>
                    <TableHead className="font-semibold">Precision</TableHead>
                    <TableHead className="font-semibold">Recall</TableHead>
                    <TableHead className="font-semibold">F1 Score</TableHead>
                    <TableHead className="font-semibold">Support</TableHead>
                  </TableRow>
                </TableHeader>
                <TableBody>
                  {classMetrics.map((metric, index) => (
                    <TableRow
                      key={index}
                      className="hover:bg-slate-50 dark:hover:bg-slate-900/50"
                    >
                      <TableCell>
                        <Badge
                          variant="outline"
                          className="bg-blue-50 dark:bg-blue-900/30 text-blue-700 dark:text-blue-300 border-blue-200 dark:border-blue-800 font-medium"
                        >
                          {metric.className}
                        </Badge>
                      </TableCell>
                      <TableCell className="font-medium">
                        {metric.precision}%
                      </TableCell>
                      <TableCell className="font-medium">
                        {metric.recall}%
                      </TableCell>
                      <TableCell className="font-medium">
                        {metric.f1Score}%
                      </TableCell>
                      <TableCell className="font-medium">
                        {metric.support}
                      </TableCell>
                    </TableRow>
                  ))}
                </TableBody>
              </Table>
            </div>
          </CardContent>
        </Card>
      </motion.div>

      <motion.div variants={itemVariants}>
        <Card className="overflow-hidden border-blue-200 dark:border-blue-900 shadow-lg">
          <CardHeader className="bg-gradient-to-r from-blue-50 to-blue-100 dark:from-blue-900/30 dark:to-blue-800/20 border-b border-blue-200 dark:border-blue-800">
            <div className="flex items-center gap-2">
              <BarChart3 className="h-5 w-5 text-blue-600 dark:text-blue-400" />
              <CardTitle>Feature Importance</CardTitle>
            </div>
            <CardDescription>
              Top 10 most influential features in the model
            </CardDescription>
          </CardHeader>
          <CardContent className="p-6">
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
                    <CartesianGrid strokeDasharray="3 3" stroke="#e2e8f0" />
                    <XAxis
                      type="number"
                      label={{
                        value: "Importance (%)",
                        position: "insideBottom",
                        offset: -5,
                      }}
                      stroke="#64748b"
                    />
                    <YAxis
                      type="category"
                      dataKey="feature"
                      width={90}
                      tick={{ fontSize: 12 }}
                      stroke="#64748b"
                    />
                    <Tooltip
                      content={<ChartTooltipContent />}
                      cursor={{ fill: "rgba(147, 197, 253, 0.1)" }}
                    />
                    <Bar dataKey="importance" radius={[0, 4, 4, 0]}>
                      {featureImportanceData.map((entry, index) => (
                        <Cell
                          key={`cell-${index}`}
                          fill={barColors[index % barColors.length]}
                        />
                      ))}
                    </Bar>
                  </BarChart>
                </ResponsiveContainer>
              </ChartContainer>
            </div>
          </CardContent>
        </Card>
      </motion.div>

      <motion.div variants={itemVariants}>
        <Card className="overflow-hidden border-blue-200 dark:border-blue-900 shadow-lg">
          <CardHeader className="bg-gradient-to-r from-blue-50 to-blue-100 dark:from-blue-900/30 dark:to-blue-800/20 border-b border-blue-200 dark:border-blue-800">
            <div className="flex items-center gap-2">
              <ListFilter className="h-5 w-5 text-blue-600 dark:text-blue-400" />
              <CardTitle>Prediction Analysis</CardTitle>
            </div>
            <CardDescription>
              Comparison of actual vs predicted values
            </CardDescription>
          </CardHeader>
          <CardContent className="p-6">
            <div className="mb-6 p-4 bg-blue-50 dark:bg-blue-900/20 rounded-xl border border-blue-100 dark:border-blue-800">
              <h3 className="text-lg font-medium mb-2 text-slate-900 dark:text-white">
                Prediction Summary
              </h3>
              <p className="text-slate-700 dark:text-slate-300">
                The model made{" "}
                <span className="font-semibold">{results.y_test.length}</span>{" "}
                predictions with an accuracy of{" "}
                <span className="font-semibold text-blue-600 dark:text-blue-400">
                  {(results.accuracy * 100).toFixed(2)}%
                </span>
                .{" "}
                <span className="font-semibold">
                  {
                    results.y_test.filter(
                      (val, idx) => val === results.y_pred[idx]
                    ).length
                  }
                </span>{" "}
                predictions were correct.
              </p>
            </div>

            <div className="overflow-x-auto rounded-xl border border-slate-200 dark:border-slate-800">
              <Table>
                <TableHeader className="bg-slate-50 dark:bg-slate-900">
                  <TableRow>
                    <TableHead className="font-semibold">Sample ID</TableHead>
                    <TableHead className="font-semibold">Actual</TableHead>
                    <TableHead className="font-semibold">Predicted</TableHead>
                    <TableHead className="font-semibold">Correct</TableHead>
                  </TableRow>
                </TableHeader>
                <TableBody>
                  {results.y_test.slice(0, 10).map((actual, idx) => (
                    <TableRow
                      key={idx}
                      className="hover:bg-slate-50 dark:hover:bg-slate-900/50"
                    >
                      <TableCell className="font-medium">#{idx + 1}</TableCell>
                      <TableCell>
                        <Badge
                          variant="outline"
                          className="border-slate-200 dark:border-slate-700"
                        >
                          {actual}
                        </Badge>
                      </TableCell>
                      <TableCell>
                        <Badge
                          variant="outline"
                          className="border-slate-200 dark:border-slate-700"
                        >
                          {results.y_pred[idx]}
                        </Badge>
                      </TableCell>
                      <TableCell>
                        {actual === results.y_pred[idx] ? (
                          <Badge className="bg-green-100 text-green-800 dark:bg-green-900/30 dark:text-green-400 flex items-center gap-1">
                            <CheckCircle2 className="h-3 w-3" /> Correct
                          </Badge>
                        ) : (
                          <Badge
                            variant="destructive"
                            className="flex items-center gap-1"
                          >
                            <XCircle className="h-3 w-3" /> Incorrect
                          </Badge>
                        )}
                      </TableCell>
                    </TableRow>
                  ))}
                </TableBody>
              </Table>
            </div>
          </CardContent>
        </Card>
      </motion.div>
    </motion.div>
  );
}
