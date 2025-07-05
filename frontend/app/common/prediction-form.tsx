"use client";

import type React from "react";
import { useState, useEffect } from "react";
import { Button } from "@/components/ui/button";
import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from "@/components/ui/card";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Alert, AlertDescription, AlertTitle } from "@/components/ui/alert";
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
  AlertCircle,
  CheckCircle2,
  Download,
  Upload,
  FileUp,
  Database,
  Sparkles,
} from "lucide-react";
import { motion, AnimatePresence } from "framer-motion";

interface PredictionFormProps {
  apiUrl: string;
}

export default function PredictionForm({ apiUrl }: PredictionFormProps) {
  const [file, setFile] = useState<File | null>(null);
  const [loading, setLoading] = useState(false);
  const [progress, setProgress] = useState(0);
  const [error, setError] = useState<string | null>(null);
  const [results, setResults] = useState<any>(null);
  const [modelStatus, setModelStatus] = useState<{
    modelExists: boolean;
    featureColumnsExist: boolean;
    modelReady: boolean;
  } | null>(null);
  const [dragActive, setDragActive] = useState(false);

  // Check if model exists
  const checkModelStatus = async () => {
    try {
      const response = await fetch(`${apiUrl}/model-status`);
      if (response.ok) {
        const data = await response.json();
        setModelStatus(data);
      }
    } catch (err) {
      console.error("Error checking model status:", err);
    }
  };

  // Call checkModelStatus on component mount
  useEffect(() => {
    checkModelStatus();
  }, []);

  // Simulate progress bar
  useEffect(() => {
    let interval: NodeJS.Timeout;
    if (loading) {
      interval = setInterval(() => {
        setProgress((prev) => {
          const newProgress = prev + Math.random() * 5;
          return newProgress >= 95 ? 95 : newProgress;
        });
      }, 300);
    } else if (progress > 0 && progress < 100) {
      setProgress(100);
    }

    return () => {
      if (interval) clearInterval(interval);
    };
  }, [loading, progress]);

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files.length > 0) {
      setFile(e.target.files[0]);
    }
  };

  const handleDrag = (e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
    if (e.type === "dragenter" || e.type === "dragover") {
      setDragActive(true);
    } else if (e.type === "dragleave") {
      setDragActive(false);
    }
  };

  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
    setDragActive(false);
    if (e.dataTransfer.files && e.dataTransfer.files.length > 0) {
      setFile(e.dataTransfer.files[0]);
    }
  };

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!file) {
      setError("Please select a file to upload");
      return;
    }

    try {
      setLoading(true);
      setProgress(0);
      setError(null);

      // Create form data
      const formData = new FormData();
      formData.append("file", file);

      // Simulate API call for demo purposes
      await new Promise((resolve) => setTimeout(resolve, 3000));

      // Mock response
      const mockData = {
        success: true,
        total_predictions: 10,
        predictions: Array(5)
          .fill(0)
          .map((_, i) => ({
            predicted_class:
              Math.random() > 0.5 ? "Hepatitis A" : "Hepatitis C",
            probability_Hepatitis_A: Math.random().toFixed(4),
            probability_Hepatitis_C: Math.random().toFixed(4),
          })),
        downloadUrl: "/download-predictions.csv",
      };

      setResults(mockData);
      setProgress(100);
    } catch (err: any) {
      setError(err.message || "An unexpected error occurred");
      setProgress(0);
    } finally {
      setLoading(false);
    }
  };

  const downloadModel = () => {
    window.open(`${apiUrl}/download-model`, "_blank");
  };

  const downloadPredictions = () => {
    if (results && results.downloadUrl) {
      window.open(`${apiUrl}${results.downloadUrl}`, "_blank");
    }
  };

  const containerVariants = {
    hidden: { opacity: 0 },
    visible: {
      opacity: 1,
      transition: {
        staggerChildren: 0.1,
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

  return (
    <div className="relative overflow-hidden py-12 px-6 sm:px-10">
      {/* Background elements */}
      <div className="absolute inset-0 overflow-hidden">
        <div className="absolute top-40 -right-20 w-80 h-80 bg-blue-500/5 rounded-full blur-3xl" />
        <div className="absolute -bottom-20 -left-20 w-80 h-80 bg-indigo-500/5 rounded-full blur-3xl" />
      </div>

      <motion.div
        className="max-w-4xl mx-auto text-center mb-10 relative z-10"
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.5 }}
      >
        <div className="inline-flex items-center gap-2 px-4 py-2 rounded-full bg-blue-500/10 text-blue-600 dark:text-blue-400 mb-4">
          <Sparkles className="h-4 w-4" />
          <span className="text-sm font-medium">Batch Predictions</span>
        </div>
        <h2 className="text-4xl font-bold text-slate-900 dark:text-white mb-4">
          Make Predictions
        </h2>
        <p className="text-lg text-slate-600 dark:text-slate-300 max-w-2xl mx-auto">
          Upload a dataset to make predictions using our trained hepatitis
          prediction model.
        </p>
      </motion.div>

      <motion.div
        className="space-y-6 max-w-4xl mx-auto relative z-10"
        initial="hidden"
        animate="visible"
        variants={containerVariants}
      >
        <motion.div variants={itemVariants}>
          <Card className="border-blue-200 dark:border-blue-900 shadow-lg overflow-hidden">
            <CardHeader className="bg-gradient-to-r from-blue-50 to-blue-100 dark:from-blue-900/30 dark:to-blue-800/20 border-b border-blue-200 dark:border-blue-800">
              <div className="flex items-center gap-2">
                <Database className="h-5 w-5 text-blue-600 dark:text-blue-400" />
                <CardTitle>Make Predictions</CardTitle>
              </div>
              <CardDescription>
                Upload a dataset to make predictions using the trained model
              </CardDescription>
            </CardHeader>
            <CardContent className="p-6">
              <AnimatePresence mode="wait">
                {!modelStatus?.modelReady && (
                  <motion.div
                    initial={{ opacity: 0, y: 10 }}
                    animate={{ opacity: 1, y: 0 }}
                    exit={{ opacity: 0, y: -10 }}
                    transition={{ duration: 0.3 }}
                  >
                    <Alert variant="destructive" className="mb-6 shadow-md">
                      <div className="flex items-center">
                        <AlertCircle className="h-5 w-5 mr-2" />
                        <div>
                          <AlertTitle>Model Not Found</AlertTitle>
                          <AlertDescription>
                            Please train the model first before making
                            predictions.
                          </AlertDescription>
                        </div>
                      </div>
                    </Alert>
                  </motion.div>
                )}

                {modelStatus?.modelReady && (
                  <motion.div
                    initial={{ opacity: 0, y: 10 }}
                    animate={{ opacity: 1, y: 0 }}
                    exit={{ opacity: 0, y: -10 }}
                    transition={{ duration: 0.3 }}
                  >
                    <Alert className="mb-6 bg-green-50 dark:bg-green-900/20 border-green-200 dark:border-green-800 shadow-md">
                      <div className="flex items-center">
                        <CheckCircle2 className="h-5 w-5 text-green-600 dark:text-green-400 mr-2" />
                        <div className="flex-1">
                          <AlertTitle className="text-green-800 dark:text-green-300 font-medium">
                            Model Ready
                          </AlertTitle>
                          <AlertDescription className="text-green-700 dark:text-green-400 flex flex-wrap items-center gap-2">
                            The trained model is ready for making predictions.
                            <Button
                              variant="outline"
                              size="sm"
                              onClick={downloadModel}
                              className="ml-auto border-green-200 dark:border-green-800 text-green-700 dark:text-green-400 hover:bg-green-50 dark:hover:bg-green-900/30"
                            >
                              <Download className="h-4 w-4 mr-1" /> Download
                              Model
                            </Button>
                          </AlertDescription>
                        </div>
                      </div>
                    </Alert>
                  </motion.div>
                )}

                <form onSubmit={handleSubmit} className="space-y-6">
                  <motion.div
                    className="space-y-3"
                    variants={itemVariants}
                    onDragEnter={handleDrag}
                    onDragOver={handleDrag}
                    onDragLeave={handleDrag}
                    onDrop={handleDrop}
                  >
                    <Label
                      htmlFor="file"
                      className="text-slate-900 dark:text-white"
                    >
                      Upload Dataset (CSV)
                    </Label>
                    <div
                      className={`border-2 border-dashed rounded-xl p-8 text-center transition-colors ${
                        dragActive
                          ? "border-blue-500 bg-blue-50 dark:bg-blue-900/20"
                          : "border-slate-200 dark:border-slate-700 hover:border-blue-300 dark:hover:border-blue-700"
                      }`}
                    >
                      <div className="flex flex-col items-center justify-center gap-3">
                        <div
                          className={`p-3 rounded-full ${
                            dragActive
                              ? "bg-blue-100 dark:bg-blue-800/50 text-blue-600 dark:text-blue-400"
                              : "bg-slate-100 dark:bg-slate-800 text-slate-500 dark:text-slate-400"
                          }`}
                        >
                          <FileUp className="h-6 w-6" />
                        </div>
                        <div>
                          <p className="text-sm font-medium text-slate-900 dark:text-white mb-1">
                            {file
                              ? file.name
                              : "Drag and drop your file here or click to browse"}
                          </p>
                          <p className="text-xs text-slate-500 dark:text-slate-400">
                            Supports CSV files up to 10MB
                          </p>
                        </div>
                        <Input
                          id="file"
                          type="file"
                          accept=".csv"
                          onChange={handleFileChange}
                          disabled={loading || !modelStatus?.modelReady}
                          className="hidden"
                        />
                        <Button
                          type="button"
                          variant="outline"
                          size="sm"
                          onClick={() =>
                            document.getElementById("file")?.click()
                          }
                          disabled={loading || !modelStatus?.modelReady}
                          className="mt-2 border-blue-200 dark:border-blue-800 text-blue-600 dark:text-blue-400 hover:bg-blue-50 dark:hover:bg-blue-900/20"
                        >
                          Browse Files
                        </Button>
                      </div>
                    </div>
                    <p className="text-sm text-slate-500 dark:text-slate-400">
                      The CSV file should have the same structure as the
                      training data.
                    </p>
                  </motion.div>

                  <motion.div variants={itemVariants}>
                    <Button
                      type="submit"
                      disabled={loading || !file || !modelStatus?.modelReady}
                      className="w-full bg-blue-600 hover:bg-blue-700 text-white shadow-lg shadow-blue-500/20 transition-all duration-300 hover:shadow-xl hover:shadow-blue-500/30 hover:-translate-y-0.5"
                    >
                      {loading ? (
                        <div className="flex items-center">
                          <div className="w-5 h-5 border-2 border-white border-t-transparent rounded-full animate-spin mr-2"></div>
                          Processing...
                        </div>
                      ) : (
                        <div className="flex items-center">
                          Make Predictions
                          <Upload className="ml-2 h-4 w-4" />
                        </div>
                      )}
                    </Button>
                  </motion.div>

                  <AnimatePresence>
                    {loading && (
                      <motion.div
                        className="mt-6"
                        initial={{ opacity: 0, y: 10 }}
                        animate={{ opacity: 1, y: 0 }}
                        exit={{ opacity: 0, y: -10 }}
                        transition={{ duration: 0.3 }}
                      >
                        <p className="mb-2 text-sm font-medium text-slate-900 dark:text-white flex justify-between">
                          Processing
                          <span className="text-blue-600 dark:text-blue-400">
                            {Math.round(progress)}%
                          </span>
                        </p>
                        <div className="h-2 w-full bg-slate-100 dark:bg-slate-800 rounded-full overflow-hidden">
                          <div
                            className="h-full bg-gradient-to-r from-blue-600 to-blue-400 rounded-full transition-all duration-300"
                            style={{ width: `${progress}%` }}
                          ></div>
                        </div>
                        <p className="mt-2 text-xs text-slate-500 dark:text-slate-400">
                          {progress < 100
                            ? "Processing data and making predictions..."
                            : "Processing complete!"}
                        </p>
                      </motion.div>
                    )}
                  </AnimatePresence>

                  <AnimatePresence>
                    {error && (
                      <motion.div
                        initial={{ opacity: 0, y: 10 }}
                        animate={{ opacity: 1, y: 0 }}
                        exit={{ opacity: 0, y: -10 }}
                        transition={{ duration: 0.3 }}
                      >
                        <Alert variant="destructive" className="mt-6 shadow-md">
                          <AlertCircle className="h-4 w-4" />
                          <AlertTitle>Error</AlertTitle>
                          <AlertDescription>{error}</AlertDescription>
                        </Alert>
                      </motion.div>
                    )}
                  </AnimatePresence>
                </form>

                <AnimatePresence>
                  {results && (
                    <motion.div
                      className="mt-8 space-y-6"
                      initial={{ opacity: 0, y: 20 }}
                      animate={{ opacity: 1, y: 0 }}
                      exit={{ opacity: 0, y: -20 }}
                      transition={{ duration: 0.5 }}
                    >
                      <Alert className="bg-green-50 dark:bg-green-900/20 border-green-200 dark:border-green-800 shadow-md">
                        <div className="flex items-center">
                          <CheckCircle2 className="h-5 w-5 text-green-600 dark:text-green-400 mr-2" />
                          <div className="flex-1">
                            <AlertTitle className="text-green-800 dark:text-green-300 font-medium">
                              Predictions Complete
                            </AlertTitle>
                            <AlertDescription className="text-green-700 dark:text-green-400 flex flex-wrap items-center gap-2">
                              Successfully generated {results.total_predictions}{" "}
                              predictions.
                              <Button
                                variant="outline"
                                size="sm"
                                onClick={downloadPredictions}
                                className="ml-auto border-green-200 dark:border-green-800 text-green-700 dark:text-green-400 hover:bg-green-50 dark:hover:bg-green-900/30"
                              >
                                <Download className="h-4 w-4 mr-1" /> Download
                                CSV
                              </Button>
                            </AlertDescription>
                          </div>
                        </div>
                      </Alert>

                      <div>
                        <h3 className="text-lg font-medium mb-4 text-slate-900 dark:text-white flex items-center gap-2">
                          <Database className="h-5 w-5 text-blue-600 dark:text-blue-400" />
                          Sample Predictions
                        </h3>
                        <div className="overflow-x-auto rounded-xl border border-slate-200 dark:border-slate-700 shadow-sm">
                          <Table>
                            <TableHeader className="bg-slate-50 dark:bg-slate-900">
                              <TableRow>
                                <TableHead className="font-semibold">
                                  #
                                </TableHead>
                                <TableHead className="font-semibold">
                                  Predicted Class
                                </TableHead>
                                {results.predictions[0] &&
                                  Object.keys(results.predictions[0])
                                    .filter((key) =>
                                      key.startsWith("probability_")
                                    )
                                    .map((key) => (
                                      <TableHead
                                        key={key}
                                        className="font-semibold"
                                      >
                                        {key.replace("probability_", "")}
                                      </TableHead>
                                    ))}
                              </TableRow>
                            </TableHeader>
                            <TableBody>
                              {results.predictions.map(
                                (pred: any, idx: number) => (
                                  <TableRow
                                    key={idx}
                                    className="hover:bg-slate-50 dark:hover:bg-slate-900/50"
                                  >
                                    <TableCell className="font-medium">
                                      {idx + 1}
                                    </TableCell>
                                    <TableCell>
                                      <Badge
                                        className={
                                          pred.predicted_class === "Hepatitis A"
                                            ? "bg-blue-100 text-blue-800 dark:bg-blue-900/30 dark:text-blue-300 border-blue-200 dark:border-blue-800"
                                            : "bg-purple-100 text-purple-800 dark:bg-purple-900/30 dark:text-purple-300 border-purple-200 dark:border-purple-800"
                                        }
                                      >
                                        {pred.predicted_class}
                                      </Badge>
                                    </TableCell>
                                    {Object.entries(pred)
                                      .filter(([key]) =>
                                        key.startsWith("probability_")
                                      )
                                      .map(([key, value]) => (
                                        <TableCell
                                          key={key}
                                          className="font-medium"
                                        >
                                          {typeof value === "number"
                                            ? (value as number).toFixed(4)
                                            : String(value)}
                                        </TableCell>
                                      ))}
                                  </TableRow>
                                )
                              )}
                            </TableBody>
                          </Table>
                        </div>
                      </div>
                    </motion.div>
                  )}
                </AnimatePresence>
              </AnimatePresence>
            </CardContent>
          </Card>
        </motion.div>
      </motion.div>
    </div>
  );
}
