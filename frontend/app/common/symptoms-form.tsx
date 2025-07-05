"use client";

import type React from "react";
import { useState, useEffect } from "react";
import {
  Card,
  CardContent,
  CardDescription,
  CardFooter,
  CardHeader,
  CardTitle,
} from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Slider } from "@/components/ui/slider";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { Switch } from "@/components/ui/switch";
import { Label } from "@/components/ui/label";
import { useToast } from "../hooks/use-toast";
import {
  Loader2,
  ChevronRight,
  Download,
  AlertCircle,
  CheckCircle2,
  Sparkles,
} from "lucide-react";
import { motion, AnimatePresence } from "framer-motion";

const symptoms = [
  { id: "fatigue", label: "Fatigue", type: "slider" },
  { id: "nausea", label: "Nausea", type: "slider" },
  { id: "abdominalPain", label: "Abdominal Pain", type: "slider" },
  {
    id: "jaundice",
    label: "Yellowing of Skin/Eyes (Jaundice)",
    type: "switch",
  },
  { id: "darkUrine", label: "Dark Urine", type: "switch" },
  { id: "jointPain", label: "Joint Pain", type: "switch" },
  { id: "appetite", label: "Loss of Appetite", type: "switch" },
  { id: "fever", label: "Fever", type: "slider" },
];

const riskFactors = [
  { id: "recentTravel", label: "Recent Travel to High-Risk Areas" },
  { id: "bloodTransfusion", label: "History of Blood Transfusion" },
  { id: "unsafeInjection", label: "History of Unsafe Injection Practices" },
  { id: "contactWithInfected", label: "Contact with Infected Person" },
];

type FormState = {
  symptoms: Record<string, number | boolean>;
  age: string;
  gender: string;
  riskFactors: string[];
  hepatitisType: string; // Added for direct selection
};

type PredictionResult = {
  result: "hepatitisB" | "hepatitisC" | "unlikely" | null;
  score: number;
  details: {
    symptomScore: number;
    riskScore: number;
  };
  predictions?: Array<{
    predicted_class: string;
    [key: string]: any;
  }>;
  downloadUrl?: string;
};

const SymptomsForm = () => {
  const { toast } = useToast();
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
  });
  const [showResults, setShowResults] = useState(false);
  const [predictionResult, setPredictionResult] =
    useState<PredictionResult | null>(null);
  const [loading, setLoading] = useState(false);
  const [progress, setProgress] = useState(0);

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

  const handleSymptomChange = (id: string, value: number | boolean) => {
    setFormState((prev) => ({
      ...prev,
      symptoms: {
        ...prev.symptoms,
        [id]: value,
      },
    }));
  };

  const handleRiskFactorToggle = (id: string) => {
    setFormState((prev) => {
      const riskFactors = [...prev.riskFactors];
      if (riskFactors.includes(id)) {
        return {
          ...prev,
          riskFactors: riskFactors.filter((factorId) => factorId !== id),
        };
      } else {
        return {
          ...prev,
          riskFactors: [...riskFactors, id],
        };
      }
    });
  };

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();

    // Validation
    if (!formState.age || !formState.gender) {
      toast({
        title: "Missing Information",
        description: "Please select your age group and gender to continue.",
        variant: "destructive",
      });
      return;
    }

    try {
      setLoading(true);
      setProgress(0);

      // Create form data to send to the API
      const formData = new FormData();
      formData.append("age", formState.age);
      formData.append("gender", formState.gender);
      formData.append("symptoms", JSON.stringify(formState.symptoms));
      formData.append("riskFactors", JSON.stringify(formState.riskFactors));

      // Call the API
      const response = await fetch("/api/predict", {
        method: "POST",
        body: formData,
      });

      if (!response.ok) {
        const errorData = await response.json();
        throw new Error(errorData.error || "Failed to get prediction");
      }

      const data = await response.json();
      setPredictionResult(data);
      setShowResults(true);

      // Scroll to results
      setTimeout(() => {
        const resultsElement = document.getElementById("prediction-results");
        if (resultsElement) {
          resultsElement.scrollIntoView({ behavior: "smooth" });
        }
      }, 100);
    } catch (error) {
      toast({
        title: "Error",
        description:
          "There was an error processing your symptoms. Please try again.",
        variant: "destructive",
      });
    } finally {
      setLoading(false);
    }
  };

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
    });
    setShowResults(false);
    setPredictionResult(null);
  };

  const downloadResults = () => {
    if (predictionResult?.downloadUrl) {
      const API_URL =
        process.env.NEXT_PUBLIC_API_URL || "http://localhost:5000/api";
      window.open(`${API_URL}${predictionResult.downloadUrl}`, "_blank");
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
      return value ? `- ${key}: Yes` : `- ${key}: No`;
    } else {
      return `- ${key}: ${value}%`;
    }
  })
  .join("\n")}

Risk Factors:
${
  formState.riskFactors.length > 0
    ? formState.riskFactors.map((factor) => `- ${factor}`).join("\n")
    : "- None reported"
}

Important Note:
This prediction is not a medical diagnosis. It's based on the information provided and is intended to guide your next steps. 
For accurate diagnosis, please consult with a healthcare professional.
      `;

      const blob = new Blob([resultText], { type: "text/plain" });
      const url = URL.createObjectURL(blob);
      const a = document.createElement("a");
      a.href = url;
      a.download = "HepaPredict_Results.txt";
      document.body.appendChild(a);
      a.click();
      document.body.removeChild(a);
      URL.revokeObjectURL(url);
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
    <section
      id="predict"
      className="py-20 px-6 sm:px-10 relative overflow-hidden"
    >
      {/* Background elements */}
      <div className="absolute inset-0 overflow-hidden">
        <div className="absolute top-40 -right-20 w-80 h-80 bg-blue-500/5 rounded-full blur-3xl" />
        <div className="absolute -bottom-20 -left-20 w-80 h-80 bg-indigo-500/5 rounded-full blur-3xl" />
      </div>

      <motion.div
        className="max-w-4xl mx-auto relative z-10"
        initial="hidden"
        whileInView="visible"
        viewport={{ once: true, margin: "-100px" }}
        variants={containerVariants}
      >
        <motion.div className="text-center mb-12" variants={itemVariants}>
          <div className="inline-flex items-center gap-2 px-4 py-2 rounded-full bg-blue-500/10 text-blue-600 dark:text-blue-400 mb-4">
            <Sparkles className="h-4 w-4" />
            <span className="text-sm font-medium">Symptom Assessment</span>
          </div>
          <h2 className="text-4xl font-bold text-slate-900 dark:text-white mb-4">
            Check Your Symptoms
          </h2>
          <p className="text-lg text-slate-600 dark:text-slate-300 max-w-2xl mx-auto">
            Answer these questions about your symptoms and risk factors to get
            insights about potential hepatitis types.
          </p>
        </motion.div>

        <AnimatePresence mode="wait">
          {!showResults ? (
            <motion.div
              key="form"
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0, y: -20 }}
              transition={{ duration: 0.5 }}
            >
              <Card className="border-blue-200 dark:border-blue-900 shadow-lg overflow-hidden">
                <CardHeader className="bg-gradient-to-r from-blue-50 to-blue-100 dark:from-blue-900/30 dark:to-blue-800/20 border-b border-blue-200 dark:border-blue-800">
                  <CardTitle>Symptom Checker</CardTitle>
                  <CardDescription>
                    Rate your symptoms and provide some basic information
                  </CardDescription>
                </CardHeader>
                <CardContent className="p-6 pt-8">
                  <form onSubmit={handleSubmit}>
                    <div className="grid gap-8">
                      <motion.div className="space-y-6" variants={itemVariants}>
                        <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
                          <div>
                            <Label
                              htmlFor="age"
                              className="text-slate-900 dark:text-white"
                            >
                              Age Group
                            </Label>
                            <Select
                              value={formState.age}
                              onValueChange={(value) =>
                                setFormState((prev) => ({
                                  ...prev,
                                  age: value,
                                }))
                              }
                            >
                              <SelectTrigger
                                id="age"
                                className="w-full mt-2 border-slate-200 dark:border-slate-700 focus:ring-blue-500"
                              >
                                <SelectValue placeholder="Select age group" />
                              </SelectTrigger>
                              <SelectContent>
                                <SelectItem value="under18">
                                  Under 18
                                </SelectItem>
                                <SelectItem value="18-30">18-30</SelectItem>
                                <SelectItem value="31-45">31-45</SelectItem>
                                <SelectItem value="46-60">46-60</SelectItem>
                                <SelectItem value="over60">Over 60</SelectItem>
                              </SelectContent>
                            </Select>
                          </div>
                          <div>
                            <Label
                              htmlFor="gender"
                              className="text-slate-900 dark:text-white"
                            >
                              Gender
                            </Label>
                            <Select
                              value={formState.gender}
                              onValueChange={(value) =>
                                setFormState((prev) => ({
                                  ...prev,
                                  gender: value,
                                }))
                              }
                            >
                              <SelectTrigger
                                id="gender"
                                className="w-full mt-2 border-slate-200 dark:border-slate-700 focus:ring-blue-500"
                              >
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
                            <Label
                              htmlFor="hepatitisType"
                              className="text-slate-900 dark:text-white"
                            >
                              Hepatitis Type
                            </Label>
                            <Select
                              value={formState.hepatitisType}
                              onValueChange={(value) =>
                                setFormState((prev) => ({
                                  ...prev,
                                  hepatitisType: value,
                                }))
                              }
                            >
                              <SelectTrigger
                                id="hepatitisType"
                                className="w-full mt-2 border-slate-200 dark:border-slate-700 focus:ring-blue-500"
                              >
                                <SelectValue placeholder="Select hepatitis type" />
                              </SelectTrigger>
                              <SelectContent>
                                <SelectItem value="hepatitisB">
                                  Hepatitis B
                                </SelectItem>
                                <SelectItem value="hepatitisC">
                                  Hepatitis C
                                </SelectItem>
                                <SelectItem value="unlikely">
                                  No Hepatitis
                                </SelectItem>
                              </SelectContent>
                            </Select>
                          </div>
                        </div>

                        <div className="mt-10">
                          <div className="flex items-center gap-2 mb-6">
                            <div className="p-1.5 bg-blue-100 dark:bg-blue-900/50 rounded-lg">
                              <AlertCircle className="h-5 w-5 text-blue-600 dark:text-blue-400" />
                            </div>
                            <h3 className="text-xl font-semibold text-slate-900 dark:text-white">
                              Symptoms
                            </h3>
                          </div>
                          <div className="space-y-8">
                            {symptoms.map((symptom, index) => (
                              <motion.div
                                key={symptom.id}
                                className="space-y-3"
                                initial={{ opacity: 0, y: 10 }}
                                animate={{ opacity: 1, y: 0 }}
                                transition={{ delay: 0.2 + index * 0.05 }}
                              >
                                {symptom.type === "slider" ? (
                                  <>
                                    <div className="flex justify-between items-center">
                                      <Label className="text-slate-900 dark:text-white">
                                        {symptom.label}
                                      </Label>
                                      <span className="text-sm font-medium px-2 py-1 rounded-md bg-blue-100 dark:bg-blue-900/30 text-blue-700 dark:text-blue-300">
                                        {
                                          formState.symptoms[
                                            symptom.id
                                          ] as number
                                        }
                                        %
                                      </span>
                                    </div>
                                    <Slider
                                      value={[
                                        formState.symptoms[
                                          symptom.id
                                        ] as number,
                                      ]}
                                      min={0}
                                      max={100}
                                      step={1}
                                      onValueChange={(value) =>
                                        handleSymptomChange(
                                          symptom.id,
                                          value[0]
                                        )
                                      }
                                      className="py-1"
                                    />
                                    <div className="flex justify-between text-xs text-slate-500 dark:text-slate-400 px-1">
                                      <span>None</span>
                                      <span>Mild</span>
                                      <span>Moderate</span>
                                      <span>Severe</span>
                                    </div>
                                  </>
                                ) : (
                                  <div className="flex items-center justify-between p-3 rounded-lg border border-slate-200 dark:border-slate-700 hover:bg-slate-50 dark:hover:bg-slate-800/50 transition-colors">
                                    <Label
                                      htmlFor={symptom.id}
                                      className="flex-1 cursor-pointer text-slate-900 dark:text-white"
                                    >
                                      {symptom.label}
                                    </Label>
                                    <Switch
                                      id={symptom.id}
                                      checked={
                                        formState.symptoms[
                                          symptom.id
                                        ] as boolean
                                      }
                                      onCheckedChange={(checked) =>
                                        handleSymptomChange(symptom.id, checked)
                                      }
                                    />
                                  </div>
                                )}
                              </motion.div>
                            ))}
                          </div>
                        </div>

                        <div className="mt-10">
                          <div className="flex items-center gap-2 mb-6">
                            <div className="p-1.5 bg-blue-100 dark:bg-blue-900/50 rounded-lg">
                              <AlertCircle className="h-5 w-5 text-blue-600 dark:text-blue-400" />
                            </div>
                            <h3 className="text-xl font-semibold text-slate-900 dark:text-white">
                              Risk Factors
                            </h3>
                          </div>
                          <div className="space-y-3">
                            {riskFactors.map((factor, index) => (
                              <motion.div
                                key={factor.id}
                                className="flex items-center justify-between p-3 rounded-lg border border-slate-200 dark:border-slate-700 hover:bg-slate-50 dark:hover:bg-slate-800/50 transition-colors"
                                initial={{ opacity: 0, y: 10 }}
                                animate={{ opacity: 1, y: 0 }}
                                transition={{ delay: 0.4 + index * 0.05 }}
                              >
                                <Label
                                  htmlFor={factor.id}
                                  className="flex-1 cursor-pointer text-slate-900 dark:text-white"
                                >
                                  {factor.label}
                                </Label>
                                <Switch
                                  id={factor.id}
                                  checked={formState.riskFactors.includes(
                                    factor.id
                                  )}
                                  onCheckedChange={() =>
                                    handleRiskFactorToggle(factor.id)
                                  }
                                />
                              </motion.div>
                            ))}
                          </div>
                        </div>
                      </motion.div>
                    </div>

                    <div className="mt-10 flex justify-end">
                      <Button
                        type="submit"
                        className="bg-blue-600 hover:bg-blue-700 text-white shadow-lg shadow-blue-500/20 transition-all duration-300 hover:shadow-xl hover:shadow-blue-500/30 hover:-translate-y-0.5"
                        disabled={loading}
                      >
                        {loading ? (
                          <>
                            <Loader2 className="mr-2 h-4 w-4 animate-spin" />{" "}
                            Processing...
                          </>
                        ) : (
                          <>
                            Submit Information{" "}
                            <ChevronRight className="ml-1 h-4 w-4" />
                          </>
                        )}
                      </Button>
                    </div>

                    {loading && (
                      <div className="mt-6 p-4 rounded-lg bg-blue-50 dark:bg-blue-900/20 border border-blue-200 dark:border-blue-800">
                        <p className="mb-2 text-sm font-medium text-blue-700 dark:text-blue-300">
                          Processing
                        </p>
                        <div className="h-2 w-full bg-blue-200 dark:bg-blue-800 rounded-full overflow-hidden">
                          <div
                            className="h-full bg-gradient-to-r from-blue-600 to-blue-400 rounded-full transition-all duration-300"
                            style={{ width: `${progress}%` }}
                          ></div>
                        </div>
                        <p className="mt-2 text-xs text-blue-600 dark:text-blue-400">
                          {progress < 100
                            ? "Processing data and analyzing symptoms..."
                            : "Processing complete!"}
                        </p>
                      </div>
                    )}
                  </form>
                </CardContent>
              </Card>
            </motion.div>
          ) : (
            <motion.div
              id="prediction-results"
              key="results"
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0, y: -20 }}
              transition={{ duration: 0.5 }}
              className="relative"
            >
              <Card className="border-t-4 border-t-blue-600 dark:border-t-blue-500 shadow-lg overflow-hidden">
                <CardHeader className="bg-gradient-to-r from-blue-50 to-blue-100 dark:from-blue-900/30 dark:to-blue-800/20 border-b border-blue-200 dark:border-blue-800">
                  <CardTitle>Your Prediction Results</CardTitle>
                  <CardDescription>
                    Based on the information you provided
                  </CardDescription>
                </CardHeader>
                <CardContent className="p-6 pt-8">
                  <div className="mb-8 p-6 rounded-xl bg-gradient-to-br from-blue-50 to-slate-50 dark:from-blue-900/20 dark:to-slate-900/50 border border-blue-200 dark:border-blue-800">
                    <div className="text-center">
                      <motion.div
                        initial={{ scale: 0.8, opacity: 0 }}
                        animate={{ scale: 1, opacity: 1 }}
                        transition={{
                          type: "spring",
                          stiffness: 200,
                          damping: 15,
                        }}
                      >
                        <h3 className="text-2xl font-bold mb-4 text-slate-900 dark:text-white">
                          {predictionResult?.result === "hepatitisB" &&
                            "Potential Signs of Hepatitis B"}
                          {predictionResult?.result === "hepatitisC" &&
                            "Potential Signs of Hepatitis C"}
                          {predictionResult?.result === "unlikely" &&
                            "Low Risk of Hepatitis"}
                        </h3>
                      </motion.div>

                      <motion.p
                        className="text-slate-600 dark:text-slate-300 mb-6"
                        initial={{ y: 10, opacity: 0 }}
                        animate={{ y: 0, opacity: 1 }}
                        transition={{ delay: 0.2 }}
                      >
                        {predictionResult?.result === "hepatitisB" &&
                          "Your symptoms suggest possible signs of Hepatitis B. This does not replace professional medical advice."}
                        {predictionResult?.result === "hepatitisC" &&
                          "Your symptoms suggest possible signs of Hepatitis C. This does not replace professional medical advice."}
                        {predictionResult?.result === "unlikely" &&
                          "Your symptoms suggest a lower likelihood of hepatitis. However, if symptoms persist, please consult a doctor."}
                      </motion.p>

                      <motion.div
                        initial={{ y: 10, opacity: 0 }}
                        animate={{ y: 0, opacity: 1 }}
                        transition={{ delay: 0.3 }}
                      >
                        <div
                          className={`inline-block px-6 py-3 rounded-full font-semibold text-white ${
                            predictionResult?.result === "unlikely"
                              ? "bg-gradient-to-r from-green-500 to-emerald-600"
                              : "bg-gradient-to-r from-amber-500 to-orange-600"
                          }`}
                        >
                          {predictionResult?.result === "hepatitisB" && (
                            <div className="flex items-center gap-2">
                              <AlertCircle className="h-5 w-5" />
                              <span>Hepatitis B Possible</span>
                            </div>
                          )}
                          {predictionResult?.result === "hepatitisC" && (
                            <div className="flex items-center gap-2">
                              <AlertCircle className="h-5 w-5" />
                              <span>Hepatitis C Possible</span>
                            </div>
                          )}
                          {predictionResult?.result === "unlikely" && (
                            <div className="flex items-center gap-2">
                              <CheckCircle2 className="h-5 w-5" />
                              <span>Low Risk Detected</span>
                            </div>
                          )}
                        </div>
                      </motion.div>
                    </div>
                  </div>

                  <motion.div
                    className="p-6 rounded-xl bg-blue-50 dark:bg-blue-900/20 border border-blue-200 dark:border-blue-800"
                    initial={{ y: 20, opacity: 0 }}
                    animate={{ y: 0, opacity: 1 }}
                    transition={{ delay: 0.4 }}
                  >
                    <h4 className="font-semibold mb-3 text-slate-900 dark:text-white flex items-center gap-2">
                      <AlertCircle className="h-5 w-5 text-blue-600 dark:text-blue-400" />
                      Important Note:
                    </h4>
                    <p className="text-slate-700 dark:text-slate-300">
                      This prediction is not a medical diagnosis. It's based on
                      the information you provided and is intended to guide your
                      next steps. For accurate diagnosis, please consult with a
                      healthcare professional.
                    </p>
                  </motion.div>

                  <motion.div
                    className="mt-8"
                    initial={{ y: 20, opacity: 0 }}
                    animate={{ y: 0, opacity: 1 }}
                    transition={{ delay: 0.5 }}
                  >
                    <h4 className="font-semibold mb-4 text-slate-900 dark:text-white flex items-center gap-2">
                      <ChevronRight className="h-5 w-5 text-blue-600 dark:text-blue-400" />
                      Next Steps:
                    </h4>
                    <ul className="space-y-3">
                      {[
                        "Consult with a healthcare provider for proper testing and diagnosis",
                        "Mention these specific symptoms to your doctor",
                        "Stay hydrated and get plenty of rest in the meantime",
                        "Learn more about hepatitis in our education section below",
                      ].map((step, i) => (
                        <motion.li
                          key={i}
                          className="flex items-start p-3 rounded-lg hover:bg-slate-50 dark:hover:bg-slate-800/50 transition-colors"
                          initial={{ x: -10, opacity: 0 }}
                          animate={{ x: 0, opacity: 1 }}
                          transition={{ delay: 0.6 + i * 0.1 }}
                        >
                          <CheckCircle2 className="h-5 w-5 mr-3 text-blue-600 dark:text-blue-400 flex-shrink-0 mt-0.5" />
                          <span className="text-slate-700 dark:text-slate-300">
                            {step}
                          </span>
                        </motion.li>
                      ))}
                    </ul>
                  </motion.div>
                </CardContent>
                <CardFooter className="flex justify-between p-6 bg-gradient-to-r from-slate-50 to-blue-50 dark:from-slate-900 dark:to-blue-900/20 border-t border-blue-200 dark:border-blue-800">
                  <Button
                    variant="outline"
                    onClick={resetForm}
                    className="border-blue-200 dark:border-blue-800 text-blue-600 dark:text-blue-400 hover:bg-blue-50 dark:hover:bg-blue-900/20"
                  >
                    Start Over
                  </Button>

                  <Button
                    onClick={downloadResults}
                    className="bg-blue-600 hover:bg-blue-700 text-white shadow-lg shadow-blue-500/20 transition-all duration-300 hover:shadow-xl hover:shadow-blue-500/30 hover:-translate-y-0.5"
                  >
                    <Download className="h-4 w-4 mr-2" /> Download Results
                  </Button>
                </CardFooter>
              </Card>
            </motion.div>
          )}
        </AnimatePresence>
      </motion.div>
    </section>
  );
};

export default SymptomsForm;
