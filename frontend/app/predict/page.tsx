"use client";

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
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { Label } from "@/components/ui/label";
import { RadioGroup, RadioGroupItem } from "@/components/ui/radio-group";
import { Checkbox } from "@/components/ui/checkbox";
import { Slider } from "@/components/ui/slider";
import {
  Download,
  Loader2,
  ChevronRight,
  ChevronLeft,
  AlertCircle,
  CheckCircle2,
} from "lucide-react";
import { useToast } from "../hooks/use-toast";
import { Badge } from "@/components/ui/badge";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import Header from "../common/header";
import Footer from "../common/footer";
import { motion, AnimatePresence } from "framer-motion";
import { Progress } from "@/components/ui/progress";

type UserData = {
  age: string;
  gender: string;
  hepatitisType?: string | null;
  symptoms: {
    jaundice: boolean;
    darkUrine: boolean;
    abdominalPain: number;
    fatigue: number;
    fever: boolean;
    nausea: boolean;
    jointPain: boolean;
    appetite: boolean;
  };
  riskFactors: string[];
};

type PredictionResult = {
  success: boolean;
  prediction: {
    success: boolean;
    message: string;
    predictions: Array<{
      predicted_class: string;
      "probability_Hepatitis A": number;
      "probability_Hepatitis C": number;
    }>;
    total_predictions: number;
  };
};

export default function PredictionSelector() {
  const { toast } = useToast();
  const [activeTab, setActiveTab] = useState("basic");
  const [loading, setLoading] = useState(false);
  const [progress, setProgress] = useState(0);
  const [showResults, setShowResults] = useState(false);
  const [predictionResult, setPredictionResult] =
    useState<PredictionResult | null>(null);
  const [userData, setUserData] = useState<UserData>({
    age: "",
    gender: "",
    symptoms: {
      jaundice: false,
      darkUrine: false,
      abdominalPain: 0,
      fatigue: 0,
      fever: false,
      nausea: false,
      jointPain: false,
      appetite: false,
    },
    riskFactors: [],
  });

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

  const handleSymptomChange = (id: string, value: boolean | number) => {
    setUserData((prev) => ({
      ...prev,
      symptoms: {
        ...prev.symptoms,
        [id]: value,
      },
    }));
  };

  const handleRiskFactorToggle = (id: string) => {
    setUserData((prev) => {
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

  const handleSubmit = async () => {
    if (!userData.age || !userData.gender) {
      toast({
        title: "Missing Information",
        description: "Please provide all required information.",
        variant: "destructive",
      });
      return;
    }

    try {
      setLoading(true);
      setProgress(0);

      // Get API URL from environment variable
      const API_URL =
        process.env.NEXT_PUBLIC_API_URL || "http://localhost:5000/api";

      // Call the API with JSON data
      const response = await fetch(`${API_URL}/predict`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({
          age: userData.age,
          gender: userData.gender,
          symptoms: {
            jaundice: userData.symptoms.jaundice,
            dark_urine: userData.symptoms.darkUrine,
            pain: userData.symptoms.abdominalPain > 0,
            fatigue: userData.symptoms.fatigue > 0,
            nausea: userData.symptoms.nausea,
            vomiting: false, // Add if needed
            fever: userData.symptoms.fever,
            loss_of_appetite: userData.symptoms.appetite,
            joint_pain: userData.symptoms.jointPain,
          },
          riskFactors: userData.riskFactors,
        }),
      });

      if (!response.ok) {
        const errorData = await response.json();
        throw new Error(errorData.error || "Failed to process prediction");
      }

      const data = await response.json();
      console.log("Prediction data received:", data);
      setPredictionResult(data);
      setShowResults(true);

      // Scroll to results
      setTimeout(() => {
        const resultsElement = document.getElementById("prediction-results");
        if (resultsElement) {
          resultsElement.scrollIntoView({ behavior: "smooth" });
        }
      }, 100);
    } catch (error: any) {
      console.error("Prediction error:", error);
      toast({
        title: "Error",
        description:
          error.message || "There was an error processing your information.",
        variant: "destructive",
      });
    } finally {
      setLoading(false);
    }
  };

  const resetForm = () => {
    setUserData({
      age: "",
      gender: "",
      hepatitisType: "",
      symptoms: {
        jaundice: false,
        darkUrine: false,
        abdominalPain: 0,
        fatigue: 0,
        fever: false,
        nausea: false,
        jointPain: false,
        appetite: false,
      },
      riskFactors: [],
    });
    setShowResults(false);
    setActiveTab("basic");
  };

  const downloadResults = () => {
    const resultText = `
HepaPredict Results
-------------------
Date: ${new Date().toLocaleString()}

Patient Information:
- Age Group: ${formatAgeGroup(userData.age)}
- Gender: ${formatGender(userData.gender)}

Selected Hepatitis Type: ${formatHepatitisType(userData.hepatitisType)}

Reported Symptoms:
- Jaundice (yellowing of skin/eyes): ${
      userData.symptoms.jaundice ? "Yes" : "No"
    }
- Dark urine: ${userData.symptoms.darkUrine ? "Yes" : "No"}
- Abdominal pain level: ${userData.symptoms.abdominalPain / 10}/10
- Fatigue level: ${userData.symptoms.fatigue / 10}/10
- Fever: ${userData.symptoms.fever ? "Yes" : "No"}
- Nausea: ${userData.symptoms.nausea ? "Yes" : "No"}
- Joint pain: ${userData.symptoms.jointPain ? "Yes" : "No"}
- Loss of appetite: ${userData.symptoms.appetite ? "Yes" : "No"}

Risk Factors:
${
  userData.riskFactors.length > 0
    ? userData.riskFactors
        .map((factor) => `- ${formatRiskFactor(factor)}`)
        .join("\n")
    : "- None reported"
}

Prediction Results:
${
  predictionResult?.prediction.predictions
    .map(
      (pred) =>
        `${pred.predicted_class}:
- Probability of Hepatitis A: ${(pred["probability_Hepatitis A"] * 100).toFixed(
          1
        )}%
- Probability of Hepatitis C: ${(pred["probability_Hepatitis C"] * 100).toFixed(
          1
        )}%`
    )
    .join("\n\n") || "No prediction results available."
}

Important Note:
This is based on your selection and is not a medical diagnosis. For accurate diagnosis, please consult with a healthcare professional.
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

    toast({
      title: "Results Downloaded",
      description:
        "Your assessment results have been downloaded as a text file.",
    });
  };

  const nextTab = () => {
    if (activeTab === "basic") {
      if (!userData.age || !userData.gender) {
        toast({
          title: "Missing Information",
          description:
            "Please select your age group and gender before continuing.",
          variant: "destructive",
        });
        return;
      }
      setActiveTab("symptoms");
    } else if (activeTab === "symptoms") {
      setActiveTab("risk");
    }
  };

  const prevTab = () => {
    if (activeTab === "symptoms") {
      setActiveTab("basic");
    } else if (activeTab === "risk") {
      setActiveTab("symptoms");
    }
  };

  const getProgressPercentage = () => {
    switch (activeTab) {
      case "basic":
        return 33;
      case "symptoms":
        return 66;
      case "risk":
        return 100;
      default:
        return 0;
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
    <div className="min-h-screen flex flex-col bg-white dark:bg-slate-950">
      <Header />
      <main className="flex-grow container mx-auto px-4 py-12">
        <motion.div
          className="max-w-4xl mx-auto"
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.5 }}
        >
          <Card className="w-full max-w-4xl mx-auto border shadow-sm overflow-hidden">
            <CardHeader className="bg-blue-600 dark:bg-blue-700 p-6">
              <CardTitle className="text-2xl font-bold text-white">
                HepaPredict
              </CardTitle>
              <CardDescription className="text-blue-100">
                {!showResults
                  ? "Complete the assessment to get your results"
                  : "Your prediction results"}
              </CardDescription>
              {!showResults && (
                <div className="mt-4">
                  <div className="flex justify-between text-xs text-blue-100 mb-2">
                    <span>Basic Info</span>
                    <span>Symptoms</span>
                    <span>Risk Factors</span>
                  </div>
                  <Progress
                    value={getProgressPercentage()}
                    className="h-2 bg-blue-500/30"
                  />
                </div>
              )}
            </CardHeader>
            <CardContent className="bg-white dark:bg-slate-900 p-6">
              <AnimatePresence mode="wait">
                {!showResults ? (
                  <motion.div
                    key="form"
                    initial={{ opacity: 0, y: 20 }}
                    animate={{ opacity: 1, y: 0 }}
                    exit={{ opacity: 0, y: -20 }}
                    transition={{ duration: 0.5 }}
                  >
                    <Tabs
                      value={activeTab}
                      onValueChange={setActiveTab}
                      className="w-full"
                    >
                      <TabsList className="grid grid-cols-3 mb-8 bg-slate-100 dark:bg-slate-800 p-1 rounded-lg">
                        <TabsTrigger
                          value="basic"
                          className="data-[state=active]:bg-blue-600 data-[state=active]:text-white dark:data-[state=active]:bg-blue-700"
                        >
                          Basic Info
                        </TabsTrigger>
                        <TabsTrigger
                          value="symptoms"
                          className="data-[state=active]:bg-blue-600 data-[state=active]:text-white dark:data-[state=active]:bg-blue-700"
                        >
                          Symptoms
                        </TabsTrigger>
                        <TabsTrigger
                          value="risk"
                          className="data-[state=active]:bg-blue-600 data-[state=active]:text-white dark:data-[state=active]:bg-blue-700"
                        >
                          Risk Factors
                        </TabsTrigger>
                      </TabsList>

                      <TabsContent value="basic">
                        <motion.div
                          className="space-y-6"
                          initial="hidden"
                          animate="visible"
                          variants={containerVariants}
                        >
                          <motion.div variants={itemVariants}>
                            <Label
                              htmlFor="age-group"
                              className="text-slate-900 dark:text-white"
                            >
                              Age Group
                            </Label>
                            <Select
                              value={userData.age}
                              onValueChange={(value) =>
                                setUserData((prev) => ({ ...prev, age: value }))
                              }
                            >
                              <SelectTrigger
                                id="age-group"
                                className="w-full mt-2 border-slate-200 dark:border-slate-700"
                              >
                                <SelectValue placeholder="Select your age group" />
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
                          </motion.div>

                          <motion.div variants={itemVariants}>
                            <Label
                              htmlFor="gender"
                              className="text-slate-900 dark:text-white"
                            >
                              Gender
                            </Label>
                            <RadioGroup
                              value={userData.gender}
                              onValueChange={(value: any) =>
                                setUserData((prev) => ({
                                  ...prev,
                                  gender: value,
                                }))
                              }
                              className="flex flex-col sm:flex-row gap-4 mt-2"
                            >
                              <div className="flex items-center space-x-2 p-3 rounded-lg border border-slate-200 dark:border-slate-700 hover:bg-slate-50 dark:hover:bg-slate-800/50 transition-colors">
                                <RadioGroupItem value="male" id="male" />
                                <Label
                                  htmlFor="male"
                                  className="cursor-pointer text-slate-900 dark:text-white"
                                >
                                  Male
                                </Label>
                              </div>
                              <div className="flex items-center space-x-2 p-3 rounded-lg border border-slate-200 dark:border-slate-700 hover:bg-slate-50 dark:hover:bg-slate-800/50 transition-colors">
                                <RadioGroupItem value="female" id="female" />
                                <Label
                                  htmlFor="female"
                                  className="cursor-pointer text-slate-900 dark:text-white"
                                >
                                  Female
                                </Label>
                              </div>
                              <div className="flex items-center space-x-2 p-3 rounded-lg border border-slate-200 dark:border-slate-700 hover:bg-slate-50 dark:hover:bg-slate-800/50 transition-colors">
                                <RadioGroupItem value="other" id="other" />
                                <Label
                                  htmlFor="other"
                                  className="cursor-pointer text-slate-900 dark:text-white"
                                >
                                  Other
                                </Label>
                              </div>
                            </RadioGroup>
                          </motion.div>
                        </motion.div>
                      </TabsContent>

                      <TabsContent value="symptoms">
                        <motion.div
                          className="space-y-6"
                          initial="hidden"
                          animate="visible"
                          variants={containerVariants}
                        >
                          <motion.div variants={itemVariants}>
                            <div className="flex items-center gap-2 mb-6">
                              <div className="p-1.5 bg-blue-100 dark:bg-blue-900/50 rounded-lg">
                                <AlertCircle className="h-5 w-5 text-blue-600 dark:text-blue-400" />
                              </div>
                              <h3 className="text-xl font-semibold text-slate-900 dark:text-white">
                                Select Your Symptoms
                              </h3>
                            </div>
                          </motion.div>

                          <motion.div variants={itemVariants}>
                            <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                              <div className="space-y-4">
                                {[
                                  {
                                    id: "jaundice",
                                    label: "Yellowing of skin/eyes (Jaundice)",
                                  },
                                  { id: "darkUrine", label: "Dark Urine" },
                                  { id: "fever", label: "Fever" },
                                  { id: "jointPain", label: "Joint Pain" },
                                ].map((symptom) => (
                                  <div
                                    key={symptom.id}
                                    className="flex items-center space-x-2 p-3 rounded-lg border border-slate-200 dark:border-slate-700 hover:bg-slate-50 dark:hover:bg-slate-800/50 transition-colors"
                                  >
                                    <Checkbox
                                      id={symptom.id}
                                      checked={
                                        userData.symptoms[
                                          symptom.id as keyof typeof userData.symptoms
                                        ] as boolean
                                      }
                                      onCheckedChange={(checked: boolean) =>
                                        handleSymptomChange(
                                          symptom.id,
                                          checked === true
                                        )
                                      }
                                      className="data-[state=checked]:bg-blue-600 data-[state=checked]:border-blue-600"
                                    />
                                    <Label
                                      htmlFor={symptom.id}
                                      className="flex-1 cursor-pointer text-slate-900 dark:text-white"
                                    >
                                      {symptom.label}
                                    </Label>
                                  </div>
                                ))}
                              </div>

                              <div className="space-y-4">
                                {[
                                  { id: "nausea", label: "Nausea" },
                                  { id: "appetite", label: "Loss of Appetite" },
                                ].map((symptom) => (
                                  <div
                                    key={symptom.id}
                                    className="flex items-center space-x-2 p-3 rounded-lg border border-slate-200 dark:border-slate-700 hover:bg-slate-50 dark:hover:bg-slate-800/50 transition-colors"
                                  >
                                    <Checkbox
                                      id={symptom.id}
                                      checked={
                                        userData.symptoms[
                                          symptom.id as keyof typeof userData.symptoms
                                        ] as boolean
                                      }
                                      onCheckedChange={(checked: boolean) =>
                                        handleSymptomChange(
                                          symptom.id,
                                          checked === true
                                        )
                                      }
                                      className="data-[state=checked]:bg-blue-600 data-[state=checked]:border-blue-600"
                                    />
                                    <Label
                                      htmlFor={symptom.id}
                                      className="flex-1 cursor-pointer text-slate-900 dark:text-white"
                                    >
                                      {symptom.label}
                                    </Label>
                                  </div>
                                ))}
                              </div>
                            </div>
                          </motion.div>

                          <motion.div
                            className="space-y-6 mt-8"
                            variants={itemVariants}
                          >
                            <div>
                              <div className="flex justify-between">
                                <Label className="text-slate-900 dark:text-white">
                                  Abdominal Pain Level
                                </Label>
                                <span className="text-sm font-medium px-2 py-1 rounded-md bg-blue-100 dark:bg-blue-900/30 text-blue-700 dark:text-blue-300">
                                  {userData.symptoms.abdominalPain / 10}/10
                                </span>
                              </div>
                              <Slider
                                value={[userData.symptoms.abdominalPain]}
                                min={0}
                                max={100}
                                step={10}
                                onValueChange={(value) =>
                                  handleSymptomChange("abdominalPain", value[0])
                                }
                                className="mt-2"
                              />
                              <div className="flex justify-between text-xs text-slate-500 dark:text-slate-400 px-1 mt-1">
                                <span>None</span>
                                <span>Mild</span>
                                <span>Moderate</span>
                                <span>Severe</span>
                              </div>
                            </div>

                            <div>
                              <div className="flex justify-between">
                                <Label className="text-slate-900 dark:text-white">
                                  Fatigue Level
                                </Label>
                                <span className="text-sm font-medium px-2 py-1 rounded-md bg-blue-100 dark:bg-blue-900/30 text-blue-700 dark:text-blue-300">
                                  {userData.symptoms.fatigue / 10}/10
                                </span>
                              </div>
                              <Slider
                                value={[userData.symptoms.fatigue]}
                                min={0}
                                max={100}
                                step={10}
                                onValueChange={(value) =>
                                  handleSymptomChange("fatigue", value[0])
                                }
                                className="mt-2"
                              />
                              <div className="flex justify-between text-xs text-slate-500 dark:text-slate-400 px-1 mt-1">
                                <span>None</span>
                                <span>Mild</span>
                                <span>Moderate</span>
                                <span>Severe</span>
                              </div>
                            </div>
                          </motion.div>
                        </motion.div>
                      </TabsContent>

                      <TabsContent value="risk">
                        <motion.div
                          className="space-y-6"
                          initial="hidden"
                          animate="visible"
                          variants={containerVariants}
                        >
                          <motion.div variants={itemVariants}>
                            <div className="flex items-center gap-2 mb-6">
                              <div className="p-1.5 bg-blue-100 dark:bg-blue-900/50 rounded-lg">
                                <AlertCircle className="h-5 w-5 text-blue-600 dark:text-blue-400" />
                              </div>
                              <h3 className="text-xl font-semibold text-slate-900 dark:text-white">
                                Risk Factors
                              </h3>
                            </div>
                            <p className="text-slate-600 dark:text-slate-300 mb-6">
                              Select any risk factors that apply to you:
                            </p>
                          </motion.div>

                          <motion.div
                            className="space-y-4"
                            variants={itemVariants}
                          >
                            {[
                              {
                                id: "recentTravel",
                                label: "Recent Travel to High-Risk Areas",
                                description:
                                  "Travel to regions with known hepatitis outbreaks in the past 6 months",
                              },
                              {
                                id: "bloodTransfusion",
                                label: "History of Blood Transfusion",
                                description:
                                  "Received blood products before comprehensive screening was implemented",
                              },
                              {
                                id: "unsafeInjection",
                                label: "History of Unsafe Injection Practices",
                                description:
                                  "Shared needles or received injections with potentially unsterilized equipment",
                              },
                              {
                                id: "contactWithInfected",
                                label: "Contact with Infected Person",
                                description:
                                  "Close contact with someone diagnosed with hepatitis",
                              },
                            ].map((factor) => (
                              <div
                                key={factor.id}
                                className="p-4 rounded-lg border border-slate-200 dark:border-slate-700 hover:bg-slate-50 dark:hover:bg-slate-800/50 transition-colors"
                              >
                                <div className="flex items-center space-x-3">
                                  <Checkbox
                                    id={factor.id}
                                    checked={userData.riskFactors.includes(
                                      factor.id
                                    )}
                                    onCheckedChange={() =>
                                      handleRiskFactorToggle(factor.id)
                                    }
                                    className="data-[state=checked]:bg-blue-600 data-[state=checked]:border-blue-600"
                                  />
                                  <div className="flex-1">
                                    <Label
                                      htmlFor={factor.id}
                                      className="font-medium cursor-pointer text-slate-900 dark:text-white"
                                    >
                                      {factor.label}
                                    </Label>
                                    <p className="text-sm text-slate-500 dark:text-slate-400 mt-1">
                                      {factor.description}
                                    </p>
                                  </div>
                                </div>
                              </div>
                            ))}
                          </motion.div>

                          <motion.div
                            className="p-6 mt-6 rounded-xl bg-slate-50 dark:bg-slate-800 border border-slate-200 dark:border-slate-700"
                            variants={itemVariants}
                          >
                            <h4 className="font-semibold mb-3 text-slate-900 dark:text-white flex items-center gap-2">
                              <AlertCircle className="h-5 w-5 text-blue-600 dark:text-blue-400" />
                              Important Note:
                            </h4>
                            <p className="text-slate-700 dark:text-slate-300">
                              This assessment is not a medical diagnosis. It's
                              based on the information you provided and is
                              intended to guide your next steps. For accurate
                              diagnosis, please consult with a healthcare
                              professional.
                            </p>
                          </motion.div>
                        </motion.div>
                      </TabsContent>
                    </Tabs>

                    {loading && (
                      <motion.div
                        className="mt-6 p-4 rounded-lg bg-slate-50 dark:bg-slate-800 border border-slate-200 dark:border-slate-700"
                        initial={{ opacity: 0, y: 10 }}
                        animate={{ opacity: 1, y: 0 }}
                        transition={{ duration: 0.3 }}
                      >
                        <p className="mb-2 text-sm font-medium text-slate-900 dark:text-white flex justify-between">
                          Processing
                          <span>{Math.round(progress)}%</span>
                        </p>
                        <Progress value={progress} className="h-2" />
                        <p className="mt-2 text-xs text-slate-500 dark:text-slate-400">
                          {progress < 100
                            ? "Processing data and analyzing symptoms..."
                            : "Processing complete!"}
                        </p>
                      </motion.div>
                    )}
                  </motion.div>
                ) : (
                  <motion.div
                    id="prediction-results"
                    key="results"
                    initial={{ opacity: 0, y: 20 }}
                    animate={{ opacity: 1, y: 0 }}
                    exit={{ opacity: 0, y: -20 }}
                    transition={{ duration: 0.5 }}
                  >
                    <div className="mb-6 p-6 rounded-xl bg-white dark:bg-slate-800 border border-slate-200 dark:border-slate-700">
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
                            Prediction Results
                          </h3>
                        </motion.div>

                        {predictionResult?.prediction.predictions.map(
                          (pred, index) => (
                            <motion.div
                              key={index}
                              className="mt-6"
                              initial={{ opacity: 0, y: 20 }}
                              animate={{ opacity: 1, y: 0 }}
                              transition={{ delay: 0.3 + index * 0.2 }}
                            >
                              <Badge className="mb-4 bg-blue-600 hover:bg-blue-700 text-white px-4 py-2 text-lg">
                                {pred.predicted_class}
                              </Badge>

                              <motion.div
                                className="mt-6 space-y-4 bg-slate-50 dark:bg-slate-900 p-6 rounded-xl border border-slate-200 dark:border-slate-700"
                                initial={{ opacity: 0, y: 10 }}
                                animate={{ opacity: 1, y: 0 }}
                                transition={{ delay: 0.5 + index * 0.2 }}
                              >
                                <div>
                                  <p className="text-sm text-slate-500 dark:text-slate-400 mb-1">
                                    Probability of Hepatitis A
                                  </p>
                                  <div className="flex items-center gap-4">
                                    <div className="w-full bg-slate-200 dark:bg-slate-700 rounded-full h-4">
                                      <div
                                        className="h-4 rounded-full bg-blue-600"
                                        style={{
                                          width: `${
                                            pred["probability_Hepatitis A"] *
                                            100
                                          }%`,
                                        }}
                                      ></div>
                                    </div>
                                    <span className="font-medium text-blue-600 dark:text-blue-400 min-w-[80px] text-right">
                                      {(
                                        pred["probability_Hepatitis A"] * 100
                                      ).toFixed(1)}
                                      %
                                    </span>
                                  </div>
                                </div>

                                <div>
                                  <p className="text-sm text-slate-500 dark:text-slate-400 mb-1">
                                    Probability of Hepatitis C
                                  </p>
                                  <div className="flex items-center gap-4">
                                    <div className="w-full bg-slate-200 dark:bg-slate-700 rounded-full h-4">
                                      <div
                                        className="h-4 rounded-full bg-blue-600"
                                        style={{
                                          width: `${
                                            pred["probability_Hepatitis C"] *
                                            100
                                          }%`,
                                        }}
                                      ></div>
                                    </div>
                                    <span className="font-medium text-blue-600 dark:text-blue-400 min-w-[80px] text-right">
                                      {(
                                        pred["probability_Hepatitis C"] * 100
                                      ).toFixed(1)}
                                      %
                                    </span>
                                  </div>
                                </div>
                              </motion.div>
                            </motion.div>
                          )
                        )}

                        <motion.div
                          className="text-left mt-8"
                          initial={{ opacity: 0, y: 20 }}
                          animate={{ opacity: 1, y: 0 }}
                          transition={{ delay: 0.7 }}
                        >
                          <p className="mb-6 text-slate-700 dark:text-slate-300">
                            <strong className="text-slate-900 dark:text-white">
                              Important Note:
                            </strong>{" "}
                            This prediction is based on the symptoms and risk
                            factors you provided. It is not a medical diagnosis.
                            Please consult with a healthcare professional for
                            proper medical advice.
                          </p>

                          <motion.div
                            className="p-6 rounded-xl bg-slate-50 dark:bg-slate-800 border border-slate-200 dark:border-slate-700"
                            initial={{ opacity: 0, y: 10 }}
                            animate={{ opacity: 1, y: 0 }}
                            transition={{ delay: 0.8 }}
                          >
                            <h4 className="font-semibold mb-4 text-slate-900 dark:text-white flex items-center gap-2">
                              <CheckCircle2 className="h-5 w-5 text-blue-600 dark:text-blue-400" />
                              Next Steps:
                            </h4>
                            <ul className="space-y-3">
                              {[
                                "Schedule an appointment with your healthcare provider",
                                "Share these results with your doctor",
                                "Get proper medical testing and diagnosis",
                              ].map((step, i) => (
                                <motion.li
                                  key={i}
                                  className="flex items-start p-3 rounded-lg hover:bg-slate-100 dark:hover:bg-slate-700 transition-colors"
                                  initial={{ x: -10, opacity: 0 }}
                                  animate={{ x: 0, opacity: 1 }}
                                  transition={{ delay: 0.9 + i * 0.1 }}
                                >
                                  <CheckCircle2 className="h-5 w-5 mr-3 text-blue-600 dark:text-blue-400 flex-shrink-0 mt-0.5" />
                                  <span className="text-slate-700 dark:text-slate-300">
                                    {step}
                                  </span>
                                </motion.li>
                              ))}
                            </ul>
                          </motion.div>
                        </motion.div>
                      </div>
                    </div>
                  </motion.div>
                )}
              </AnimatePresence>
            </CardContent>
            <CardFooter className="flex justify-between bg-blue-600 dark:bg-blue-700 p-6">
              {!showResults ? (
                <>
                  {activeTab !== "basic" && (
                    <Button
                      variant="outline"
                      onClick={prevTab}
                      className="border-blue-500 bg-transparent text-white hover:bg-blue-500/20"
                    >
                      <ChevronLeft className="h-4 w-4 mr-2" /> Previous
                    </Button>
                  )}
                  {activeTab === "risk" ? (
                    <Button
                      onClick={handleSubmit}
                      disabled={loading}
                      className="bg-white text-blue-600 hover:bg-blue-50"
                    >
                      {loading ? (
                        <>
                          <Loader2 className="mr-2 h-4 w-4 animate-spin" />
                          Processing...
                        </>
                      ) : (
                        "Submit"
                      )}
                    </Button>
                  ) : (
                    <Button
                      onClick={nextTab}
                      className="bg-white text-blue-600 hover:bg-blue-50"
                    >
                      Next <ChevronRight className="h-4 w-4 ml-2" />
                    </Button>
                  )}
                  {activeTab === "basic" && <div></div>}
                </>
              ) : (
                <>
                  <Button
                    variant="outline"
                    onClick={resetForm}
                    className="border-blue-500 bg-transparent text-white hover:bg-blue-500/20"
                  >
                    Start Over
                  </Button>
                  <Button
                    onClick={downloadResults}
                    className="bg-white text-blue-600 hover:bg-blue-50"
                  >
                    <Download className="h-4 w-4 mr-2" /> Download Results
                  </Button>
                </>
              )}
            </CardFooter>
          </Card>
        </motion.div>
      </main>
      <Footer />
    </div>
  );
}

// Helper functions
function formatAgeGroup(age: string): string {
  switch (age) {
    case "under18":
      return "Under 18";
    case "18-30":
      return "18-30";
    case "31-45":
      return "31-45";
    case "46-60":
      return "46-60";
    case "over60":
      return "Over 60";
    default:
      return age;
  }
}

function formatGender(gender: string): string {
  return gender.charAt(0).toUpperCase() + gender.slice(1);
}

function formatHepatitisType(type: string | null | undefined): string {
  if (!type) return "Not specified";
  switch (type) {
    case "hepatitisA":
      return "Hepatitis A";
    case "hepatitisB":
      return "Hepatitis B";
    case "hepatitisC":
      return "Hepatitis C";
    case "hepatitisD":
      return "Hepatitis D";
    case "hepatitisE":
      return "Hepatitis E";
    case "unlikely":
      return "No Hepatitis / Not Sure";
    default:
      return type;
  }
}

function formatRiskFactor(factor: string): string {
  switch (factor) {
    case "recentTravel":
      return "Recent Travel to High-Risk Areas";
    case "bloodTransfusion":
      return "History of Blood Transfusion";
    case "unsafeInjection":
      return "History of Unsafe Injection Practices";
    case "contactWithInfected":
      return "Contact with Infected Person";
    default:
      return factor;
  }
}
