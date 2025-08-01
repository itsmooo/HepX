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
import { Alert, AlertDescription } from "@/components/ui/alert";

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

      // Get API URL from environment variable - pointing to Python FastAPI server
      const API_URL =
        process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";

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
            <CardHeader className="bg-blue-600 p-6">
              <div className="flex items-center gap-3 mb-4">
                <div className="p-2 bg-white/20 rounded-lg">
                  <div className="text-2xl">🏥</div>
                </div>
                <div>
                  <CardTitle className="text-2xl font-bold text-white mb-1">
                    HepaPredict
                  </CardTitle>
                  <CardDescription className="text-blue-100">
                    {!showResults
                      ? "Complete the assessment to get your results"
                      : "Your prediction results"}
                  </CardDescription>
                </div>
              </div>
              
              {!showResults && (
                <div className="mt-4">
                  <div className="flex justify-between text-sm text-blue-100 mb-2 font-medium">
                    <span>Basic Info</span>
                    <span>Symptoms</span>
                    <span>Risk Factors</span>
                  </div>
                  <Progress
                    value={getProgressPercentage()}
                    className="h-2 bg-white/20 rounded-full"
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
                              className="text-slate-900 dark:text-white font-medium mb-3 block"
                            >
                              Age Group
                            </Label>
                            <div className="grid grid-cols-2 md:grid-cols-3 gap-3">
                              {[
                                { value: "under18", label: "Under 18" },
                                { value: "18-30", label: "18-30" },
                                { value: "31-45", label: "31-45" },
                                { value: "46-60", label: "46-60" },
                                { value: "over60", label: "Over 60" },
                              ].map((ageGroup) => (
                                <div
                                  key={ageGroup.value}
                                  className={`relative p-3 rounded-lg border-2 transition-all duration-200 cursor-pointer ${
                                    userData.age === ageGroup.value
                                      ? "border-blue-500 bg-blue-50 dark:bg-blue-900/20"
                                      : "border-slate-200 dark:border-slate-700 hover:border-blue-300 dark:hover:border-blue-600"
                                  }`}
                                  onClick={() => setUserData((prev) => ({ ...prev, age: ageGroup.value }))}
                                >
                                  <div className="flex items-center justify-between">
                                    <span className={`font-medium ${
                                      userData.age === ageGroup.value
                                        ? "text-blue-700 dark:text-blue-300"
                                        : "text-slate-900 dark:text-white"
                                    }`}>
                                      {ageGroup.label}
                                    </span>
                                    {userData.age === ageGroup.value && (
                                      <div className="w-4 h-4 bg-blue-500 rounded-full flex items-center justify-center">
                                        <CheckCircle2 className="w-3 h-3 text-white" />
                                      </div>
                                    )}
                                  </div>
                                </div>
                              ))}
                            </div>
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
                              className="grid grid-cols-2 gap-4 mt-2"
                            >
                              <div
                                className={`relative p-3 rounded-lg border-2 transition-all duration-200 cursor-pointer ${
                                  userData.gender === "male"
                                    ? "border-blue-500 bg-blue-50 dark:bg-blue-900/20"
                                    : "border-slate-200 dark:border-slate-700 hover:border-blue-300 dark:hover:border-blue-600"
                                }`}
                              >
                                <RadioGroupItem value="male" id="male" className="sr-only" />
                                <div className="flex items-center space-x-3">
                                  <div className={`w-4 h-4 rounded-full border-2 flex items-center justify-center ${
                                    userData.gender === "male"
                                      ? "border-blue-500 bg-blue-500"
                                      : "border-slate-300 dark:border-slate-600"
                                  }`}>
                                    {userData.gender === "male" && (
                                      <div className="w-2 h-2 bg-white rounded-full" />
                                    )}
                                  </div>
                                  <div className="flex-1">
                                    <Label
                                      htmlFor="male"
                                      className={`cursor-pointer font-medium ${
                                        userData.gender === "male"
                                          ? "text-blue-700 dark:text-blue-300"
                                          : "text-slate-900 dark:text-white"
                                      }`}
                                    >
                                      Male
                                    </Label>
                                  </div>
                                </div>
                              </div>

                              <div
                                className={`relative p-3 rounded-lg border-2 transition-all duration-200 cursor-pointer ${
                                  userData.gender === "female"
                                    ? "border-pink-500 bg-pink-50 dark:bg-pink-900/20"
                                    : "border-slate-200 dark:border-slate-700 hover:border-pink-300 dark:hover:border-pink-600"
                                }`}
                              >
                                <RadioGroupItem value="female" id="female" className="sr-only" />
                                <div className="flex items-center space-x-3">
                                  <div className={`w-4 h-4 rounded-full border-2 flex items-center justify-center ${
                                    userData.gender === "female"
                                      ? "border-pink-500 bg-pink-500"
                                      : "border-slate-300 dark:border-slate-600"
                                  }`}>
                                    {userData.gender === "female" && (
                                      <div className="w-2 h-2 bg-white rounded-full" />
                                    )}
                                  </div>
                                  <div className="flex-1">
                                    <Label
                                      htmlFor="female"
                                      className={`cursor-pointer font-medium ${
                                        userData.gender === "female"
                                          ? "text-pink-700 dark:text-pink-300"
                                          : "text-slate-900 dark:text-white"
                                      }`}
                                    >
                                      Female
                                    </Label>
                                  </div>
                                </div>
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
                                    icon: "🟡",
                                    color: "from-yellow-400 to-orange-500"
                                  },
                                  { 
                                    id: "darkUrine", 
                                    label: "Dark Urine",
                                    icon: "🟤",
                                    color: "from-amber-600 to-brown-600"
                                  },
                                  { 
                                    id: "fever", 
                                    label: "Fever",
                                    icon: "🌡️",
                                    color: "from-red-400 to-pink-500"
                                  },
                                  { 
                                    id: "jointPain", 
                                    label: "Joint Pain",
                                    icon: "🦴",
                                    color: "from-gray-400 to-slate-500"
                                  },
                                ].map((symptom) => (
                                  <motion.div
                                    key={symptom.id}
                                    className={`relative p-4 rounded-2xl border-2 transition-all duration-300 cursor-pointer group ${
                                      userData.symptoms[symptom.id as keyof typeof userData.symptoms] as boolean
                                        ? `border-transparent bg-gradient-to-r ${symptom.color} shadow-lg scale-105`
                                        : "border-slate-200 dark:border-slate-700 hover:border-slate-300 dark:hover:border-slate-600 hover:bg-slate-50/50 dark:hover:bg-slate-800/50"
                                    }`}
                                    whileHover={{ scale: 1.02 }}
                                    whileTap={{ scale: 0.98 }}
                                    onClick={() => handleSymptomChange(
                                      symptom.id,
                                      !(userData.symptoms[symptom.id as keyof typeof userData.symptoms] as boolean)
                                    )}
                                  >
                                    <Checkbox
                                      id={symptom.id}
                                      checked={userData.symptoms[symptom.id as keyof typeof userData.symptoms] as boolean}
                                      onCheckedChange={(checked: boolean) => handleSymptomChange(symptom.id, checked === true)}
                                      className="sr-only"
                                    />
                                    <div className="flex items-center space-x-3">
                                      <div className="text-2xl">{symptom.icon}</div>
                                      <div className="flex-1">
                                        <Label
                                          htmlFor={symptom.id}
                                          className={`cursor-pointer font-medium ${
                                            userData.symptoms[symptom.id as keyof typeof userData.symptoms] as boolean
                                              ? "text-white"
                                              : "text-slate-900 dark:text-white"
                                          }`}
                                        >
                                          {symptom.label}
                                        </Label>
                                      </div>
                                      {userData.symptoms[symptom.id as keyof typeof userData.symptoms] as boolean && (
                                        <motion.div
                                          className="w-5 h-5 bg-white/20 rounded-full flex items-center justify-center"
                                          initial={{ scale: 0 }}
                                          animate={{ scale: 1 }}
                                          transition={{ type: "spring", stiffness: 300, damping: 20 }}
                                        >
                                          <CheckCircle2 className="w-3 h-3 text-white" />
                                        </motion.div>
                                      )}
                                    </div>
                                  </motion.div>
                                ))}
                              </div>

                              <div className="space-y-4">
                                {[
                                  { 
                                    id: "nausea", 
                                    label: "Nausea",
                                    icon: "🤢",
                                    color: "from-green-400 to-emerald-500"
                                  },
                                  { 
                                    id: "appetite", 
                                    label: "Loss of Appetite",
                                    icon: "🍽️",
                                    color: "from-purple-400 to-indigo-500"
                                  },
                                ].map((symptom) => (
                                  <motion.div
                                    key={symptom.id}
                                    className={`relative p-4 rounded-2xl border-2 transition-all duration-300 cursor-pointer group ${
                                      userData.symptoms[symptom.id as keyof typeof userData.symptoms] as boolean
                                        ? `border-transparent bg-gradient-to-r ${symptom.color} shadow-lg scale-105`
                                        : "border-slate-200 dark:border-slate-700 hover:border-slate-300 dark:hover:border-slate-600 hover:bg-slate-50/50 dark:hover:bg-slate-800/50"
                                    }`}
                                    whileHover={{ scale: 1.02 }}
                                    whileTap={{ scale: 0.98 }}
                                    onClick={() => handleSymptomChange(
                                      symptom.id,
                                      !(userData.symptoms[symptom.id as keyof typeof userData.symptoms] as boolean)
                                    )}
                                  >
                                    <Checkbox
                                      id={symptom.id}
                                      checked={userData.symptoms[symptom.id as keyof typeof userData.symptoms] as boolean}
                                      onCheckedChange={(checked: boolean) => handleSymptomChange(symptom.id, checked === true)}
                                      className="sr-only"
                                    />
                                    <div className="flex items-center space-x-3">
                                      <div className="text-2xl">{symptom.icon}</div>
                                      <div className="flex-1">
                                        <Label
                                          htmlFor={symptom.id}
                                          className={`cursor-pointer font-medium ${
                                            userData.symptoms[symptom.id as keyof typeof userData.symptoms] as boolean
                                              ? "text-white"
                                              : "text-slate-900 dark:text-white"
                                          }`}
                                        >
                                          {symptom.label}
                                        </Label>
                                      </div>
                                      {userData.symptoms[symptom.id as keyof typeof userData.symptoms] as boolean && (
                                        <motion.div
                                          className="w-5 h-5 bg-white/20 rounded-full flex items-center justify-center"
                                          initial={{ scale: 0 }}
                                          animate={{ scale: 1 }}
                                          transition={{ type: "spring", stiffness: 300, damping: 20 }}
                                        >
                                          <CheckCircle2 className="w-3 h-3 text-white" />
                                        </motion.div>
                                      )}
                                    </div>
                                  </motion.div>
                                ))}
                              </div>
                            </div>
                          </motion.div>

                          <motion.div
                            className="space-y-8 mt-8"
                            variants={itemVariants}
                          >
                            <div className="p-6 rounded-2xl bg-gradient-to-br from-red-50/80 via-white/90 to-orange-50/80 dark:from-red-900/20 dark:via-slate-800/90 dark:to-orange-900/20 border border-red-200/50 dark:border-red-700/50 shadow-xl">
                              <div className="flex items-center gap-3 mb-4">
                                <div className="text-2xl">🤕</div>
                                <div className="flex-1">
                                  <Label className="text-slate-900 dark:text-white font-semibold text-lg">
                                    Abdominal Pain Level
                                  </Label>
                                  <p className="text-sm text-slate-500 dark:text-slate-400">
                                    Rate your abdominal pain intensity
                                  </p>
                                </div>
                                <motion.div
                                  className="px-4 py-2 rounded-full bg-gradient-to-r from-red-500 to-orange-500 text-white font-bold text-lg shadow-lg"
                                  animate={{ scale: [1, 1.05, 1] }}
                                  transition={{ duration: 2, repeat: Infinity }}
                                >
                                  {userData.symptoms.abdominalPain / 10}/10
                                </motion.div>
                              </div>
                              <Slider
                                value={[userData.symptoms.abdominalPain]}
                                min={0}
                                max={100}
                                step={10}
                                onValueChange={(value) =>
                                  handleSymptomChange("abdominalPain", value[0])
                                }
                                className="mt-4"
                              />
                              <div className="flex justify-between text-sm font-medium text-slate-600 dark:text-slate-300 px-1 mt-3">
                                <span className="flex items-center gap-1">
                                  <div className="w-2 h-2 bg-green-400 rounded-full"></div>
                                  None
                                </span>
                                <span className="flex items-center gap-1">
                                  <div className="w-2 h-2 bg-yellow-400 rounded-full"></div>
                                  Mild
                                </span>
                                <span className="flex items-center gap-1">
                                  <div className="w-2 h-2 bg-orange-400 rounded-full"></div>
                                  Moderate
                                </span>
                                <span className="flex items-center gap-1">
                                  <div className="w-2 h-2 bg-red-400 rounded-full"></div>
                                  Severe
                                </span>
                              </div>
                            </div>

                            <div className="p-6 rounded-2xl bg-gradient-to-br from-blue-50/80 via-white/90 to-indigo-50/80 dark:from-blue-900/20 dark:via-slate-800/90 dark:to-indigo-900/20 border border-blue-200/50 dark:border-blue-700/50 shadow-xl">
                              <div className="flex items-center gap-3 mb-4">
                                <div className="text-2xl">😴</div>
                                <div className="flex-1">
                                  <Label className="text-slate-900 dark:text-white font-semibold text-lg">
                                    Fatigue Level
                                  </Label>
                                  <p className="text-sm text-slate-500 dark:text-slate-400">
                                    Rate your fatigue and tiredness level
                                  </p>
                                </div>
                                <motion.div
                                  className="px-4 py-2 rounded-full bg-gradient-to-r from-blue-500 to-indigo-500 text-white font-bold text-lg shadow-lg"
                                  animate={{ scale: [1, 1.05, 1] }}
                                  transition={{ duration: 2, repeat: Infinity, delay: 1 }}
                                >
                                  {userData.symptoms.fatigue / 10}/10
                                </motion.div>
                              </div>
                              <Slider
                                value={[userData.symptoms.fatigue]}
                                min={0}
                                max={100}
                                step={10}
                                onValueChange={(value) =>
                                  handleSymptomChange("fatigue", value[0])
                                }
                                className="mt-4"
                              />
                              <div className="flex justify-between text-sm font-medium text-slate-600 dark:text-slate-300 px-1 mt-3">
                                <span className="flex items-center gap-1">
                                  <div className="w-2 h-2 bg-green-400 rounded-full"></div>
                                  None
                                </span>
                                <span className="flex items-center gap-1">
                                  <div className="w-2 h-2 bg-blue-400 rounded-full"></div>
                                  Mild
                                </span>
                                <span className="flex items-center gap-1">
                                  <div className="w-2 h-2 bg-indigo-400 rounded-full"></div>
                                  Moderate
                                </span>
                                <span className="flex items-center gap-1">
                                  <div className="w-2 h-2 bg-purple-400 rounded-full"></div>
                                  Severe
                                </span>
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
                                description: "Travel to regions with known hepatitis outbreaks in the past 6 months",
                                icon: "✈️",
                                color: "from-blue-500 to-cyan-500"
                              },
                              {
                                id: "bloodTransfusion",
                                label: "History of Blood Transfusion",
                                description: "Received blood products before comprehensive screening was implemented",
                                icon: "🩸",
                                color: "from-red-500 to-pink-500"
                              },
                              {
                                id: "unsafeInjection",
                                label: "History of Unsafe Injection Practices",
                                description: "Shared needles or received injections with potentially unsterilized equipment",
                                icon: "💉",
                                color: "from-orange-500 to-red-500"
                              },
                              {
                                id: "contactWithInfected",
                                label: "Contact with Infected Person",
                                description: "Close contact with someone diagnosed with hepatitis",
                                icon: "👥",
                                color: "from-purple-500 to-indigo-500"
                              },
                            ].map((factor) => (
                              <motion.div
                                key={factor.id}
                                className={`relative p-6 rounded-2xl border-2 transition-all duration-300 cursor-pointer group ${
                                  userData.riskFactors.includes(factor.id)
                                    ? `border-transparent bg-gradient-to-r ${factor.color} shadow-lg scale-105`
                                    : "border-slate-200 dark:border-slate-700 hover:border-slate-300 dark:hover:border-slate-600 hover:bg-slate-50/50 dark:hover:bg-slate-800/50"
                                }`}
                                whileHover={{ scale: 1.02 }}
                                whileTap={{ scale: 0.98 }}
                                onClick={() => handleRiskFactorToggle(factor.id)}
                              >
                                <Checkbox
                                  id={factor.id}
                                  checked={userData.riskFactors.includes(factor.id)}
                                  onCheckedChange={() => handleRiskFactorToggle(factor.id)}
                                  className="sr-only"
                                />
                                <div className="flex items-start space-x-4">
                                  <div className="text-3xl flex-shrink-0">{factor.icon}</div>
                                  <div className="flex-1">
                                    <Label
                                      htmlFor={factor.id}
                                      className={`font-semibold cursor-pointer text-lg ${
                                        userData.riskFactors.includes(factor.id)
                                          ? "text-white"
                                          : "text-slate-900 dark:text-white"
                                      }`}
                                    >
                                      {factor.label}
                                    </Label>
                                    <p className={`text-sm mt-2 leading-relaxed ${
                                      userData.riskFactors.includes(factor.id)
                                        ? "text-white/90"
                                        : "text-slate-500 dark:text-slate-400"
                                    }`}>
                                      {factor.description}
                                    </p>
                                  </div>
                                  {userData.riskFactors.includes(factor.id) && (
                                    <motion.div
                                      className="w-6 h-6 bg-white/20 rounded-full flex items-center justify-center flex-shrink-0"
                                      initial={{ scale: 0 }}
                                      animate={{ scale: 1 }}
                                      transition={{ type: "spring", stiffness: 300, damping: 20 }}
                                    >
                                      <CheckCircle2 className="w-4 h-4 text-white" />
                                    </motion.div>
                                  )}
                                </div>
                                {userData.riskFactors.includes(factor.id) && (
                                  <motion.div
                                    className="absolute inset-0 bg-gradient-to-r from-white/10 to-transparent rounded-2xl"
                                    initial={{ opacity: 0 }}
                                    animate={{ opacity: 1 }}
                                    transition={{ duration: 0.3 }}
                                  />
                                )}
                              </motion.div>
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
                    <div className="mb-8 p-8 rounded-3xl bg-gradient-to-br from-slate-50/90 via-white/95 to-blue-50/80 dark:from-slate-900/90 dark:via-slate-800/95 dark:to-blue-900/80 backdrop-blur-xl border border-blue-200/50 dark:border-blue-800/50 shadow-2xl relative overflow-hidden">
                      {/* Enhanced animated background with multiple layers */}
                      <div className="absolute inset-0 z-0 pointer-events-none">
                        <div className="absolute -top-16 -left-16 w-80 h-80 bg-gradient-to-br from-blue-400/20 via-indigo-300/15 to-purple-400/10 rounded-full blur-3xl animate-pulse" />
                        <div className="absolute top-1/2 right-0 w-64 h-64 bg-gradient-to-bl from-emerald-300/15 via-blue-400/10 to-transparent rounded-full blur-2xl animate-pulse" style={{ animationDelay: '2s' }} />
                        <div className="absolute bottom-0 left-1/3 w-48 h-48 bg-gradient-to-tr from-violet-400/20 via-blue-300/15 to-transparent rounded-full blur-2xl animate-pulse" style={{ animationDelay: '3s' }} />
                        {/* Floating particles */}
                        <div className="absolute top-10 left-10 w-2 h-2 bg-blue-400/40 rounded-full animate-bounce" style={{ animationDelay: '0.5s' }} />
                        <div className="absolute top-20 right-20 w-1.5 h-1.5 bg-indigo-400/40 rounded-full animate-bounce" style={{ animationDelay: '1.2s' }} />
                        <div className="absolute bottom-16 left-16 w-1 h-1 bg-purple-400/40 rounded-full animate-bounce" style={{ animationDelay: '2.1s' }} />
                      </div>
                      <div className="text-center relative z-10">
                        <motion.div
                          initial={{ scale: 0.8, opacity: 0, y: 20 }}
                          animate={{ scale: 1, opacity: 1, y: 0 }}
                          transition={{ type: "spring", stiffness: 300, damping: 20, delay: 0.2 }}
                        >
                          <div className="mb-6">
                            <motion.div
                              className="inline-flex items-center justify-center w-20 h-20 mb-4 rounded-full bg-gradient-to-r from-emerald-400 via-blue-500 to-purple-600 shadow-xl"
                              animate={{ 
                                rotate: [0, 360],
                                scale: [1, 1.1, 1]
                              }}
                              transition={{ 
                                rotate: { duration: 20, repeat: Infinity, ease: "linear" },
                                scale: { duration: 2, repeat: Infinity, ease: "easeInOut" }
                              }}
                            >
                              <CheckCircle2 className="h-10 w-10 text-white" />
                            </motion.div>
                            <h3 className="text-4xl font-extrabold mb-2 text-slate-900 dark:text-white tracking-tight">
                              <span className="inline-block bg-gradient-to-r from-blue-600 via-indigo-500 to-purple-600 bg-clip-text text-transparent">
                                Analysis Complete
                              </span>
                            </h3>
                            <p className="text-lg text-slate-600 dark:text-slate-300 font-medium">
                              Your hepatitis risk assessment results
                            </p>
                          </div>
                        </motion.div>

                        {predictionResult?.prediction.predictions.map((pred, index) => (
                          <motion.div
                            key={index}
                            className="mt-8"
                            initial={{ opacity: 0, y: 30 }}
                            animate={{ opacity: 1, y: 0 }}
                            transition={{ 
                              delay: 0.5 + index * 0.2,
                              type: "spring",
                              stiffness: 100,
                              damping: 15
                            }}
                          >
                            {/* Prediction Classification Badge */}
                            <motion.div
                              className="inline-flex items-center gap-3 px-8 py-4 rounded-2xl bg-gradient-to-r from-emerald-500 via-blue-500 to-purple-600 text-white text-xl font-bold shadow-xl mb-8 relative overflow-hidden"
                              whileHover={{ 
                                scale: 1.05,
                                boxShadow: "0 25px 50px -12px rgba(0, 0, 0, 0.25)"
                              }}
                              whileTap={{ scale: 0.98 }}
                            >
                              {/* Shimmer effect */}
                              <div className="absolute inset-0 bg-gradient-to-r from-transparent via-white/20 to-transparent transform -translate-x-full animate-shimmer" />
                              <motion.div
                                animate={{ rotate: [0, 360] }}
                                transition={{ duration: 3, repeat: Infinity, ease: "linear" }}
                              >
                                <CheckCircle2 className="h-7 w-7 text-white drop-shadow-lg" />
                              </motion.div>
                              <span className="relative z-10">{pred.predicted_class}</span>
                            </motion.div>

                            {/* Enhanced Results Cards */}
                            <motion.div
                              className="grid gap-6 md:grid-cols-2"
                              initial={{ opacity: 0, y: 20 }}
                              animate={{ opacity: 1, y: 0 }}
                              transition={{ delay: 0.7 + index * 0.2 }}
                            >
                              {/* Hepatitis A Card */}
                              <motion.div
                                className="group p-6 rounded-2xl bg-gradient-to-br from-blue-50/80 via-white/90 to-cyan-50/80 dark:from-blue-900/30 dark:via-slate-800/90 dark:to-cyan-900/30 border border-blue-200/50 dark:border-blue-700/50 shadow-xl hover:shadow-2xl transition-all duration-300 relative overflow-hidden"
                                whileHover={{ y: -5, scale: 1.02 }}
                                whileTap={{ scale: 0.98 }}
                              >
                                {/* Card background effect */}
                                <div className="absolute inset-0 bg-gradient-to-br from-blue-400/5 via-transparent to-cyan-400/5 opacity-0 group-hover:opacity-100 transition-opacity duration-300" />
                                
                                <div className="relative z-10">
                                  <div className="flex items-center justify-between mb-4">
                                    <h4 className="text-lg font-bold text-slate-800 dark:text-slate-200 flex items-center gap-2">
                                      <div className="w-3 h-3 rounded-full bg-gradient-to-r from-blue-400 to-cyan-500"></div>
                                      Hepatitis A
                                    </h4>
                                    <Badge variant="secondary" className="bg-blue-100 text-blue-700 dark:bg-blue-900 dark:text-blue-300">
                                      Type A
                                    </Badge>
                                  </div>
                                  
                                  <div className="space-y-3">
                                    <div className="flex justify-between items-center">
                                      <span className="text-sm font-medium text-slate-600 dark:text-slate-400">Probability</span>
                                      <span className="text-2xl font-bold text-blue-600 dark:text-blue-400">
                                        {(pred["probability_Hepatitis A"] * 100).toFixed(1)}%
                                      </span>
                                    </div>
                                    
                                    <div className="relative">
                                      <div className="w-full h-4 bg-slate-200 dark:bg-slate-700 rounded-full overflow-hidden shadow-inner">
                                        <motion.div
                                          className="h-full bg-gradient-to-r from-blue-500 via-cyan-400 to-blue-600 shadow-lg relative overflow-hidden"
                                          style={{ width: `${pred["probability_Hepatitis A"] * 100}%` }}
                                          initial={{ width: 0 }}
                                          animate={{ width: `${pred["probability_Hepatitis A"] * 100}%` }}
                                          transition={{ duration: 1.5, delay: 0.8, ease: "easeInOut" }}
                                        >
                                          {/* Animated shine effect */}
                                          <motion.div
                                            className="absolute inset-0 bg-gradient-to-r from-transparent via-white/30 to-transparent"
                                            animate={{ x: ["-100%", "100%"] }}
                                            transition={{ duration: 2, repeat: Infinity, delay: 1.5 }}
                                          />
                                        </motion.div>
                                      </div>
                                    </div>
                                  </div>
                                </div>
                              </motion.div>

                              {/* Hepatitis C Card */}
                              <motion.div
                                className="group p-6 rounded-2xl bg-gradient-to-br from-purple-50/80 via-white/90 to-indigo-50/80 dark:from-purple-900/30 dark:via-slate-800/90 dark:to-indigo-900/30 border border-purple-200/50 dark:border-purple-700/50 shadow-xl hover:shadow-2xl transition-all duration-300 relative overflow-hidden"
                                whileHover={{ y: -5, scale: 1.02 }}
                                whileTap={{ scale: 0.98 }}
                              >
                                {/* Card background effect */}
                                <div className="absolute inset-0 bg-gradient-to-br from-purple-400/5 via-transparent to-indigo-400/5 opacity-0 group-hover:opacity-100 transition-opacity duration-300" />
                                
                                <div className="relative z-10">
                                  <div className="flex items-center justify-between mb-4">
                                    <h4 className="text-lg font-bold text-slate-800 dark:text-slate-200 flex items-center gap-2">
                                      <div className="w-3 h-3 rounded-full bg-gradient-to-r from-purple-400 to-indigo-500"></div>
                                      Hepatitis C
                                    </h4>
                                    <Badge variant="secondary" className="bg-purple-100 text-purple-700 dark:bg-purple-900 dark:text-purple-300">
                                      Type C
                                    </Badge>
                                  </div>
                                  
                                  <div className="space-y-3">
                                    <div className="flex justify-between items-center">
                                      <span className="text-sm font-medium text-slate-600 dark:text-slate-400">Probability</span>
                                      <span className="text-2xl font-bold text-purple-600 dark:text-purple-400">
                                        {(pred["probability_Hepatitis C"] * 100).toFixed(1)}%
                                      </span>
                                    </div>
                                    
                                    <div className="relative">
                                      <div className="w-full h-4 bg-slate-200 dark:bg-slate-700 rounded-full overflow-hidden shadow-inner">
                                        <motion.div
                                          className="h-full bg-gradient-to-r from-purple-500 via-indigo-400 to-purple-600 shadow-lg relative overflow-hidden"
                                          style={{ width: `${pred["probability_Hepatitis C"] * 100}%` }}
                                          initial={{ width: 0 }}
                                          animate={{ width: `${pred["probability_Hepatitis C"] * 100}%` }}
                                          transition={{ duration: 1.5, delay: 1.0, ease: "easeInOut" }}
                                        >
                                          {/* Animated shine effect */}
                                          <motion.div
                                            className="absolute inset-0 bg-gradient-to-r from-transparent via-white/30 to-transparent"
                                            animate={{ x: ["-100%", "100%"] }}
                                            transition={{ duration: 2, repeat: Infinity, delay: 2.0 }}
                                          />
                                        </motion.div>
                                      </div>
                                    </div>
                                  </div>
                                </div>
                              </motion.div>
                            </motion.div>
                          </motion.div>
                        ))}

                        {/* Important Notice */}
                        <motion.div
                          className="text-center mt-12"
                          initial={{ opacity: 0, y: 20 }}
                          animate={{ opacity: 1, y: 0 }}
                          transition={{ delay: 1.2 }}
                        >
                          <Alert className="mb-8 border-amber-200 bg-gradient-to-r from-amber-50/80 via-yellow-50/90 to-orange-50/80 dark:from-amber-900/20 dark:via-yellow-900/30 dark:to-orange-900/20 shadow-lg">
                            <AlertCircle className="h-5 w-5 text-amber-600 dark:text-amber-400" />
                            <AlertDescription className="text-left">
                              <strong className="text-amber-800 dark:text-amber-200">Medical Disclaimer:</strong> This AI-powered analysis is for informational purposes only and should not replace professional medical advice. The results are based on the symptoms and risk factors you provided and are meant to guide your next steps.
                            </AlertDescription>
                          </Alert>
                        </motion.div>

                        {/* Enhanced Recommendations Section */}
                        <motion.div
                          className="mt-8"
                          initial={{ opacity: 0, y: 30 }}
                          animate={{ opacity: 1, y: 0 }}
                          transition={{ delay: 1.4 }}
                        >
                          <motion.div
                            className="p-8 rounded-3xl bg-gradient-to-br from-emerald-50/80 via-white/95 to-blue-50/80 dark:from-emerald-900/20 dark:via-slate-800/90 dark:to-blue-900/20 border border-emerald-200/50 dark:border-emerald-700/50 shadow-2xl backdrop-blur-md relative overflow-hidden"
                            whileHover={{ scale: 1.01 }}
                            transition={{ type: "spring", stiffness: 300, damping: 30 }}
                          >
                            {/* Background decorations */}
                            <div className="absolute top-0 right-0 w-32 h-32 bg-gradient-to-bl from-emerald-300/10 via-blue-300/5 to-transparent rounded-full blur-2xl" />
                            <div className="absolute bottom-0 left-0 w-24 h-24 bg-gradient-to-tr from-blue-300/10 via-emerald-300/5 to-transparent rounded-full blur-xl" />
                            
                            <div className="relative z-10">
                              <motion.h4 
                                className="font-bold mb-6 text-slate-900 dark:text-white flex items-center gap-3 text-2xl"
                                initial={{ opacity: 0, x: -20 }}
                                animate={{ opacity: 1, x: 0 }}
                                transition={{ delay: 1.6 }}
                              >
                                <motion.div
                                  className="p-2 bg-gradient-to-r from-emerald-500 to-blue-600 rounded-xl shadow-lg"
                                  animate={{ rotate: [0, 5, -5, 0] }}
                                  transition={{ duration: 2, repeat: Infinity, ease: "easeInOut" }}
                                >
                                  <CheckCircle2 className="h-6 w-6 text-white" />
                                </motion.div>
                                Recommended Next Steps
                              </motion.h4>
                              
                              <div className="space-y-4">
                                {[
                                  {
                                    icon: "🏥",
                                    title: "Consult Healthcare Provider",
                                    description: "Schedule an appointment with your doctor or hepatologist for professional evaluation",
                                    priority: "high"
                                  },
                                  {
                                    icon: "📋",
                                    title: "Share Assessment Results",
                                    description: "Bring these results and your symptoms history to your medical consultation",
                                    priority: "high"
                                  },
                                  {
                                    icon: "🧪",
                                    title: "Medical Testing",
                                    description: "Get proper blood tests and liver function tests for accurate diagnosis",
                                    priority: "medium"
                                  },
                                  {
                                    icon: "📱",
                                    title: "Monitor Symptoms",
                                    description: "Keep track of any changes in symptoms and overall health condition",
                                    priority: "medium"
                                  }
                                ].map((step, i) => (
                                  <motion.div
                                    key={i}
                                    className={`group flex items-start gap-4 p-4 rounded-2xl transition-all duration-300 cursor-pointer relative overflow-hidden ${
                                      step.priority === 'high' 
                                        ? 'bg-gradient-to-r from-red-50/80 to-orange-50/80 dark:from-red-900/20 dark:to-orange-900/20 border border-red-200/30 dark:border-red-700/30 hover:shadow-lg' 
                                        : 'bg-gradient-to-r from-blue-50/80 to-emerald-50/80 dark:from-blue-900/20 dark:to-emerald-900/20 border border-blue-200/30 dark:border-blue-700/30 hover:shadow-md'
                                    }`}
                                    initial={{ x: -20, opacity: 0 }}
                                    animate={{ x: 0, opacity: 1 }}
                                    transition={{ 
                                      delay: 1.8 + i * 0.15,
                                      type: "spring",
                                      stiffness: 100,
                                      damping: 20
                                    }}
                                    whileHover={{ x: 5, scale: 1.02 }}
                                    whileTap={{ scale: 0.98 }}
                                  >
                                    {/* Hover effect background */}
                                    <div className="absolute inset-0 bg-gradient-to-r from-white/20 via-transparent to-white/10 opacity-0 group-hover:opacity-100 transition-opacity duration-300" />
                                    
                                    <motion.div 
                                      className="flex-shrink-0 text-2xl p-2 rounded-xl bg-white/80 dark:bg-slate-700/80 shadow-md"
                                      whileHover={{ rotate: 15, scale: 1.1 }}
                                      transition={{ type: "spring", stiffness: 300, damping: 20 }}
                                    >
                                      {step.icon}
                                    </motion.div>
                                    
                                    <div className="flex-1 relative z-10">
                                      <div className="flex items-center gap-2 mb-1">
                                        <h5 className="font-semibold text-slate-900 dark:text-white text-lg">
                                          {step.title}
                                        </h5>
                                        {step.priority === 'high' && (
                                          <Badge className="bg-red-100 text-red-700 dark:bg-red-900 dark:text-red-300 text-xs">
                                            Priority
                                          </Badge>
                                        )}
                                      </div>
                                      <p className="text-slate-600 dark:text-slate-300 text-sm leading-relaxed">
                                        {step.description}
                                      </p>
                                    </div>
                                    
                                    <motion.div
                                      className="flex-shrink-0 opacity-0 group-hover:opacity-100 transition-opacity duration-300"
                                      initial={{ x: 10 }}
                                      whileHover={{ x: 0 }}
                                    >
                                      <ChevronRight className="h-5 w-5 text-slate-400 dark:text-slate-500" />
                                    </motion.div>
                                  </motion.div>
                                ))}
                              </div>
                            </div>
                          </motion.div>
                        </motion.div>
                      </div>
                    </div>
                  </motion.div>
                )}
              </AnimatePresence>
            </CardContent>
            <CardFooter className="flex justify-between bg-slate-50 dark:bg-slate-800 p-6 border-t border-slate-200 dark:border-slate-700">
              {!showResults ? (
                <>
                  {activeTab !== "basic" && (
                    <Button
                      variant="outline"
                      onClick={prevTab}
                      className="px-4 py-2"
                    >
                      <ChevronLeft className="h-4 w-4 mr-2" /> Previous
                    </Button>
                  )}
                  {activeTab === "risk" ? (
                    <Button
                      onClick={handleSubmit}
                      disabled={loading}
                      className="px-6 py-2"
                    >
                      {loading ? (
                        <>
                          <Loader2 className="mr-2 h-4 w-4 animate-spin" />
                          Processing...
                        </>
                      ) : (
                        <>
                          Submit Assessment
                        </>
                      )}
                    </Button>
                  ) : (
                    <Button
                      onClick={nextTab}
                      className="px-6 py-2"
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
                    className="px-4 py-2"
                  >
                    Start Over
                  </Button>
                  <Button
                    onClick={downloadResults}
                    className="px-4 py-2"
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
