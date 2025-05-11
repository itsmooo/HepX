"use client";

import { useState } from "react";
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
import { Download, Loader2 } from "lucide-react";
import { useToast } from "../hooks/use-toast";
import { Badge } from "@/components/ui/badge";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import Header from "../common/header";
import Footer from "../common/footer";

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
  const [showResults, setShowResults] = useState(false);
  const [predictionResult, setPredictionResult] = useState<PredictionResult | null>(null);
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

      // Call the API with JSON data
      const response = await fetch("http://localhost:5000/api/predict", {
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
            joint_pain: userData.symptoms.jointPain
          },
          riskFactors: userData.riskFactors
        }),
      });

      if (!response.ok) {
        const errorData = await response.json();
        throw new Error(errorData.error || "Failed to process prediction");
      }

      const data = await response.json();
      console.log('Prediction data received:', data);
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
        description: error.message || "There was an error processing your information.",
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

  return (
    <div className="min-h-screen flex flex-col">
      <Header />
      <main className="flex-grow container mx-auto px-4 py-8">
        <Card className="w-full max-w-4xl mx-auto">
          <CardHeader>
            <CardTitle>Hepatitis Type Selector</CardTitle>
            <CardDescription>
              Answer a few questions to learn about different hepatitis types
            </CardDescription>
          </CardHeader>
          <CardContent>
            {!showResults ? (
              <Tabs
                value={activeTab}
                onValueChange={setActiveTab}
                className="w-full"
              >
                <TabsList className="grid grid-cols-3 mb-8">
                  <TabsTrigger value="basic">Basic Info</TabsTrigger>
                  <TabsTrigger value="symptoms">Symptoms</TabsTrigger>
                  <TabsTrigger value="risk">Risk Factors</TabsTrigger>
                </TabsList>

                <TabsContent value="basic">
                  <div className="space-y-6">
                    <div>
                      <Label htmlFor="age-group">Age Group</Label>
                      <Select
                        value={userData.age}
                        onValueChange={(value) =>
                          setUserData((prev) => ({ ...prev, age: value }))
                        }
                      >
                        <SelectTrigger id="age-group" className="w-full mt-2">
                          <SelectValue placeholder="Select your age group" />
                        </SelectTrigger>
                        <SelectContent>
                          <SelectItem value="under18">Under 18</SelectItem>
                          <SelectItem value="18-30">18-30</SelectItem>
                          <SelectItem value="31-45">31-45</SelectItem>
                          <SelectItem value="46-60">46-60</SelectItem>
                          <SelectItem value="over60">Over 60</SelectItem>
                        </SelectContent>
                      </Select>
                    </div>

                    <div>
                      <Label htmlFor="gender">Gender</Label>
                      <RadioGroup
                        value={userData.gender}
                        onValueChange={(value: any) =>
                          setUserData((prev) => ({ ...prev, gender: value }))
                        }
                        className="flex space-x-4 mt-2"
                      >
                        <div className="flex items-center space-x-2">
                          <RadioGroupItem value="male" id="male" />
                          <Label htmlFor="male">Male</Label>
                        </div>
                        <div className="flex items-center space-x-2">
                          <RadioGroupItem value="female" id="female" />
                          <Label htmlFor="female">Female</Label>
                        </div>
                        <div className="flex items-center space-x-2">
                          <RadioGroupItem value="other" id="other" />
                          <Label htmlFor="other">Other</Label>
                        </div>
                      </RadioGroup>
                    </div>
                  </div>
                </TabsContent>

                <TabsContent value="symptoms">
                  <div className="space-y-6">
                    <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                      <div className="space-y-4">
                        <div className="flex items-center space-x-2">
                          <Checkbox
                            id="jaundice"
                            checked={userData.symptoms.jaundice}
                            onCheckedChange={(checked: boolean) =>
                              handleSymptomChange("jaundice", checked === true)
                            }
                          />
                          <Label htmlFor="jaundice">
                            Yellowing of skin/eyes (Jaundice)
                          </Label>
                        </div>

                        <div className="flex items-center space-x-2">
                          <Checkbox
                            id="darkUrine"
                            checked={userData.symptoms.darkUrine}
                            onCheckedChange={(checked: boolean) =>
                              handleSymptomChange("darkUrine", checked === true)
                            }
                          />
                          <Label htmlFor="darkUrine">Dark Urine</Label>
                        </div>

                        <div className="flex items-center space-x-2">
                          <Checkbox
                            id="fever"
                            checked={userData.symptoms.fever}
                            onCheckedChange={(checked: boolean) =>
                              handleSymptomChange("fever", checked === true)
                            }
                          />
                          <Label htmlFor="fever">Fever</Label>
                        </div>

                        <div className="flex items-center space-x-2">
                          <Checkbox
                            id="jointPain"
                            checked={userData.symptoms.jointPain}
                            onCheckedChange={(checked: boolean) =>
                              handleSymptomChange("jointPain", checked === true)
                            }
                          />
                          <Label htmlFor="jointPain">Joint Pain</Label>
                        </div>
                      </div>

                      <div className="space-y-4">
                        <div className="flex items-center space-x-2">
                          <Checkbox
                            id="nausea"
                            checked={userData.symptoms.nausea}
                            onCheckedChange={(checked: boolean) =>
                              handleSymptomChange("nausea", checked === true)
                            }
                          />
                          <Label htmlFor="nausea">Nausea</Label>
                        </div>

                        <div className="flex items-center space-x-2">
                          <Checkbox
                            id="appetite"
                            checked={userData.symptoms.appetite}
                            onCheckedChange={(checked: boolean) =>
                              handleSymptomChange("appetite", checked === true)
                            }
                          />
                          <Label htmlFor="appetite">Loss of Appetite</Label>
                        </div>
                      </div>
                    </div>

                    <div className="space-y-6 mt-6">
                      <div>
                        <div className="flex justify-between">
                          <Label>Abdominal Pain Level</Label>
                          <span className="text-sm text-muted-foreground">
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
                      </div>

                      <div>
                        <div className="flex justify-between">
                          <Label>Fatigue Level</Label>
                          <span className="text-sm text-muted-foreground">
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
                      </div>
                    </div>
                  </div>
                </TabsContent>

                <TabsContent value="risk">
                  <div className="space-y-4">
                    <p className="text-muted-foreground mb-4">
                      Select any risk factors that apply to you:
                    </p>

                    <div className="space-y-3">
                      <div className="flex items-center space-x-2">
                        <Checkbox
                          id="recentTravel"
                          checked={userData.riskFactors.includes("recentTravel")}
                          onCheckedChange={() =>
                            handleRiskFactorToggle("recentTravel")
                          }
                        />
                        <Label htmlFor="recentTravel">
                          Recent Travel to High-Risk Areas
                        </Label>
                      </div>

                      <div className="flex items-center space-x-2">
                        <Checkbox
                          id="bloodTransfusion"
                          checked={userData.riskFactors.includes(
                            "bloodTransfusion"
                          )}
                          onCheckedChange={() =>
                            handleRiskFactorToggle("bloodTransfusion")
                          }
                        />
                        <Label htmlFor="bloodTransfusion">
                          History of Blood Transfusion
                        </Label>
                      </div>

                      <div className="flex items-center space-x-2">
                        <Checkbox
                          id="unsafeInjection"
                          checked={userData.riskFactors.includes("unsafeInjection")}
                          onCheckedChange={() =>
                            handleRiskFactorToggle("unsafeInjection")
                          }
                        />
                        <Label htmlFor="unsafeInjection">
                          History of Unsafe Injection Practices
                        </Label>
                      </div>

                      <div className="flex items-center space-x-2">
                        <Checkbox
                          id="contactWithInfected"
                          checked={userData.riskFactors.includes(
                            "contactWithInfected"
                          )}
                          onCheckedChange={() =>
                            handleRiskFactorToggle("contactWithInfected")
                          }
                        />
                        <Label htmlFor="contactWithInfected">
                          Contact with Infected Person
                        </Label>
                      </div>
                    </div>
                  </div>
                </TabsContent>

                <TabsContent value="selection">
                  <div className="space-y-6">
                    <div>
                      <Label
                        htmlFor="hepatitis-type"
                        className="text-lg font-medium"
                      >
                        Select Hepatitis Type
                      </Label>
                      <p className="text-muted-foreground mb-4">
                        Based on your knowledge or previous diagnosis, which type of
                        hepatitis do you think you might have?
                      </p>

                      <RadioGroup
                        value={userData.hepatitisType}
                        onValueChange={(value: any) =>
                          setUserData((prev) => ({ ...prev, hepatitisType: value }))
                        }
                        className="space-y-4 mt-2"
                      >
                        <div className="flex items-start space-x-3">
                          <RadioGroupItem
                            value="hepatitisB"
                            id="hepatitisB"
                            className="mt-1"
                          />
                          <div>
                            <Label htmlFor="hepatitisB" className="font-medium">
                              Hepatitis B
                            </Label>
                            <p className="text-sm text-muted-foreground">
                              Transmitted through blood, semen, and other body
                              fluids. Can cause both acute and chronic infection.
                            </p>
                          </div>
                        </div>

                        <div className="flex items-start space-x-3">
                          <RadioGroupItem
                            value="hepatitisC"
                            id="hepatitisC"
                            className="mt-1"
                          />
                          <div>
                            <Label htmlFor="hepatitisC" className="font-medium">
                              Hepatitis C
                            </Label>
                            <p className="text-sm text-muted-foreground">
                              Primarily spread through contact with infected blood.
                              Often becomes chronic.
                            </p>
                          </div>
                        </div>

                        <div className="flex items-start space-x-3">
                          <RadioGroupItem
                            value="unlikely"
                            id="unlikely"
                            className="mt-1"
                          />
                          <div>
                            <Label htmlFor="unlikely" className="font-medium">
                              No Hepatitis / Not Sure
                            </Label>
                            <p className="text-sm text-muted-foreground">
                              I don't think I have hepatitis or I'm not sure.
                            </p>
                          </div>
                        </div>
                      </RadioGroup>
                    </div>

                    <div className="bg-muted p-4 rounded-md mt-6">
                      <p className="text-sm">
                        <strong>Note:</strong> This selection is not a diagnosis.
                        For accurate diagnosis, please consult with a healthcare
                        professional.
                      </p>
                    </div>
                  </div>
                </TabsContent>
              </Tabs>
            ) : (
              <div id="prediction-results" className="animate-fade-in">
                <div className="mb-6 p-4 rounded-lg bg-muted">
                  <div className="text-center">
                    <h3 className="text-xl font-bold mb-2">Prediction Results</h3>
                    
                    {predictionResult?.prediction.predictions.map((pred, index) => {
                      console.log('Rendering prediction:', pred);
                      return (
                        <div key={index} className="mt-4">
                          <Badge className="mb-4 bg-primary">
                            {pred.predicted_class}
                          </Badge>
                          
                          <div className="mt-4 space-y-2">
                            <div className="flex justify-between items-center">
                              <span>Probability of Hepatitis A:</span>
                              <span className="font-semibold">
                                {(pred["probability_Hepatitis A"] * 100).toFixed(1)}%
                              </span>
                            </div>
                            <div className="flex justify-between items-center">
                              <span>Probability of Hepatitis C:</span>
                              <span className="font-semibold">
                                {(pred["probability_Hepatitis C"] * 100).toFixed(1)}%
                              </span>
                            </div>
                          </div>
                        </div>
                      );
                    })}

                    <div className="text-left mt-6">
                      <p className="mb-4">
                        <strong>Important Note:</strong> This prediction is based on the symptoms and risk factors you provided. 
                        It is not a medical diagnosis. Please consult with a healthcare professional for proper medical advice.
                      </p>
                      
                      <div className="bg-primary/10 p-4 rounded-lg">
                        <h4 className="font-semibold mb-2">Next Steps:</h4>
                        <ul className="space-y-2">
                          <li className="flex items-start">
                            <span className="mr-2 text-primary">✓</span>
                            <span>Schedule an appointment with your healthcare provider</span>
                          </li>
                          <li className="flex items-start">
                            <span className="mr-2 text-primary">✓</span>
                            <span>Share these results with your doctor</span>
                          </li>
                          <li className="flex items-start">
                            <span className="mr-2 text-primary">✓</span>
                            <span>Get proper medical testing and diagnosis</span>
                          </li>
                        </ul>
                      </div>
                    </div>
                  </div>
                </div>
              </div>
            )}
          </CardContent>
          <CardFooter className="flex justify-between">
            {!showResults ? (
              <>
                {activeTab !== "basic" && (
                  <Button variant="outline" onClick={prevTab}>
                    Previous
                  </Button>
                )}
                {activeTab === "risk" ? (
                  <Button onClick={handleSubmit} disabled={loading}>
                    {loading ? (
                      <>
                        <Loader2 className="mr-2 h-4 w-4 animate-spin" />{" "}
                        Processing...
                      </>
                    ) : (
                      "Submit"
                    )}
                  </Button>
                ) : (
                  <Button onClick={nextTab}>Next</Button>
                )}
                {activeTab === "basic" && <div></div>}
              </>
            ) : (
              <>
                <Button variant="outline" onClick={resetForm}>
                  Start Over
                </Button>
                <Button onClick={downloadResults}>
                  <Download className="h-4 w-4 mr-2" /> Download Results
                </Button>
              </>
            )}
          </CardFooter>
        </Card>
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
    case "A":
      return "Hepatitis A";
    case "B":
      return "Hepatitis B";
    case "C":
      return "Hepatitis C";
    case "D":
      return "Hepatitis D";
    case "E":
      return "Hepatitis E";
    default:
      return "Unknown Type";
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
