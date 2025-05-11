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

type UserData = {
  age: string;
  gender: string;
  hepatitisType: string;
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

export default function PredictionSelector() {
  const { toast } = useToast();
  const [activeTab, setActiveTab] = useState("basic");
  const [loading, setLoading] = useState(false);
  const [showResults, setShowResults] = useState(false);
  const [userData, setUserData] = useState<UserData>({
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
    // Validation
    if (!userData.age || !userData.gender || !userData.hepatitisType) {
      toast({
        title: "Missing Information",
        description: "Please complete all required fields before submitting.",
        variant: "destructive",
      });
      return;
    }

    try {
      setLoading(true);

      // Create form data to send to the API
      const formData = new FormData();
      formData.append("age", userData.age);
      formData.append("gender", userData.gender);
      formData.append("hepatitisType", userData.hepatitisType);
      formData.append("symptoms", JSON.stringify(userData.symptoms));
      formData.append("riskFactors", JSON.stringify(userData.riskFactors));

      // Call the API
      const response = await fetch("/api/predict", {
        method: "POST",
        body: formData,
      });

      if (!response.ok) {
        const errorData = await response.json();
        throw new Error(errorData.error || "Failed to process selection");
      }

      // Show results
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
          "There was an error processing your information. Please try again.",
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
    } else if (activeTab === "risk") {
      setActiveTab("selection");
    }
  };

  const prevTab = () => {
    if (activeTab === "symptoms") {
      setActiveTab("basic");
    } else if (activeTab === "risk") {
      setActiveTab("symptoms");
    } else if (activeTab === "selection") {
      setActiveTab("risk");
    }
  };

  return (
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
            <TabsList className="grid grid-cols-4 mb-8">
              <TabsTrigger value="basic">Basic Info</TabsTrigger>
              <TabsTrigger value="symptoms">Symptoms</TabsTrigger>
              <TabsTrigger value="risk">Risk Factors</TabsTrigger>
              <TabsTrigger value="selection">Selection</TabsTrigger>
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
                <h3 className="text-xl font-bold mb-2">
                  {userData.hepatitisType === "hepatitisB" &&
                    "Hepatitis B Information"}
                  {userData.hepatitisType === "hepatitisC" &&
                    "Hepatitis C Information"}
                  {userData.hepatitisType === "unlikely" &&
                    "Hepatitis Information"}
                </h3>

                <Badge
                  className={`mb-4 ${
                    userData.hepatitisType === "unlikely"
                      ? "bg-green-500"
                      : "bg-amber-500"
                  }`}
                >
                  {userData.hepatitisType === "hepatitisB" &&
                    "Hepatitis B Selected"}
                  {userData.hepatitisType === "hepatitisC" &&
                    "Hepatitis C Selected"}
                  {userData.hepatitisType === "unlikely" &&
                    "No Hepatitis Selected"}
                </Badge>

                <div className="text-left mt-6">
                  {userData.hepatitisType === "hepatitisB" && (
                    <div className="space-y-4">
                      <p>
                        <strong>About Hepatitis B:</strong> Hepatitis B is a
                        viral infection that attacks the liver and can cause
                        both acute and chronic disease. It is transmitted
                        through contact with the blood or other body fluids of
                        an infected person.
                      </p>
                      <p>
                        <strong>Key Facts:</strong>
                      </p>
                      <ul className="list-disc pl-5 space-y-1">
                        <li>
                          Hepatitis B is spread through blood, semen, and other
                          body fluids
                        </li>
                        <li>
                          It can cause both acute (short-term) and chronic
                          (long-term) infection
                        </li>
                        <li>A vaccine is available to prevent Hepatitis B</li>
                        <li>
                          Symptoms may include fatigue, jaundice, abdominal
                          pain, and dark urine
                        </li>
                        <li>
                          Some people with Hepatitis B don't experience any
                          symptoms
                        </li>
                      </ul>
                      <p>
                        <strong>Treatment:</strong> Acute Hepatitis B usually
                        doesn't require specific treatment. For chronic
                        Hepatitis B, medications like entecavir and tenofovir
                        can help fight the virus and slow liver damage.
                      </p>
                    </div>
                  )}

                  {userData.hepatitisType === "hepatitisC" && (
                    <div className="space-y-4">
                      <p>
                        <strong>About Hepatitis C:</strong> Hepatitis C is a
                        viral infection caused by the Hepatitis C virus (HCV)
                        that primarily affects the liver. It is often referred
                        to as a "silent epidemic" because many people don't know
                        they're infected.
                      </p>
                      <p>
                        <strong>Key Facts:</strong>
                      </p>
                      <ul className="list-disc pl-5 space-y-1">
                        <li>
                          Hepatitis C is primarily spread through contact with
                          infected blood
                        </li>
                        <li>
                          It often becomes chronic and can lead to serious liver
                          problems if untreated
                        </li>
                        <li>
                          Many people with Hepatitis C don't have symptoms until
                          liver damage occurs
                        </li>
                        <li>
                          Unlike Hepatitis B, there is no vaccine for Hepatitis
                          C
                        </li>
                        <li>
                          Effective treatments are available that can cure
                          Hepatitis C in most cases
                        </li>
                      </ul>
                      <p>
                        <strong>Treatment:</strong> Modern treatments like
                        direct-acting antivirals (DAAs) can cure Hepatitis C in
                        8-12 weeks with minimal side effects.
                      </p>
                    </div>
                  )}

                  {userData.hepatitisType === "unlikely" && (
                    <div className="space-y-4">
                      <p>
                        <strong>General Hepatitis Information:</strong>{" "}
                        Hepatitis is an inflammation of the liver. It can be
                        caused by various factors, including viral infections,
                        alcohol consumption, certain medications, and toxins.
                      </p>
                      <p>
                        <strong>Key Facts About Hepatitis:</strong>
                      </p>
                      <ul className="list-disc pl-5 space-y-1">
                        <li>
                          There are five main types of viral hepatitis: A, B, C,
                          D, and E
                        </li>
                        <li>
                          Symptoms may include fatigue, jaundice, abdominal
                          pain, and dark urine
                        </li>
                        <li>
                          Some types of hepatitis can be prevented with vaccines
                        </li>
                        <li>
                          Maintaining good hygiene and avoiding risk factors can
                          help prevent hepatitis
                        </li>
                      </ul>
                      <p>
                        <strong>Prevention:</strong> Practice good hygiene, get
                        vaccinated for Hepatitis A and B if available, avoid
                        sharing personal items that may have blood on them, and
                        practice safe sex.
                      </p>
                    </div>
                  )}
                </div>
              </div>
            </div>

            <div className="bg-primary/10 p-4 rounded-lg">
              <h4 className="font-semibold mb-2">Important Note:</h4>
              <p>
                This information is based on your selection and is not a medical
                diagnosis. For accurate diagnosis, please consult with a
                healthcare professional.
              </p>
            </div>

            <div className="mt-6">
              <h4 className="font-semibold mb-3">Next Steps:</h4>
              <ul className="space-y-2">
                <li className="flex items-start">
                  <span className="mr-2 text-primary">✓</span>
                  <span>
                    Consult with a healthcare provider for proper testing and
                    diagnosis
                  </span>
                </li>
                <li className="flex items-start">
                  <span className="mr-2 text-primary">✓</span>
                  <span>Mention your symptoms and concerns to your doctor</span>
                </li>
                <li className="flex items-start">
                  <span className="mr-2 text-primary">✓</span>
                  <span>
                    Learn more about hepatitis in our education section
                  </span>
                </li>
              </ul>
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
            {activeTab !== "selection" ? (
              <Button onClick={nextTab}>Next</Button>
            ) : (
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

function formatHepatitisType(type: string): string {
  switch (type) {
    case "hepatitisB":
      return "Hepatitis B";
    case "hepatitisC":
      return "Hepatitis C";
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
