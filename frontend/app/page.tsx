"use client";
import { useState } from "react";
import Header from "./common/header";
import Hero from "./common/hero";
import SymptomsForm from "./common/symptoms-form";
import Education from "./common/education";
import Footer from "./common/footer";
import ModelStatus from "./common/model-status";
import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from "@/components/ui/card";
import { Progress } from "@/components/ui/progress";
import { Alert, AlertDescription, AlertTitle } from "@/components/ui/alert";
import { CheckCircle2, AlertCircle } from "lucide-react";
import { useToast } from "./hooks/use-toast";
import OurMission from "./common/our-mission";

export default function Home() {
  const { toast } = useToast();
  const [isTraining, setIsTraining] = useState(false);
  const [trainProgress, setTrainProgress] = useState(0);
  const [error, setError] = useState<string | null>(null);

  const API_URL =
    process.env.NEXT_PUBLIC_API_URL || "http://localhost:5000/api";

  const trainModel = async () => {
    try {
      setIsTraining(true);
      setTrainProgress(0);
      setError(null);

      // Simulate progress while training
      const progressInterval = setInterval(() => {
        setTrainProgress((prev) => {
          const newProgress = prev + Math.random() * 10;
          return newProgress >= 95 ? 95 : newProgress;
        });
      }, 1000);

      const response = await fetch(`${API_URL}/train-model`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
      });

      clearInterval(progressInterval);

      if (!response.ok) {
        const errorData = await response.json();
        throw new Error(errorData.error || "Failed to train model");
      }

      const data = await response.json();
      setTrainProgress(100);

      toast({
        title: "Model Trained Successfully",
        description: `Model accuracy: ${(data.results.accuracy * 100).toFixed(
          2
        )}%`,
      });
    } catch (err: any) {
      setError(err.message || "An unexpected error occurred");
      setTrainProgress(0);

      toast({
        title: "Training Failed",
        description: err.message || "An unexpected error occurred",
        variant: "destructive",
      });
    } finally {
      setIsTraining(false);
    }
  };

  return (
    <div className="min-h-screen flex flex-col">
      <main className="flex-grow">
        <Hero />
        <SymptomsForm />
        <Education />
        <OurMission />
      </main>
    </div>
  );
}
