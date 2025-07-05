"use client";

import { useEffect, useState } from "react";
import { Alert, AlertDescription, AlertTitle } from "@/components/ui/alert";
import { Button } from "@/components/ui/button";
import {
  AlertCircle,
  CheckCircle2,
  Loader2,
  RefreshCw,
  Server,
} from "lucide-react";
import { motion } from "framer-motion";

interface ModelStatusProps {
  onTrainModel: () => void;
  isTraining: boolean;
}

export default function ModelStatus({
  onTrainModel,
  isTraining,
}: ModelStatusProps) {
  const [modelStatus, setModelStatus] = useState<{
    modelExists: boolean;
    featureColumnsExist: boolean;
    modelReady: boolean;
  } | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  const API_URL =
    process.env.NEXT_PUBLIC_API_URL || "http://localhost:5000/api";

  useEffect(() => {
    checkModelStatus();
  }, [isTraining]);

  const checkModelStatus = async () => {
    try {
      setLoading(true);
      setError(null);

      const response = await fetch(`${API_URL}/model-status`);

      if (!response.ok) {
        throw new Error("Failed to check model status");
      }

      const data = await response.json();
      setModelStatus(data);
    } catch (error) {
      console.error("Error checking model status:", error);
      setError("Could not connect to the backend server");
    } finally {
      setLoading(false);
    }
  };

  if (loading) {
    return (
      <motion.div
        initial={{ opacity: 0, y: 10 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.3 }}
      >
        <Alert className="bg-blue-50 dark:bg-blue-900/20 border-blue-200 dark:border-blue-800 shadow-md">
          <div className="flex items-center">
            <Loader2 className="h-5 w-5 text-blue-600 dark:text-blue-400 mr-2 animate-spin" />
            <div>
              <AlertTitle className="text-blue-800 dark:text-blue-300 font-medium">
                Checking Model Status
              </AlertTitle>
              <AlertDescription className="text-blue-700 dark:text-blue-400">
                Connecting to the backend server...
              </AlertDescription>
            </div>
          </div>
        </Alert>
      </motion.div>
    );
  }

  if (error) {
    return (
      <motion.div
        initial={{ opacity: 0, y: 10 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.3 }}
      >
        <Alert variant="destructive" className="shadow-md">
          <div className="flex items-center">
            <AlertCircle className="h-5 w-5 mr-2" />
            <div>
              <AlertTitle>Connection Error</AlertTitle>
              <AlertDescription className="flex items-center flex-wrap gap-2">
                {error}. Please make sure the backend server is running.
                <Button
                  variant="outline"
                  size="sm"
                  onClick={checkModelStatus}
                  className="ml-2"
                >
                  <RefreshCw className="h-3 w-3 mr-1" /> Retry
                </Button>
              </AlertDescription>
            </div>
          </div>
        </Alert>
      </motion.div>
    );
  }

  if (!modelStatus?.modelReady) {
    return (
      <motion.div
        initial={{ opacity: 0, y: 10 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.3 }}
      >
        <Alert variant="destructive" className="shadow-md">
          <div className="flex items-center">
            <Server className="h-5 w-5 mr-2" />
            <div>
              <AlertTitle>Model Not Found</AlertTitle>
              <AlertDescription className="flex items-center flex-wrap gap-2">
                The hepatitis prediction model is not ready. Please train the
                model first.
                <Button
                  className="ml-2 bg-blue-600 hover:bg-blue-700 text-white shadow-md"
                  size="sm"
                  onClick={onTrainModel}
                  disabled={isTraining}
                >
                  {isTraining ? (
                    <>
                      <Loader2 className="mr-2 h-4 w-4 animate-spin" />{" "}
                      Training...
                    </>
                  ) : (
                    <>Train Model</>
                  )}
                </Button>
              </AlertDescription>
            </div>
          </div>
        </Alert>
      </motion.div>
    );
  }

  return (
    <motion.div
      initial={{ opacity: 0, y: 10 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.3 }}
    >
      <Alert className="bg-green-50 dark:bg-green-900/20 border-green-200 dark:border-green-800 shadow-md">
        <div className="flex items-center">
          <CheckCircle2 className="h-5 w-5 text-green-600 dark:text-green-400 mr-2" />
          <div>
            <AlertTitle className="text-green-800 dark:text-green-300 font-medium">
              Model Ready
            </AlertTitle>
            <AlertDescription className="text-green-700 dark:text-green-400">
              The hepatitis prediction model is trained and ready to use.
            </AlertDescription>
          </div>
        </div>
      </Alert>
    </motion.div>
  );
}
