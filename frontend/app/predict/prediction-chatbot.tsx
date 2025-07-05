"use client"

import { useState, useRef, useEffect } from "react"
import { Card, CardContent, CardDescription, CardFooter, CardHeader, CardTitle } from "@/components/ui/card"
import { Button } from "@/components/ui/button"
import { Input } from "@/components/ui/input"
import { Avatar, AvatarFallback, AvatarImage } from "@/components/ui/avatar"
import { Send, RefreshCw, Download, Bot, User, Sparkles } from "lucide-react"
import { useToast } from "../hooks/use-toast"
import { motion, AnimatePresence } from "framer-motion"

type Message = {
  id: string
  content: string
  role: "user" | "assistant" | "system"
  timestamp: Date
}

type ChatbotState =
  | "greeting"
  | "asking_age"
  | "asking_gender"
  | "asking_symptoms"
  | "asking_jaundice"
  | "asking_dark_urine"
  | "asking_abdominal_pain"
  | "asking_fatigue"
  | "asking_fever"
  | "asking_risk_factors"
  | "confirming"
  | "predicting"
  | "result"
  | "followup"

type UserData = {
  age?: string
  gender?: string
  symptoms: {
    jaundice?: boolean
    darkUrine?: boolean
    abdominalPain?: number
    fatigue?: number
    fever?: boolean
    nausea?: boolean
    jointPain?: boolean
    appetite?: boolean
  }
  riskFactors: string[]
}

export default function PredictionChatbot() {
  const [messages, setMessages] = useState<Message[]>([])
  const [input, setInput] = useState("")
  const [loading, setLoading] = useState(false)
  const [chatState, setChatState] = useState<ChatbotState>("greeting")
  const [userData, setUserData] = useState<UserData>({
    symptoms: {},
    riskFactors: [],
  })
  const [prediction, setPrediction] = useState<string | null>(null)
  const messagesEndRef = useRef<HTMLDivElement>(null)
  const { toast } = useToast()
  const API_URL = process.env.NEXT_PUBLIC_API_URL || "http://localhost:5000/api"
  const [thinking, setThinking] = useState(false)

  // Initialize chat with greeting
  useEffect(() => {
    const initialMessage: Message = {
      id: "1",
      content:
        "Hello! I'm your HepaPredict assistant. I can help assess your risk of hepatitis based on your symptoms. Would you like to start?",
      role: "assistant",
      timestamp: new Date(),
    }
    setMessages([initialMessage])
  }, [])

  // Scroll to bottom of chat
  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" })
  }, [messages])

  const addMessage = (content: string, role: "user" | "assistant" | "system") => {
    const newMessage: Message = {
      id: Date.now().toString(),
      content,
      role,
      timestamp: new Date(),
    }
    setMessages((prev) => [...prev, newMessage])
  }

  const handleSendMessage = async () => {
    if (!input.trim()) return

    // Add user message
    addMessage(input, "user")

    // Clear input field
    setInput("")

    // Process user input based on current state
    await processUserInput(input)
  }

  const processUserInput = async (userInput: string) => {
    setLoading(true)
    setThinking(true)

    // Delay to simulate thinking
    await new Promise((resolve) => setTimeout(resolve, 800))
    setThinking(false)

    try {
      switch (chatState) {
        case "greeting":
          if (isAffirmative(userInput)) {
            addMessage(
              "Great! Let's start by getting some basic information. What is your age group? (Under 18, 18-30, 31-45, 46-60, or Over 60)",
              "assistant",
            )
            setChatState("asking_age")
          } else {
            addMessage(
              "No problem. If you change your mind, just let me know and we can start the assessment.",
              "assistant",
            )
          }
          break

        case "asking_age":
          const age = parseAgeGroup(userInput)
          if (age) {
            setUserData((prev) => ({
              ...prev,
              age,
            }))
            addMessage(`Thanks! And what is your gender? (Male, Female, or Other)`, "assistant")
            setChatState("asking_gender")
          } else {
            addMessage(
              "I didn't catch that. Please specify your age group: Under 18, 18-30, 31-45, 46-60, or Over 60.",
              "assistant",
            )
          }
          break

        case "asking_gender":
          const gender = parseGender(userInput)
          if (gender) {
            setUserData((prev) => ({ ...prev, gender }))
            addMessage(
              "Now, let's talk about your symptoms. Do you have yellowing of the skin or eyes (jaundice)? (Yes/No)",
              "assistant",
            )
            setChatState("asking_jaundice")
          } else {
            addMessage("I didn't catch that. Please specify your gender: Male, Female, or Other.", "assistant")
          }
          break

        case "asking_jaundice":
          if (isAffirmative(userInput)) {
            setUserData((prev) => ({
              ...prev,
              symptoms: { ...prev.symptoms, jaundice: true },
            }))
            addMessage("I see. Do you have dark urine? (Yes/No)", "assistant")
          } else if (isNegative(userInput)) {
            setUserData((prev) => ({
              ...prev,
              symptoms: { ...prev.symptoms, jaundice: false },
            }))
            addMessage("Okay. Do you have dark urine? (Yes/No)", "assistant")
          } else {
            addMessage(
              "Please answer with Yes or No. Do you have yellowing of the skin or eyes (jaundice)?",
              "assistant",
            )
            break
          }
          setChatState("asking_dark_urine")
          break

        case "asking_dark_urine":
          if (isAffirmative(userInput) || isNegative(userInput)) {
            setUserData((prev) => ({
              ...prev,
              symptoms: { ...prev.symptoms, darkUrine: isAffirmative(userInput) },
            }))
            addMessage("On a scale of 0-10, how would you rate your abdominal pain, if any?", "assistant")
            setChatState("asking_abdominal_pain")
          } else {
            addMessage("Please answer with Yes or No. Do you have dark urine?", "assistant")
          }
          break

        case "asking_abdominal_pain":
          const painLevel = Number.parseInt(userInput.match(/\d+/)?.[0] || "")
          if (!isNaN(painLevel) && painLevel >= 0 && painLevel <= 10) {
            setUserData((prev) => ({
              ...prev,
              symptoms: { ...prev.symptoms, abdominalPain: painLevel * 10 },
            }))
            addMessage("On a scale of 0-10, how would you rate your fatigue or tiredness?", "assistant")
            setChatState("asking_fatigue")
          } else {
            addMessage("Please provide a number between 0 and 10 for your abdominal pain level.", "assistant")
          }
          break

        case "asking_fatigue":
          const fatigueLevel = Number.parseInt(userInput.match(/\d+/)?.[0] || "")
          if (!isNaN(fatigueLevel) && fatigueLevel >= 0 && fatigueLevel <= 10) {
            setUserData((prev) => ({
              ...prev,
              symptoms: { ...prev.symptoms, fatigue: fatigueLevel * 10 },
            }))
            addMessage("Do you have a fever? (Yes/No)", "assistant")
            setChatState("asking_fever")
          } else {
            addMessage("Please provide a number between 0 and 10 for your fatigue level.", "assistant")
          }
          break

        case "asking_fever":
          if (isAffirmative(userInput) || isNegative(userInput)) {
            setUserData((prev) => ({
              ...prev,
              symptoms: { ...prev.symptoms, fever: isAffirmative(userInput) },
            }))
            addMessage(
              "Let's talk about risk factors. Have you traveled to areas with high hepatitis rates, had blood transfusions, or been exposed to contaminated needles? (Yes/No)",
              "assistant",
            )
            setChatState("asking_risk_factors")
          } else {
            addMessage("Please answer with Yes or No. Do you have a fever?", "assistant")
          }
          break

        case "asking_risk_factors":
          if (isAffirmative(userInput)) {
            setUserData((prev) => ({
              ...prev,
              riskFactors: ["exposureRisk"],
            }))
          }

          // Summarize the information
          const summary = generateSummary(userData)
          addMessage(
            `Thank you for providing this information. Here's what I understand:\n\n${summary}\n\nIs this correct? (Yes/No)`,
            "assistant",
          )
          setChatState("confirming")
          break

        case "confirming":
          if (isAffirmative(userInput)) {
            addMessage(
              "Great! I'll analyze this information to assess your hepatitis risk. One moment please...",
              "assistant",
            )
            setChatState("predicting")
            await makePrediction()
          } else {
            addMessage("I understand. Let's start over to make sure we get accurate information.", "assistant")
            resetChat()
          }
          break

        case "followup":
          if (isAffirmative(userInput)) {
            addMessage(
              "I recommend consulting with a healthcare professional for proper testing and diagnosis. They can provide personalized advice based on your specific situation.",
              "assistant",
            )
            addMessage("Is there anything else you'd like to know about hepatitis?", "assistant")
          } else {
            addMessage(
              "Thank you for using HepaPredict. If you have any more questions in the future, feel free to come back!",
              "assistant",
            )
          }
          break

        default:
          addMessage("I'm not sure how to respond to that. Let's continue with the assessment.", "assistant")
      }
    } catch (error) {
      console.error("Error processing message:", error)
      addMessage("I'm having trouble processing your response. Let's try again.", "assistant")
    } finally {
      setLoading(false)
    }
  }

  const makePrediction = async () => {
    try {
      setLoading(true)

      // Create form data to send to the API
      const formData = new FormData()
      formData.append("age", userData.age || "")
      formData.append("gender", userData.gender || "")
      formData.append(
        "symptoms",
        JSON.stringify({
          fatigue: userData.symptoms.fatigue || 0,
          nausea: userData.symptoms.nausea ? 70 : 0,
          abdominalPain: userData.symptoms.abdominalPain || 0,
          jaundice: userData.symptoms.jaundice || false,
          darkUrine: userData.symptoms.darkUrine || false,
          jointPain: userData.symptoms.jointPain || false,
          appetite: userData.symptoms.appetite || false,
          fever: userData.symptoms.fever ? 70 : 0,
        }),
      )
      formData.append("riskFactors", JSON.stringify(userData.riskFactors))

      // Simulate API call delay
      await new Promise((resolve) => setTimeout(resolve, 2000))

      // Mock response for demo purposes
      const mockResult =
        userData.symptoms.jaundice && userData.symptoms.darkUrine
          ? "hepatitisB"
          : (userData.symptoms.fatigue || 0) > 70
            ? "hepatitisC"
            : "unlikely"

      setPrediction(mockResult)

      // Display the result
      let resultMessage = ""
      if (mockResult === "hepatitisB") {
        resultMessage =
          "Based on my analysis of your symptoms, there are indicators that suggest a possibility of Hepatitis B. The key symptoms that contribute to this assessment include:"
        if (userData.symptoms.jaundice) resultMessage += "\n- Yellowing of skin/eyes (jaundice)"
        if (userData.symptoms.darkUrine) resultMessage += "\n- Dark urine"
        if ((userData.symptoms.abdominalPain || 0) > 50) resultMessage += "\n- Significant abdominal pain"
        if (userData.riskFactors.length > 0) resultMessage += "\n- Exposure to risk factors"

        resultMessage +=
          "\n\nIt's important to note that this is not a diagnosis, but rather an indication that you should consult with a healthcare provider for proper testing and evaluation."
      } else if (mockResult === "hepatitisC") {
        resultMessage =
          "Based on my analysis of your symptoms, there are indicators that suggest a possibility of Hepatitis C. The key symptoms that contribute to this assessment include:"
        if ((userData.symptoms.fatigue || 0) > 70) resultMessage += "\n- Significant fatigue"
        if (userData.symptoms.darkUrine) resultMessage += "\n- Dark urine"
        if (userData.riskFactors.length > 0) resultMessage += "\n- Exposure to risk factors"

        resultMessage +=
          "\n\nIt's important to note that this is not a diagnosis, but rather an indication that you should consult with a healthcare provider for proper testing and evaluation."
      } else {
        resultMessage =
          "Based on my analysis of your symptoms, your symptoms don't strongly indicate hepatitis. However, if you're experiencing persistent symptoms, it's always a good idea to consult with a healthcare provider."
      }

      addMessage(resultMessage, "assistant")
      addMessage("Would you like me to provide more information about next steps?", "assistant")
      setChatState("followup")
    } catch (error) {
      console.error("Error making prediction:", error)
      addMessage(
        "I'm sorry, I encountered an error while analyzing your symptoms. Please try again later.",
        "assistant",
      )
      setChatState("greeting")
    } finally {
      setLoading(false)
    }
  }

  const resetChat = () => {
    setUserData({
      symptoms: {},
      riskFactors: [],
    })
    setPrediction(null)
    setChatState("greeting")
    setMessages([
      {
        id: "reset",
        content:
          "Hello! I'm your HepaPredict assistant. I can help assess your risk of hepatitis based on your symptoms. Would you like to start?",
        role: "assistant",
        timestamp: new Date(),
      },
    ])
  }

  const downloadResults = () => {
    if (!prediction) return

    const resultText = `
HepaPredict Chatbot Results
---------------------------
Date: ${new Date().toLocaleString()}

Patient Information:
- Age Group: ${userData.age || "Not specified"}
- Gender: ${userData.gender || "Not specified"}

Symptoms Reported:
- Jaundice (yellowing of skin/eyes): ${userData.symptoms.jaundice ? "Yes" : "No"}
- Dark urine: ${userData.symptoms.darkUrine ? "Yes" : "No"}
- Abdominal pain level: ${userData.symptoms.abdominalPain ? (userData.symptoms.abdominalPain / 10) + "/10" : "Not reported"}
- Fatigue level: ${userData.symptoms.fatigue ? (userData.symptoms.fatigue / 10) + "/10" : "Not reported"}
- Fever: ${userData.symptoms.fever ? "Yes" : "No"}

Risk Factors:
${userData.riskFactors.length > 0 ? "- Exposure to risk factors reported" : "- No risk factors reported"}

Assessment Result:
${
  prediction === "hepatitisB"
    ? "Potential indicators of Hepatitis B detected"
    : prediction === "hepatitisC"
      ? "Potential indicators of Hepatitis C detected"
      : "Low risk of hepatitis based on reported symptoms"
}

Important Note:
This assessment is not a medical diagnosis. It's based on the information provided and is intended to guide your next steps. 
For accurate diagnosis, please consult with a healthcare professional.
    `

    const blob = new Blob([resultText], { type: "text/plain" })
    const url = URL.createObjectURL(blob)
    const a = document.createElement("a")
    a.href = url
    a.download = "HepaPredict_Chatbot_Results.txt"
    document.body.appendChild(a)
    a.click()
    document.body.removeChild(a)
    URL.revokeObjectURL(url)

    toast({
      title: "Results Downloaded",
      description: "Your assessment results have been downloaded as a text file.",
    })
  }

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
        <div className="inline-flex items-center gap-2 px-4 py-2 rounded-full bg-gradient-to-r from-blue-500 via-blue-400 to-indigo-400 text-white shadow-lg mb-4 animate-gradient-x">
          <Sparkles className="h-4 w-4" />
          <span className="text-sm font-medium">AI-Powered Assessment</span>
        </div>
        <h2 className="text-4xl font-bold text-slate-900 dark:text-white mb-4 drop-shadow-lg">Chat with HepaPredict</h2>
        <p className="text-lg text-slate-600 dark:text-slate-300 max-w-2xl mx-auto">
          Our AI assistant will guide you through a series of questions to assess your hepatitis risk based on your
          symptoms.
        </p>
      </motion.div>

      <motion.div
        className="max-w-4xl mx-auto relative z-10"
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.5, delay: 0.2 }}
      >
        <Card className="border-0 shadow-2xl overflow-hidden bg-white/70 dark:bg-slate-900/80 backdrop-blur-xl rounded-3xl">
          <CardHeader className="bg-gradient-to-r from-blue-500 via-blue-600 to-indigo-500 border-b-0 flex flex-row items-center gap-4 p-6">
            <Avatar className="h-12 w-12 shadow-lg">
              <AvatarImage src="/placeholder.svg?height=40&width=40" />
              <AvatarFallback className="bg-blue-600 text-white">
                <Bot className="h-6 w-6" />
              </AvatarFallback>
            </Avatar>
            <div>
              <CardTitle className="text-white drop-shadow-lg">HepaPredict Assistant</CardTitle>
              <CardDescription className="text-blue-100">Chat with our AI to assess your hepatitis risk</CardDescription>
            </div>
          </CardHeader>
          <CardContent className="p-0">
            <div className="h-[450px] overflow-y-auto p-6 bg-gradient-to-b from-slate-50/80 to-white/80 dark:from-slate-900/80 dark:to-slate-800/80">
              <AnimatePresence initial={false}>
                {messages.map((message) => (
                  <motion.div
                    key={message.id}
                    className={`mb-4 flex ${message.role === "user" ? "justify-end" : "justify-start"}`}
                    initial={{ opacity: 0, y: 10, scale: 0.95 }}
                    animate={{ opacity: 1, y: 0, scale: 1 }}
                    transition={{ duration: 0.3 }}
                  >
                    <div
                      className={`flex gap-3 max-w-[85%] ${message.role === "user" ? "flex-row-reverse" : "flex-row"}`}
                    >
                      <Avatar className="h-10 w-10 flex-shrink-0 mt-1 shadow-md">
                        {message.role === "user" ? (
                          <AvatarFallback className="bg-blue-600 text-white">
                            <User className="h-5 w-5" />
                          </AvatarFallback>
                        ) : (
                          <AvatarFallback className="bg-slate-200 dark:bg-slate-700 text-slate-700 dark:text-slate-200">
                            <Bot className="h-5 w-5" />
                          </AvatarFallback>
                        )}
                      </Avatar>
                      <div
                        className={`rounded-2xl p-4 shadow-md transition-all duration-200 ${
                          message.role === "user"
                            ? "bg-gradient-to-r from-blue-600 via-blue-500 to-indigo-500 text-white"
                            : "bg-white/90 dark:bg-slate-800/90 border border-slate-200 dark:border-slate-700"
                        }`}
                      >
                        {message.content.split("\n").map((line, i) => (
                          <p key={i} className={i > 0 ? "mt-2" : ""}>
                            {line}
                          </p>
                        ))}
                        <div
                          className={`text-xs mt-2 ${
                            message.role === "user" ? "text-blue-100" : "text-slate-400 dark:text-slate-500"
                          }`}
                        >
                          {message.timestamp.toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" })}
                        </div>
                      </div>
                    </div>
                  </motion.div>
                ))}
              </AnimatePresence>
              {thinking && (
                <motion.div
                  className="flex justify-start mb-4"
                  initial={{ opacity: 0, y: 10, scale: 0.95 }}
                  animate={{ opacity: 1, y: 0, scale: 1 }}
                  exit={{ opacity: 0, scale: 0.95 }}
                  transition={{ duration: 0.3 }}
                >
                  <div className="flex gap-3">
                    <Avatar className="h-10 w-10 flex-shrink-0 mt-1 shadow-md">
                      <AvatarFallback className="bg-slate-200 dark:bg-slate-700 text-slate-700 dark:text-slate-200">
                        <Bot className="h-5 w-5" />
                      </AvatarFallback>
                    </Avatar>
                    <div className="rounded-2xl p-4 bg-white/90 dark:bg-slate-800/90 shadow-md border border-slate-200 dark:border-slate-700 flex items-center">
                      <div className="flex space-x-1">
                        <div
                          className="w-2 h-2 bg-blue-600 dark:bg-blue-400 rounded-full animate-bounce"
                          style={{ animationDelay: "0ms" }}
                        ></div>
                        <div
                          className="w-2 h-2 bg-blue-600 dark:bg-blue-400 rounded-full animate-bounce"
                          style={{ animationDelay: "150ms" }}
                        ></div>
                        <div
                          className="w-2 h-2 bg-blue-600 dark:bg-blue-400 rounded-full animate-bounce"
                          style={{ animationDelay: "300ms" }}
                        ></div>
                      </div>
                    </div>
                  </div>
                </motion.div>
              )}
              <div ref={messagesEndRef} />
            </div>
            <div className="p-4 border-t-0 bg-gradient-to-r from-blue-500 via-blue-600 to-indigo-500">
              <div className="flex gap-2">
                <Input
                  placeholder="Type your message..."
                  value={input}
                  onChange={(e) => setInput(e.target.value)}
                  onKeyDown={(e) => {
                    if (e.key === "Enter" && !e.shiftKey) {
                      e.preventDefault()
                      handleSendMessage()
                    }
                  }}
                  disabled={loading}
                  className="border-slate-200 dark:border-slate-700 focus-visible:ring-blue-500 bg-white/80 dark:bg-slate-900/80"
                />
                <Button
                  onClick={handleSendMessage}
                  disabled={loading || !input.trim()}
                  className="bg-gradient-to-r from-blue-600 via-blue-500 to-indigo-500 hover:from-blue-700 hover:to-indigo-700 text-white shadow-md transition-all duration-300 hover:-translate-y-0.5"
                >
                  <Send className="h-4 w-4" />
                </Button>
              </div>
            </div>
          </CardContent>
          <CardFooter className="flex justify-between p-4 bg-gradient-to-r from-blue-500 via-blue-600 to-indigo-500 border-t-0">
            <Button
              variant="outline"
              onClick={resetChat}
              className="border-blue-200 dark:border-blue-800 text-white hover:bg-blue-50/20 dark:hover:bg-blue-900/20 hover:text-blue-700 dark:hover:text-blue-300"
            >
              <RefreshCw className="h-4 w-4 mr-2" /> Reset Chat
            </Button>
            {prediction && (
              <Button
                onClick={downloadResults}
                className="bg-white text-blue-600 hover:bg-blue-50 shadow-md transition-all duration-300 hover:-translate-y-0.5"
              >
                <Download className="h-4 w-4 mr-2" /> Download Results
              </Button>
            )}
          </CardFooter>
        </Card>
      </motion.div>
    </div>
  )
}

// Helper functions
function isAffirmative(input: string): boolean {
  const affirmativeTerms = ["yes", "yeah", "yep", "sure", "ok", "okay", "y", "yup", "correct", "right", "true"]
  return affirmativeTerms.some((term) => input.toLowerCase().includes(term))
}

function isNegative(input: string): boolean {
  const negativeTerms = ["no", "nope", "nah", "n", "not", "incorrect", "wrong", "false"]
  return negativeTerms.some((term) => input.toLowerCase().includes(term))
}

function parseAgeGroup(input: string): string | null {
  const lowerInput = input.toLowerCase()

  if (lowerInput.includes("under 18") || lowerInput.includes("<18") || lowerInput.match(/\b(0|1[0-7])\b/)) {
    return "under18"
  } else if (lowerInput.includes("18-30") || lowerInput.match(/\b(1[8-9]|2[0-9]|30)\b/)) {
    return "18-30"
  } else if (lowerInput.includes("31-45") || lowerInput.match(/\b(3[1-9]|4[0-5])\b/)) {
    return "31-45"
  } else if (lowerInput.includes("46-60") || lowerInput.match(/\b(4[6-9]|5[0-9]|60)\b/)) {
    return "46-60"
  } else if (
    lowerInput.includes("over 60") ||
    lowerInput.includes(">60") ||
    lowerInput.match(/\b(6[1-9]|[7-9][0-9]|[1-9][0-9]{2})\b/)
  ) {
    return "over60"
  }

  return null
}

function parseGender(input: string): string | null {
  const lowerInput = input.toLowerCase()

  if (lowerInput.includes("male") || lowerInput === "m") {
    return "male"
  } else if (lowerInput.includes("female") || lowerInput === "f") {
    return "female"
  } else if (lowerInput.includes("other") || lowerInput.includes("non-binary") || lowerInput.includes("nonbinary")) {
    return "other"
  }

  return null
}

function generateSummary(userData: UserData): string {
  let summary = ""

  if (userData.age) {
    summary += `Age Group: ${userData.age.replace(/([A-Z])/g, " $1").trim()}\n`
  }

  if (userData.gender) {
    summary += `Gender: ${userData.gender.charAt(0).toUpperCase() + userData.gender.slice(1)}\n`
  }

  summary += "\nSymptoms:\n"
  if (userData.symptoms.jaundice !== undefined) {
    summary += `- Jaundice (yellowing of skin/eyes): ${userData.symptoms.jaundice ? "Yes" : "No"}\n`
  }
  if (userData.symptoms.darkUrine !== undefined) {
    summary += `- Dark urine: ${userData.symptoms.darkUrine ? "Yes" : "No"}\n`
  }
  if (userData.symptoms.abdominalPain !== undefined) {
    summary += `- Abdominal pain: ${userData.symptoms.abdominalPain / 10}/10\n`
  }
  if (userData.symptoms.fatigue !== undefined) {
    summary += `- Fatigue: ${userData.symptoms.fatigue / 10}/10\n`
  }
  if (userData.symptoms.fever !== undefined) {
    summary += `- Fever: ${userData.symptoms.fever ? "Yes" : "No"}\n`
  }

  summary += "\nRisk Factors:\n"
  if (userData.riskFactors.length > 0) {
    summary += "- Exposure to risk factors reported\n"
  } else {
    summary += "- No risk factors reported\n"
  }

  return summary
}
