"use client"

import { useState, useRef, useEffect } from "react"
import { Card, CardContent, CardDescription, CardFooter, CardHeader, CardTitle } from "@/components/ui/card"
import { Button } from "@/components/ui/button"
import { Input } from "@/components/ui/input"
import { Avatar, AvatarFallback, AvatarImage } from "@/components/ui/avatar"
import { Loader2, Send, RefreshCw, Download } from "lucide-react"
import { useToast } from "../hooks/use-toast"

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
  | "asking_hepatitis_type" // Added state for asking hepatitis type
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
  hepatitisType?: string // Added for direct selection
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

  // Initialize chat with greeting
  useEffect(() => {
    const initialMessage: Message = {
      id: "1",
      content:
        "Hello! I'm your HepaPredict assistant. I can help assess your hepatitis type based on your symptoms. Would you like to start?",
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

    // Delay to simulate thinking
    await new Promise((resolve) => setTimeout(resolve, 500))

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
            setUserData((prev) => ({ ...prev, age }))
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

          // Now ask for hepatitis type directly
          addMessage(
            "Based on your knowledge or previous diagnosis, which type of hepatitis do you think you might have? (Hepatitis B, Hepatitis C, or Not Sure/None)",
            "assistant",
          )
          setChatState("asking_hepatitis_type")
          break

        case "asking_hepatitis_type":
          const hepatitisType = parseHepatitisType(userInput)
          if (hepatitisType) {
            setUserData((prev) => ({ ...prev, hepatitisType }))

            // Summarize the information
            const summary = generateSummary(userData)
            addMessage(
              `Thank you for providing this information. Here's what I understand:\n\n${summary}\n\nHepatitis Type: ${formatHepatitisType(hepatitisType)}\n\nIs this correct? (Yes/No)`,
              "assistant",
            )
            setChatState("confirming")
          } else {
            addMessage(
              "I didn't catch that. Please specify the hepatitis type: Hepatitis B, Hepatitis C, or Not Sure/None.",
              "assistant",
            )
          }
          break

        case "confirming":
          if (isAffirmative(userInput)) {
            addMessage("Great! I'll process this information. One moment please...", "assistant")
            setChatState("predicting")
            await processResult()
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

  const processResult = async () => {
    try {
      setLoading(true)

      // Use the directly selected hepatitis type
      const selectedType = userData.hepatitisType || "unlikely"
      setPrediction(selectedType)

      // Display the result based on the selected type
      let resultMessage = ""
      if (selectedType === "hepatitisB") {
        resultMessage =
          "Based on your selection, we'll provide information about Hepatitis B. Here are some key points about Hepatitis B:"
        resultMessage += "\n\n- Hepatitis B is caused by the Hepatitis B virus (HBV)"
        resultMessage += "\n- It's transmitted through contact with infected blood, semen, and other body fluids"
        resultMessage += "\n- It can cause both acute and chronic infection"
        resultMessage += "\n- Symptoms may include fatigue, jaundice, abdominal pain, and dark urine"
        resultMessage += "\n- A vaccine is available to prevent Hepatitis B"

        if (userData.symptoms.jaundice)
          resultMessage += "\n\nYou mentioned having jaundice, which is a common symptom of Hepatitis B."
        if (userData.symptoms.darkUrine)
          resultMessage += "\n\nYou mentioned having dark urine, which is often seen in Hepatitis B cases."

        resultMessage += "\n\nIt's important to consult with a healthcare provider for proper testing and evaluation."
      } else if (selectedType === "hepatitisC") {
        resultMessage =
          "Based on your selection, we'll provide information about Hepatitis C. Here are some key points about Hepatitis C:"
        resultMessage += "\n\n- Hepatitis C is caused by the Hepatitis C virus (HCV)"
        resultMessage += "\n- It's primarily transmitted through contact with infected blood"
        resultMessage += "\n- It often becomes chronic and can lead to serious liver problems if untreated"
        resultMessage += "\n- Many people with Hepatitis C don't have symptoms until liver damage occurs"
        resultMessage += "\n- Effective treatments are available that can cure Hepatitis C in most cases"

        if ((userData.symptoms.fatigue || 0) > 70)
          resultMessage += "\n\nYou mentioned experiencing significant fatigue, which is common in Hepatitis C."

        resultMessage += "\n\nIt's important to consult with a healthcare provider for proper testing and evaluation."
      } else {
        resultMessage =
          "Based on your selection, you don't believe you have hepatitis or you're not sure. This is good to know, but it's still important to be aware of hepatitis symptoms and risk factors."
        resultMessage += "\n\nIf you're experiencing persistent symptoms like:"
        resultMessage += "\n- Fatigue"
        resultMessage += "\n- Jaundice (yellowing of skin/eyes)"
        resultMessage += "\n- Abdominal pain"
        resultMessage += "\n- Dark urine"
        resultMessage +=
          "\n\nIt's always a good idea to consult with a healthcare provider, even if you don't think it's hepatitis."
      }

      addMessage(resultMessage, "assistant")
      addMessage("Would you like me to provide more information about next steps?", "assistant")
      setChatState("followup")
    } catch (error) {
      console.error("Error processing result:", error)
      addMessage(
        "I'm sorry, I encountered an error while processing your information. Please try again later.",
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
          "Hello! I'm your HepaPredict assistant. I can help assess your hepatitis type based on your symptoms. Would you like to start?",
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

Selected Hepatitis Type:
${formatHepatitisType(prediction)}

Important Note:
This assessment is based on the information you provided. For accurate diagnosis, please consult with a healthcare professional.
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
    <Card className="w-full max-w-4xl mx-auto">
      <CardHeader>
        <CardTitle className="flex items-center">
          <Avatar className="h-8 w-8 mr-2">
            <AvatarImage src="/placeholder.svg?height=40&width=40" />
            <AvatarFallback className="bg-primary text-primary-foreground">HP</AvatarFallback>
          </Avatar>
          HepaPredict Chatbot
        </CardTitle>
        <CardDescription>Chat with our AI assistant to learn about hepatitis based on your symptoms</CardDescription>
      </CardHeader>
      <CardContent>
        <div className="h-[400px] overflow-y-auto border rounded-md p-4 mb-4">
          {messages.map((message) => (
            <div key={message.id} className={`mb-4 flex ${message.role === "user" ? "justify-end" : "justify-start"}`}>
              <div
                className={`max-w-[80%] rounded-lg p-3 ${
                  message.role === "user" ? "bg-primary text-primary-foreground" : "bg-muted"
                }`}
              >
                {message.content.split("\n").map((line, i) => (
                  <p key={i} className={i > 0 ? "mt-2" : ""}>
                    {line}
                  </p>
                ))}
                <div className="text-xs mt-1 opacity-70">
                  {message.timestamp.toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" })}
                </div>
              </div>
            </div>
          ))}
          {loading && (
            <div className="flex justify-start mb-4">
              <div className="bg-muted rounded-lg p-3 flex items-center">
                <Loader2 className="h-4 w-4 animate-spin mr-2" />
                <span>Thinking...</span>
              </div>
            </div>
          )}
          <div ref={messagesEndRef} />
        </div>
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
          />
          <Button onClick={handleSendMessage} disabled={loading || !input.trim()}>
            <Send className="h-4 w-4" />
          </Button>
        </div>
      </CardContent>
      <CardFooter className="flex justify-between">
        <Button variant="outline" onClick={resetChat}>
          <RefreshCw className="h-4 w-4 mr-2" /> Reset Chat
        </Button>
        {prediction && (
          <Button onClick={downloadResults}>
            <Download className="h-4 w-4 mr-2" /> Download Results
          </Button>
        )}
      </CardFooter>
    </Card>
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

function parseHepatitisType(input: string): string | null {
  const lowerInput = input.toLowerCase()

  if (lowerInput.includes("b") || lowerInput.includes("hbv")) {
    return "hepatitisB"
  } else if (lowerInput.includes("c") || lowerInput.includes("hcv")) {
    return "hepatitisC"
  } else if (
    lowerInput.includes("no") ||
    lowerInput.includes("none") ||
    lowerInput.includes("not") ||
    lowerInput.includes("unsure") ||
    lowerInput.includes("don't know")
  ) {
    return "unlikely"
  }

  return null
}

function formatHepatitisType(type: string): string {
  if (type === "hepatitisB") return "Hepatitis B"
  if (type === "hepatitisC") return "Hepatitis C"
  return "No Hepatitis / Unsure"
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
