"use client"
import { Button } from "@/components/ui/button"
import { HeartPulse, ShieldCheck } from "lucide-react"

const Hero = () => {
  const scrollToPredict = () => {
    const predictSection = document.getElementById("predict")
    if (predictSection) {
      predictSection.scrollIntoView({ behavior: "smooth" })
    }
  }

  return (
    <section className="py-20 px-6 sm:px-10 mb-10">
      <div className="max-w-6xl mx-auto">
        <div className="flex flex-col md:flex-row items-center">
          <div className="md:w-1/2 mb-10 md:mb-0 md:pr-10 animate-fade-in">
            <h1 className="text-4xl md:text-5xl font-bold text-hepa-navy mb-6 leading-tight">
              Your Friendly Hepatitis Prediction Tool
            </h1>
            <p className="text-lg text-gray-600 mb-8">
              Wondering if your symptoms might be related to Hepatitis? HepaPredict helps identify whether you may have
              Hepatitis B or C based on your symptoms - all in a simple, caring way.
            </p>
            <div className="flex flex-col sm:flex-row gap-4">
              <Button onClick={scrollToPredict} className="hepa-button bg-hepa-teal hover:bg-hepa-navy">
                Check Symptoms
              </Button>
              <Button variant="outline" className="border-hepa-teal text-hepa-teal hover:bg-hepa-teal hover:text-white">
                Learn More
              </Button>
            </div>
          </div>
          <div className="md:w-1/2 relative animate-fade-in" style={{ animationDelay: "0.2s" }}>
            <div className="relative bg-gradient-to-br from-hepa-lightTeal/30 to-hepa-teal/30 rounded-2xl p-10 shadow-lg">
              <div className="absolute -top-4 -left-4 bg-white rounded-lg shadow-md p-4">
                <HeartPulse className="h-8 w-8 text-hepa-coral" />
              </div>
              <div className="absolute -bottom-4 -right-4 bg-white rounded-lg shadow-md p-4">
                <ShieldCheck className="h-8 w-8 text-hepa-teal" />
              </div>
              <div className="space-y-6">
                <div className="hepa-card">
                  <h3 className="text-xl font-semibold text-hepa-navy mb-2">Fast & Caring</h3>
                  <p className="text-gray-600">Get insights based on real medical data without stress or waiting.</p>
                </div>
                <div className="hepa-card">
                  <h3 className="text-xl font-semibold text-hepa-navy mb-2">Not a Diagnosis</h3>
                  <p className="text-gray-600">A smart first step toward proper medical care and attention.</p>
                </div>
                <div className="hepa-card">
                  <h3 className="text-xl font-semibold text-hepa-navy mb-2">Educational</h3>
                  <p className="text-gray-600">Learn about different types of hepatitis while you use the tool.</p>
                </div>
              </div>
            </div>
          </div>
        </div>
      </div>
    </section>
  )
}

export default Hero
