"use client"
import Header from "../common/header"
import Footer from "../common/footer"
import { Avatar, AvatarFallback, AvatarImage } from "@/components/ui/avatar"
import { Users, Award, Heart, HeartPulse, Microscope } from "lucide-react"

const AboutPage = () => {
  const teamMembers = [
    {
      name: "Dr. Amina Hassan",
      role: "Medical Director",
      bio: "Hepatologist with 15 years of experience in liver disease research and treatment.",
      image: "/placeholder.svg?height=200&width=200",
      initials: "AH",
    },
    {
      name: "Dr. Mohamed Abdi",
      role: "Research Lead",
      bio: "Specializes in viral hepatitis epidemiology and public health interventions.",
      image: "/placeholder.svg?height=200&width=200",
      initials: "MA",
    },
    {
      name: "Fartun Omar",
      role: "Data Scientist",
      bio: "Expert in machine learning algorithms for medical diagnostics and prediction models.",
      image: "/placeholder.svg?height=200&width=200",
      initials: "FO",
    },
    {
      name: "Ahmed Jama",
      role: "Patient Advocate",
      bio: "Former hepatitis patient dedicated to improving awareness and support resources.",
      image: "/placeholder.svg?height=200&width=200",
      initials: "AJ",
    },
    {
      name: "Dr. Hodan Farah",
      role: "Clinical Researcher",
      bio: "Focused on developing new treatment protocols and clinical trials for hepatitis.",
      image: "/placeholder.svg?height=200&width=200",
      initials: "HF",
    },
    {
      name: "Abdikarim Mohamud",
      role: "Technology Director",
      bio: "Leads our digital health initiatives and ensures data security and accessibility.",
      image: "/placeholder.svg?height=200&width=200",
      initials: "AM",
    },
  ]

  return (
    <div className="min-h-screen flex flex-col">
      <Header />
      <main className="flex-grow">
        <section className="py-16 px-6 bg-gradient-to-b from-hepa-lightTeal/30 to-white">
          <div className="max-w-6xl mx-auto text-center">
            <div className="inline-block p-3 bg-white rounded-full mb-6">
              <Users className="h-8 w-8 text-hepa-teal" />
            </div>
            <h1 className="text-4xl md:text-5xl font-bold text-hepa-navy mb-6">Our Team</h1>
            <p className="text-lg text-gray-600 max-w-3xl mx-auto mb-8">
              We're a dedicated group of Somali medical professionals, researchers, and patient advocates working
              together to improve hepatitis awareness, prediction, and care.
            </p>
          </div>
        </section>

        <section className="py-16 px-6">
          <div className="max-w-6xl mx-auto">
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-8">
              {teamMembers.map((member, index) => (
                <div
                  key={index}
                  className="feature-card animate-fade-in flex flex-col items-center text-center"
                  style={{ animationDelay: `${index * 0.1}s` }}
                >
                  <Avatar className="h-24 w-24 mb-4 border-2 border-hepa-teal">
                    <AvatarImage src={member.image || "/placeholder.svg"} alt={member.name} />
                    <AvatarFallback className="bg-hepa-teal text-white text-xl">{member.initials}</AvatarFallback>
                  </Avatar>
                  <h3 className="text-xl font-semibold text-hepa-navy mb-2">{member.name}</h3>
                  <p className="text-hepa-teal font-medium mb-3">{member.role}</p>
                  <p className="text-gray-600">{member.bio}</p>
                </div>
              ))}
            </div>
          </div>
        </section>

        <section className="py-16 px-6 bg-gradient-to-b from-white to-hepa-lightTeal/30">
          <div className="max-w-6xl mx-auto">
            <div className="text-center mb-12">
              <div className="inline-block p-3 bg-white rounded-full mb-6">
                <Award className="h-8 w-8 text-hepa-teal" />
              </div>
              <h2 className="text-3xl md:text-4xl font-bold text-hepa-navy mb-6">Our Mission</h2>
              <p className="text-lg text-gray-600 max-w-3xl mx-auto">
                We're committed to providing accessible tools and information that help identify potential hepatitis
                cases early and connect people with the care they need.
              </p>
            </div>

            <div className="grid grid-cols-1 md:grid-cols-3 gap-8">
              <div className="feature-card animate-fade-in" style={{ animationDelay: "0.1s" }}>
                <div className="flex items-center mb-4">
                  <div className="p-3 bg-hepa-lightTeal/50 rounded-full mr-4">
                    <Heart className="h-6 w-6 text-hepa-coral" />
                  </div>
                  <h3 className="text-xl font-semibold text-hepa-navy">Care</h3>
                </div>
                <p className="text-gray-600">
                  Supporting patients through their diagnosis journey with compassion and understanding.
                </p>
              </div>

              <div className="feature-card animate-fade-in" style={{ animationDelay: "0.2s" }}>
                <div className="flex items-center mb-4">
                  <div className="p-3 bg-hepa-lightTeal/50 rounded-full mr-4">
                    <HeartPulse className="h-6 w-6 text-hepa-coral" />
                  </div>
                  <h3 className="text-xl font-semibold text-hepa-navy">Prevention</h3>
                </div>
                <p className="text-gray-600">
                  Promoting awareness and education to prevent the spread of hepatitis in communities.
                </p>
              </div>

              <div className="feature-card animate-fade-in" style={{ animationDelay: "0.3s" }}>
                <div className="flex items-center mb-4">
                  <div className="p-3 bg-hepa-lightTeal/50 rounded-full mr-4">
                    <Microscope className="h-6 w-6 text-hepa-coral" />
                  </div>
                  <h3 className="text-xl font-semibold text-hepa-navy">Research</h3>
                </div>
                <p className="text-gray-600">
                  Advancing scientific understanding of hepatitis through rigorous research and data analysis.
                </p>
              </div>
            </div>
          </div>
        </section>
      </main>
      <Footer />
    </div>
  )
}

export default AboutPage
