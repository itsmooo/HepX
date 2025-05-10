import { Activity, Heart } from "lucide-react"
import Link from "next/link"

const Footer = () => {
  const currentYear = new Date().getFullYear()

  return (
    <footer className="bg-foreground text-background py-10 px-6 sm:px-10">
      <div className="max-w-6xl mx-auto">
        <div className="grid grid-cols-1 md:grid-cols-4 gap-8">
          <div className="md:col-span-1">
            <div className="flex items-center space-x-2 mb-4">
              <Activity className="h-6 w-6 text-primary" />
              <h2 className="text-xl font-bold">
                <span className="text-primary">Hepa</span>Predict
              </h2>
            </div>
            <p className="text-background/80 text-sm mb-4">
              A friendly tool to help identify potential hepatitis types based on your symptoms.
            </p>
            <div className="flex items-center text-sm text-background/80">
              <Heart className="h-4 w-4 mr-2 text-red-400" />
              <span>Made with care for your health</span>
            </div>
          </div>

          <div>
            <h3 className="font-semibold mb-4 text-primary">Quick Links</h3>
            <ul className="space-y-2 text-sm text-background/80">
              <li>
                <Link href="/" className="hover:text-primary transition-colors">
                  Home
                </Link>
              </li>
              <li>
                <a href="#about" className="hover:text-primary transition-colors">
                  About
                </a>
              </li>
              <li>
                <a href="#symptoms" className="hover:text-primary transition-colors">
                  Symptoms
                </a>
              </li>
              <li>
                <a href="#predict" className="hover:text-primary transition-colors">
                  Predict
                </a>
              </li>
              <li>
                <a href="#education" className="hover:text-primary transition-colors">
                  Education
                </a>
              </li>
            </ul>
          </div>

          <div>
            <h3 className="font-semibold mb-4 text-primary">Resources</h3>
            <ul className="space-y-2 text-sm text-background/80">
              <li>
                <Link href="#" className="hover:text-primary transition-colors">
                  Hepatitis Information
                </Link>
              </li>
              <li>
                <Link href="#" className="hover:text-primary transition-colors">
                  Prevention Tips
                </Link>
              </li>
              <li>
                <Link href="#" className="hover:text-primary transition-colors">
                  Finding Care
                </Link>
              </li>
              <li>
                <Link href="/faq" className="hover:text-primary transition-colors">
                  FAQ
                </Link>
              </li>
              <li>
                <Link href="#" className="hover:text-primary transition-colors">
                  Medical Disclaimer
                </Link>
              </li>
            </ul>
          </div>

          <div>
            <h3 className="font-semibold mb-4 text-primary">Important</h3>
            <p className="text-sm text-background/80 mb-4">
              HepaPredict is not a medical diagnosis tool. Always consult with a healthcare professional for proper
              diagnosis and treatment.
            </p>
            <p className="text-sm text-background/80">
              If you're experiencing severe symptoms, please seek immediate medical attention.
            </p>
          </div>
        </div>

        <div className="border-t border-background/20 mt-8 pt-6 flex flex-col md:flex-row justify-between items-center">
          <p className="text-sm text-background/60 mb-4 md:mb-0">
            &copy; {currentYear} HepaPredict. All rights reserved.
          </p>
          <div className="flex space-x-4">
            <Link href="#" className="text-sm text-background/60 hover:text-primary transition-colors">
              Privacy Policy
            </Link>
            <Link href="#" className="text-sm text-background/60 hover:text-primary transition-colors">
              Terms of Service
            </Link>
            <Link href="#" className="text-sm text-background/60 hover:text-primary transition-colors">
              Contact
            </Link>
          </div>
        </div>
      </div>
    </footer>
  )
}

export default Footer
