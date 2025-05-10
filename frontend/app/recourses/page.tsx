"use client";
import Header from "@/components/header";
import Footer from "@/components/footer";
import { Button } from "@/components/ui/button";
import {
  FileText,
  ExternalLink,
  BookOpen,
  Download,
  Phone,
  Globe,
  HelpCircle,
} from "lucide-react";

const ResourcesPage = () => {
  const resources = [
    {
      title: "Understanding Hepatitis B",
      type: "Guide",
      description:
        "A comprehensive guide to Hepatitis B causes, symptoms, and treatment options.",
      icon: FileText,
      url: "#",
      button: "Download PDF",
    },
    {
      title: "Hepatitis C Treatment Advances",
      type: "Article",
      description:
        "Learn about the latest treatment advances for Hepatitis C and success rates.",
      icon: BookOpen,
      url: "#",
      button: "Read Article",
    },
    {
      title: "Prevention Guidelines",
      type: "Checklist",
      description:
        "Step-by-step preventive measures to reduce your risk of contracting viral hepatitis.",
      icon: FileText,
      url: "#",
      button: "Download Checklist",
    },
    {
      title: "Living With Hepatitis",
      type: "Support Guide",
      description:
        "Practical advice for managing daily life after a hepatitis diagnosis.",
      icon: BookOpen,
      url: "#",
      button: "View Guide",
    },
    {
      title: "Nutrition for Liver Health",
      type: "Dietary Guide",
      description:
        "Recommended diet plans and nutritional guidance for supporting liver function.",
      icon: FileText,
      url: "#",
      button: "Download Guide",
    },
    {
      title: "Hepatitis Myths & Facts",
      type: "Fact Sheet",
      description:
        "Addressing common misconceptions and providing evidence-based facts about hepatitis.",
      icon: HelpCircle,
      url: "#",
      button: "Read Facts",
    },
  ];

  const organizations = [
    {
      name: "World Health Organization (WHO)",
      description:
        "Access global health guidelines and hepatitis information resources.",
      url: "https://www.who.int/health-topics/hepatitis",
      icon: Globe,
    },
    {
      name: "Centers for Disease Control (CDC)",
      description:
        "U.S. guidance on hepatitis prevention, testing, and treatment.",
      url: "https://www.cdc.gov/hepatitis/index.htm",
      icon: Globe,
    },
    {
      name: "National Hepatitis Hotline",
      description:
        "Get answers to your questions from trained health professionals.",
      phone: "1-800-HEP-ABC1",
      icon: Phone,
    },
    {
      name: "Hepatitis B Foundation",
      description: "Support and advocacy for people affected by hepatitis B.",
      url: "https://www.hepb.org/",
      icon: Globe,
    },
  ];

  return (
    <div className="min-h-screen flex flex-col">
      <Header />
      <main className="flex-grow">
        <section className="py-16 px-6 bg-gradient-to-b from-hepa-lightTeal/30 to-white">
          <div className="max-w-6xl mx-auto text-center">
            <div className="inline-block p-3 bg-white rounded-full mb-6">
              <BookOpen className="h-8 w-8 text-hepa-teal" />
            </div>
            <h1 className="text-4xl md:text-5xl font-bold text-hepa-navy mb-6">
              Hepatitis Resources
            </h1>
            <p className="text-lg text-gray-600 max-w-3xl mx-auto mb-8">
              Access reliable information, educational materials, and support
              resources to better understand and manage hepatitis.
            </p>
          </div>
        </section>

        <section className="py-16 px-6">
          <div className="max-w-6xl mx-auto">
            <h2 className="text-3xl font-bold text-hepa-navy mb-8">
              Educational Materials
            </h2>
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-8">
              {resources.map((resource, index) => (
                <div
                  key={index}
                  className="feature-card animate-fade-in flex flex-col h-full"
                  style={{ animationDelay: `${index * 0.1}s` }}
                >
                  <div className="flex items-start mb-4">
                    <div className="p-2 bg-hepa-lightTeal/50 rounded-full mr-3">
                      <resource.icon className="h-5 w-5 text-hepa-teal" />
                    </div>
                    <div>
                      <span className="text-sm text-hepa-teal font-medium">
                        {resource.type}
                      </span>
                      <h3 className="text-xl font-semibold text-hepa-navy">
                        {resource.title}
                      </h3>
                    </div>
                  </div>
                  <p className="text-gray-600 mb-6 flex-grow">
                    {resource.description}
                  </p>
                  <Button
                    variant="outline"
                    className="mt-auto border-hepa-teal text-hepa-teal hover:bg-hepa-teal hover:text-white"
                  >
                    {resource.button === "Download PDF" ||
                    resource.button === "Download Guide" ||
                    resource.button === "Download Checklist" ? (
                      <Download className="h-4 w-4 mr-2" />
                    ) : (
                      <ExternalLink className="h-4 w-4 mr-2" />
                    )}
                    {resource.button}
                  </Button>
                </div>
              ))}
            </div>
          </div>
        </section>

        <section className="py-16 px-6 bg-gradient-to-b from-white to-hepa-lightTeal/30">
          <div className="max-w-6xl mx-auto">
            <h2 className="text-3xl font-bold text-hepa-navy mb-8">
              Support Organizations
            </h2>
            <div className="grid grid-cols-1 md:grid-cols-2 gap-8">
              {organizations.map((org, index) => (
                <div
                  key={index}
                  className="feature-card animate-fade-in"
                  style={{ animationDelay: `${index * 0.1}s` }}
                >
                  <div className="flex items-start">
                    <div className="p-3 bg-hepa-lightTeal/50 rounded-full mr-4 mt-1">
                      <org.icon className="h-6 w-6 text-hepa-teal" />
                    </div>
                    <div>
                      <h3 className="text-xl font-semibold text-hepa-navy mb-2">
                        {org.name}
                      </h3>
                      <p className="text-gray-600 mb-4">{org.description}</p>
                      {org.url ? (
                        <a
                          href={org.url}
                          target="_blank"
                          rel="noopener noreferrer"
                          className="inline-flex items-center text-hepa-teal hover:text-hepa-navy transition-colors"
                        >
                          <Globe className="h-4 w-4 mr-2" />
                          Visit Website
                          <ExternalLink className="h-3 w-3 ml-1" />
                        </a>
                      ) : (
                        <div className="inline-flex items-center text-hepa-teal">
                          <Phone className="h-4 w-4 mr-2" />
                          {org.phone}
                        </div>
                      )}
                    </div>
                  </div>
                </div>
              ))}
            </div>
          </div>
        </section>
      </main>
      <Footer />
    </div>
  );
};

export default ResourcesPage;
