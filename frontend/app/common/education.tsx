import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { Card, CardContent } from "@/components/ui/card";
import {
  ShieldCheck,
  AlertCircle,
  BookOpen,
  Pill,
  HeartPulse,
  Bug,
} from "lucide-react";

const Education = () => {
  return (
    <section
      id="education"
      className="py-16 px-6 sm:px-10 bg-gradient-to-b from-white to-hepa-lightTeal/10"
    >
      <div className="max-w-5xl mx-auto">
        <div className="text-center mb-10">
          <h2 className="text-3xl font-bold text-hepa-navy mb-3">
            Learn About Hepatitis
          </h2>
          <p className="text-lg text-gray-600 max-w-2xl mx-auto">
            Understanding hepatitis types, symptoms, and treatments can help you
            make better healthcare decisions.
          </p>
        </div>

        <Tabs defaultValue="what-is-hepatitis">
          <div className="flex justify-center mb-8">
            <TabsList className="grid grid-cols-2 md:grid-cols-3 bg-hepa-lightTeal/30">
              <TabsTrigger
                value="what-is-hepatitis"
                className="data-[state=active]:bg-white"
              >
                What is Hepatitis?
              </TabsTrigger>
              <TabsTrigger
                value="hepatitis-b"
                className="data-[state=active]:bg-white"
              >
                Hepatitis B
              </TabsTrigger>
              <TabsTrigger
                value="hepatitis-c"
                className="data-[state=active]:bg-white"
              >
                Hepatitis C
              </TabsTrigger>
            </TabsList>
          </div>

          <TabsContent value="what-is-hepatitis">
            <Card>
              <CardContent className="p-6">
                <div className="grid grid-cols-1 md:grid-cols-2 gap-10">
                  <div>
                    <div className="flex items-center gap-3 mb-4">
                      <Bug className="h-6 w-6 text-hepa-teal" />
                      <h3 className="text-xl font-semibold">
                        Understanding Hepatitis
                      </h3>
                    </div>
                    <p className="text-gray-700 mb-4">
                      Hepatitis is an inflammation of the liver. It can be
                      caused by various factors, including viral infections,
                      alcohol consumption, certain medications, and toxins.
                    </p>
                    <p className="text-gray-700 mb-4">
                      The liver is vital for digesting food, filtering toxins
                      from the blood, and storing energy. When it becomes
                      inflamed, these functions may be affected.
                    </p>
                    <div className="flex items-center gap-3 mb-4 mt-6">
                      <AlertCircle className="h-6 w-6 text-hepa-amber" />
                      <h4 className="text-lg font-semibold">Common Symptoms</h4>
                    </div>
                    <ul className="list-disc list-inside space-y-2 text-gray-700">
                      <li>Fatigue and general weakness</li>
                      <li>Jaundice (yellowing of skin and eyes)</li>
                      <li>Abdominal pain, especially in the liver area</li>
                      <li>Nausea and vomiting</li>
                      <li>Dark urine and pale stool</li>
                      <li>Loss of appetite</li>
                    </ul>
                  </div>

                  <div>
                    <div className="flex items-center gap-3 mb-4">
                      <BookOpen className="h-6 w-6 text-hepa-teal" />
                      <h3 className="text-xl font-semibold">
                        Types of Viral Hepatitis
                      </h3>
                    </div>
                    <div className="space-y-5">
                      <div className="p-4 rounded-lg bg-hepa-lightTeal/20">
                        <h4 className="font-semibold mb-2">Hepatitis A</h4>
                        <p className="text-gray-700 text-sm">
                          Typically spread through contaminated food or water.
                          Usually acute and resolves without treatment.
                        </p>
                      </div>

                      <div className="p-4 rounded-lg bg-hepa-lightTeal/20">
                        <h4 className="font-semibold mb-2">Hepatitis B</h4>
                        <p className="text-gray-700 text-sm">
                          Spread through blood, semen, and other body fluids.
                          Can be both acute and chronic.
                        </p>
                      </div>

                      <div className="p-4 rounded-lg bg-hepa-lightTeal/20">
                        <h4 className="font-semibold mb-2">Hepatitis C</h4>
                        <p className="text-gray-700 text-sm">
                          Primarily spread through contact with infected blood.
                          Often becomes chronic.
                        </p>
                      </div>

                      <div className="p-4 rounded-lg bg-hepa-lightTeal/20">
                        <h4 className="font-semibold mb-2">Hepatitis D & E</h4>
                        <p className="text-gray-700 text-sm">
                          Less common types. Hepatitis D only occurs with
                          Hepatitis B infection. Hepatitis E is typically spread
                          through contaminated water.
                        </p>
                      </div>
                    </div>
                  </div>
                </div>
              </CardContent>
            </Card>
          </TabsContent>

          <TabsContent value="hepatitis-b">
            <Card>
              <CardContent className="p-6">
                <div className="grid grid-cols-1 md:grid-cols-2 gap-10">
                  <div>
                    <div className="flex items-center gap-3 mb-4">
                      <Bug className="h-6 w-6 text-hepa-teal" />
                      <h3 className="text-xl font-semibold">
                        Hepatitis B Overview
                      </h3>
                    </div>
                    <p className="text-gray-700 mb-4">
                      Hepatitis B is a viral infection that attacks the liver
                      and can cause both acute and chronic disease. It is
                      transmitted through contact with the blood or other body
                      fluids of an infected person.
                    </p>
                    <p className="text-gray-700 mb-4">
                      Hepatitis B is a major global health problem and can cause
                      chronic infection, leading to a high risk of death from
                      cirrhosis and liver cancer.
                    </p>

                    <div className="flex items-center gap-3 mb-4 mt-6">
                      <AlertCircle className="h-6 w-6 text-hepa-amber" />
                      <h4 className="text-lg font-semibold">
                        Specific Symptoms
                      </h4>
                    </div>
                    <ul className="list-disc list-inside space-y-2 text-gray-700">
                      <li>Jaundice (often more pronounced)</li>
                      <li>Fatigue that may last for weeks or months</li>
                      <li>
                        Abdominal pain, especially in the right upper quadrant
                      </li>
                      <li>Joint pain (may be more common in Hepatitis B)</li>
                      <li>Fever and chills</li>
                    </ul>
                  </div>

                  <div>
                    <div className="flex items-center gap-3 mb-4">
                      <ShieldCheck className="h-6 w-6 text-hepa-teal" />
                      <h3 className="text-xl font-semibold">
                        Prevention & Treatment
                      </h3>
                    </div>

                    <div className="p-4 rounded-lg bg-hepa-lightTeal/20 mb-5">
                      <h4 className="font-semibold mb-2">Prevention</h4>
                      <ul className="list-disc list-inside space-y-2 text-gray-700 text-sm">
                        <li>Hepatitis B vaccine is highly effective</li>
                        <li>Avoid sharing needles or personal items</li>
                        <li>Practice safe sex</li>
                        <li>Be cautious about body piercing and tattoos</li>
                      </ul>
                    </div>

                    <div className="flex items-center gap-3 mb-4">
                      <Pill className="h-6 w-6 text-hepa-teal" />
                      <h3 className="text-lg font-semibold">
                        Treatment Options
                      </h3>
                    </div>

                    <div className="space-y-4">
                      <p className="text-gray-700">
                        Acute Hepatitis B usually doesn't require specific
                        treatment except for supportive care. For chronic
                        Hepatitis B, treatments include:
                      </p>

                      <div className="p-4 rounded-lg bg-gray-50">
                        <h5 className="font-semibold mb-2 text-sm">
                          Antiviral Medications
                        </h5>
                        <p className="text-gray-700 text-sm">
                          Medications like entecavir and tenofovir can help
                          fight the virus and slow liver damage.
                        </p>
                      </div>

                      <div className="p-4 rounded-lg bg-gray-50">
                        <h5 className="font-semibold mb-2 text-sm">
                          Immune System Modulators
                        </h5>
                        <p className="text-gray-700 text-sm">
                          Peginterferon alfa-2a boosts your immune system to
                          fight the virus.
                        </p>
                      </div>

                      <div className="p-4 rounded-lg bg-gray-50">
                        <h5 className="font-semibold mb-2 text-sm">
                          Regular Monitoring
                        </h5>
                        <p className="text-gray-700 text-sm">
                          Regular liver function tests and possibly liver
                          ultrasounds are important.
                        </p>
                      </div>
                    </div>
                  </div>
                </div>
              </CardContent>
            </Card>
          </TabsContent>

          <TabsContent value="hepatitis-c">
            <Card>
              <CardContent className="p-6">
                <div className="grid grid-cols-1 md:grid-cols-2 gap-10">
                  <div>
                    <div className="flex items-center gap-3 mb-4">
                      <Bug className="h-6 w-6 text-hepa-teal" />
                      <h3 className="text-xl font-semibold">
                        Hepatitis C Overview
                      </h3>
                    </div>
                    <p className="text-gray-700 mb-4">
                      Hepatitis C is a viral infection caused by the Hepatitis C
                      virus (HCV) that primarily affects the liver. It is often
                      referred to as a "silent epidemic" because many people
                      don't know they're infected.
                    </p>
                    <p className="text-gray-700 mb-4">
                      Hepatitis C is primarily spread through contact with blood
                      from an infected person. It can cause both acute and
                      chronic hepatitis, ranging in severity from a mild illness
                      lasting a few weeks to a serious, lifelong condition.
                    </p>

                    <div className="flex items-center gap-3 mb-4 mt-6">
                      <AlertCircle className="h-6 w-6 text-hepa-amber" />
                      <h4 className="text-lg font-semibold">
                        Specific Symptoms
                      </h4>
                    </div>
                    <p className="text-gray-700 mb-3">
                      Many people with Hepatitis C don't experience symptoms
                      until liver damage has occurred, which may be many years
                      after infection. When symptoms do appear, they may
                      include:
                    </p>
                    <ul className="list-disc list-inside space-y-2 text-gray-700">
                      <li>Fatigue (often the most prominent symptom)</li>
                      <li>Mild abdominal pain</li>
                      <li>Loss of appetite</li>
                      <li>Dark urine</li>
                      <li>Jaundice (less common than in Hepatitis B)</li>
                    </ul>
                  </div>

                  <div>
                    <div className="flex items-center gap-3 mb-4">
                      <ShieldCheck className="h-6 w-6 text-hepa-teal" />
                      <h3 className="text-xl font-semibold">
                        Prevention & Treatment
                      </h3>
                    </div>

                    <div className="p-4 rounded-lg bg-hepa-lightTeal/20 mb-5">
                      <h4 className="font-semibold mb-2">Prevention</h4>
                      <ul className="list-disc list-inside space-y-2 text-gray-700 text-sm">
                        <li>No vaccine is available for Hepatitis C</li>
                        <li>Avoid sharing needles or personal items</li>
                        <li>
                          Ensure proper sterilization of medical equipment
                        </li>
                        <li>
                          Practice safe sex, especially if you have multiple
                          partners
                        </li>
                      </ul>
                    </div>

                    <div className="flex items-center gap-3 mb-4">
                      <HeartPulse className="h-6 w-6 text-hepa-teal" />
                      <h3 className="text-lg font-semibold">
                        Treatment Success
                      </h3>
                    </div>

                    <p className="text-gray-700 mb-4">
                      Unlike Hepatitis B, Hepatitis C is usually curable with
                      antiviral medications. Current treatments are highly
                      effective, with cure rates of over 95%.
                    </p>

                    <div className="space-y-4">
                      <div className="p-4 rounded-lg bg-gray-50">
                        <h5 className="font-semibold mb-2 text-sm">
                          Direct-Acting Antivirals (DAAs)
                        </h5>
                        <p className="text-gray-700 text-sm">
                          Modern treatments like sofosbuvir, ledipasvir, and
                          others can cure Hepatitis C in 8-12 weeks with minimal
                          side effects.
                        </p>
                      </div>

                      <div className="p-4 rounded-lg bg-gray-50">
                        <h5 className="font-semibold mb-2 text-sm">
                          Regular Monitoring
                        </h5>
                        <p className="text-gray-700 text-sm">
                          During and after treatment, your healthcare provider
                          will likely monitor your viral load and liver
                          function.
                        </p>
                      </div>

                      <div className="p-4 rounded-lg bg-gray-50">
                        <h5 className="font-semibold mb-2 text-sm">
                          Lifestyle Changes
                        </h5>
                        <p className="text-gray-700 text-sm">
                          Avoiding alcohol, maintaining a healthy diet, and
                          getting regular exercise can help support liver
                          health.
                        </p>
                      </div>
                    </div>
                  </div>
                </div>
              </CardContent>
            </Card>
          </TabsContent>
        </Tabs>
      </div>
    </section>
  );
};

export default Education;
