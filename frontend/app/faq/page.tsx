"use client";

import { HelpCircle, Sparkles, Shield, Activity, Brain } from "lucide-react";
import {
  Accordion,
  AccordionContent,
  AccordionItem,
  AccordionTrigger,
} from "@/components/ui/accordion";
import { motion } from "framer-motion";

const FaqPage = () => {
  const faqCategories = [
    {
      category: "Ku Saabsan Cagaarshowga",
      icon: <Shield className="h-6 w-6 text-blue-600 dark:text-blue-400" />,
      questions: [
        {
          question: "Maxay tahay Cagaarshowga?",
          answer:
            "Cagaarshowgu waa xanuun beerka ku dhaca kaasoo keena bararka beerka. Waxa keena fayruusyo kala duwan, khamriga xad-dhaafka ah, daawooyin, ama waxyaabo kale oo sun ah. Noocyada ugu muhiimsan ee aan ku saabsannahay waa cagaarshowga fayruuska ah A iyo C, kuwaas oo midkasta leeyahay sababihiisa gaarka ah.",
        },
        {
          question: "Sidee ayey u kala duwan yihiin noocyada Cagaarshowga?",
          answer:
            "Cagaarshowga A: Ku faafa cuntada ama biyaha wasakhaysan, waa mid ku meel gaar ah oo badanaa is-bogsiiya. Cagaarshowga C: Inta badan ku faafa dhiigga, badanaa waa mid joogto ah, wuxuuna keeni karaa dhaawac beerka ah haddii aan la daaweyn. Labadan nooc ayaa ah kuwa ugu muhiimsan ee aan baaritaanka ku sameynno.",
        },
        {
          question: "Ma la daaweyn karaa Cagaarshowga?",
          answer:
            "Haa, laakiin daawadu way ku xiran tahay nooca. Cagaarshowga A wuu is-daaweeyaa oo ma u baahna daaweyn gaar ah. Cagaarshowga C hadda si buuxda ayaa loo daaweyn karaa inta badan kiisaska, waxaana la isticmaalaa daawooyin casri ah oo la qaato 8-12 toddobaad.",
        },
      ],
    },
    {
      category: "Astaamaha & Baaritaanka",
      icon: <Activity className="h-6 w-6 text-blue-600 dark:text-blue-400" />,
      questions: [
        {
          question: "Waa maxay astaamaha ugu caansan ee Cagaarshowga?",
          answer:
            "Astaamaha ugu muhiimsan waxaa ka mid ah: daal joogto ah, jaalloonaanta maqaarka iyo indhaha, xanuun caloosha ah, rabitaan la'aanta cuntada, lallabo iyo matag, kaadi madow, iyo saxaro midabkeedu cad yahay. Waa muhiim in la ogaado in qaar ka mid ah dadka qaba cagaarshowga aysan muujin astaamo bilowga xanuunka.",
        },
        {
          question: "Sidee loo ogaadaa Cagaarshowga?",
          answer:
            "Baaritaanka waxaa ka mid ah: baaritaanka dhiigga si loo hubiyo shaqada beerka, baaritaanka unugyada difaaca jirka, iyo baaritaanka antigen-ka fayruuska. Baaritaanno dheeraad ah sida ultrasound-ka beerka, fibroscan, ama biopsy ayaa loo isticmaali karaa si loo qiimeeyo xaaladda beerka.",
        },
        {
          question: "Sidee u shaqeeyaa nidaamka baaritaanka HepaPredict?",
          answer:
            "HepaPredict wuxuu adeegsadaa habab casri ah oo ku salaysan barashada mashiinka (AI) si loo qiimeeyo astaamahaaga. Nidaamkan wuxuu ku saleeyaa natiijooyinkiisa xog ballaaran oo caafimaad, hase yeeshee waa in la fahmo in uusan ahayn beddelka baaritaanka caafimaad ee xirfadlayaasha.",
        },
      ],
    },
    {
      category: "Ka Hortagga & Daaweynta",
      icon: <Sparkles className="h-6 w-6 text-blue-600 dark:text-blue-400" />,
      questions: [
        {
          question: "Sidee looga hortagi karaa Cagaarshowga?",
          answer:
            "Ka hortagga wuxuu ku xiran yahay nooca: Cagaarshowga A & E: Ku dhaqan nadaafad wanaagsan, iska ilaali cunto iyo biyo wasakhaysan. Cagaarshowga B: Qaado tallaalka. Cagaarshowga C: Ka fogow wadaagista waxyaabaha gaarka ah sida cirbadaha. Dhammaan noocyada: Ka fogow khamriga xad-dhaafka ah, raac nidaam cunto caafimaad leh.",
        },
        {
          question: "Ma jiraan tallaalada Cagaarshowga?",
          answer:
            "Haa, waxaa jira tallaalada ammaan ah ee Cagaarshowga A iyo B. Tallaalka Cagaarshowga B waxaa lagu taliyaa in la siiyo dhammaan carruurta iyo dadka waaweyn ee halista ku jira. Weli ma jiraan tallaalada Cagaarshowga C, D, ama E, laakiin cilmi-baarisyo ayaa socda.",
        },
        {
          question:
            "Maxay tahay habka ugu wanaagsan ee loo daaweeyo Cagaarshowga?",
          answer:
            "Daaweyntu waxay ku xiran tahay nooca iyo darnaanta xanuunka. Cagaarshowga xad-jirka ah wuxuu u baahan yahay nasasho iyo la-socod caafimaad. Cagaarshowga joogtada ah wuxuu u baahan yahay daaweyn dheer iyo daawooyin gaar ah. Waa muhiim in la raaco tilmaamaha dhakhtarka oo si joogto ah loo booqdo xarunta caafimaadka.",
        },
      ],
    },
    {
      category: "Isticmaalka HepaPredict",
      icon: <Brain className="h-6 w-6 text-blue-600 dark:text-blue-400" />,
      questions: [
        {
          question: "Sidee u shaqeeyaa barnaamijka HepaPredict?",
          answer:
            "HepaPredict waa nidaam casri ah oo isticmaala teknoolajiyada AI-ga si loo qiimeeyo astaamahaaga. Wuxuu si taxadar leh u falanqeeyaa xogtaada, isbarbardhigaa xogta caafimaad ee la hubiyay, kadibna wuxuu bixiyaa tilmaamo ku saabsan suurtagalnimada Cagaarshowga.",
        },
        {
          question:
            "Ma ammaan baa xogtayda marka aan isticmaalayo HepaPredict?",
          answer:
            "Haa, waxaan si gaar ah muhiimad u siinaa ilaalinta xogta adeegsadayaasha. Dhammaan xogta la geliyo waa la siriyaa, lama kaydiyo, lamana wadaago cid kale. Barnaamijku ma uruuriyo wax macluumaad shaqsi ah oo lagu aqoonsan karo qofka.",
        },
        {
          question:
            "Maxaan sameeyaa kadib marka aan helo natiijooyinka HepaPredict?",
          answer:
            "Iyadoo aan loo eegin natiijooyinka, haddii aad dareemayso astaamo walaac leh, waa muhiim inaad la tashatid dhakhtar. HepaPredict waa qalab caawiya wacyigelinta, ma aha mid beddela baaritaanka caafimaad. Baaritaan buuxa wuxuu u baahan yahay qiimayn xirfadle caafimaad.",
        },
      ],
    },
  ];

  const containerVariants = {
    hidden: { opacity: 0 },
    visible: {
      opacity: 1,
      transition: {
        staggerChildren: 0.1,
      },
    },
  };

  const itemVariants = {
    hidden: { opacity: 0, y: 20 },
    visible: {
      opacity: 1,
      y: 0,
      transition: {
        duration: 0.5,
      },
    },
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-white via-blue-50/5 to-white dark:from-slate-950 dark:via-blue-900/5 dark:to-slate-950">
        <section className="py-20 px-6 relative overflow-hidden">
          {/* Background Elements */}
          <div className="absolute inset-0 overflow-hidden">
            <div className="absolute top-40 -right-20 w-80 h-80 bg-blue-500/10 rounded-full blur-3xl" />
            <div className="absolute -bottom-20 -left-20 w-80 h-80 bg-indigo-500/10 rounded-full blur-3xl" />
          </div>

          <motion.div
            className="max-w-6xl mx-auto text-center relative z-10"
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.6 }}
          >
            <div className="inline-flex items-center gap-2 px-4 py-2 rounded-full bg-blue-500/10 text-blue-600 dark:text-blue-400 mb-8">
              <HelpCircle className="h-4 w-4" />
              <span className="text-sm font-medium">Su'aalaha & Jawaabaha</span>
            </div>
            <h1 className="text-4xl md:text-5xl font-bold text-slate-900 dark:text-white mb-6">
              Su'aalaha Inta Badan{" "}
              <span className="text-blue-600 dark:text-blue-400">
                La Is-weydiiyo
              </span>
            </h1>
            <p className="text-lg text-slate-600 dark:text-slate-300 max-w-3xl mx-auto mb-8">
              Halkaan ka hel jawaabaha su'aalaha muhiimka ah ee ku saabsan
              cagaarshowga iyo sida loo isticmaalo nidaamka HepaPredict.
            </p>
          </motion.div>
        </section>

        <section className="py-16 px-6">
          <motion.div
            className="max-w-4xl mx-auto"
            variants={containerVariants}
            initial="hidden"
            animate="visible"
          >
            {faqCategories.map((category, categoryIndex) => (
              <motion.div
                key={categoryIndex}
                className="mb-12"
                variants={itemVariants}
              >
                <div className="flex items-center gap-3 mb-6">
                  <div className="p-2 bg-blue-100 dark:bg-blue-900/50 rounded-xl">
                    {category.icon}
                  </div>
                  <h2 className="text-2xl font-bold text-slate-900 dark:text-white">
                    {category.category}
                  </h2>
                </div>
                <Accordion
                  type="single"
                  collapsible
                  className="border border-blue-200 dark:border-blue-900 rounded-2xl overflow-hidden bg-white/50 dark:bg-slate-900/50 backdrop-blur-sm"
                >
                  {category.questions.map((faq, faqIndex) => (
                    <AccordionItem
                      key={faqIndex}
                      value={`item-${categoryIndex}-${faqIndex}`}
                      className="border-b border-blue-200 dark:border-blue-900 last:border-b-0"
                    >
                      <AccordionTrigger className="px-6 py-4 text-left text-slate-900 dark:text-white hover:text-blue-600 dark:hover:text-blue-400 hover:bg-blue-50/50 dark:hover:bg-blue-900/20 transition-all duration-300">
                        {faq.question}
                      </AccordionTrigger>
                      <AccordionContent className="px-6 py-4 text-slate-600 dark:text-slate-300">
                        {faq.answer}
                      </AccordionContent>
                    </AccordionItem>
                  ))}
                </Accordion>
              </motion.div>
            ))}
          </motion.div>
        </section>
    </div>
  );
};

export default FaqPage;
